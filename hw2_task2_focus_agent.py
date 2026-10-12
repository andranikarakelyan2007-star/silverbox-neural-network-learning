"""hw2_task2_focus_agent.py

Homework 1 / Task 6 multi-agent system (orchestrator, scientist, general, executor)
with a NEW TOOL for the executor: `focus_beam` -> runs the DMD focusing search on the simulated
bench (dmd_bench.py) and returns the enhancement and the camera images.

The user only asks "focus the beam". After that the agent works alone:
  orchestrator -> scientist (plan) -> executor (LLM chooses the settings, calls the tool) -> orchestrator -> ...
The executor chooses macropixel size, region sizes, number of iterations, ... by itself.
Every tool result is kept in the state and written to results/experiments.md / .csv.

GOOGLE COLAB USAGE
------------------
1. Secrets (key icon) -> add GEMINI_API_KEY, enable notebook access (or type it when asked).
2. Upload dmd_bench.py and this file to /content.
3. In a cell:   %run hw2_task2_focus_agent.py
   (optional: LECTURE_ENHANCEMENT=<number> in the environment to print the comparison with the lecture)
"""

import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict


def _ensure(mod: str, pkg: str) -> None:
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure("langgraph", "langgraph")
_ensure("langchain_google_genai", "langchain-google-genai")

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import dmd_bench  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
RESULTS_DIR = Path("./results").resolve()
MAX_DECISIONS = 16
MIN_EXPERIMENTS = 5          # the orchestrator may finish only after this many successful tool runs
MAX_EXPERIMENTS = 10
MAX_TOOL_ROUNDS = 3          # tool-calling rounds inside one executor turn
LLM = None


def get_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                key = userdata.get(name)
                if key:
                    return key
            except Exception:
                pass
    except ImportError:
        pass
    import getpass

    return getpass.getpass("Gemini API key: ").strip()


def get_llm():
    return LLM


def text_of(resp) -> str:
    c = resp.content
    if isinstance(c, str):
        return c
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)


def invoke_retry(runnable, arg, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            return runnable.invoke(arg)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[llm] call failed ({e.__class__.__name__}), retry {i + 1}/{tries}")
            time.sleep(3 * (i + 1))
    raise last


# ──────────────────────────────────────────────────────────────────────────────
# 2. THE TOOL: focus_beam  (only the executor calls it)
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: list = []       # every successful tool run (also mirrored into the graph state)


class FocusArgs(BaseModel):
    macropixel_size: int = Field(
        description=f"Side of one macropixel in DMD micromirrors. One of {dmd_bench.VALID_MACRO}. "
        f"The DMD has {dmd_bench.DMD_N}x{dmd_bench.DMD_N} micromirrors, so size m gives (64/m)^2 macropixels."
    )
    focus_radius: float = Field(
        description="Radius of the circular focus region on the camera, in camera pixels (0 = a single pixel). "
        "The camera is 48x48 pixels."
    )
    bg_gap: float = Field(description="Gap between the focus region and the background ring, in pixels (>= 0).")
    bg_width: float = Field(description="Width of the proximal background ring, in pixels (> 0).")
    n_bg_regions: int = Field(description="Number of regions (ring sectors) the background is split into, 1..12.")
    n_iterations: int = Field(description="Number of search iterations (1..200000). Each iteration is 2 camera shots.")
    flips_per_iteration: int = Field(description="Number of macropixels flipped in each iteration (>= 1).")
    seed: int = Field(default=0, description="Random seed of the search (start pattern, flips, camera noise).")


def _focus_beam(macropixel_size: int, focus_radius: float, bg_gap: float, bg_width: float,
                n_bg_regions: int, n_iterations: int, flips_per_iteration: int, seed: int = 0) -> str:
    """Run the focusing search on the simulated DMD + scattering medium + camera bench."""
    run_no = len(EXPERIMENTS) + 1
    name = f"exp{run_no:02d}"
    try:
        res = dmd_bench.run_focusing(
            macropixel_size=macropixel_size, focus_radius=focus_radius, bg_gap=bg_gap, bg_width=bg_width,
            n_bg_regions=n_bg_regions, n_iterations=n_iterations, flips_per_iteration=flips_per_iteration,
            seed=seed, out_dir=str(RESULTS_DIR), run_name=name,
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"{e.__class__.__name__}: {e}"})
    settings = dict(macropixel_size=macropixel_size, focus_radius=focus_radius, bg_gap=bg_gap, bg_width=bg_width,
                    n_bg_regions=n_bg_regions, n_iterations=n_iterations,
                    flips_per_iteration=flips_per_iteration, seed=seed)
    rec = {"run": run_no, "name": name, **settings, **res}
    EXPERIMENTS.append(rec)
    return json.dumps({"run": run_no, **res})


FOCUS_TOOL = StructuredTool.from_function(
    func=_focus_beam, name="focus_beam", args_schema=FocusArgs,
    description="Run one beam-focusing experiment on the simulated bench (laser -> DMD -> scattering medium -> "
                "camera). Starts from a random binary DMD pattern, flips macropixels and keeps flips that raise "
                "the enhancement (focus intensity / background intensity). Returns the measured enhancement "
                "(averaged over 5 fresh camera shots), the enhancement of the random start pattern, and the path "
                "of the saved camera images.",
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. STATE
# ──────────────────────────────────────────────────────────────────────────────


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content, "timestamp": datetime.now().strftime("%H:%M:%S")}


def add_lists(existing: list, new_items: list) -> list:
    return existing + new_items


class GraphState(TypedDict, total=False):
    messages: Annotated[List[dict], add_lists]
    experiments: Annotated[List[dict], add_lists]
    decision_count: int
    next_role: str
    instruction: str


class OrchestratorDecision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"] = Field(description="Next destination.")
    instruction: str = Field(description="Instruction for the chosen role (words, not exact numbers, unless needed).")
    reason: str = Field(description="One-sentence reason.")


def table_text(exps: list) -> str:
    if not exps:
        return "(no experiments yet)"
    rows = ["run | macropx | focus_r | gap | width | bg_regions | iters | flips/it | enhancement | start_enh"]
    for e in exps:
        rows.append(
            f"{e['run']} | {e['macropixel_size']} | {e['focus_radius']} | {e['bg_gap']} | {e['bg_width']} | "
            f"{e['n_bg_regions']} | {e['n_iterations']} | {e['flips_per_iteration']} | {e['enhancement']} | "
            f"{e['enhancement_initial_random_pattern']}"
        )
    return "\n".join(rows)


def history_text(messages: List[dict], limit: int = 3000) -> str:
    out = []
    for m in messages:
        c = m["content"] if len(m["content"]) <= limit else m["content"][:limit] + "\n...[truncated]"
        out.append(f"[{m['role'].upper()}]: {c}")
    return "\n\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 4. NODES
# ──────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = f"""You are the Orchestrator of a multi-agent system that focuses a laser beam through a
scattering medium on a simulated optical bench (laser -> DMD -> scattering medium -> camera).
Choose the next destination STRICTLY from the dialogue so far:
- scientist: explains the physics and writes the search plan: what to vary and why, which settings to try first.
  No tools.
- general: answers high-level questions beside the search (what enhancement means, what a macropixel is, ...).
- executor: the ONLY role with the tool `focus_beam`. It chooses all settings itself (macropixel size, region sizes,
  number of iterations, ...), runs ONE experiment and reports the enhancement the tool returned.
- finish: stop and report the experiments.
Rules:
- Typical run: scientist first (a plan), then executor several times, each time asking for a different idea
  (e.g. coarser vs finer macropixels, smaller vs larger focus region, more iterations, more flips per iteration,
  different number of background regions), then finish.
- You do not run experiments and do not type numeric settings for the tool unless you must correct an error;
  describe the direction in words, the executor picks the numbers.
- Run at least {MIN_EXPERIMENTS} successful experiments before 'finish'; at most {MAX_EXPERIMENTS}.
- Use what the table shows: if one direction clearly helped, explore around it; if a run returned an error, ask the
  executor to fix the settings.
- Hard limit: {MAX_DECISIONS} decisions in total."""


def check_decision(d: OrchestratorDecision, n_exp: int):
    if d.next_role == "finish" and n_exp < MIN_EXPERIMENTS:
        return f"finish is allowed only after {MIN_EXPERIMENTS} successful experiments (now {n_exp})."
    if d.next_role == "executor" and n_exp >= MAX_EXPERIMENTS:
        return f"the experiment budget ({MAX_EXPERIMENTS}) is used up; choose finish."
    return None


def orchestrator_node(state: GraphState) -> dict:
    n = state.get("decision_count", 0)
    exps = state.get("experiments", [])
    n_exp = len(exps)
    has_plan = any(m["role"] == "scientist" for m in state["messages"])
    prompt = (
        f"{ORCHESTRATOR_PROMPT}\n\nDecisions used: {n}/{MAX_DECISIONS}\nSuccessful experiments: {n_exp}\n"
        f"Scientist plan already written: {has_plan}\n\nExperiment table so far:\n{table_text(exps)}\n\n"
        f"Dialogue:\n{history_text(state['messages'])}\n\nMake your decision:"
    )
    structured = get_llm().with_structured_output(OrchestratorDecision)
    d, feedback = None, ""
    for _ in range(3):
        try:
            cand = structured.invoke(prompt + feedback)
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] LLM call failed, retrying: {e}")
            time.sleep(3)
            continue
        if cand is None:
            continue
        problem = check_decision(cand, n_exp)
        if problem is None:
            d = cand
            break
        feedback = f"\n\nYour previous answer was rejected: {problem}\nDecide again."
    if d is None:  # safe fallback
        if not has_plan:
            d = OrchestratorDecision(next_role="scientist", instruction="Write the search plan.", reason="fallback")
        elif n_exp < MIN_EXPERIMENTS:
            d = OrchestratorDecision(next_role="executor", instruction="Run one more experiment with a new idea.",
                                     reason="fallback")
        else:
            d = OrchestratorDecision(next_role="finish", instruction="", reason="fallback")
    if n + 1 >= MAX_DECISIONS and d.next_role != "finish":
        d = OrchestratorDecision(next_role="finish", instruction="", reason="decision limit reached")
    content = f"NEXT ROLE: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
    return {"messages": [msg("orchestrator", content)], "decision_count": n + 1,
            "next_role": d.next_role, "instruction": d.instruction}


SCIENTIST_PROMPT = """You are the Scientist. Context: a DMD (digital micromirror device) modulates only the AMPLITUDE of
the laser field (binary macropixels on/off). Light then crosses a scattering medium with transmission matrix T and
the camera records I = |T E_in|^2 plus noise. Goal: a bright spot (focus) in a chosen camera region, standing out of
the speckle. Algorithm: split the DMD into macropixels, random start pattern, read the intensity in the focus region
and in the proximal background (split into several regions), enhancement = focus / background, flip some macropixels,
keep the change if the enhancement grew, repeat.
Write a short plan (max ~250 words): expected scaling of the enhancement with the number of macropixels N, the focus
size and noise; which settings to scan (macropixel size, focus radius, background ring gap/width, number of background
regions, iterations, flips per iteration) and in what order; what result would be good.
You have NO tool. Never claim a number was measured."""


def scientist_node(state: GraphState) -> dict:
    prompt = (f"{SCIENTIST_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\n"
              f"Available tool settings: macropixel_size in {dmd_bench.VALID_MACRO}, camera 48x48 px, "
              f"background must fit: focus_radius + bg_gap + bg_width <= 23.\n\nWrite the plan:")
    return {"messages": [msg("scientist", text_of(invoke_retry(get_llm(), prompt)))]}


GENERAL_PROMPT = """You are the General Agent. Answer high-level questions beside the search (what enhancement is, what
a macropixel or the transmission matrix is, why binary amplitude modulation gives a lower enhancement than phase
modulation, ...). No tools. Do not plan the search and do not run experiments. Be concise."""


def general_node(state: GraphState) -> dict:
    prompt = f"{GENERAL_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\nAnswer:"
    return {"messages": [msg("general", text_of(invoke_retry(get_llm(), prompt)))]}


EXECUTOR_PROMPT = f"""You are the Executor, the only role with the tool `focus_beam`.
You decide ALL settings of the experiment yourself (macropixel size, focus radius, background gap and width, number of
background regions, iterations, flips per iteration, seed) from the Orchestrator's instruction, the Scientist's plan and the
table of earlier experiments. Do not repeat a setting combination that is already in the table unless you change the seed
on purpose. Constraints: macropixel_size in {dmd_bench.VALID_MACRO}; camera 48x48 px;
focus_radius + bg_gap + bg_width <= 23; a run costs about 1 s per 3000 iterations, keep n_iterations <= 40000.
Call `focus_beam` exactly ONCE per turn. If it returns an error, fix the settings and call it again.
After the tool answers, write 2-3 sentences: the settings you chose, why, and the enhancement the tool returned.
Never state an enhancement the tool did not return."""


def executor_node(state: GraphState) -> dict:
    msgs = state["messages"]
    exps = state.get("experiments", [])
    plan = next((m["content"] for m in reversed(msgs) if m["role"] == "scientist"), "(no plan yet)")
    instr = state.get("instruction") or "Run one experiment."
    prompt = (f"{EXECUTOR_PROMPT}\n\n=== SCIENTIST PLAN ===\n{plan}\n\n=== EARLIER EXPERIMENTS ===\n{table_text(exps)}\n\n"
              f"=== ORCHESTRATOR INSTRUCTION ===\n{instr}\n")
    llm_tools = get_llm().bind_tools([FOCUS_TOOL])
    messages = [HumanMessage(content=prompt)]
    before = len(EXPERIMENTS)
    report, calls = [], []
    for _round in range(MAX_TOOL_ROUNDS + 1):
        ai = invoke_retry(llm_tools, messages)
        messages.append(ai)
        if not ai.tool_calls:
            if len(EXPERIMENTS) == before and _round < MAX_TOOL_ROUNDS:
                messages.append(HumanMessage(content="You must call the tool focus_beam now."))
                continue
            report.append(text_of(ai))
            break
        for tc in ai.tool_calls:
            if tc["name"] != "focus_beam":
                out = json.dumps({"error": f"unknown tool {tc['name']}"})
            elif len(EXPERIMENTS) > before:
                out = json.dumps({"error": "only one experiment per turn; stop calling the tool and summarise"})
            else:
                try:
                    out = FOCUS_TOOL.invoke(tc["args"])
                except Exception as e:  # noqa: BLE001
                    out = json.dumps({"error": f"{e.__class__.__name__}: {e}"})
            calls.append(f"focus_beam({json.dumps(tc['args'])}) -> {out}")
            messages.append(ToolMessage(content=out, tool_call_id=tc["id"], name=tc["name"]))
        if len(EXPERIMENTS) > before and _round >= MAX_TOOL_ROUNDS - 1:
            pass
    new = EXPERIMENTS[before:]
    text = "TOOL CALLS:\n" + ("\n".join(calls) if calls else "(none)") + "\n\n" + "\n".join(report)
    if new:
        e = new[-1]
        text += (f"\n\nEXPERIMENT {e['run']} KEPT: enhancement = {e['enhancement']} "
                 f"(random start {e['enhancement_initial_random_pattern']}); images: {e['image_path']}")
    else:
        text += "\n\nNO SUCCESSFUL EXPERIMENT in this turn."
    return {"messages": [msg("executor", text)], "experiments": new}


# ──────────────────────────────────────────────────────────────────────────────
# 5. GRAPH
# ──────────────────────────────────────────────────────────────────────────────


def after_specialist(state: GraphState) -> str:
    return "end" if state.get("decision_count", 0) >= MAX_DECISIONS else "orchestrator"


def build_app():
    b = StateGraph(GraphState)
    b.add_node("orchestrator", orchestrator_node)
    b.add_node("scientist", scientist_node)
    b.add_node("general", general_node)
    b.add_node("executor", executor_node)
    b.set_entry_point("orchestrator")
    b.add_conditional_edges("orchestrator", lambda s: s.get("next_role", "finish"),
                            {"scientist": "scientist", "general": "general", "executor": "executor", "finish": END})
    for node in ("scientist", "general", "executor"):
        b.add_conditional_edges(node, after_specialist, {"orchestrator": "orchestrator", "end": END})
    return b.compile(checkpointer=MemorySaver())


# ──────────────────────────────────────────────────────────────────────────────
# 6. RUNNER + REPORT FILES
# ──────────────────────────────────────────────────────────────────────────────

REQUEST = "Focus the beam."


def show_update(upd: dict) -> None:
    for node, delta in upd.items():
        if not isinstance(delta, dict):
            continue
        for m in delta.get("messages", []):
            print(f"\n--- [{m['role'].upper()}] {m['timestamp']} ---")
            body = m["content"]
            print(body[:1800] + ("\n...[full text in transcript.md]" if len(body) > 1800 else ""))


def write_reports(exps: list, final: dict, request: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    head = ("| Run | Macropixel (px) | N macropixels | Focus radius | BG gap | BG width | BG regions | Iterations | "
            "Flips/iter | Start enhancement | **Enhancement** | Image |\n|" + "---|" * 12)
    rows = [head]
    for e in exps:
        rows.append(
            f"| {e['run']} | {e['macropixel_size']} | {e['n_macropixels']} | {e['focus_radius']} | {e['bg_gap']} | "
            f"{e['bg_width']} | {e['n_bg_regions']} | {e['n_iterations']} | {e['flips_per_iteration']} | "
            f"{e['enhancement_initial_random_pattern']} | **{e['enhancement']}** | {Path(e['image_path']).name} |")
    (RESULTS_DIR / "experiments.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    import csv

    if exps:
        with open(RESULTS_DIR / "experiments.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(exps[0].keys()))
            w.writeheader()
            w.writerows(exps)
        (RESULTS_DIR / "experiments.json").write_text(json.dumps(exps, indent=2), encoding="utf-8")
    lines = [f"# Transcript\n", f"Request: {request}\n"]
    for i, m in enumerate(final.get("messages", []), 1):
        lines.append(f"## Turn {i} [{m['role'].upper()}] {m['timestamp']}\n\n{m['content']}\n")
    (RESULTS_DIR / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global LLM
    api_key = get_api_key()
    os.environ["GEMINI_API_KEY"] = api_key
    LLM = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.2, max_retries=3)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.clear()
    request = os.environ.get("FOCUS_REQUEST") or REQUEST
    print(f"REQUEST: {request}")
    app = build_app()
    config = {"configurable": {"thread_id": "focus-" + datetime.now().strftime("%H%M%S")}, "recursion_limit": 100}
    init = {"messages": [msg("user", request)], "experiments": [], "decision_count": 0}
    for upd in app.stream(init, config, stream_mode="updates"):
        show_update(upd)
    final = app.get_state(config).values
    exps = final.get("experiments", [])
    write_reports(exps, final, request)

    print("\n" + "=" * 70)
    print(table_text(exps))
    if exps:
        best = max(exps, key=lambda e: e["enhancement"])
        print(f"\nBest: run {best['run']} enhancement {best['enhancement']} ({best['image_path']})")
        lec = os.environ.get("LECTURE_ENHANCEMENT")
        if lec:
            print(f"Lecture enhancement: {lec} -> ratio ours/lecture = {best['enhancement'] / float(lec):.2f}")
    print(f"Files: {RESULTS_DIR}/experiments.md, experiments.csv, transcript.md, exp*.png")


if __name__ == "__main__":
    main()
