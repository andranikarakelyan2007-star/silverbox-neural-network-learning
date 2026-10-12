"""hw2_task1_hitl.py

Homework 1 / Task 6 multi-agent system (orchestrator, scientist, general, executor)
extended with a fifth destination: `human`  (LangGraph interrupt + checkpointer).

GOOGLE COLAB USAGE
------------------
1. Secrets (key icon) -> add GEMINI_API_KEY, enable notebook access (or type it when asked).
2. Upload this file (and SNLS80mV.mat, optional) to /content.
3. In a cell:   %run hw2_task1_hitl.py
   Choose scenario a (ambiguous request) or b (hard decision), or c (your own text).
   The graph pauses, prints the question and waits for your reply in the input box.
   Reply /quit to abandon the run.

Every run uses its own folder ./run_workspace/<thread>/ with the generated scripts,
test_prediction.png and transcript.md.
"""

import importlib
import os
import re
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
_ensure("scipy", "scipy")

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
WORKSPACE_DIR = Path("./run_workspace").resolve()   # replaced per run in __main__
MAX_DECISIONS = 8
EXEC_ATTEMPTS = 2
RUN_TIMEOUT = 300
TRAIN_SECONDS = int(os.environ.get("TRAIN_SECONDS", "200"))
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


def _is_mat(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(6) == b"MATLAB" and p.stat().st_size > 1_000_000
    except OSError:
        return False


def find_mat() -> str:
    cands = [
        os.environ.get("SILVERBOX_MAT"),
        "silverboxfiles/SNLS80mV.mat",
        "SNLS80mV.mat",
        "/content/silverboxfiles/SNLS80mV.mat",
        "/content/SNLS80mV.mat",
        "/content/sample_data/SNLS80mV.mat",
        "/content/drive/MyDrive/silverboxfiles/SNLS80mV.mat",
        "/content/drive/MyDrive/SNLS80mV.mat",
    ]
    for c in cands:
        if c and _is_mat(Path(c)):
            return str(Path(c).resolve())
    for root in (".", "/content"):
        for pat in ("SNLS80mV.mat", "*/SNLS80mV.mat"):
            for p in Path(root).glob(pat):
                if _is_mat(p):
                    return str(p.resolve())
    try:
        from google.colab import files  # type: ignore

        print("A valid SNLS80mV.mat was not found -> please upload it.")
        up = files.upload()
        for name in up:
            if name.lower().endswith(".mat") and _is_mat(Path(name)):
                return str(Path(name).resolve())
    except ImportError:
        pass
    raise FileNotFoundError("Valid SNLS80mV.mat not found. Set SILVERBOX_MAT to its path.")


def check_mat(path: str) -> None:
    import scipy.io as sio

    m = sio.loadmat(path)
    for k in ("V1", "V2"):
        if k not in m:
            raise KeyError(f"{k} not in .mat file, keys: {list(m.keys())}")
    n = len(m["V1"].ravel())
    if n < 46746:
        raise ValueError(f"file has only {n} samples, need >= 46746")
    print(f"Data OK: {path} ({n} samples)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. HELPER LIBRARY WRITTEN INTO THE RUN DIRECTORY (data, RK4, windows, metric)
# ──────────────────────────────────────────────────────────────────────────────

HARNESS_CODE = r'''
import numpy as np
import torch
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 610.35
DT = 1.0 / FS
START = 42650
N = 4096
N_TRAIN = 3072


def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def load_data(path):
    """Centered excerpt [42650:46746]: returns u, y as float64 numpy arrays (N,)."""
    m = sio.loadmat(path)
    u = np.asarray(m["V1"], dtype=np.float64).ravel()[START:START + N]
    y = np.asarray(m["V2"], dtype=np.float64).ravel()[START:START + N]
    assert len(u) == N and len(y) == N, "excerpt shorter than 4096 samples"
    return u - u.mean(), y - y.mean()


def rk4_step(f, y, v, u0, um, u1, dt):
    """One RK4 step of y'=v, v'=f(y,v,u); u0/um/u1 = input at t, t+dt/2, t+dt."""
    k1y = v
    k1v = f(y, v, u0)
    k2y = v + 0.5 * dt * k1v
    k2v = f(y + 0.5 * dt * k1y, k2y, um)
    k3y = v + 0.5 * dt * k2v
    k3v = f(y + 0.5 * dt * k2y, k3y, um)
    k4y = v + dt * k3v
    k4v = f(y + dt * k3y, k4y, u1)
    y_new = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_new = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_new, v_new


def sample_windows(u, y, L, B, device="cpu"):
    """B random windows of L+1 samples inside the TRAIN part.
    Returns yw (B,L+1), uw (B,L+1), v0 (B,) finite-difference initial velocity."""
    u = _np(u)
    y = _np(y)
    starts = np.random.randint(0, N_TRAIN - L, size=B)
    idx = starts[:, None] + np.arange(L + 1)[None, :]
    yw = torch.tensor(y[idx], dtype=torch.float32, device=device)
    uw = torch.tensor(u[idx], dtype=torch.float32, device=device)
    v0 = (yw[:, 1] - yw[:, 0]) / DT
    return yw, uw, v0


def rollout(model, y0, v0, uw, dt=DT):
    """Differentiable RK4 rollout, returns predicted positions (B, L+1)."""
    y, v = y0, v0
    ys = [y]
    for k in range(uw.shape[1] - 1):
        u0 = uw[:, k]
        u1 = uw[:, k + 1]
        um = 0.5 * (u0 + u1)
        y, v = rk4_step(model, y, v, u0, um, u1, dt)
        ys.append(y)
    return torch.stack(ys, dim=1)


def evaluate(model, u, y, tag="model", out_png="test_prediction.png"):
    """Free run of the WHOLE excerpt from sample 0 (measured y0, finite-difference v0),
    reading only u. Prints Training RMSE (first 3072) and Test RMSE (last 1024)."""
    u = _np(u)
    y = _np(y)
    model = model.cpu().eval()
    ut = torch.tensor(u, dtype=torch.float32)
    yc = torch.tensor([y[0]], dtype=torch.float32)
    vc = torch.tensor([(y[1] - y[0]) / DT], dtype=torch.float32)
    yhat = np.empty(N)
    yhat[0] = y[0]
    with torch.no_grad():
        for k in range(N - 1):
            u0 = ut[k:k + 1]
            u1 = ut[k + 1:k + 2]
            um = 0.5 * (u0 + u1)
            yc, vc = rk4_step(model, yc, vc, u0, um, u1, DT)
            yhat[k + 1] = float(yc)
    rtr = float(np.sqrt(np.mean((yhat[:N_TRAIN] - y[:N_TRAIN]) ** 2)))
    rte = float(np.sqrt(np.mean((yhat[N_TRAIN:] - y[N_TRAIN:]) ** 2)))
    base = float(np.sqrt(np.mean(y[N_TRAIN:] ** 2)))
    t = np.arange(N_TRAIN, N) * DT
    plt.figure(figsize=(10, 4))
    plt.plot(t, y[N_TRAIN:], label="measured")
    plt.plot(t, yhat[N_TRAIN:], "--", label="free-run prediction")
    plt.xlabel("t, s")
    plt.ylabel("y, V")
    plt.title(f"{tag}: test free run, RMSE={rte:.5f} V")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Training RMSE: {rtr:.6f}")
    print(f"Test RMSE: {rte:.6f}")
    print(f"Baseline (predict zero) RMSE on test: {base:.6f}")
    return rtr, rte
'''


def write_harness() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "silverbox_harness.py").write_text(HARNESS_CODE, encoding="utf-8")


def disk_listing() -> str:
    if not WORKSPACE_DIR.exists():
        return "(nothing yet)"
    names = sorted(
        p.name for p in WORKSPACE_DIR.iterdir() if p.name not in ("silverbox_harness.py", "__pycache__")
    )
    return ", ".join(names) if names else "(nothing yet)"


# ──────────────────────────────────────────────────────────────────────────────
# 3. CODING TOOL (only the executor calls it)
# ──────────────────────────────────────────────────────────────────────────────


def run_python_code(filename: str, code: str, mat_path: str) -> dict:
    """Tool: (1) write a .py file into the run directory, (2) run it with a
    5-minute limit and return exit code, stdout and stderr."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = WORKSPACE_DIR / filename
    file_path.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    env["SILVERBOX_MAT"] = mat_path
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
            cwd=str(WORKSPACE_DIR),
            env=env,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return {"exit_code": -1, "stdout": out, "stderr": f"Execution timed out after {RUN_TIMEOUT} seconds."}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 4. STATE (plain dict messages -> always serializable by the checkpointer)
# ──────────────────────────────────────────────────────────────────────────────


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content, "timestamp": datetime.now().strftime("%H:%M:%S")}


def message_reducer(existing: List[dict], new_items: List[dict]) -> List[dict]:
    return existing + new_items


class GraphState(TypedDict, total=False):
    messages: Annotated[List[dict], message_reducer]
    decision_count: int
    next_role: str
    instruction: str
    pending_kind: str          # "ambiguity" | "hard_decision"
    pending_role: str          # step that waits for approval (hard decision)
    pending_instruction: str
    pending_question: str
    gate_result: str           # "answered" | "accepted" | "refused"
    last_code: str
    last_error: str
    run_id: int
    mat_path: str


class OrchestratorDecision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish", "human"] = Field(
        description="Destination of the next turn."
    )
    instruction: str = Field(
        description="Instruction for the chosen role. For a hard decision: the instruction the proposed step "
        "gets if the human accepts it. For an ambiguity: empty string."
    )
    reason: str = Field(description="One-sentence reason for this decision.")
    question: str = Field(
        description="Question to ask the human when next_role is human (readings to choose from, or the step "
        "to approve). Empty string otherwise."
    )
    human_kind: Literal["ambiguity", "hard_decision", "none"] = Field(
        description="Why the human is asked; 'none' when next_role is not human."
    )
    proposed_role: Literal["executor", "finish", "none"] = Field(
        description="For a hard decision: the step that waits for approval; 'none' otherwise."
    )


class ApprovalVerdict(BaseModel):
    approved: bool = Field(description="True only if the human clearly agrees to proceed with the step.")
    reason: str = Field(description="One short sentence.")


VERIFIED_RE = re.compile(r"VERIFIED TEST RMSE \(exit code 0\):\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
STDOUT_RMSE_RE = re.compile(r"^Test RMSE:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M)


def get_llm() -> ChatGoogleGenerativeAI:
    return LLM


def text_of(resp) -> str:
    c = resp.content
    if isinstance(c, str):
        return c
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)


def history_text(messages: List[dict], limit: int = 3500) -> str:
    out = []
    for m in messages:
        c = m["content"] if len(m["content"]) <= limit else m["content"][:limit] + "\n...[truncated]"
        out.append(f"[{m['role'].upper()}]: {c}")
    return "\n\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 5. NODES
# ──────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are the Orchestrator of a multi-agent system working on the Silverbox PINN task (Homework 1).
You talk to the user and choose the next destination STRICTLY from the dialogue so far.
Destinations:
- scientist: writes the differential equation, network, state equation, training plan, free-run test and a numbered
  implementation list. No coding tool.
- general: answers high-level questions beside the fit (what a record in the file is, what a term or a reported error
  means). Does NOT write the fitting plan.
- executor: the only role with the coding tool; implements the latest plan in a NEW file, runs it, reports
  exit code / stdout / stderr / RMSE. Tracebacks go here.
- finish: end the dialogue and report the measurement.
- human: pause the graph and ask the person a question.
Roles of other speakers in the dialogue: 'user' (first request), 'human' (the person's replies to your questions),
'human_gate' (graph notes: ACCEPTED or REFUSED a proposed step).

Choose 'human' in exactly two situations:
(1) AMBIGUITY (human_kind='ambiguity', proposed_role='none', instruction=''): the latest request has more than one
    reasonable reading AND the readings would send the work to different roles or change the number that is reported
    (e.g. train vs test RMSE, explain a term vs compute a number, reuse an old result vs run again). The question lists
    the readings and asks which one to follow. Ask only once: if a 'human' message already settled the point, follow it.
(2) HARD DECISION (human_kind='hard_decision'): the next step would run the coding tool (executor), overwrite a result
    that is already on disk (see the file list below), or end the dialogue and report a measurement (finish).
    Set proposed_role to 'executor' or 'finish', instruction to the instruction that step gets if accepted, and make the
    question state the step plainly (what will run or be overwritten or reported) and ask whether to proceed.
    NEVER choose 'executor' or 'finish' directly as next_role: ask first. An accepted step runs right away without
    coming back to you.
After 'HUMAN_GATE: REFUSED' the step did NOT run. Take the human's words into account: revise the instruction and
propose the step again as a new hard decision, or choose another destination. Do not re-propose an unchanged step.
Other rules:
- 'finish' may be proposed only after an executor turn contains 'VERIFIED TEST RMSE (exit code 0): <number>'.
- You write no theory and no code and have no tools. Hard limit: 8 decisions in total, so be economical; a typical run is
  scientist -> human(run?) -> executor -> human(finish?) -> finish.
- If the last executor turn failed (non-zero exit code, timeout, no Test RMSE), propose the executor again (hard decision)
  with a short instruction about what to fix; if the plan itself is wrong, call scientist."""


def check_decision(d: OrchestratorDecision, has_rmse: bool):
    if d.next_role in ("executor", "finish"):
        return (
            f"You may not choose '{d.next_role}' directly because it is a hard decision. Choose next_role='human', "
            f"human_kind='hard_decision', proposed_role='{d.next_role}' and ask whether to proceed."
        )
    if d.next_role == "human":
        if d.human_kind not in ("ambiguity", "hard_decision"):
            return "next_role='human' needs human_kind 'ambiguity' or 'hard_decision'."
        if not d.question.strip():
            return "next_role='human' needs a non-empty question."
        if d.human_kind == "hard_decision":
            if d.proposed_role not in ("executor", "finish"):
                return "a hard_decision needs proposed_role 'executor' or 'finish'."
            if d.proposed_role == "finish" and not has_rmse:
                return "finish can be proposed only after a verified test RMSE exists."
    return None


def decide(prompt: str, has_plan: bool, has_rmse: bool) -> OrchestratorDecision:
    structured = get_llm().with_structured_output(OrchestratorDecision)
    feedback = ""
    for _ in range(3):
        try:
            d = structured.invoke(prompt + feedback)
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] LLM call failed, retrying: {e}")
            time.sleep(2)
            continue
        if d is None:
            continue
        problem = check_decision(d, has_rmse)
        if problem is None:
            return d
        feedback = f"\n\nYour previous answer was rejected: {problem}\nDecide again."
    # Safe fallback after repeated invalid answers: never run or finish without asking.
    if has_rmse:
        return OrchestratorDecision(
            next_role="human", instruction="Finish and report the verified test RMSE.",
            reason="Fallback: ending the dialogue needs approval.",
            question="I am about to end the dialogue and report the measured test RMSE. Proceed?",
            human_kind="hard_decision", proposed_role="finish",
        )
    if has_plan:
        return OrchestratorDecision(
            next_role="human", instruction="Implement the latest plan, run it and report the free-run RMSE.",
            reason="Fallback: running the coding tool needs approval.",
            question="I am about to write a Python script and run it (up to 5 minutes). Proceed?",
            human_kind="hard_decision", proposed_role="executor",
        )
    return OrchestratorDecision(
        next_role="scientist", instruction="Write the full plan for the Silverbox PINN task.",
        reason="Fallback: a plan is needed first.", question="", human_kind="none", proposed_role="none",
    )


def orchestrator_node(state: GraphState) -> dict:
    n = state.get("decision_count", 0)
    msgs = state["messages"]
    has_rmse = any(m["role"] == "executor" and VERIFIED_RE.search(m["content"]) for m in msgs)
    has_plan = any(m["role"] == "scientist" for m in msgs)
    prompt = (
        f"{ORCHESTRATOR_PROMPT}\n\n"
        f"Decisions used so far: {n}/{MAX_DECISIONS}\n"
        f"Verified test RMSE present: {has_rmse}\n"
        f"Files already on disk in the run directory: {disk_listing()}\n\n"
        f"Dialogue:\n{history_text(msgs)}\n\nMake your decision:"
    )
    d = decide(prompt, has_plan, has_rmse)

    if d.next_role == "human":
        content = (
            f"NEXT ROLE: human\nKIND: {d.human_kind}\nPROPOSED STEP: {d.proposed_role}\n"
            f"QUESTION: {d.question}\nREASON: {d.reason}"
        )
        return {
            "messages": [msg("orchestrator", content)],
            "decision_count": n + 1,
            "next_role": "human",
            "pending_kind": d.human_kind,
            "pending_role": d.proposed_role,
            "pending_instruction": d.instruction,
            "pending_question": d.question,
        }
    content = f"NEXT ROLE: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
    return {
        "messages": [msg("orchestrator", content)],
        "decision_count": n + 1,
        "next_role": d.next_role,
        "instruction": d.instruction,
        "pending_kind": "",
        "pending_role": "",
        "pending_instruction": "",
        "pending_question": "",
    }


# ── human node: LangGraph interrupt ───────────────────────────────────────────


def judge_reply(question: str, reply: str) -> bool:
    prompt = (
        "A system asked a person to approve a step.\n"
        f"QUESTION: {question}\nPERSON'S REPLY: {reply}\n"
        "Is the reply a clear approval to proceed with the step exactly as proposed? "
        "A refusal, a request to wait, or a request to change something first means NOT approved."
    )
    try:
        v = get_llm().with_structured_output(ApprovalVerdict).invoke(prompt)
        if v is not None:
            return bool(v.approved)
    except Exception as e:  # noqa: BLE001
        print(f"[human] verdict call failed, using plain parse: {e}")
    return reply.strip().lower() in {"y", "yes", "ok", "okay", "proceed", "да", "ок", "давай"}


def human_node(state: GraphState) -> dict:
    kind = state.get("pending_kind", "ambiguity")
    role = state.get("pending_role", "none")
    question = state.get("pending_question", "Please clarify how to proceed.")
    # The graph pauses here until the same thread is resumed with Command(resume=reply).
    reply = str(interrupt({"kind": kind, "proposed_role": role, "question": question})).strip()
    out = [msg("human", reply)]
    if kind != "hard_decision":
        return {"messages": out, "gate_result": "answered"}
    if judge_reply(question, reply):
        out.append(msg("human_gate", f"ACCEPTED: the human approved the step '{role}'. It runs now."))
        return {
            "messages": out,
            "gate_result": "accepted",
            "next_role": role,
            "instruction": state.get("pending_instruction", ""),
        }
    out.append(
        msg(
            "human_gate",
            f"REFUSED: the human did not approve the step '{role}'. The step was NOT run "
            "(no code executed, no file written, nothing reported).",
        )
    )
    return {"messages": out, "gate_result": "refused"}


# ── scientist / general ───────────────────────────────────────────────────────

SCIENTIST_PROMPT = f"""You are the Scientist. You own the theory and the plan for the Silverbox PINN task.
Write:
1. The differential equation y'' + a*y' + b*y + c*y^3 = g*u(t) (centered signals, meaning of a, b, c, g).
2. The network you propose (MLP, depth/width/activation, input and output scaling) producing the acceleration.
3. The state equation: dy/dt = v, dv/dt = alpha_theta(y, v, u), integrated by RK4 with dt = 1/610.35 s,
   input at half steps = average of neighbouring samples.
4. Training: multiple shooting on random windows of the first 3072 samples (e.g. 64 steps), Adam with a decaying
   learning rate, loss on position. Wall-clock budget for training: about {TRAIN_SECONDS} s (the whole script has a
   300 s limit).
5. Free-run test: start at sample 0 with measured y and finite-difference v, integrate through train and test,
   u may be read but y may not; RMSE on samples [0:3072] and [3072:4096]; baseline about 0.06 V.
6. A numbered list of steps specific enough to code directly.
Constraints: you have NO coding tool. Never claim a number was measured unless an executor turn printed it."""


def scientist_node(state: GraphState) -> dict:
    prompt = (
        f"{SCIENTIST_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\n"
        "Write the scientific formulation and the numbered implementation plan:"
    )
    return {"messages": [msg("scientist", text_of(get_llm().invoke(prompt)))]}


GENERAL_PROMPT = """You are the General Agent. You answer high-level questions that sit beside the fit:
- what a record in SNLS80mV.mat is (V1 = input voltage u, V2 = output voltage y, fs = 610.35 Hz),
- what a term in the problem means (damping a, linear stiffness b, cubic stiffness c, gain g),
- what a reported error is measuring (RMSE, train vs test, the 0.06 V zero-prediction baseline).
Constraints: no coding tool. Do NOT write the fitting plan (Scientist's job) and do not debug tracebacks
(Executor's job). Answer the orchestrator's instruction concisely."""


def general_node(state: GraphState) -> dict:
    prompt = f"{GENERAL_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\nAnswer:"
    return {"messages": [msg("general", text_of(get_llm().invoke(prompt)))]}


# ── executor ──────────────────────────────────────────────────────────────────

CODE_RULES = f"""You write ONE complete, self-contained Python script. A helper module `silverbox_harness.py`
already exists next to the script. Use it, do not re-implement it:
  from silverbox_harness import load_data, sample_windows, rollout, evaluate, rk4_step, DT, N, N_TRAIN
  u, y = load_data(os.environ["SILVERBOX_MAT"])      # centered numpy float64, shape (4096,); keep them numpy
  yw, uw, v0 = sample_windows(u, y, L, B, device)    # random TRAIN windows: yw,uw (B,L+1), v0 (B,)
  pred = rollout(model, yw[:, 0], v0, uw)            # differentiable RK4 rollout -> (B,L+1)
  evaluate(model, u, y, tag="...")                   # free run of the whole excerpt, PRINTS the RMSE lines
MANDATORY:
1. Define an nn.Module `AccelNet` whose forward(y, v, u) takes three float32 tensors of shape (B,) and returns the
   acceleration, shape (B,). Do the input/output scaling INSIDE the module:
   Y_SCALE=y[:N_TRAIN].std(), V_SCALE=2*pi*70*Y_SCALE, U_SCALE=u[:N_TRAIN].std(), ACC_SCALE=(2*pi*70)**2*Y_SCALE
   (inputs divided by the scales, output multiplied by ACC_SCALE).
2. Train on random windows with Adam, gradient clipping 1.0 and a cosine learning-rate decay driven by the elapsed
   wall-clock fraction (lr from 1e-3 down to 1e-5). Loss = mean squared error of the predicted positions divided by
   Y_SCALE**2. Use cuda for training if available.
3. Stop training by wall clock: stop when time.time()-t0 > {TRAIN_SECONDS} (check every step) or at a max step count.
   Print the loss every ~100 steps. The whole script must finish in well under 300 s.
4. The script must actually execute its training at top level (or via main() called under __main__) and print progress.
   At the end call evaluate(model, u, y, tag=...) exactly once. It prints "Training RMSE:" and "Test RMSE:" itself;
   do not print those lines yourself and do not fake numbers.
5. No input(), no plt.show(), no pip install, no network. Follow the Scientist's plan for architecture, window length
   and optimizer; if the plan does not fit the time budget, shrink it and say so in a comment.
6. Output ONLY the code in one ```python block."""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def tail(s: str, n: int = 3500) -> str:
    return s if len(s) <= n else "...[cut]...\n" + s[-n:]


def executor_node(state: GraphState) -> dict:
    msgs = state["messages"]
    plan = next((m["content"] for m in reversed(msgs) if m["role"] == "scientist"), "")
    instr = state.get("instruction") or next(
        (m["content"] for m in reversed(msgs) if m["role"] == "orchestrator"), ""
    )
    mat_path = state["mat_path"]
    run_id = state.get("run_id", 0) + 1
    prev_code = state.get("last_code", "")
    prev_err = state.get("last_error", "")
    write_harness()

    report, ok_rmse = [], None
    for attempt in range(1, EXEC_ATTEMPTS + 1):
        prompt = (
            f"{CODE_RULES}\n\n=== SCIENTIST PLAN (latest) ===\n{plan or '(no plan yet - follow the instruction)'}\n\n"
            f"=== INSTRUCTION ===\n{instr}\n"
        )
        if prev_code and prev_err:
            prompt += (
                "\n=== PREVIOUS SCRIPT FAILED - REPAIR IT, KEEP THE SAME PLAN ===\n"
                f"--- previous code ---\n{prev_code}\n--- error / output ---\n{prev_err}\n"
            )
        code = extract_code(text_of(get_llm().invoke(prompt)))
        fname = f"silverbox_pinn_v{run_id}_a{attempt}.py"
        res = run_python_code(fname, code, mat_path)
        match = STDOUT_RMSE_RE.search(res["stdout"])
        good = res["exit_code"] == 0 and match is not None
        report.append(
            f"=== ATTEMPT {attempt}/{EXEC_ATTEMPTS} | file {fname} ===\n"
            f"EXIT CODE: {res['exit_code']}\n"
            f"--- STDOUT ---\n{tail(res['stdout'])}\n"
            f"--- STDERR ---\n{tail(res['stderr'])}\n"
        )
        if good:
            ok_rmse = match.group(1)
            break
        prev_code = code
        prev_err = (
            f"exit code {res['exit_code']}\nSTDERR:\n{tail(res['stderr'], 3000)}\nSTDOUT:\n{tail(res['stdout'], 1500)}"
        )
        if res["exit_code"] == 0:
            prev_err += "\nThe script did not print 'Test RMSE:' (call evaluate once at the end)."

    text = "\n".join(report)
    if ok_rmse is not None:
        text += f"\nVERIFIED TEST RMSE (exit code 0): {ok_rmse}\n"
    return {
        "messages": [msg("executor", text)],
        "last_code": prev_code if ok_rmse is None else "",
        "last_error": prev_err if ok_rmse is None else "",
        "run_id": run_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. GRAPH (compiled WITH a checkpointer: required for interrupt/resume)
# ──────────────────────────────────────────────────────────────────────────────


def route_after_orchestrator(state: GraphState) -> str:
    return state.get("next_role", "finish")


def after_specialist(state: GraphState) -> str:
    return "end" if state.get("decision_count", 0) >= MAX_DECISIONS else "orchestrator"


def route_after_human(state: GraphState) -> str:
    if state.get("gate_result") == "accepted":
        role = state.get("next_role", "")
        return "end" if role in ("finish", "") else role
    return "end" if state.get("decision_count", 0) >= MAX_DECISIONS else "orchestrator"


def build_app():
    b = StateGraph(GraphState)
    b.add_node("orchestrator", orchestrator_node)
    b.add_node("scientist", scientist_node)
    b.add_node("general", general_node)
    b.add_node("executor", executor_node)
    b.add_node("human", human_node)
    b.set_entry_point("orchestrator")
    b.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "human": "human",
            "finish": END,
        },
    )
    for node in ("scientist", "general", "executor"):
        b.add_conditional_edges(node, after_specialist, {"orchestrator": "orchestrator", "end": END})
    b.add_conditional_edges(
        "human",
        route_after_human,
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            "orchestrator": "orchestrator",
            "end": END,
        },
    )
    return b.compile(checkpointer=MemorySaver())


# ──────────────────────────────────────────────────────────────────────────────
# 7. RUNNER: stream, pause on interrupt, ask the person, resume the SAME thread
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "a": "Check how the Silverbox model does and give me the error.",          # ambiguous
    "b": "Run the Silverbox PINN from Homework 1 and report the free-run test RMSE.",  # hard decision
}


def show_update(upd: dict) -> None:
    for node, delta in upd.items():
        if node == "__interrupt__" or not isinstance(delta, dict):
            continue
        for m in delta.get("messages", []):
            head = f"--- [{m['role'].upper()}] {m['timestamp']} ---"
            body = m["content"]
            print("\n" + head)
            print(body[:1800] + ("\n...[full text in transcript.md]" if len(body) > 1800 else ""))


def pending_interrupt(app, config):
    snap = app.get_state(config)
    for t in snap.tasks:
        for i in getattr(t, "interrupts", ()) or ():
            return i.value
    return None


def run_session(app, request: str, thread_id: str, mat: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    init = {
        "messages": [msg("user", request)],
        "decision_count": 0,
        "run_id": 0,
        "mat_path": mat,
    }
    stream_input = init
    while True:
        for upd in app.stream(stream_input, config, stream_mode="updates"):
            show_update(upd)
        payload = pending_interrupt(app, config)
        if payload is None:
            break
        print("\n" + "=" * 70)
        print(f"GRAPH PAUSED ({payload['kind']}; proposed step: {payload['proposed_role']})")
        print(f"QUESTION: {payload['question']}")
        print("=" * 70)
        reply = input("Your reply (/quit to abandon the run): ").strip()
        if reply == "/quit":
            print("Run abandoned; the graph stays paused on this thread.")
            break
        stream_input = Command(resume=reply)
    return app.get_state(config).values


if __name__ == "__main__":
    api_key = get_api_key()
    os.environ["GEMINI_API_KEY"] = api_key
    LLM = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.1, max_retries=3)
    mat = find_mat()
    check_mat(mat)

    choice = os.environ.get("SCENARIO") or input(
        "Scenario: [a] ambiguous request, [b] hard decision, [c] custom text: "
    ).strip().lower()
    if choice in SCENARIOS:
        request = SCENARIOS[choice]
    else:
        choice = "c"
        request = input("Your request: ").strip()
    print(f"\nREQUEST: {request}")

    stamp = datetime.now().strftime("%H%M%S")
    thread_id = f"hitl-{choice}-{stamp}"
    WORKSPACE_DIR = (Path("./run_workspace") / thread_id).resolve()
    write_harness()
    app = build_app()

    print("=" * 70)
    print(f"HITL multi-agent system | {MODEL_NAME} | thread: {thread_id} | dir: {WORKSPACE_DIR}")
    print("=" * 70)

    final = run_session(app, request, thread_id, mat)

    lines = [f"# Transcript ({thread_id})\n", f"Request: {request}\n"]
    for i, m in enumerate(final.get("messages", []), 1):
        lines.append(f"## Turn {i} [{m['role'].upper()}] {m['timestamp']}\n\n{m['content']}\n")
    (WORKSPACE_DIR / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    rm = [VERIFIED_RE.search(m["content"]) for m in final.get("messages", []) if m["role"] == "executor"]
    rm = [r.group(1) for r in rm if r]
    n_human = sum(1 for m in final.get("messages", []) if m["role"] == "human")
    print("\n" + "=" * 70)
    print(f"Decisions used: {final.get('decision_count')}/{MAX_DECISIONS} | human replies: {n_human}")
    print(f"Verified free-run Test RMSE: {rm[-1] if rm else 'none (no executor run completed)'}")
    print(f"Transcript: {WORKSPACE_DIR / 'transcript.md'}")
