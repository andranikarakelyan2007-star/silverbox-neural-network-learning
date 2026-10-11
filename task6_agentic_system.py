
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
from langgraph.graph import END, StateGraph  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
WORKSPACE_DIR = Path("./run_workspace").resolve()
MAX_DECISIONS = 8          # orchestrator decisions
EXEC_ATTEMPTS = 2          # write+run attempts inside one executor turn
RUN_TIMEOUT = 300          # seconds, hard limit from the assignment
TRAIN_SECONDS = 200        # wall-clock training budget given to generated code
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


def find_mat() -> str:
    cands = [
        os.environ.get("SILVERBOX_MAT"),
        "silverboxfiles/SNLS80mV.mat",
        "SNLS80mV.mat",
        "/content/silverboxfiles/SNLS80mV.mat",
        "/content/SNLS80mV.mat",
        "/content/drive/MyDrive/silverboxfiles/SNLS80mV.mat",
        "/content/drive/MyDrive/SNLS80mV.mat",
    ]
    for c in cands:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    for root in (".", "/content"):
        for pat in ("SNLS80mV.mat", "*/SNLS80mV.mat"):
            for p in Path(root).glob(pat):
                return str(p.resolve())
    try:
        from google.colab import files  # type: ignore

        print("SNLS80mV.mat not found -> please upload it.")
        up = files.upload()
        for name in up:
            if name.lower().endswith(".mat"):
                return str(Path(name).resolve())
    except ImportError:
        pass
    raise FileNotFoundError(
        "SNLS80mV.mat not found. Put it next to this script or set SILVERBOX_MAT."
    )


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
# 2. HELPER LIBRARY WRITTEN INTO THE WORKSPACE (not by the LLM)
#    Data loading, RK4, windows, and the official free-run metric.
#    The executor's script imports it and only writes network + training.
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
        return {
            "exit_code": -1,
            "stdout": out,
            "stderr": f"Execution timed out after {RUN_TIMEOUT} seconds.",
        }
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 4. STATE
# ──────────────────────────────────────────────────────────────────────────────


class AgentMessage(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


def message_reducer(existing: List[AgentMessage], new_items: List[AgentMessage]) -> List[AgentMessage]:
    return existing + new_items


class GraphState(TypedDict, total=False):
    messages: Annotated[List[AgentMessage], message_reducer]
    decision_count: int
    next_role: str
    test_rmse_found: bool
    last_code: str
    last_error: str
    run_id: int
    mat_path: str


class OrchestratorDecision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"] = Field(
        description="Next role to call, or finish if a verified test RMSE exists."
    )
    instruction: str = Field(description="Direct instruction for the chosen role.")
    reason: str = Field(description="One-sentence reason for this decision.")


VERIFIED_RE = re.compile(r"VERIFIED TEST RMSE \(exit code 0\):\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
STDOUT_RMSE_RE = re.compile(
    r"^Test RMSE:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$", re.M
)


def get_llm() -> ChatGoogleGenerativeAI:
    return LLM


def text_of(resp) -> str:
    c = resp.content
    if isinstance(c, str):
        return c
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)


def history_text(messages: List[AgentMessage], limit: int = 3500) -> str:
    out = []
    for m in messages:
        c = m.content if len(m.content) <= limit else m.content[:limit] + "\n...[truncated]"
        out.append(f"[{m.role.upper()}]: {c}")
    return "\n\n".join(out)


# ──────────────────────────────────────────────────────────────────────────────
# 5. NODES
# ──────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """You are the Orchestrator of a multi-agent system that solves Task 5 (Silverbox PINN).
You talk to the user and choose the next role STRICTLY from the dialogue so far.
Roles:
- scientist: writes the differential equation, network, state equation, training plan,
  free-run test and a numbered implementation list. Has no coding tool.
- general: answers high-level questions beside the fit (file channels, meaning of terms
  or of a reported error). Does NOT write the fitting plan.
- executor: the only role with the coding tool; implements the scientist's latest plan in a
  NEW file, runs it and reports exit code / stdout / stderr / RMSE. Failures and tracebacks go here.
- finish: end the workflow.
Rules:
1. You may choose 'finish' ONLY after an executor turn contains the line
   'VERIFIED TEST RMSE (exit code 0): <number>' (a script that really ran and printed it).
2. You write no theory and no code and have no tools.
3. Hard limit: 8 decisions in total. Be economical: a typical good run is
   scientist -> executor -> (executor repair if it failed) -> finish.
4. If the last executor turn failed (non-zero exit code, timeout, no Test RMSE), call executor again with a short
   instruction about what to fix; if the plan itself is wrong, call scientist."""


def orchestrator_node(state: GraphState) -> dict:
    n = state.get("decision_count", 0)
    msgs = state["messages"]
    has_rmse = any(m.role == "executor" and VERIFIED_RE.search(m.content) for m in msgs)
    has_plan = any(m.role == "scientist" for m in msgs)

    prompt = (
        f"{ORCHESTRATOR_PROMPT}\n\n"
        f"Decisions used so far: {n}/{MAX_DECISIONS}\n"
        f"Verified test RMSE present: {has_rmse}\n\n"
        f"Dialogue:\n{history_text(msgs)}\n\n"
        "Make your decision (next_role, instruction, reason):"
    )
    structured = get_llm().with_structured_output(OrchestratorDecision)
    decision = None
    for _ in range(3):
        try:
            decision = structured.invoke(prompt)
            if decision is not None:
                break
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] LLM call failed, retrying: {e}")
            time.sleep(2)
    if decision is None:  # only if the API keeps failing
        decision = OrchestratorDecision(
            next_role="executor" if has_plan else "scientist",
            instruction="Continue with the task.",
            reason="Fallback after repeated model errors.",
        )

    if decision.next_role == "finish" and not has_rmse:
        decision = OrchestratorDecision(
            next_role="executor" if has_plan else "scientist",
            instruction=(
                "Run the PINN script and print the free-run Test RMSE before finishing."
                if has_plan
                else "Write the full plan for the PINN task first."
            ),
            reason="Cannot finish: no executor turn has a verified test RMSE yet.",
        )

    content = (
        f"NEXT ROLE: {decision.next_role}\n"
        f"INSTRUCTION: {decision.instruction}\n"
        f"REASON: {decision.reason}"
    )
    return {
        "messages": [AgentMessage(role="orchestrator", content=content)],
        "decision_count": n + 1,
        "next_role": decision.next_role,
        "test_rmse_found": has_rmse,
    }


SCIENTIST_PROMPT = f"""You are the Scientist. You own the theory and the plan for the Silverbox PINN task.
Write:
1. The differential equation y'' + a*y' + b*y + c*y^3 = g*u(t) (centered signals, meaning of a, b, c, g).
2. The network you propose (MLP, depth/width/activation, input and output scaling) producing the acceleration.
3. The state equation: dy/dt = v, dv/dt = alpha_theta(y, v, u), integrated by RK4 with dt = 1/610.35 s,
   input at half steps = average of neighbouring samples.
4. Training: multiple shooting on random windows of the first 3072 samples (e.g. 64 steps), Adam, lr schedule,
   loss on position. Wall-clock budget for training: about {TRAIN_SECONDS} s (the whole script has a 300 s limit).
5. Free-run test: start at sample 0 with measured y and finite-difference v, integrate through train and test,
   u may be read but y may not; RMSE on samples [0:3072] and [3072:4096]; baseline about 0.06 V.
6. A numbered list of steps specific enough to code directly.
Constraints: you have NO coding tool. Never claim a number was measured unless an executor turn printed it."""


def scientist_node(state: GraphState) -> dict:
    prompt = (
        f"{SCIENTIST_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\n"
        "Write the scientific formulation and the numbered implementation plan:"
    )
    return {"messages": [AgentMessage(role="scientist", content=text_of(get_llm().invoke(prompt)))]}


GENERAL_PROMPT = """You are the General Agent. You answer high-level questions that sit beside the fit:
- what a record in SNLS80mV.mat is (V1 = input voltage u, V2 = output voltage y, fs = 610.35 Hz),
- what a term in the problem means (damping a, linear stiffness b, cubic stiffness c, gain g),
- what a reported error measures (RMSE, the 0.06 V zero-prediction baseline).
Constraints: no coding tool. Do NOT write the fitting plan (that is the Scientist's job) and do not debug tracebacks
(that is the Executor's job). Answer the orchestrator's instruction concisely."""


def general_node(state: GraphState) -> dict:
    prompt = f"{GENERAL_PROMPT}\n\nDialogue:\n{history_text(state['messages'])}\n\nAnswer:"
    return {"messages": [AgentMessage(role="general", content=text_of(get_llm().invoke(prompt)))]}


CODE_RULES = f"""You write ONE complete, self-contained Python script. A helper module `silverbox_harness.py`
already exists next to the script. Use it, do not re-implement it:
  from silverbox_harness import load_data, sample_windows, rollout, evaluate, rk4_step, DT, N, N_TRAIN
  u, y = load_data(os.environ["SILVERBOX_MAT"])      # centered numpy float64, shape (4096,)
  yw, uw, v0 = sample_windows(u, y, L, B, device)    # random TRAIN windows: yw,uw (B,L+1), v0 (B,)
  pred = rollout(model, yw[:, 0], v0, uw)            # differentiable RK4 rollout -> (B,L+1)
  evaluate(model, u, y, tag="...")                   # free run of the whole excerpt, PRINTS the RMSE lines
MANDATORY:
1. Define an nn.Module `AccelNet` whose forward(y, v, u) takes three float32 tensors of shape (B,) and returns the
   acceleration, shape (B,). Do the input/output scaling INSIDE the module:
   Y_SCALE=y[:N_TRAIN].std(), V_SCALE=2*pi*70*Y_SCALE, U_SCALE=u[:N_TRAIN].std(), ACC_SCALE=(2*pi*70)**2*Y_SCALE
   (inputs divided by the scales, output multiplied by ACC_SCALE).
2. Train on random windows with Adam, gradient clipping 1.0 and an lr schedule. Loss = mean squared error of the
   predicted positions divided by Y_SCALE**2. Use cuda for training if available.
3. Stop training by wall clock: stop when time.time()-t0 > {TRAIN_SECONDS} (check every step) or at a max step count.
   Print the loss every ~100 steps. The whole script must finish in well under 300 s.
4. At the end call evaluate(model, u, y, tag=...) exactly once. It prints "Training RMSE:" and "Test RMSE:" itself;
   do not print those lines yourself and do not fake numbers.
5. No input(), no plt.show(), no pip install, no network. Follow the Scientist's plan for architecture, window length,
   optimizer and lr; if the plan does not fit the time budget, shrink it and say so in a comment.
6. Output ONLY the code in one ```python block."""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def tail(s: str, n: int = 3500) -> str:
    return s if len(s) <= n else "...[cut]...\n" + s[-n:]


def executor_node(state: GraphState) -> dict:
    msgs = state["messages"]
    plan = next((m.content for m in reversed(msgs) if m.role == "scientist"), "")
    instr = next((m.content for m in reversed(msgs) if m.role == "orchestrator"), "")
    mat_path = state["mat_path"]
    run_id = state.get("run_id", 0) + 1
    prev_code = state.get("last_code", "")
    prev_err = state.get("last_error", "")
    write_harness()

    report, ok_rmse = [], None
    for attempt in range(1, EXEC_ATTEMPTS + 1):
        prompt = (
            f"{CODE_RULES}\n\n=== SCIENTIST PLAN (latest) ===\n{plan or '(no plan yet - follow the orchestrator)'}\n\n"
            f"=== ORCHESTRATOR INSTRUCTION ===\n{instr}\n"
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
        prev_err = f"exit code {res['exit_code']}\nSTDERR:\n{tail(res['stderr'], 3000)}\nSTDOUT:\n{tail(res['stdout'], 1500)}"
        if res["exit_code"] == 0:
            prev_err += "\nThe script did not print 'Test RMSE:' (call evaluate once at the end)."

    text = "\n".join(report)
    if ok_rmse is not None:
        text += f"\nVERIFIED TEST RMSE (exit code 0): {ok_rmse}\n"
    return {
        "messages": [AgentMessage(role="executor", content=text)],
        "last_code": prev_code if ok_rmse is None else "",
        "last_error": prev_err if ok_rmse is None else "",
        "run_id": run_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. GRAPH
# ──────────────────────────────────────────────────────────────────────────────


def route_after_orchestrator(state: GraphState) -> str:
    return state.get("next_role", "finish")


def after_specialist(state: GraphState) -> str:
    return "end" if state.get("decision_count", 0) >= MAX_DECISIONS else "orchestrator"


def build_app():
    b = StateGraph(GraphState)
    b.add_node("orchestrator", orchestrator_node)
    b.add_node("scientist", scientist_node)
    b.add_node("general", general_node)
    b.add_node("executor", executor_node)
    b.set_entry_point("orchestrator")
    b.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"scientist": "scientist", "general": "general", "executor": "executor", "finish": END},
    )
    for node in ("scientist", "general", "executor"):
        b.add_conditional_edges(node, after_specialist, {"orchestrator": "orchestrator", "end": END})
    return b.compile()


# ──────────────────────────────────────────────────────────────────────────────
# 7. RUN
# ──────────────────────────────────────────────────────────────────────────────

USER_PROMPT = (
    "Solve Task 5: PINN for the Silverbox oscillator.\n"
    "Data: 4096 samples [42650:46746] of SNLS80mV.mat (V1=u, V2=y), fs=610.35 Hz, subtract the means of u and y.\n"
    "Equation: y'' + a*y' + b*y + c*y^3 = g*u(t). State: dy/dt = v, dv/dt = alpha_theta(y, v, u), integrate with RK4.\n"
    "Train on the first 3072 samples, free-run the whole excerpt from sample 0, report the free-run RMSE on the\n"
    "training (3072) and test (1024) samples."
)

if __name__ == "__main__":
    api_key = get_api_key()
    os.environ["GEMINI_API_KEY"] = api_key
    LLM = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=api_key, temperature=0.1, max_retries=3
    )
    mat = find_mat()
    check_mat(mat)
    write_harness()
    app = build_app()

    print("=" * 70)
    print(f"LangGraph multi-agent system | {MODEL_NAME} | workspace: {WORKSPACE_DIR}")
    print("=" * 70)

    init: GraphState = {
        "messages": [AgentMessage(role="user", content=USER_PROMPT)],
        "decision_count": 0,
        "test_rmse_found": False,
        "run_id": 0,
        "mat_path": mat,
    }
    final = app.invoke(init, config={"recursion_limit": 60})

    lines = ["# Transcript\n"]
    for i, m in enumerate(final["messages"], 1):
        head = f"--- Turn {i} [{m.role.upper()}] {m.timestamp} ---"
        lines.append(f"## {head}\n\n{m.content}\n")
        print("\n" + head)
        print(m.content[:1500] + ("\n...[see transcript.md]" if len(m.content) > 1500 else ""))
    (WORKSPACE_DIR / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

    rm = [VERIFIED_RE.search(m.content) for m in final["messages"] if m.role == "executor"]
    rm = [r.group(1) for r in rm if r]
    print("\n" + "=" * 70)
    print(f"Decisions used: {final.get('decision_count')}/{MAX_DECISIONS}")
    print(f"Verified free-run Test RMSE: {rm[-1] if rm else 'NONE (not reached)'}")
    print(f"Full transcript: {WORKSPACE_DIR / 'transcript.md'}")
    png = WORKSPACE_DIR / "test_prediction.png"
    if png.exists():
        try:
            from IPython.display import Image, display  # type: ignore

            display(Image(filename=str(png)))
        except Exception:
            print(f"Plot: {png}")
