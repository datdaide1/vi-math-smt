# %% [markdown]
# ## 0. Cài đặt

# %%
# !pip install langgraph langchain langchain-openai z3-solver openai

# %% [markdown]
# ## 1. Imports & Config

# %%
import json, os, re, time
from typing import TypedDict, Literal

import z3 as z3_lib
from openai import OpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from tqdm import tqdm
import time

# ====================== CONFIG ======================
# Set NVIDIA_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL   = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "openai/gpt-oss-120b"

# Assumes this notebook is run from its own directory (src/formailize/)
INPUT_FILE   = os.path.join("..", "..", "..", "data", "generate", "gsm8k", "train.jsonl")
# FAIL_FILE    = os.path.join("..", "..", "..", "data", "formalize_output", "gsm8k", "gsm8k-fail.jsonl")
OUTPUT_FILE  = os.path.join("..", "..", "..", "data", "formalize_output", "gsm8k", "gsm8k-smt-nvidia-FULL.jsonl")
# REPAIR_FILE  = os.path.join("..", "..", "..", "data", "formalize_output", "gsm8k", "gsm8k-repair.jsonl")

TARGET_RPM    = 8
DELAY_BETWEEN = (60 / TARGET_RPM) + 1
MAX_RETRIES   = 3
MAX_TOKENS    = 4096
TEMPERATURE   = 0

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"Config OK — delay {DELAY_BETWEEN:.1f}s/req | max_retry={MAX_RETRIES}")
# print(f"INPUT (FAIL)  : {FAIL_FILE}")
print(f"OUTPUT (new)  : {OUTPUT_FILE}")


# %% [markdown]
# ## 2. Prompt & Helpers

# %%
SYSTEM_PROMPT = """\
<persona>
You are a formal verification expert specializing in SMT-LIB 2.6 and mathematical reasoning.
Your sole task is to convert math word problems into correct, complete SMT-LIB 2.6 scripts.
You are NOT a general-purpose assistant. You do NOT explain — you formalize.
</persona>

<rules>
- ALWAYS wrap the SMT-LIB code strictly inside [SMT-CODE] and [/SMT-CODE] tags.
- ALWAYS include ALL required sections in the exact order specified below.
- NEVER omit any required section — a missing section causes validation failure.
- NEVER use non-linear arithmetic (e.g. x * y where both are variables).
- NEVER use (push), (pop), or incremental SMT commands.
- Brief reasoning (< 5 lines) before [SMT-CODE] is allowed; nothing after [/SMT-CODE].
- If a previous attempt failed, read the error message carefully and fix the root cause.
</rules>

<required_sections>
The SMT-LIB script MUST contain ALL of the following, in order:
  1. (set-logic QF_LRA)
  2. (declare-fun <name> () Real)  — one per variable; MUST include `answer`
  3. (assert ...)                  — one or more constraints encoding the problem
  4. (check-sat)
  5. (get-value (answer))          — OR (get-model), choose ONE
</required_sections>

<output_format>
Use this exact template:

[SMT-CODE]
(set-logic QF_LRA)
(declare-fun x () Real)
(declare-fun answer () Real)
(assert (= x 5.0))
(assert (= answer (* x 2.0)))
(check-sat)
(get-value (answer))
[/SMT-CODE]
</output_format>

<error_guide>
When retrying after a FAIL, apply the following fixes:
- incomplete_smtlib   → add the missing sections listed in the error.
- syntax_error        → fix parentheses, declare-fun types (must be Real), assert syntax.
- unsat               → constraints are contradictory; revisit the assert logic.
- wrong_answer        → re-check the arithmetic in assert expressions.
- missing tags        → wrap all SMT code inside [SMT-CODE]...[/SMT-CODE].
- answer_not_in_model → add (declare-fun answer () Real) and (assert (= answer <expr>)).
</error_guide>
"""

_REQUIRED_SECTIONS = [
    ("set-logic",           r"\(set-logic"),
    ("declare-fun",         r"\(declare-fun"),
    ("assert",              r"\(assert"),
    ("check-sat",           r"\(check-sat\)"),
    ("get-value|get-model", r"\(get-value|\(get-model\)"),
]

def _check_completeness(code: str) -> list:
    missing = []
    for name, pattern in _REQUIRED_SECTIONS:
        if not re.search(pattern, code, re.IGNORECASE):
            missing.append(name)
    if "answer" not in code:
        missing.append("variable:answer")
    return missing

def _extract_smt(raw: str) -> str:
    if not raw:
        return "ERROR: empty_response"
    match = re.search(
        r'\[SMT-CODE\]\s*(.*?)\s*\[/SMT-CODE\]',
        raw, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return "ERROR: missing [SMT-CODE]/[/SMT-CODE] tags"
    text = match.group(1)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("("):
            line = line.split(";")[0].strip()
            if line:
                lines.append(line)
    code = "\n".join(lines).strip()
    if not code:
        return "ERROR: empty_smtlib_block"
    missing = _check_completeness(code)
    if missing:
        return (
            f"ERROR: incomplete_smtlib — missing: {', '.join(missing)}. "
            "Required: set-logic, declare-fun (answer), assert, check-sat, get-value or get-model."
        )
    return code

def _validate_with_z3(smt_code: str, ground_truth: str) -> str:
    if not smt_code or smt_code.startswith("ERROR:"):
        return f"FAIL: upstream_error — {smt_code}"
    missing = _check_completeness(smt_code)
    if missing:
        return (
            f"FAIL: incomplete_smtlib — missing: {', '.join(missing)}. "
            "Must include: set-logic, declare-fun answer, assert, check-sat, get-value or get-model."
        )
    try:
        _STRIP = {"check-sat", "get-value", "get-model", "exit", "get-info", "set-option"}
        parse_code = "\n".join(
            l for l in smt_code.split("\n")
            if not any(t in l for t in _STRIP)
        )
        s = z3_lib.Solver()
        s.set("timeout", 5000)
        try:
            s.add(z3_lib.parse_smt2_string(parse_code))
        except Exception as pe:
            return (
                f"FAIL: syntax_error — {str(pe)[:100]}. "
                "Check declare-fun types (Real), assert syntax, parentheses."
            )
        result = s.check()
        if result == z3_lib.unsat:
            return "FAIL: unsat — constraints are contradictory. Re-check assert statements."
        if result == z3_lib.unknown:
            return "FAIL: unknown — solver timed out."
        m = s.model()
        ans_var = next((d() for d in m.decls() if d.name() == "answer"), None)
        if ans_var is None:
            return (
                "FAIL: answer_not_in_model — "
                "add (declare-fun answer () Real) and (assert (= answer <expr>))."
            )
        eval_val = m.eval(ans_var, model_completion=True)
        try:
            val = (
                float(eval_val.numerator_as_long()) / float(eval_val.denominator_as_long())
                if z3_lib.is_rational_value(eval_val)
                else float(str(eval_val).replace("?", ""))
            )
        except Exception:
            return f"FAIL: cannot_parse_answer_value '{eval_val}'."
        gt_val = float(str(ground_truth).replace(",", ""))
        if abs(val - gt_val) < 1e-4:
            return "PASS"
        return (
            f"FAIL: wrong_answer — got {val}, expected {gt_val}. "
            "Re-check assert constraints and arithmetic."
        )
    except Exception as e:
        return f"FAIL: crash — {str(e)[:80]}"

print("Prompt & helpers defined.")


# %% [markdown]
# ## 3. Tool Definition & Agent Setup

# %%
@tool
def tool_convert(problem: str, retry_count: int = 0, last_error: str = "") -> str:
    """Uses LLM to convert a math word problem into a formal SMT-LIB 2.6 script.
    Always call this FIRST.
    Returns: A JSON string containing 'smt_code'.
    """
    if last_error and retry_count > 0:
        user_content = (
            f"Previous attempt (#{retry_count}) FAILED with the following error:\n"
            f"  {last_error}\n\n"
            f"Please carefully fix the issue and regenerate the complete SMT-LIB script.\n\n"
            f"Problem:\n{problem}"
        )
    else:
        user_content = f"Problem:\n{problem}"

    max_api_retries = 5
    for attempt in range(max_api_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0,
                top_p=1,
                max_tokens=4096,
                stream=False
            )
            
            raw = completion.choices[0].message.content or ""

            if not raw:
                return json.dumps({"smt_code": "ERROR: empty_response"}, ensure_ascii=False)
            
            smt_code = _extract_smt(raw)
            time.sleep(DELAY_BETWEEN)
            return json.dumps({"smt_code": smt_code}, ensure_ascii=False)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg:
                wait_time = 2 ** attempt + 5
                print(f"\n[!] API 429 Too Many Requests, sleeping {wait_time}s... (attempt {attempt+1}/{max_api_retries})")
                time.sleep(wait_time)
                continue
            return json.dumps({"smt_code": f"ERROR: agent_request_failed — {err_msg[:60]}"}, ensure_ascii=False)
    return json.dumps({"smt_code": f"ERROR: agent_request_failed — 429 Too Many Requests max retries exceeded"}, ensure_ascii=False)

@tool
def tool_validate(smt_code: str, ground_truth: str) -> str:
    """Validates the generated SMT-LIB 2.6 code against the ground truth answer using Z3.
    Call this AFTER tool_convert to verify correctness.
    Returns: A JSON string containing 'status' ("pass" or "fail"), and 'error_msg'.
    """
    if not smt_code or smt_code.startswith("ERROR"):
        return json.dumps({"status": "fail", "error_msg": smt_code}, ensure_ascii=False)

    result = _validate_with_z3(smt_code, ground_truth)
    if result == "PASS":
        return json.dumps({"status": "pass", "error_msg": ""}, ensure_ascii=False)
    return json.dumps({"status": "fail", "error_msg": result}, ensure_ascii=False)

print("Tools initialized.")

# ====================================================================
# AGENT SETUP
# ====================================================================
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    max_retries=10
)

agent_executor = create_react_agent(llm, tools=[tool_convert, tool_validate])
print("Agent Executor ready.")


# %%
def run_agent_for_sample(sample: dict) -> dict:
    """Chạy React agent cho 1 sample. Trả về record kết quả."""
    idx          = sample["index"]
    question     = sample["question"]
    ground_truth = str(sample.get("ground_truth") or sample.get("answer") or "")

    sys_prompt = f"""\
<persona>
You are a formalization orchestrator. Use the explicit tools provided to solve this math problem.
</persona>

<problem_context>
<problem>{question}</problem>
<ground_truth>{ground_truth}</ground_truth>
</problem_context>

<workflow>
Follow this strict algorithm:
1. Call `tool_convert(problem="{question}")`.
2. Extract the `smt_code` from the JSON response and call `tool_validate` with that `smt_code` and `ground_truth="{ground_truth}"`.
3. If tool_validate returns status "fail", you MUST retry by calling `tool_convert` again with an incremented `retry_count`, and the `error_msg` returned from validation as `last_error`.
4. You may retry up to {MAX_RETRIES} times. Stop immediately when tool_validate returns status "pass".
</workflow>
"""

    try:
        max_agent_retries = 10
        result = None
        for attempt in range(max_agent_retries):
            try:
                result = agent_executor.invoke({"messages": [{"role": "user", "content": sys_prompt}]}, {"recursion_limit": 20})
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower():
                    wait_time = 2 ** attempt + 5
                    import time
                    print(f"\n[!] LangGraph 429 Too Many Requests, sleeping {wait_time}s... (attempt {attempt+1}/{max_agent_retries})")
                    time.sleep(wait_time)
                    if attempt == max_agent_retries - 1:
                        raise e
                    continue
                else:
                    raise e
        
        msgs = result["messages"]
        smt_code = ""
        status = "fail"
        error_msg = "Agent failed to complete validation."
        attempts = 0

        for msg in msgs:
            if getattr(msg, "type", "") == "tool":
                if msg.name == "tool_convert":
                    attempts += 1
                    try:
                        d = json.loads(msg.content)
                        smt_code = d.get("smt_code", smt_code)
                    except: pass
                elif msg.name == "tool_validate":
                    try:
                        d = json.loads(msg.content)
                        status = d.get("status", status)
                        error_msg = d.get("error_msg", error_msg)
                    except: pass

        passed = (status == "pass")
        return {
            "index":    idx,
            "question": question,
            "smt_lib":  smt_code,
            "z3_score": 1 if passed else 0,
            "z3_error": "NONE" if passed else error_msg,
            "retries":  attempts - 1 if attempts > 0 else 0,
        }
    except Exception as e:
        return {
            "index":    idx,
            "question": question,
            "smt_lib":  "",
            "z3_score": 0,
            "z3_error": f"agent_crash: {str(e)[:120]}",
            "retries":  0,
        }

print("Runner defined.")


# %% [markdown]
# ## 5. Main Pipeline

# %%
def load_all_samples():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Không tìm thấy {INPUT_FILE}")
        return []
    
    samples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    
    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")
    return samples

def load_fail_samples():
    if not os.path.exists(FAIL_FILE):
        print(f"❌ Không tìm thấy {FAIL_FILE}")
        return []
    
    samples = []
    with open(FAIL_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    
    print(f"Loaded {len(samples)} fail samples from {FAIL_FILE}")
    return samples

def load_processed_results():
    if not os.path.exists(OUTPUT_FILE):
        return {}
    
    processed = {}
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed[result.get("index")] = result
                except: pass
    except: pass
    return processed

def run_full_pipeline():
    all_samples = load_all_samples()
    if not all_samples: return
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    processed_results = load_processed_results()
    passed_indices = set()
    for idx, result in processed_results.items():
        if result.get("z3_score") == 1:
            passed_indices.add(idx)
    
    pending_samples = [s for s in all_samples if s.get("index") not in passed_indices]
    
    pass_count = fail_count = 0
    new_results = []
    
    print(f"\n{'='*60}")
    print(f"  Already PASS: {len(passed_indices)}/{len(all_samples)}")
    print(f"  PROCESSING: {len(pending_samples)} pending | max_retry={MAX_RETRIES}")
    print(f"{'='*60}\n")
    
    for i, sample in enumerate(pending_samples):
        idx = sample.get("index")
        print(f"[{i+1}/{len(pending_samples)}] Index {idx}", end=" ... ", flush=True)
        
        result = run_agent_for_sample(sample)
        new_results.append(result)
        
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        if result["z3_score"] == 1:
            pass_count += 1
            print(f"✅ PASS  (retries={result['retries']})")
        else:
            fail_count += 1
            err_short = str(result["z3_error"])[:60]
            print(f"❌ FAIL  {err_short}")
            
    total_pass = len(passed_indices) + pass_count
    total_fail = len(all_samples) - total_pass
    
    print(f"\n{'='*60}\n  KẾT QUẢ TOÀN BỘ ({len(all_samples)} samples)\n{'='*60}")
    print(f"NEW — PASS     : {pass_count}/{len(pending_samples)}")
    print(f"NEW — FAIL     : {fail_count}/{len(pending_samples)}")
    if len(pending_samples) > 0: print(f"NEW — Rate     : {100*pass_count/len(pending_samples):.1f}%")
    print(f"\nTOTAL — PASS   : {total_pass}/{len(all_samples)}")
    print(f"TOTAL — FAIL   : {total_fail}/{len(all_samples)}")
    print(f"TOTAL — Rate   : {100*total_pass/len(all_samples):.1f}%")
    print(f"\n✅ Output → {OUTPUT_FILE}")

# def run_repair_pipeline():
#     fail_samples = load_fail_samples()
#     if not fail_samples: return
    
#     repair_dir = os.path.dirname(REPAIR_FILE)
#     if repair_dir: os.makedirs(repair_dir, exist_ok=True)
    
#     with open(REPAIR_FILE, "w", encoding="utf-8") as f: pass
    
#     pass_count = fail_count = 0
#     results = []
    
#     print(f"\n{'='*60}")
#     print(f"  RERUN: {len(fail_samples)} câu fail | max_retry={MAX_RETRIES}")
#     print(f"{'='*60}\n")
    
#     for i, sample in enumerate(fail_samples):
#         idx = sample.get("index")
#         print(f"[{i+1}/{len(fail_samples)}] Index {idx}", end=" ... ", flush=True)
        
#         result = run_agent_for_sample(sample)
#         results.append(result)
        
#         with open(REPAIR_FILE, "a", encoding="utf-8") as f:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
#         if result["z3_score"] == 1:
#             pass_count += 1
#             print(f"✅ PASS  (retries={result['retries']})")
#         else:
#             fail_count += 1
#             err_short = str(result["z3_error"])[:60]
#             print(f"❌ FAIL  {err_short}")
            
#     print(f"\n{'='*60}\n  KẾT QUẢ RERUN 22 CÂU\n{'='*60}")
#     print(f"PASS     : {pass_count}/{len(fail_samples)}")
#     if len(fail_samples) > 0: print(f"Rate     : {100*pass_count/len(fail_samples):.1f}%")
#     print(f"\n✅ Output → {OUTPUT_FILE}")
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")


# %% [markdown]
# ## 6. Execute Run

# %%
# run_full_pipeline()
# run_repair_pipeline()

# %% [markdown]
# ## 7. Metrics & Stats

# %%
import pandas as pd

records = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

if records:
    df = pd.DataFrame(records)
    print(f"📊 RERUN RESULTS:")
    print(f"Total  : {len(df)}")
    print(f"PASS   : {int(df['z3_score'].sum())}")
    print(f"FAIL   : {int((df['z3_score']==0).sum())}")
    print(f"Rate   : {df['z3_score'].mean()*100:.2f}%")
    
    if "z3_error" in df.columns:
        print("\n❌ Lỗi chi tiết:")
        fails = df[df["z3_score"]==0]
        for idx, row in fails.iterrows():
            print(f"  Index {row['index']}: {row['z3_error'][:80]}")
    
    if "retries" in df.columns:
        rd = df[df["retries"] > 0]
        if len(rd) > 0:
            print(f"\n🔄 Retried: {len(rd)} | avg={rd['retries'].mean():.2f}")
else:
    print("❌ Không có data.")
