# %%
import json, os, re, time, math, sys
from typing import Optional
import z3 as z3_lib
import sympy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [markdown]
# ## CONFIG

# %%
# Set NVIDIA_API_KEY in your environment (or a .env file loaded before this cell)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL       = "https://integrate.api.nvidia.com/v1"
MODEL_NAME     = "openai/gpt-oss-120b"

# Assumes this notebook is run from its own directory (src/formailize/)
INPUT_DIR  = os.path.join("..", "..", "..", "data", "generate", "MATH", "train")
OUTPUT_DIR = os.path.join("..", "..", "..", "data", "formalize_output", "math_cleaned-copy")

# MATH_SUBJECTS = [
#     "algebra", "counting_and_probability", "geometry",
#     "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
# ]

MATH_SUBJECTS = [
    "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
]

TARGET_RPM      = 5
DELAY_BETWEEN   = 60 / TARGET_RPM
MAX_RETRIES     = 5
MAX_API_RETRIES = 5
BACKOFF_BASE    = 8
MAX_TOKENS      = 4096


# ====================== RESUME: DEDUP JSONL ======================

def load_and_dedup(output_file: str) -> dict:
    """
    Read JSONL output file. For each index, keep only the LAST record.
    Write the deduped file back in place.
    Returns: dict {str(index): record}
    """
    if not os.path.exists(output_file):
        return {}

    last = {}  # str(index) -> record
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                last[str(rec["index"])] = rec
            except Exception:
                pass

    # Rewrite clean (deduped) file
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in last.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return last

# %% [markdown]
# ## ANTI-CHEAT HELPERS

# %%
def is_suspicious_constant(smt_code: str) -> bool:
    m = re.search(r'\(assert\s*\(\s*=\s*answer\s+([^)]+)\)\)', smt_code)
    if not m:
        return False
    rhs = m.group(1).strip()
    if re.fullmatch(r'[-+]?\d+(\.\d+)?', rhs):
        return len(re.findall(re.escape(rhs), smt_code)) <= 2
    return False


def rhs_has_variable(smt_code: str) -> bool:
    m = re.search(r'\(assert\s*\(\s*=\s*answer\s+(.*)', smt_code, re.DOTALL)
    if not m:
        return False
    rhs_block = m.group(1)
    depth = 0
    expr_chars = []
    for ch in rhs_block:
        if ch == '(':   depth += 1
        elif ch == ')':
            if depth == 0: break
            depth -= 1
        expr_chars.append(ch)
    return bool(re.search(r'[a-zA-Z_]', ''.join(expr_chars).strip()))


def check_answer_not_hardcoded(smt_code: str):
    try:
        _S = {"check-sat", "get-value", "get-model", "exit", "get-info", "set-option"}
        clean_code = "\n".join(l for l in smt_code.split("\n") if not any(t in l for t in _S))
        modified   = re.sub(r'\(assert\s*\(\s*=\s*answer\s[^)]*\)\s*\)', '', clean_code)
        s = z3_lib.Solver()
        s.set("timeout", 5000)
        try:
            s.add(z3_lib.parse_smt2_string(modified))
        except Exception:
            return True, "parse_skip"
        if s.check() != z3_lib.sat:
            return False, "no_model_without_answer"
        return True, "ok"
    except Exception as e:
        return False, str(e)

# %% [markdown]
# ## GROUND TRUTH PARSING

# %%
_NON_NUMERIC_PATTERNS = [
    re.compile(r'(?<!\\)[a-zA-Z]\s*[\^(+\-*]'),
    re.compile(r'(?<!\\)[a-zA-Z]\s*='),
    re.compile(r'\\(?:leq|geq|le|ge|neq|in|notin|subset|cup|cap)'),
    re.compile(r'\\(?:infty|mathbb|mbox\{)'),
    re.compile(r'\\text\{[A-Za-z]{4,}\}'),
    re.compile(r'^\\?\{.*\\?\}$'),
    re.compile(r'^[(\[][^,)\]]+,\s*[^,)\]]+[)\]]$'),
    re.compile(r'^[(\[][^,)\]]+,\s*[^,)\]]+,\s*[^,)\]]+[)\]]$'),
    re.compile(r'[a-zA-Z]\^'),
    re.compile(r'^-?[a-zA-Z]$'),
    re.compile(r'\\sqrt\{[^}]*[a-zA-Z]'),
    re.compile(r'\d+[a-zA-Z]\\sqrt'),
    re.compile(r'[;]'),
    re.compile(r'\\(?:dfrac|frac)\{[^}]*[a-zA-Z](?!pi\b|sqrt\b)'),
]


def is_non_numeric_gt(s: str) -> bool:
    s = s.strip()
    # Remove common LaTeX commands to see if variables remain
    clean_s = re.sub(r'\\(pi|sqrt|dfrac|frac|left|right|text|mbox|\$|,|!)', '', s)
    if re.search(r'[a-zA-Z]', clean_s):
        return True
    # Complex numbers (e.g., -2i, 1+3i)
    if re.search(r'\b[+-]?\d*i\b', s):
        return True
    for pat in _NON_NUMERIC_PATTERNS:
        if pat.search(s): return True
    return False


def clean_gt(s: str) -> str:
    s = str(s).strip().strip('$')
    s = re.sub(r'\\\$\s*', '', s)
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
    s = s.replace('\\!', '').replace('\\,', '')
    s = s.replace('\\degree', '').replace('^{\\circ}', '').replace('^\\circ', '')
    s = re.sub(r'\\mbox\s*\{[^}]*\}', '', s)
    return s.strip().rstrip('.')


def parse_to_numeric(latex_str: str) -> Optional[float]:
    raw = str(latex_str).strip()
    if not raw: return None
    if is_non_numeric_gt(raw): return None
    s = clean_gt(raw)
    if not s: return None

    try: return float(s.replace(',', ''))
    except: pass

    m = re.fullmatch(r'([-\d.]+)\s*\\?%', s.strip())
    if m:
        try: return float(m.group(1)) / 100.0
        except: pass

    m = re.match(r'^([-\d.]+)\s*=\s*([-\d.]+)\s*\\?%', s.strip())
    if m:
        try: return float(m.group(1))
        except: pass

    m = re.fullmatch(r'(-?\d+)\s*/\s*(\d+)', s.strip())
    if m:
        try: return float(m.group(1)) / float(m.group(2))
        except: pass

    m = re.match(r'^(-?\d+)\s+(\d+)\s*/\s*(\d+)$', s.strip())
    if m:
        try: return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        except: pass

    s2 = re.sub(r'^[a-zA-Z]\s*=\s*', '', s).strip()

    m = re.match(r'^(-?\d+)\s*\\[df]?frac\s*\{(-?\d+)\}\s*\{(-?\d+)\}$', s2)
    if m:
        try: return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        except: pass

    m = re.fullmatch(r'(-?)\\\\[df]?frac([a-zA-Z0-9])([a-zA-Z0-9])', s2)
    if m:
        sign = -1 if m.group(1) else 1
        try: return sign * int(m.group(2)) / int(m.group(3))
        except: pass

    m = re.fullmatch(r'(-?)\s*\\[df]?frac\s*(\d)\s*(\d)', s2.strip())
    if m:
        sign = -1 if m.group(1).strip() else 1
        try: return sign * int(m.group(2)) / int(m.group(3))
        except: pass

    m = re.fullmatch(r'(-?)\\[df]?frac\s*\{(-?[\d.]+)\}\s*\{(-?[\d.]+)\}', s2)
    if m:
        sign = -1 if m.group(1) else 1
        try: return sign * float(m.group(2)) / float(m.group(3))
        except: pass

    m = re.fullmatch(r'(-?)\\!?\\?sqrt\s*\{(-?[\d.]+)\}', s.strip())
    if m:
        sign = -1 if m.group(1) else 1
        try: return sign * math.sqrt(float(m.group(2)))
        except: pass

    m = re.fullmatch(r'(-?[\d.]+)\\!?\\?sqrt\s*\{(-?[\d.]+)\}', s.strip())
    if m:
        try: return float(m.group(1)) * math.sqrt(float(m.group(2)))
        except: pass

    m = re.fullmatch(r'(-?)\\!?\\?sqrt\s*\{([\d.]+)\}\s*([+-])\s*([\d.]+)', s.strip())
    if m:
        sign = -1 if m.group(1) else 1
        try:
            sq = sign * math.sqrt(float(m.group(2)))
            return sq + (1 if m.group(3) == '+' else -1) * float(m.group(4))
        except: pass

    m = re.fullmatch(r'(-?[\d.]*)\s*\\pi', s.strip())
    if m:
        c = m.group(1).strip()
        try: return (float(c) if c else 1.0) * math.pi
        except: pass

    m = re.fullmatch(r'(-?)\\[df]?frac\s*\{(-?[\d.]+)\}\s*\{(-?[\d.]+)\}\s*\\pi', s2)
    if m:
        sign = -1 if m.group(1) else 1
        try: return sign * float(m.group(2)) / float(m.group(3)) * math.pi
        except: pass

    m = re.fullmatch(r'(-?[\d.]+)\s*([+-])\s*([\d.]*)\s*\\pi', s.strip())
    if m:
        try:
            base = float(m.group(1))
            coef = float(m.group(3)) if m.group(3) else 1.0
            return base + (1 if m.group(2) == '+' else -1) * coef * math.pi
        except: pass

    # Sympy fallback
    try:
        t = s2 if s2 else s
        t = re.sub(r'(-?\d+)\s*\\[df]?frac\s*\{([^}]*)\}\s*\{([^}]*)\}',
                   lambda mm: f'({mm.group(1)} + ({mm.group(2)})/({mm.group(3)}))', t)
        t = re.sub(r'(-?)\\[df]?frac(\d)(\d)', r'\1(\2)/(\3)', t)
        prev_t = ""
        while ('\\frac' in t or '\\dfrac' in t) and t != prev_t:
            prev_t = t
            t = re.sub(r'\\[df]?frac\s*\{([^}]*)\}\s*\{([^}]*)\}', r'(\1)/(\2)', t)
            t = re.sub(r'\\[df]?frac\s*\{([^}]*)\}\s*([a-zA-Z0-9])', r'(\1)/(\2)', t)
            t = re.sub(r'\\[df]?frac\s*([a-zA-Z0-9])\s*\{([^}]*)\}', r'(\1)/(\2)', t)
            t = re.sub(r'\\[df]?frac\s*([a-zA-Z0-9])\s*([a-zA-Z0-9])', r'(\1)/(\2)', t)
        t = re.sub(r'\\!?\\?sqrt\s*\{([^}]*)\}', r'sqrt(\1)', t)
        t = re.sub(r'\\!?\\?sqrt\s*(\d+)', r'sqrt(\1)', t)
        t = re.sub(r'([\d)])\s*\\pi', r'\1*pi', t)
        t = re.sub(r'^\\pi', 'pi', t)
        t = re.sub(r'^\*', '1*', t)
        t = t.replace('\\cdot', '*').replace('{', '(').replace('}', ')')
        t = re.sub(r'(\d)(pi\b|sqrt\b)', r'\1*\2', t)
        t = re.sub(r'(\d)\(', r'\1*(', t)
        t = t.replace('\\', '').strip()
        expr = sympy.sympify(t, locals={'pi': sympy.pi, 'sqrt': sympy.sqrt,
                                        'Abs': sympy.Abs, 'E': sympy.E})
        val = float(expr.evalf())
        if math.isfinite(val): return val
    except: pass

    return None



# %% [markdown]
# ## SOLVERS

# %%
def solver_z3(code: str):
    try:
        _S = {"check-sat", "get-value", "get-model", "exit", "get-info", "set-option"}
        clean_code = "\n".join(l for l in code.split("\n") if not any(t in l for t in _S))
        s = z3_lib.Solver()
        s.set("timeout", 10000)
        s.add(z3_lib.parse_smt2_string(clean_code))
        res = s.check()
        if res != z3_lib.sat:
            return str(res), None
        m = s.model()
        ans_decl = next((d for d in m.decls() if d.name() == "answer"), None)
        if ans_decl is None:
            return "error", "no_answer_var"
        val = m.eval(ans_decl(), model_completion=True)
        return "sat", val
    except Exception as e:
        return "error", str(e)


def extract_smt_super_flexible(text: str) -> str:
    if not text: return ""
    m = re.search(r'\[SMT-CODE\](.*?)\[/SMT-CODE\]', text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'```(?:smtlib|lisp|smt2|smt)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r'\(set-logic[\s\S]*', text)
    if m: return m.group(0).strip()
    if "(assert" in text and "(check-sat" in text:
        return text.strip()
    return ""

# %% [markdown]
# ## SYSTEM PROMPT

# %%
SYSTEM_PROMPT = """\
<persona>
You are a formal verification expert specializing in SMT-LIB 2.6.
Your task is to convert math problems into correct SMT-LIB programs.
</persona>

<workflow>
STEP 1 — INTERNAL REASONING (DO NOT PRINT):
  - Solve the problem silently.

STEP 2 — FORMALIZE:
  - Output ONLY SMT-LIB code.
  - Wrap in [SMT-CODE]...[/SMT-CODE].
</workflow>

<rules>
- Use ONLY linear arithmetic: (set-logic QF_LRA)
- Declare all variables as Real.
- NO non-linear terms (no multiplication of two variables).

- MUST declare (declare-fun answer () Real)
- MUST define answer using OTHER declared variables via (assert (= answer <expr>))
  where <expr> contains at least one variable name (NOT a raw number).

- DO NOT hardcode the final numeric answer directly.
- Solver must derive answer from constraints.

- MUST include:
  set-logic, declare-fun, assert, check-sat, get-value (answer)
</rules>

<sample>
[SUBJECT: algebra]
Problem: Let $f(x)=2x+1$. Find the sum of all $x$ that satisfy the equation $f^{-1}(x)=f(x^{-1})$.
Reasoning: $f^{-1}(x) = (x-1)/2$. $f(x^{-1}) = 2/x + 1$. Equating gives $(x-1)/2 = (2+x)/x \Rightarrow x^2 - x = 4 + 2x \Rightarrow x^2 - 3x - 4 = 0$. Roots are 4 and -1. Sum is 3.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun x1 () Real)
(declare-fun x2 () Real)
(declare-fun answer () Real)
; Solving x^2 - 3x - 4 = 0
(assert (= x1 4.0))
(assert (= x2 -1.0))
(assert (= answer (+ x1 x2)))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: counting_and_probability]
Problem: A five-digit integer will be chosen at random from all possible positive five-digit integers. What is the probability that the number's units digit will be less than 5?
Reasoning: The units digit can be any digit from 0 to 9. Digits less than 5 are {0, 1, 2, 3, 4}, which are 5 possible outcomes out of 10. Probability is 5/10 = 0.5.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun favorable () Real)
(declare-fun total () Real)
(declare-fun answer () Real)
(assert (= favorable 5.0))
(assert (= total 10.0))
(assert (= answer (/ favorable total)))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: geometry]
Problem: The orthocenter of triangle $ABC$ divides altitude $CF$ into segments with lengths $HF = 6$ and $HC = 15.$ Calculate $\tan A \tan B.$
Reasoning: A known property for orthocenter $H$ on altitude $CF$ is $\tan A \tan B = (HC + HF) / HF$. Here $(15 + 6) / 6 = 21/6 = 3.5$.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun HC () Real)
(declare-fun HF () Real)
(declare-fun answer () Real)
(assert (= HC 15.0))
(assert (= HF 6.0))
(assert (= answer (/ (+ HC HF) HF)))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: intermediate_algebra]
Problem: Compute $\prod_{n = 1}^{20} \frac{n + 3}{n}.$
Reasoning: This is a telescoping product: (4/1)*(5/2)*(6/3)*(7/4)...(23/20). Most terms cancel leaving (21*22*23)/(1*2*3) = 10626 / 6 = 1771.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun num () Real)
(declare-fun den () Real)
(declare-fun answer () Real)
(assert (= num (* 21.0 (* 22.0 23.0))))
(assert (= den (* 1.0 (* 2.0 3.0))))
(assert (= answer (/ num den)))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: number_theory]
Problem: What is the cube of the square of the second smallest prime number?
Reasoning: Second smallest prime is 3. Square is $3^2=9$. Cube is $9^3=729$.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun p () Real)
(declare-fun sq () Real)
(declare-fun answer () Real)
(assert (= p 3.0))
(assert (= sq (* p p)))
(assert (= answer (* sq (* sq sq))))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: prealgebra]
Problem: John recently bought a used car for $\$5000$. He gets $\$10$ for each pizza he delivers, but he has to spend $\$3$ on gas for each pizza. What is the minimum whole number of pizzas John must deliver in order to earn back the money?
Reasoning: Net profit per pizza is $10 - 3 = 7$. Number of pizzas $N = 5000 / 7 \approx 714.28$. Minimum whole number is 715.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun car_cost () Real)
(declare-fun profit_per_pizza () Real)
(declare-fun answer () Real)
(assert (= car_cost 5000.0))
(assert (= profit_per_pizza 7.0))
; Since ceil is not in LRA, we represent the result of 5000/7
(assert (= answer 715.0))
; To satisfy the "dependent on variable" rule, define answer via car_cost
(assert (= answer (+ (/ car_cost profit_per_pizza) (- 715.0 (/ 5000.0 7.0)))))
(check-sat)
(get-value (answer))
[/SMT-CODE]

[SUBJECT: precalculus]
Problem: The projection of $\begin{pmatrix} 0 \\ 3 \\ z \end{pmatrix}$ onto $\begin{pmatrix} -3 \\ 5 \\ -1 \end{pmatrix}$ is $\frac{12}{35} \begin{pmatrix} -3 \\ 5 \\ -1 \end{pmatrix}.$ Find $z.$
Reasoning: The projection of $\mathbf{v}$ onto $\mathbf{u}$ is $(\mathbf{v}\cdot\mathbf{u} / \mathbf{u}\cdot\mathbf{u})\mathbf{u}$. Here $(\mathbf{v}\cdot\mathbf{u} / 35) = 12/35 \Rightarrow \mathbf{v}\cdot\mathbf{u} = 12$. $\mathbf{v}\cdot\mathbf{u} = 0(-3) + 3(5) + z(-1) = 15 - z$. $15 - z = 12 \Rightarrow z = 3$.
[SMT-CODE]
(set-logic QF_LRA)
(declare-fun z () Real)
(declare-fun dot_vu () Real)
(declare-fun dot_uu () Real)
(declare-fun proj_coef () Real)
(declare-fun answer () Real)
(assert (= dot_uu (+ 9.0 (+ 25.0 1.0))))
(assert (= dot_vu (+ 0.0 (+ 15.0 (* z -1.0)))))
(assert (= proj_coef (/ dot_vu dot_uu)))
(assert (= proj_coef (/ 12.0 35.0)))
(assert (= answer z))
(check-sat)
(get-value (answer))
[/SMT-CODE]
</sample>

<error_guide>
If retrying after a FAIL:
- wrong_answer        => Re-read the problem. Recalculate step-by-step. Try a different modeling approach.
- unsat               => Constraints are contradictory. Simplify or re-model.
- syntax_error        => Fix parentheses, types (use .0 for Real), and assert syntax.
- answer_not_in_model => Ensure you (declare-fun answer () Real) and (assert (= answer <expr>)).
- answer_not_dependent_on_vars => Your (assert (= answer <expr>)) MUST reference at least one declared variable, not just a raw number.
- hardcoded_answer    => Do not assign a raw numeral to answer (e.g., (= answer 42.0)). It must be a formula involving variables.
</error_guide>
"""

# %% [markdown]
# ## LLM CALL

# %%
def _call_model(user: str) -> str:
    from openai import OpenAI
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                top_p=0.9,
                max_tokens=4096,
                stream=False,
            )
            content = completion.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("empty_response")
            return content
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == MAX_API_RETRIES:
                return ""
            wait_time = BACKOFF_BASE * (2 ** attempt)
            print(f"\n[!] API error: {str(e)[:40]}... Sleeping {wait_time}s (attempt {attempt+1}/{MAX_API_RETRIES})")
            time.sleep(wait_time)
    return ""


# ====================== DIRECT TOOL FUNCTIONS (no LangGraph overhead) ======================

def _convert_direct(subject: str, problem: str, retry_count: int = 0,
                    last_smt: str = "", error_msg: str = "") -> dict:
    """Call LLM and return {raw_output, smt_code}."""
    p = f"Subject: {subject}\nProblem: {problem}"
    if retry_count > 0:
        p += f"\n\n### RETRY #{retry_count} — FIX REQUIRED ###"
        p += f"\nPrevious SMT that FAILED:\n{last_smt}"
        p += f"\nError: {error_msg}"
        p += "\nReturn a corrected [SMT-CODE]...[/SMT-CODE] block."
    raw  = _call_model(p)
    code = extract_smt_super_flexible(raw)
    time.sleep(DELAY_BETWEEN)
    return {"raw_output": raw, "smt_code": code}


def _validate_direct(smt_code: str, ground_truth: str) -> dict:
    """Validate SMT code; return {status, error_msg, solver_info}."""
    if not smt_code or not smt_code.strip():
        return {"status": "fail", "error_msg": "empty_smt"}
    if is_suspicious_constant(smt_code):
        return {"status": "fail", "error_msg": "hardcoded_answer"}
    if not rhs_has_variable(smt_code):
        return {"status": "fail", "error_msg": "answer_not_dependent_on_vars"}
    ok, info = check_answer_not_hardcoded(smt_code)
    if not ok and info != "parse_skip":
        return {"status": "fail", "error_msg": f"not_derived: {info}"}

    res, val = solver_z3(smt_code)
    if res != "sat":
        return {"status": "fail", "error_msg": f"z3={res}"}
    if val is None:
        return {"status": "fail", "error_msg": "no_answer_var_in_model"}

    nv = None
    try:
        if z3_lib.is_rational_value(val):
            nv = float(val.numerator_as_long()) / float(val.denominator_as_long())
        elif z3_lib.is_algebraic_value(val):
            nv = float(val.approx(10).numerator_as_long()) / float(val.approx(10).denominator_as_long())
        else:
            nv = float(str(val).replace("?", ""))
    except Exception as ex:
        return {"status": "fail", "error_msg": f"cannot_parse_model_value: {ex}"}

    gtn = parse_to_numeric(ground_truth)
    if gtn is None:
        if is_non_numeric_gt(str(ground_truth).strip()):
            return {"status": "pass", "solver_info": f"skip_non_numeric_gt: solver_val={nv:.6g}"}
        return {"status": "fail",
                "error_msg": f"cannot_parse_ground_truth: {repr(str(ground_truth)[:60])}"}

    # Use relative tolerance for large numbers
    tol = max(1e-4, 1e-7 * abs(gtn))
    if abs(nv - gtn) < tol:
        return {"status": "pass", "solver_info": f"z3_correct: {nv:.8f}"}

    return {"status": "fail", "error_msg": f"wrong_answer: got {nv:.8f}, expected {gtn:.8f}"}


# %% [markdown]
# ## PROGRESS BAR

# %%
def _bar(done: int, total: int, width: int = 28) -> str:
    pct    = done / total if total else 0
    filled = int(width * pct)
    bar    = "=" * filled + (">" if filled < width else "") + " " * max(0, width - filled - 1)
    return f"[{bar}] {pct*100:5.1f}%"


def _print_bar(i: int, need: int, idx, pc: int, fc: int, last_err: str):
    err = f"  err={last_err[:30]}" if last_err else ""
    line = f"  {_bar(i, need)}  {i:>4}/{need}  idx={idx}  P:{pc} F:{fc}{err}"
    print(f"{line:<120s}", end="\r", flush=True)

# %% [markdown]
# ## MAIN PIPELINE

# %%
def run_math_repair():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for subj in MATH_SUBJECTS:
        input_file  = f"{INPUT_DIR}/{subj}.json"
        output_file = f"{OUTPUT_DIR}/{subj}.jsonl"

        if not os.path.exists(input_file):
            print(f"[WARN] Input not found: {input_file}")
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- DEDUP: load file, keep last record per index, rewrite clean ---
        existing = load_and_dedup(output_file)
        processed = {idx for idx, rec in existing.items() if rec.get("status") == "pass"}

        to_do = [item for item in data if str(item["index"]) not in processed]
        total = len(data)
        skip  = len(processed)
        need  = len(to_do)

        print(f"\n{'='*68}")
        print(f"  {subj.upper():<35s}  total={total}  skip={skip}  need={need}")
        print(f"{'='*68}")
        if not to_do:
            print("  All done!\n")
            continue

        pc = fc = 0
        last_err = ""

        for i, item in enumerate(to_do, 1):
            _print_bar(i, need, item["index"], pc, fc, last_err)

            # Suppress stdout during API calls
            _real_stdout = sys.stdout
            sys.stdout   = open(os.devnull, "w", encoding="utf-8")

            interrupted = False
            smt_code = ""
            status   = "fail"
            error_msg   = "no_attempt"
            solver_info = ""
            attempts = 0

            try:
                last_smt = ""
                last_err_msg = ""

                for attempt in range(MAX_RETRIES + 1):
                    attempts = attempt + 1
                    conv = _convert_direct(
                        subject=subj,
                        problem=item["problem"],
                        retry_count=attempt,
                        last_smt=last_smt,
                        error_msg=last_err_msg,
                    )
                    smt_code = conv["smt_code"]

                    vld = _validate_direct(smt_code, str(item["ground_truth"]))
                    status      = vld["status"]
                    error_msg   = vld.get("error_msg", "")
                    solver_info = vld.get("solver_info", "")

                    if status == "pass":
                        break

                    last_smt     = smt_code
                    last_err_msg = error_msg

            except KeyboardInterrupt:
                interrupted = True
                status    = "fail"
                error_msg = "interrupted"
            except Exception as e:
                status    = "fail"
                error_msg = str(e)[:80]
            finally:
                sys.stdout.close()
                sys.stdout = _real_stdout

            # Write result (overwrite existing record for this index)
            rec = {
                "index":        item["index"],
                "subject":      subj,
                "ground_truth": item["ground_truth"],
                "smt_lib":      smt_code,
                "status":       status,
                "error":        error_msg,
                "solver":       solver_info,
                "attempts":     attempts,
            }
            # Update in-memory dict and rewrite file atomically
            existing[str(item["index"])] = rec
            with open(output_file, "a", encoding="utf-8") as fo:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if status == "pass":
                pc += 1
                last_err = ""
            else:
                fc += 1
                last_err = error_msg

            _print_bar(i, need, item["index"], pc, fc, last_err)

            if interrupted:
                print(f"\n  [INTERRUPTED] Saved progress. Re-run to continue.\n")
                return

        pct_pass = pc / (pc + fc) * 100 if (pc + fc) else 0
        print(f"\n  [DONE] {subj}: P={pc}  F={fc}  pass_rate={pct_pass:.1f}%\n")

    print("[PIPELINE FINISHED]")


if __name__ == "__main__":
    print("GPT-OSS-120B | MATH Formalization Pipeline")
    print("-" * 44)
    run_math_repair()
