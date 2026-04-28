"""
KAGGLE NOTEBOOK 5: Inference — Vi-MathLM-Aug (in-distribution + OOD)
=====================================================================
Runtime: T4 GPU, ~5 hours
Input: Vi-MathLM-Aug adapter (from 03_dpo_train_augmented.py) + test data
Output: gsm8k_predictions.jsonl, math_predictions.jsonl (in-distribution) and
        svamp_predictions.jsonl, asdiv_predictions.jsonl (out-of-distribution)

Only Vi-MathLM-Aug is evaluated out-of-distribution (Vi-SVAMP, Vi-ASDiv) —
these two sets never appear in any training stage, per Chương 3.

Instructions:
1. Upload the Vi-MathLM-Aug adapter (output of 03_dpo_train_augmented.py) as a
   Kaggle Dataset (e.g. "model-c-dpo")
2. Upload test data as a Kaggle Dataset:
   - gsm8k_test_vietnamese.json
   - math/ folder (7 _vi.jsonl files)
   - svamp-vi.jsonl
   - asdiv-validation-vi.jsonl
3. Enable GPU (T4), Internet ON
4. Paste this entire script and run
"""

# ====================== CELL 1: Install ======================
# !pip uninstall unsloth unsloth-zoo -y
# !pip install --upgrade --no-cache-dir unsloth unsloth-zoo
# !pip install --no-deps trl peft accelerate bitsandbytes

# ====================== CELL 2: Config =======================
import os, json, re, warnings, unicodedata
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -- Paths (adjust Dataset names) --
MODEL_PATH = "/kaggle/input/model-c-dpo/model_aug_dpo"
GSM8K_TEST = "/kaggle/input/smt-test-data/gsm8k_test_vietnamese.json"
MATH_TEST_DIR = "/kaggle/input/smt-test-data/math"
SVAMP_TEST = "/kaggle/input/smt-test-data/svamp-vi.jsonl"
ASDIV_TEST = "/kaggle/input/smt-test-data/asdiv-validation-vi.jsonl"
OUTPUT_DIR = "/kaggle/working/predictions/model_c_dpo"

MAX_NEW_TOKENS = 1024

# Prompt MUST MATCH training format
INSTR = "Bạn là trợ lý toán học. Giải bài toán sau từng bước bằng Tiếng Việt. Đưa kết quả cuối cùng theo định dạng \\boxed{kết quả}."

PROMPT = """<s>[INST] """ + INSTR + """

{problem} [/INST]
"""

# ====================== CELL 3: Utils ========================
def normalize_unicode(text: str) -> str:
    """NFC normalization to fix Vietnamese Mojibake."""
    if not isinstance(text, str): return text
    return unicodedata.normalize('NFC', text)

def clean_problem(problem: str) -> str:
    """Clean problem text to match training distribution."""
    if not problem: return ""
    problem = normalize_unicode(problem)
    problem = re.sub(
        r'^(Hãy\s+)?Giải\s+bài\s+toán\s+sau\.?\s*\n*',
        '', problem, flags=re.IGNORECASE
    ).strip()
    return problem

def extract_answer(text):
    """Extract final answer from model output — priority-ordered."""
    if not text:
        return None
    m = re.search(r'Đáp án cuối cùng là:\s*\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m: return m.group(1).strip()
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m: return m.group(1).strip()
    m = re.search(r'Đáp án cuối cùng là:\s*(.+?)(?:\n|$)', text)
    if m:
        ans = re.sub(r'[*_`]', '', m.group(1).strip())
        if ans: return ans
    m = re.search(r'####\s*(.+?)$', text, re.MULTILINE)
    if m: return m.group(1).strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if nums: return nums[-1]
    return None


def normalize_answer(ans):
    if ans is None: return ""
    ans = str(ans).strip()
    ans = re.sub(r'^\$\$?|\$\$?$', '', ans)
    ans = re.sub(r'\\(text|mathrm|mathbf|unit)\{(.+?)\}', r'\2', ans)
    ans = re.sub(r'(cm\^2|cm|inch|độ|degrees|đơn vị diện tích|đơn vị khối)', '', ans, flags=re.IGNORECASE)
    # Also strip trailing Vietnamese-unit parentheses, e.g. "9 (quả táo)" -> "9"
    ans = re.sub(r'\([^)]*\)', '', ans)
    ans = re.sub(r'\\frac(\d)(\d)', r'(\1)/(\2)', ans)
    for _ in range(3):
        ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', ans)
    ans = ans.replace('−', '-')
    ans = re.sub(r'\s+', '', ans)
    ans = re.sub(r',', '', ans)
    ans = re.sub(r'\\', '', ans)
    ans = re.sub(r'[{}]', '', ans)
    try:
        m = re.match(r'^\(?(-?\d+\.?\d*)\)?/\(?(-?\d+\.?\d*)\)?$', ans)
        if m:
            v = float(m.group(1)) / float(m.group(2))
            return str(int(v)) if v == int(v) else str(round(v, 6))
        v = float(ans)
        return str(int(v)) if v == int(v) else str(round(v, 6))
    except:
        pass
    return ans.lower().strip()


def check_answer(pred, gt):
    p, g = normalize_answer(pred), normalize_answer(str(gt))
    if not p or not g: return False
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return False


def check_format(text):
    has_boxed = bool(re.search(r'\\boxed\{', text))
    has_final = bool(re.search(r'Đáp án cuối cùng là', text))
    return has_boxed or has_final


def post_process(text):
    if "Đáp án cuối cùng là" in text:
        parts = text.split("Đáp án cuối cùng là")
        after = parts[-1].split('\n')[0]
        text = parts[0] + "Đáp án cuối cùng là" + after
    return text.strip()


# ====================== CELL 4: Load Test Data ===============
def load_gsm8k(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    entries = []
    for i, item in enumerate(data):
        gt = item.get("ground_truth", "")
        if not gt and "answer" in item:
            m = re.search(r'####\s*(.+?)$', item["answer"], re.MULTILINE)
            gt = m.group(1).strip() if m else ""
        entries.append({
            "index": i, "subject": "arithmetic", "level": 0,
            "problem": item.get("question", item.get("problem", "")),
            "solution": item.get("answer", ""), "ground_truth": str(gt),
        })
    return entries


def load_math(directory):
    entries, idx = [], 0
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".jsonl"): continue
        for line in open(os.path.join(directory, fn), encoding="utf-8"):
            line = line.strip()
            if not line: continue
            try: item = json.loads(line)
            except: continue
            gt = ""
            sol = item.get("solution", "")
            m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', sol)
            if m: gt = m.group(1)
            entries.append({
                "index": idx,
                "subject": item.get("subject", item.get("type", "unknown")).lower(),
                "level": item.get("level", "Unknown"),
                "problem": item.get("problem", ""), "solution": sol,
                "ground_truth": gt,
            })
            idx += 1
    return entries


def load_svamp(path):
    """Vi-SVAMP: fields Body/Question/Answer/Equation (capitalized)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            problem = f"{item.get('Body', '')} {item.get('Question', '')}".strip()
            entries.append({
                "index": i, "subject": "svamp_ood",
                "level": item.get("Type", "unknown"),
                "problem": problem, "solution": item.get("Equation", ""),
                "ground_truth": str(item.get("Answer", "")),
            })
    return entries


def load_asdiv(path):
    """Vi-ASDiv: fields body/question/answer/formula (lowercase)."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            problem = f"{item.get('body', '')} {item.get('question', '')}".strip()
            entries.append({
                "index": i, "subject": "asdiv_ood",
                "level": item.get("solution_type", "unknown"),
                "problem": problem, "solution": item.get("formula", ""),
                "ground_truth": str(item.get("answer", "")),
            })
    return entries

# ====================== CELL 5: Load Model ===================
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Vi-MathLM-Aug loaded")

# ====================== CELL 6: Inference ====================
def run_inference(entries, output_file, desc=""):
    results = []
    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip():
                        results.append(json.loads(line))
                except: continue
        start_idx = len(results)
        if start_idx > 0:
            print(f"Resuming {desc} from {start_idx}/{len(entries)}")

    if start_idx >= len(entries):
        print(f"All {len(entries)} entries already processed for {desc}.")
        return results

    print(f"Starting inference for {desc}. Results will be saved to {output_file}")
    with open(output_file, "a", encoding="utf-8") as f:
        for entry in tqdm(entries[start_idx:], desc=desc, initial=start_idx, total=len(entries)):
            prob = clean_problem(entry["problem"])
            prompt = PROMPT.format(problem=prob)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                max_length=None,
            )
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reason = full.split("[/INST]", 1)[1].strip() if "[/INST]" in full else full
            reason = post_process(reason)
            answer = extract_answer(reason) or ""
            fmt_ok = check_format(reason)
            res = {
                "index": entry["index"], "subject": entry["subject"],
                "level": entry["level"], "problem": entry["problem"],
                "solution": entry["solution"], "ground_truth": entry["ground_truth"],
                "model_reason": reason, "model_answer": str(answer),
                "format_ok": fmt_ok,
            }
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    return results


def report(preds, name):
    if not preds:
        print(f"{name}: no predictions")
        return
    c = sum(1 for p in preds if check_answer(p["model_answer"], p["ground_truth"]))
    fmt = sum(1 for p in preds if p.get("format_ok", False))
    print(f"{name} Accuracy:  {c}/{len(preds)} = {c/len(preds)*100:.1f}%")
    print(f"{name} Format OK: {fmt}/{len(preds)} = {fmt/len(preds)*100:.1f}%")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-distribution: GSM8K + MATH
print("=" * 50)
gsm8k = load_gsm8k(GSM8K_TEST)
gsm8k_preds = run_inference(gsm8k, os.path.join(OUTPUT_DIR, "gsm8k_predictions.jsonl"), "GSM8K")
report(gsm8k_preds, "GSM8K")

print("=" * 50)
math_entries = load_math(MATH_TEST_DIR)
math_preds = run_inference(math_entries, os.path.join(OUTPUT_DIR, "math_predictions.jsonl"), "MATH")
report(math_preds, "MATH")

# Out-of-distribution: SVAMP + ASDiv (Vi-MathLM-Aug only)
print("=" * 50)
svamp_entries = load_svamp(SVAMP_TEST)
svamp_preds = run_inference(svamp_entries, os.path.join(OUTPUT_DIR, "svamp_predictions.jsonl"), "SVAMP (OOD)")
report(svamp_preds, "SVAMP (OOD)")

print("=" * 50)
asdiv_entries = load_asdiv(ASDIV_TEST)
asdiv_preds = run_inference(asdiv_entries, os.path.join(OUTPUT_DIR, "asdiv_predictions.jsonl"), "ASDiv (OOD)")
report(asdiv_preds, "ASDiv (OOD)")

print(f"\nPredictions saved to {OUTPUT_DIR}/")
