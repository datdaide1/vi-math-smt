"""
KAGGLE NOTEBOOK 3: Inference — Model A
=======================================
Runtime: T4 GPU, ~3 hours
Input: Model A adapter (from NB1) + test data
Output: gsm8k_predictions.jsonl + math_predictions.jsonl

Instructions:
1. Upload Model A adapter output (from NB1) as Kaggle Dataset (e.g. "model-a-sft")
2. Upload test data:
   - gsm8k_test_vietnamese.json
   - math/ folder (7 _vi.jsonl files)
   as Kaggle Dataset (e.g. "smt-test-data")
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
MODEL_PATH = "/kaggle/input/model-a-sft/model_a_sft"
GSM8K_TEST = "/kaggle/input/smt-test-data/gsm8k_test_vietnamese.json"
MATH_TEST_DIR = "/kaggle/input/smt-test-data/math"
OUTPUT_DIR = "/kaggle/working/predictions/model_a_sft"

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
    # Normalize unicode first
    problem = normalize_unicode(problem)
    # Remove common Vietnamese prefixes
    problem = re.sub(
        r'^(Hãy\s+)?Giải\s+bài\s+toán\s+sau\.?\s*\n*', 
        '', problem, flags=re.IGNORECASE
    ).strip()
    return problem

def extract_answer(text):
    """Extract final answer from model output — priority-ordered."""
    if not text:
        return None
    
    # Priority 1: "Đáp án cuối cùng là: \boxed{...}"
    m = re.search(r'Đáp án cuối cùng là:\s*\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m: return m.group(1).strip()
    
    # Priority 2: Any \boxed{...}
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m: return m.group(1).strip()
    
    # Priority 3: "Đáp án cuối cùng là: X" (without boxed)
    m = re.search(r'Đáp án cuối cùng là:\s*(.+?)(?:\n|$)', text)
    if m:
        ans = re.sub(r'[*_`]', '', m.group(1).strip())
        if ans: return ans
    
    # Priority 4: #### (GSM8K style fallback)
    m = re.search(r'####\s*(.+?)$', text, re.MULTILINE)
    if m: return m.group(1).strip()
    
    # Priority 5: Last number
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if nums: return nums[-1]
    
    return None


def normalize_answer(ans):
    if ans is None: return ""
    ans = str(ans).strip()
    # Remove LaTeX math delimiters
    ans = re.sub(r'^\$\$?|\$\$?$', '', ans)
    # Remove units and common text labels
    ans = re.sub(r'\\(text|mathrm|mathbf|unit)\{(.+?)\}', r'\2', ans)
    ans = re.sub(r'(cm\^2|cm|inch|độ|degrees|đơn vị diện tích|đơn vị khối)', '', ans, flags=re.IGNORECASE)
    
    # Convert \frac{a}{b} to (a)/(b)
    ans = re.sub(r'\\frac(\d)(\d)', r'(\1)/(\2)', ans) # Handle \frac12
    for _ in range(3): # Handle nesting
        ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', ans)
    
    # Standardize math symbols
    ans = ans.replace('−', '-') # Unicode minus
    ans = re.sub(r'\s+', '', ans)
    ans = re.sub(r',', '', ans) # Thousand separators
    ans = re.sub(r'\\', '', ans) # Remove remaining backslashes
    ans = re.sub(r'[{}]', '', ans) # Remove remaining braces
    
    # Try numeric conversion
    try:
        # Handle simple fractions like (1)/(2)
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
    """Check if model output follows the trained format."""
    has_boxed = bool(re.search(r'\\boxed\{', text))
    has_final = bool(re.search(r'Đáp án cuối cùng là', text))
    return has_boxed or has_final


def post_process(text):
    """Cut garbage after the final answer marker."""
    if "Đáp án cuối cùng là" in text:
        parts = text.split("Đáp án cuối cùng là")
        # Keep everything up to and including the answer line
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

# ====================== CELL 5: Load Model ===================
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model A loaded")

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
            # Clean problem to match training distribution
            prob = clean_problem(entry["problem"])
            prompt = PROMPT.format(problem=prob)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,    # Anti-repetition
                no_repeat_ngram_size=4,    # Block 4-gram repeats
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

# Run GSM8K
print("=" * 50)
os.makedirs(OUTPUT_DIR, exist_ok=True)
gsm8k_file = os.path.join(OUTPUT_DIR, "gsm8k_predictions.jsonl")

gsm8k = load_gsm8k(GSM8K_TEST)
print(f"GSM8K test: {len(gsm8k)} entries")
gsm8k_preds = run_inference(gsm8k, gsm8k_file, "GSM8K")

c = sum(1 for p in gsm8k_preds if check_answer(p["model_answer"], p["ground_truth"]))
fmt = sum(1 for p in gsm8k_preds if p.get("format_ok", False))
print(f"GSM8K Accuracy:  {c}/{len(gsm8k_preds)} = {c/len(gsm8k_preds)*100:.1f}%")
print(f"GSM8K Format OK: {fmt}/{len(gsm8k_preds)} = {fmt/len(gsm8k_preds)*100:.1f}%")

# Run MATH
print("=" * 50)
math_file = os.path.join(OUTPUT_DIR, "math_predictions.jsonl")

math = load_math(MATH_TEST_DIR)
print(f"MATH test: {len(math)} entries")
math_preds = run_inference(math, math_file, "MATH")

c = sum(1 for p in math_preds if check_answer(p["model_answer"], p["ground_truth"]))
fmt = sum(1 for p in math_preds if p.get("format_ok", False))
print(f"MATH Accuracy:  {c}/{len(math_preds)} = {c/len(math_preds)*100:.1f}%")
print(f"MATH Format OK: {fmt}/{len(math_preds)} = {fmt/len(math_preds)*100:.1f}%")

# Per-subject breakdown
subjects = {}
for p in math_preds:
    s = p["subject"]
    if s not in subjects: subjects[s] = {"correct": 0, "total": 0}
    subjects[s]["total"] += 1
    if check_answer(p["model_answer"], p["ground_truth"]):
        subjects[s]["correct"] += 1
print("\nMATH per-subject:")
for s in sorted(subjects):
    acc = subjects[s]["correct"] / subjects[s]["total"] * 100
    print(f"  {s:30s}: {subjects[s]['correct']:4d}/{subjects[s]['total']:4d} = {acc:.1f}%")

# Combined
total = len(gsm8k_preds) + len(math_preds)
correct = sum(1 for p in gsm8k_preds + math_preds if check_answer(p["model_answer"], p["ground_truth"]))
print(f"\nCombined: {correct}/{total} = {correct/total*100:.1f}%")
print(f"\nPredictions saved to {OUTPUT_DIR}/")
