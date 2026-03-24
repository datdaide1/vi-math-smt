"""
Cleanup Phase 3 Output: Rescue false-negative "failed" entries.
v3: Dùng multiprocessing timeout để skip câu treo.
"""

import json
import re
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

warnings.filterwarnings("ignore")

# Assumes this script is run from its own directory (src/mutation_informalize/)
INPUT_FILE = os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized.jsonl")
OUTPUT_FILE = os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized_clean.jsonl")


def enhanced_normalize(ans):
    if ans is None: return ""
    ans = str(ans).strip()
    if ans.endswith("?"): ans = ans[:-1].strip()
    ans = ans.replace(r"\displaystyle", "").strip()
    ans = re.sub(r'\\text\{[^}]*\}', '', ans)
    ans = re.sub(r'\\textrm\{[^}]*\}', '', ans)
    ans = re.sub(r'\\mathrm\{[^}]*\}', '', ans)
    ans = re.sub(r'\\overline\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\d?frac\s*(\d)\s*(\d)', r'\1/\2', ans)
    ans = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', ans)
    ans = ans.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    ans = ans.replace(r"\quad", "").replace(r"\qquad", "")
    ans = ans.replace(" ", "").replace(",", "").replace("$", "")
    ans = ans.replace(r"^\circ", "").replace(r"\circ", "")
    ans = ans.replace(r"\approx", "=")
    if "=" in ans: ans = ans.split("=")[-1].strip()
    ans = ans.replace(r"\\", "").replace("{", "").replace("}", "")
    ans = ans.rstrip("%")
    return ans.strip()


def safe_to_float(s):
    """Chuyển string thành float an toàn, hỗ trợ a/b."""
    try:
        return float(s)
    except:
        pass
    # Thử a/b
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0].strip("() ")) / float(parts[1].strip("() "))
            except:
                pass
    # Thử (a)/(b)
    m = re.match(r'^\(([^)]+)\)/\(([^)]+)\)$', s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except:
            pass
    return None


def try_sympy_compare(gt_str, pred_str):
    """So sánh SymPy — được gọi trong subprocess nên có thể bị kill."""
    if len(gt_str) > 80 or len(pred_str) > 80:
        return False
    skip = ['begin', 'aligned', 'array', 'text', 'cup', 'cap', 'max_', 'min_']
    for p in skip:
        if p in pred_str.lower() or p in gt_str.lower():
            return False
    try:
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = (standard_transformations + (implicit_multiplication_application,))
        def prep(s):
            s = str(s)
            s = re.sub(r'\\text\{[^}]*\}', '', s)
            s = s.replace(r'\pi', 'pi').replace(r'\infty', 'oo')
            s = s.replace(r'\sqrt', 'sqrt')
            s = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
            s = re.sub(r'\\d?frac\s*(\d)\s*(\d)', r'(\1)/(\2)', s)
            s = s.replace('^', '**').replace('{', '(').replace('}', ')')
            s = s.replace('\\', '').replace(',', '')
            return s
        e1 = parse_expr(prep(gt_str), transformations=transformations)
        e2 = parse_expr(prep(pred_str), transformations=transformations)
        diff = abs((e1 - e2).evalf())
        if diff < 0.01: return True
        v1 = abs(e1.evalf())
        if v1 != 0 and (diff / v1) < 0.01: return True
    except:
        pass
    return False


def _rescue_one(args):
    """Hàm chạy trong subprocess — có thể bị kill nếu treo."""
    gt, raw_ans, problem, solution = args
    
    if not problem or len(problem) < 10: return False, "missing_problem"
    if not solution or len(solution) < 30: return False, "missing_solution"
    if not raw_ans.strip(): return False, "empty_answer"
    
    gt_norm = enhanced_normalize(gt)
    if not gt_norm: return False, "empty_gt"
    
    # Tạo candidates
    candidates = [raw_ans]
    splits = re.split(r'(?:,|;|\\qquad|\\quad)', raw_ans)
    if len(splits) > 1:
        candidates.extend([s.strip() for s in splits if s.strip()])
    if "=" in raw_ans:
        candidates.extend([p.strip() for p in raw_ans.split("=") if p.strip()])
    
    for cand in candidates:
        cand_norm = enhanced_normalize(cand)
        if not cand_norm: continue
        # String match
        if gt_norm == cand_norm: return True, "string_match"
        # Float match
        v_gt = safe_to_float(gt_norm)
        v_pred = safe_to_float(cand_norm)
        if v_gt is not None and v_pred is not None:
            if abs(v_gt - v_pred) < 1e-6: return True, "numeric_match"
            if v_gt != 0 and abs((v_gt - v_pred) / v_gt) < 0.02: return True, "numeric_match"
    
    # SymPy (last resort)
    if try_sympy_compare(gt, raw_ans):
        return True, "sympy_match"
    
    return False, "truly_wrong"


def main():
    print("Loading data...", flush=True)
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"{'='*70}", flush=True)
    print(f"PHASE 3 DATA CLEANUP v3 (multiprocessing timeout)", flush=True)
    print(f"{'='*70}", flush=True)
    
    ok_count = sum(1 for d in data if d["status"] == "ok")
    failed_count = sum(1 for d in data if d["status"] == "failed")
    print(f"Total: {len(data)}, OK={ok_count}, Failed={failed_count}", flush=True)
    
    failed_entries = [(i, d) for i, d in enumerate(data) if d["status"] == "failed"]
    total_failed = len(failed_entries)
    print(f"Processing {total_failed} failed entries (timeout=3s each)...", flush=True)
    
    rescued_total = 0
    skipped = 0
    rescue_reasons = {}
    fail_reasons = {}
    removed = 0
    
    # Dùng ProcessPoolExecutor với timeout 3s cho mỗi entry
    with ProcessPoolExecutor(max_workers=1) as executor:
        for batch_start in range(0, total_failed, 50):
            batch = failed_entries[batch_start:batch_start+50]
            
            for idx_in_data, entry in batch:
                args = (
                    str(entry.get("ground_truth", "")),
                    str(entry.get("answer", "")),
                    entry.get("problem", ""),
                    entry.get("solution", ""),
                )
                try:
                    future = executor.submit(_rescue_one, args)
                    rescued, reason = future.result(timeout=3)
                except FuturesTimeoutError:
                    rescued, reason = False, "timeout_skip"
                    skipped += 1
                except Exception:
                    rescued, reason = False, "error_skip"
                    skipped += 1
                
                if rescued:
                    data[idx_in_data]["status"] = "ok"
                    data[idx_in_data]["feedback"] = f"rescued:{reason}"
                    rescued_total += 1
                    rescue_reasons[reason] = rescue_reasons.get(reason, 0) + 1
                else:
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                    if reason in ("missing_problem", "missing_solution", "empty_answer", "empty_gt"):
                        data[idx_in_data]["_remove"] = True
                        removed += 1
            
            done = min(batch_start + 50, total_failed)
            print(f"  [{done}/{total_failed}] rescued={rescued_total} skipped={skipped}", flush=True)
    
    clean_data = [d for d in data if not d.get("_remove")]
    for d in clean_data:
        d.pop("_remove", None)
    
    final_ok = sum(1 for d in clean_data if d["status"] == "ok")
    final_failed = sum(1 for d in clean_data if d["status"] == "failed")
    
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Rescued: {rescued_total}", flush=True)
    for k, v in sorted(rescue_reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}", flush=True)
    print(f"Skipped (timeout): {skipped}", flush=True)
    print(f"Removed (empty): {removed}", flush=True)
    print(f"Still failed:", flush=True)
    for k, v in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}", flush=True)
    print(f"\nFINAL: {len(clean_data)} entries ({final_ok} ok, {final_failed} failed)", flush=True)
    print(f"Success rate: {final_ok/len(clean_data)*100:.1f}%", flush=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in clean_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved to: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
