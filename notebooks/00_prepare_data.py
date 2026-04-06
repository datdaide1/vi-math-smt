"""
NOTEBOOK 0: Data Preparation (Run Locally — No GPU needed)
==========================================================
Cleans raw training data to fix the 5 critical issues:
1. Removes GSM8K <<...>> calculator tags (repetition killer)
2. Unicode NFC normalization (anti-Mojibake)
3. Unified output format: always ends with \boxed{answer}
4. Removes noisy headers/footers
5. De-duplicates repeated words

Input:  data/finetune/processed/train_cot_only.jsonl
        data/finetune/processed/train_cot_smt.jsonl
Output: data/finetune/processed/train_cot_only_clean.jsonl
        data/finetune/processed/train_cot_smt_clean.jsonl
"""

import json, re, os, unicodedata, sys

# ====================== CONFIG ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_COT = os.path.join(BASE_DIR, "data", "finetune", "processed", "train_cot_only.jsonl")
INPUT_SMT = os.path.join(BASE_DIR, "data", "finetune", "processed", "train_cot_smt.jsonl")
OUTPUT_COT = os.path.join(BASE_DIR, "data", "finetune", "processed", "train_cot_only_clean.jsonl")
OUTPUT_SMT = os.path.join(BASE_DIR, "data", "finetune", "processed", "train_cot_smt_clean.jsonl")

# Unified instruction prompts
INSTR_COT = "Bạn là trợ lý toán học. Giải bài toán sau từng bước bằng Tiếng Việt. Đưa kết quả cuối cùng theo định dạng \\boxed{kết quả}."
INSTR_SMT = "Bạn là trợ lý toán học. Giải bài toán sau từng bước bằng Tiếng Việt và viết SMT-LIB code để xác minh. Đưa kết quả cuối cùng theo định dạng \\boxed{kết quả}."


# ====================== CLEANING FUNCTIONS ====================
def normalize_unicode(text: str) -> str:
    """NFC normalization to fix Vietnamese Mojibake."""
    return unicodedata.normalize('NFC', text)


def remove_gsm8k_tags(text: str) -> str:
    """Remove GSM8K calculator tags like <<300/2=150>>."""
    return re.sub(r'<<.*?>>', '', text)


def deduplicate_words(text: str) -> str:
    """Remove consecutive repeated words: 'the the' -> 'the'."""
    return re.sub(r'\b(\w+)\b( \1\b)+', r'\1', text)


def extract_ground_truth(after_inst: str) -> str:
    """Extract the ground truth answer from the response portion."""
    gt = ""
    
    # Priority 1: ## Đáp án section
    m = re.search(r'##\s*Đáp án\s*\n(.+?)(?:\s*</s>|$)', after_inst, re.DOTALL)
    if m:
        gt = m.group(1).strip()
        if gt:
            return gt
    
    # Priority 2: #### marker (GSM8K style)
    m = re.search(r'####\s*(.+?)(?:\s*</s>|\n|$)', after_inst, re.MULTILINE)
    if m:
        gt = m.group(1).strip()
        if gt:
            return gt
    
    # Priority 3: \boxed{} in solution
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', after_inst)
    if m:
        gt = m.group(1).strip()
        if gt:
            return gt
    
    # Priority 4: Last line before </s>
    lines = after_inst.replace('</s>', '').strip().split('\n')
    gt = lines[-1].strip() if lines else ""
    return gt


def clean_solution(after_inst: str) -> str:
    """Clean solution text: remove headers, tags, noisy patterns."""
    sol = after_inst
    
    # Remove closing tags
    sol = sol.replace('</s>', '')
    
    # Remove GSM8K tags
    sol = remove_gsm8k_tags(sol)
    
    # Remove noisy headers and footers
    patterns_to_remove = [
        r'####\s*.*',                 # GSM8K answer marker
        r'##\s*Đáp án\s*\n.*',       # Vietnamese answer section (greedy to end)
        r'##\s*Lời giải\s*\n?',      # Solution header
        r'Đáp án cuối cùng là.*',    # Final answer marker
    ]
    for p in patterns_to_remove:
        sol = re.sub(p, '', sol, flags=re.DOTALL)
    
    # Deduplicate words
    sol = deduplicate_words(sol)
    
    # Clean whitespace
    sol = re.sub(r'\n{3,}', '\n\n', sol)  # Max 2 consecutive newlines
    sol = sol.strip()
    
    return sol


def process_cot_only(input_path: str, output_path: str):
    """Clean CoT-only dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    stats = {"total": 0, "cleaned": 0, "skipped": 0, "subjects": {}}
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            stats["total"] += 1
            try:
                obj = json.loads(line)
                text = normalize_unicode(obj.get("text", ""))
                subject = obj.get("subject", "unknown")
                level = obj.get("level", 0)
                index = obj.get("index", stats["total"] - 1)
                
                # Parse original structure
                inst_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
                if not inst_match:
                    stats["skipped"] += 1
                    continue
                
                # Extract problem (remove the generic instruction prefix)
                problem = inst_match.group(1).strip()
                problem = re.sub(
                    r'^(Hãy\s+)?Giải\s+bài\s+toán\s+sau\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                problem = re.sub(
                    r'^Giải\s+bài\s+toán\s+sau\s+và\s+viết\s+SMT-LIB\s+code\s+để\s+xác\s+minh\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                problem = re.sub(
                    r'^và\s+viết\s+SMT-LIB\s+code\s+để\s+xác\s+minh\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                
                if not problem:
                    stats["skipped"] += 1
                    continue
                
                # Extract response portion
                after_inst = text.split('[/INST]')[-1].strip()
                
                # Extract ground truth BEFORE cleaning (important!)
                gt = extract_ground_truth(after_inst)
                if not gt:
                    stats["skipped"] += 1
                    continue
                
                # Clean solution
                sol = clean_solution(after_inst)
                if not sol or len(sol) < 10:
                    stats["skipped"] += 1
                    continue
                
                # Build unified format
                gt_display = gt if gt.startswith("\\boxed") else f"\\boxed{{{gt}}}"
                
                new_text = (
                    f"<s>[INST] {INSTR_COT}\n\n"
                    f"{problem} [/INST]\n"
                    f"{sol}\n\n"
                    f"Đáp án cuối cùng là: {gt_display}</s>"
                )
                
                new_obj = {
                    "text": new_text,
                    "index": index,
                    "subject": subject,
                    "level": level,
                }
                fout.write(json.dumps(new_obj, ensure_ascii=False) + '\n')
                stats["cleaned"] += 1
                stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
                
            except Exception as e:
                stats["skipped"] += 1
                if stats["skipped"] <= 3:
                    print(f"  Warning: Skipped line {stats['total']}: {str(e)[:80]}")
    
    print(f"\nResults:")
    print(f"  Total:   {stats['total']}")
    print(f"  Cleaned: {stats['cleaned']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Subjects: {stats['subjects']}")
    print(f"  Output:  {output_path}")
    return stats


def process_cot_smt(input_path: str, output_path: str):
    """Clean CoT+SMT dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    stats = {"total": 0, "cleaned": 0, "skipped": 0, "subjects": {}}
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            stats["total"] += 1
            try:
                obj = json.loads(line)
                text = normalize_unicode(obj.get("text", ""))
                subject = obj.get("subject", "unknown")
                level = obj.get("level", 0)
                index = obj.get("index", stats["total"] - 1)
                
                inst_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
                if not inst_match:
                    stats["skipped"] += 1
                    continue
                
                # Extract problem
                problem = inst_match.group(1).strip()
                problem = re.sub(
                    r'^(Hãy\s+)?Giải\s+bài\s+toán\s+sau\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                problem = re.sub(
                    r'^Giải\s+bài\s+toán\s+sau\s+và\s+viết\s+SMT-LIB\s+code\s+để\s+xác\s+minh\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                # Also catch partial leftovers
                problem = re.sub(
                    r'^và\s+viết\s+SMT-LIB\s+code\s+để\s+xác\s+minh\.?\s*\n*',
                    '', problem, flags=re.IGNORECASE
                ).strip()
                
                if not problem:
                    stats["skipped"] += 1
                    continue
                
                after_inst = text.split('[/INST]')[-1].strip()
                
                # Extract SMT-LIB block if present
                smt_block = ""
                smt_match = re.search(
                    r'##\s*SMT-LIB\s*\n(```(?:smt2?)?\n(.*?)\n```)',
                    after_inst, re.DOTALL
                )
                if smt_match:
                    smt_block = smt_match.group(1).strip()
                else:
                    # Try without markdown fences
                    smt_match = re.search(
                        r'##\s*SMT-LIB\s*\n(.*?)(?=\n##|\Z)',
                        after_inst, re.DOTALL
                    )
                    if smt_match:
                        smt_block = smt_match.group(1).strip()
                
                # Extract ground truth
                gt = extract_ground_truth(after_inst)
                if not gt:
                    stats["skipped"] += 1
                    continue
                
                # Extract and clean solution (remove SMT block from solution)
                sol_text = after_inst
                if smt_block:
                    # Remove the SMT section from the solution text
                    sol_text = re.sub(r'##\s*SMT-LIB\s*\n.*?(?=\n##|\Z)', '', sol_text, flags=re.DOTALL)
                
                sol = clean_solution(sol_text)
                if not sol or len(sol) < 10:
                    stats["skipped"] += 1
                    continue
                
                gt_display = gt if gt.startswith("\\boxed") else f"\\boxed{{{gt}}}"
                
                # Build unified format with SMT block
                if smt_block:
                    new_text = (
                        f"<s>[INST] {INSTR_SMT}\n\n"
                        f"{problem} [/INST]\n"
                        f"## SMT-LIB\n{smt_block}\n\n"
                        f"## Lời giải\n{sol}\n\n"
                        f"Đáp án cuối cùng là: {gt_display}</s>"
                    )
                else:
                    new_text = (
                        f"<s>[INST] {INSTR_SMT}\n\n"
                        f"{problem} [/INST]\n"
                        f"{sol}\n\n"
                        f"Đáp án cuối cùng là: {gt_display}</s>"
                    )
                
                new_obj = {
                    "text": new_text,
                    "index": index,
                    "subject": subject,
                    "level": level,
                }
                fout.write(json.dumps(new_obj, ensure_ascii=False) + '\n')
                stats["cleaned"] += 1
                stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
                
            except Exception as e:
                stats["skipped"] += 1
                if stats["skipped"] <= 3:
                    print(f"  Warning: Skipped line {stats['total']}: {str(e)[:80]}")
    
    print(f"\nResults:")
    print(f"  Total:   {stats['total']}")
    print(f"  Cleaned: {stats['cleaned']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Subjects: {stats['subjects']}")
    print(f"  Output:  {output_path}")
    return stats


def verify_output(path: str, n=3):
    """Print first N samples to verify format."""
    print(f"\n--- Verification: {os.path.basename(path)} ---")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n: break
            obj = json.loads(line)
            text = obj["text"]
            print(f"\n[Sample {i}] subject={obj['subject']}, level={obj['level']}")
            print(text[:500])
            print("...")
            # Check format compliance
            has_inst = "[INST]" in text and "[/INST]" in text
            has_boxed = "\\boxed{" in text
            has_final = "Đáp án cuối cùng là:" in text
            has_gsm_tags = "<<" in text and ">>" in text
            print(f"  ✓ [INST]/[/INST]: {has_inst}")
            print(f"  ✓ \\boxed{{}}: {has_boxed}")
            print(f"  ✓ Final answer marker: {has_final}")
            print(f"  {'✗' if has_gsm_tags else '✓'} No GSM8K tags: {not has_gsm_tags}")


# ====================== MAIN =================================
if __name__ == "__main__":
    # Fix Windows terminal encoding for Vietnamese
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except: pass
    
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Process CoT-only
    if os.path.exists(INPUT_COT):
        cot_stats = process_cot_only(INPUT_COT, OUTPUT_COT)
        try: verify_output(OUTPUT_COT)
        except UnicodeEncodeError: print("  (Skipped verification — terminal encoding issue)")
    else:
        print(f"WARNING: {INPUT_COT} not found!")
    
    # Process CoT+SMT
    if os.path.exists(INPUT_SMT):
        smt_stats = process_cot_smt(INPUT_SMT, OUTPUT_SMT)
        try: verify_output(OUTPUT_SMT)
        except UnicodeEncodeError: print("  (Skipped verification — terminal encoding issue)")
    else:
        print(f"WARNING: {INPUT_SMT} not found!")
    
    print(f"\n{'='*60}")
    print("DONE! Upload the *_clean.jsonl files to Kaggle Dataset.")
    print(f"{'='*60}")
