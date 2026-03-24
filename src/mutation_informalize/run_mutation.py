"""
Phase 2: Run mutation on ALL SMT codes → generate ~28K new problems.
Pure CPU — no API calls needed. Very fast.
Fully resumeable via JSONL checkpoint.

Usage:
    cd src/mutation_informalize
    python run_mutation.py
"""
import json
import os
import re
import sys
import time
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INPUT_SMT, FIXED_SMT_OUTPUT, MUTATED_SMT_OUTPUT,
    OUTPUT_DIR, MUTATION_TARGET_PER_SUBJECT
)
from smt_parser import parse_smt
from mutation_engine import mutate_single
from z3_validator import validate_smt
from llm_client import load_checkpoint, append_checkpoint, print_progress

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)


def extract_smt_from_text(text: str) -> str:
    m = re.search(r'## SMT-LIB\n```smt2?\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_all_smt_data():
    """Load all SMT data, applying Phase 1 fixes where available."""
    # Load original data
    with open(INPUT_SMT, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    # Load Phase 1 fixes
    fixes = {}
    if os.path.exists(FIXED_SMT_OUTPUT):
        with open(FIXED_SMT_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("source_index", "")
                if rec.get("fix_status", "").startswith("fixed"):
                    fixes[str(idx)] = rec["smt_code"]
    
    # Build final list: use fixed SMT if available, else original
    items = []
    fixed_count = 0
    for item in all_data:
        idx = str(item.get("index", 0))
        smt = fixes.get(idx, extract_smt_from_text(item["text"]))
        if idx in fixes:
            fixed_count += 1
        items.append({
            "index": item.get("index", 0),
            "subject": item.get("subject", ""),
            "level": item.get("level", 0),
            "smt_code": smt,
        })
    
    print(f"  Loaded {len(items)} items ({fixed_count} with Phase 1 fixes)")
    return items


def run_phase2():
    """Run mutation on all SMT codes."""
    print("=" * 70)
    print("  PHASE 2: SMT Mutation (CPU only)")
    print("=" * 70)
    
    items = load_all_smt_data()
    
    # Group by subject
    by_subject = defaultdict(list)
    for item in items:
        by_subject[item["subject"]].append(item)
    
    # Load checkpoint
    checkpoint = load_checkpoint(MUTATED_SMT_OUTPUT)
    print(f"  Checkpoint: {len(checkpoint)} mutations already done")
    
    # Stats
    total_target = sum(MUTATION_TARGET_PER_SUBJECT.values())
    total_done = 0
    total_success = 0
    total_fail = 0
    
    for subject, target in MUTATION_TARGET_PER_SUBJECT.items():
        subject_items = by_subject.get(subject, [])
        if not subject_items:
            print(f"\n  [{subject}] No data, skipping")
            continue
        
        # Count already done for this subject
        subject_done = sum(1 for uid, rec in checkpoint.items() 
                          if rec.get("subject") == subject)
        remaining_target = max(0, target - subject_done)
        
        print(f"\n  [{subject}] {len(subject_items)} originals → "
              f"target {target}, done {subject_done}, remaining {remaining_target}")
        
        if remaining_target == 0:
            continue
        
        mutations_per_item = max(1, remaining_target // len(subject_items) + 1)
        generated = 0
        
        for i, item in enumerate(subject_items):
            if generated >= remaining_target:
                break
            
            smt = item["smt_code"]
            if not smt or len(smt.strip()) < 20:
                continue
            
            # Try each mutation algorithm
            for mut_idx in range(mutations_per_item):
                if generated >= remaining_target:
                    break
                
                uid = f"{subject}_{item['index']}_{mut_idx}"
                if uid in checkpoint:
                    continue
                
                # Pick a partner for composition
                partner = None
                if len(subject_items) > 1:
                    partner_item = random.choice(
                        [x for x in subject_items if x["index"] != item["index"]]
                    )
                    partner = partner_item["smt_code"]
                
                # Choose algorithm based on mutation index
                if mut_idx == 0:
                    algo = "A1_injection"
                elif mut_idx == 1 and partner:
                    algo = "A2_composition"
                elif mut_idx == 2:
                    algo = "A3_complication"
                else:
                    algo = "A4_perturbation"
                
                result = mutate_single(smt, algorithm=algo, partner_smt=partner)
                
                if result is None:
                    # Fallback to auto
                    result = mutate_single(smt, algorithm="auto", partner_smt=partner)
                
                if result:
                    mutated_smt, algo_used, z3_answer = result
                    record = {
                        "uid": uid,
                        "source_index": item["index"],
                        "subject": subject,
                        "level": item["level"],
                        "algorithm": algo_used,
                        "mutated_smt": mutated_smt,
                        "z3_answer": z3_answer,
                    }
                    append_checkpoint(MUTATED_SMT_OUTPUT, record)
                    generated += 1
                    total_success += 1
                else:
                    total_fail += 1
                
                total_done += 1
                print_progress(
                    total_done, total_target, 
                    f"[{subject[:12]:12s}]",
                    f"✓{total_success} ✗{total_fail} gen={generated}/{remaining_target}"
                )
        
        print()  # newline after subject
    
    print(f"\n{'=' * 70}")
    print(f"  Phase 2 Complete: {total_success} mutations generated, {total_fail} failed")
    print(f"  Output: {MUTATED_SMT_OUTPUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_phase2()
