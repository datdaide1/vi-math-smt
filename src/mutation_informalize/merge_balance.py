"""
Phase 4 (local): Merge original + augmented data into balanced dataset.
No API calls needed.

Usage:
    cd src/mutation_informalize
    python merge_balance.py
"""
import json
import os
import re
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INPUT_COT, INFORMALIZED_OUTPUT, FINAL_BALANCED_OUTPUT, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)


def run_merge():
    """Merge original data with augmented data into a balanced dataset."""
    print("=" * 70)
    print("  MERGE: Create balanced dataset")
    print("=" * 70)
    
    # 1. Load original data
    original = []
    with open(INPUT_COT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["source"] = "original"
                original.append(rec)
    
    print(f"\n  Original data: {len(original)}")
    
    # 2. Load augmented data
    augmented = []
    if os.path.exists(INFORMALIZED_OUTPUT):
        with open(INFORMALIZED_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") in ("ok", "ok_unverified") and rec.get("text"):
                    augmented.append({
                        "text": rec["text"],
                        "subject": rec.get("subject", ""),
                        "level": rec.get("level", 0),
                        "source": "augmented",
                        "algorithm": rec.get("algorithm", ""),
                        "verified": rec.get("verified", False),
                    })
    
    print(f"  Augmented data: {len(augmented)}")
    
    # 3. Combine
    all_data = original + augmented
    
    # 4. Print distribution
    print(f"\n  Combined: {len(all_data)}")
    print(f"\n  {'Subject':30s} {'Original':>8s} {'Augmented':>10s} {'Total':>6s}")
    print(f"  {'-'*60}")
    
    by_subject = defaultdict(lambda: {"original": 0, "augmented": 0})
    for item in all_data:
        by_subject[item.get("subject", "?")][item["source"]] += 1
    
    for s in sorted(by_subject.keys()):
        o = by_subject[s]["original"]
        a = by_subject[s]["augmented"]
        print(f"  {s:30s} {o:8d} {a:10d} {o+a:6d}")
    
    print(f"\n  Level distribution:")
    by_level = defaultdict(int)
    for item in all_data:
        by_level[str(item.get("level", 0))] += 1
    for lvl in sorted(by_level.keys()):
        pct = by_level[lvl] / len(all_data) * 100
        print(f"    Level {lvl}: {by_level[lvl]:6d} ({pct:.1f}%)")
    
    # 5. Shuffle and write
    random.shuffle(all_data)
    
    with open(FINAL_BALANCED_OUTPUT, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n  ✓ Written to: {FINAL_BALANCED_OUTPUT}")
    print(f"  Total samples: {len(all_data)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_merge()
