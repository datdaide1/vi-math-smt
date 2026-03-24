"""
Phase 2: SMT Mutation (FULL FIXED PIPELINE)
"""

import json
import os
import time
import random
import hashlib

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from smt_parser import parse_smt
from mutation_engine import mutate_single
from config import (
    PHASE1_VALIDATED_OUTPUT,
    PHASE2_MUTATED_OUTPUT,
    SKIP_SUBJECTS
)

random.seed(42)
os.makedirs(os.path.dirname(PHASE2_MUTATED_OUTPUT), exist_ok=True)


# ============================================================
# DEDUP
# ============================================================

seen = set()

def hash_code(x: str):
    return hashlib.md5(x.encode()).hexdigest()


# ============================================================
# WORKER
# ============================================================

def process(task):
    idx = task["src_idx"]
    item = task["item"]

    res_list = []
    attempts = 0

    while len(res_list) < 5 and attempts < 30:
        attempts += 1

        result = mutate_single(item["smt_code"])

        if not result:
            continue

        code = result["smt_code"]
        h = hash_code(code)

        if h in seen:
            continue

        seen.add(h)

        res_list.append({
            "source_index": idx,
            "subject": item.get("subject"),
            "level": item.get("level"),
            "smt_code": code,

            # IMPORTANT NEW FIELD
            "z3_answer": result.get("ground_truth"),

            "z3_status": result.get("z3_status"),
            "mutation_id": len(res_list)
        })

    return res_list


# ============================================================
# MAIN
# ============================================================

def run_phase2():

    print("=" * 80)
    print("PHASE 2 FIXED MUTATION PIPELINE")
    print("=" * 80)

    data = []
    with open(PHASE1_VALIDATED_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    tasks = [
        {"src_idx": i, "item": x}
        for i, x in enumerate(data)
        if x.get("subject", "").lower() not in SKIP_SUBJECTS
    ]

    print("Tasks:", len(tasks))
    print("Workers:", cpu_count())

    total = 0

    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):

            for r in res:
                with open(PHASE2_MUTATED_OUTPUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

                total += 1

    print("\nDONE")
    print("Total mutations:", total)


if __name__ == "__main__":
    run_phase2()