"""
Phase 4: Merge
Load original (14.87K) with source: original
Load informalized mutated (~X) with source: augmented
Merge + assign unified index
Output: train_augmented_final.jsonl (~14.87K + X)

FIXES:
- Reindex AFTER shuffle
- Keep smt_code
- Use canonical answer = ground_truth
- Keep mutation_strategy
- Safe division

Usage:
    python phase4_merge.py
"""

import json
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_FILE,
    PHASE3_INFORMALIZED_OUTPUT,
    PHASE4_FINAL_OUTPUT,
    OUTPUT_DIR
)

random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_phase4():

    print("=" * 70)
    print("  PHASE 4: Merge Original + Augmented")
    print("=" * 70)

    # ========================================================
    # 1. Load Original
    # ========================================================

    original = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                rec = json.loads(line)

                output_rec = {

                    "index": len(original),

                    "subject": rec.get(
                        "subject",
                        "unknown"
                    ),

                    "level": rec.get(
                        "level",
                        1
                    ),

                    "problem": rec.get(
                        "problem",
                        ""
                    ),

                    "solution": rec.get(
                        "solution",
                        ""
                    ),

                    "answer": rec.get(
                        "ground_truth",
                        ""
                    ),

                    "smt_code": rec.get(
                        "smt",
                        ""
                    ),

                    "source": "original",

                    "type": rec.get(
                        "type",
                        ""
                    ),
                }

                original.append(
                    output_rec
                )

            except json.JSONDecodeError:
                continue

    print(
        f"\n  Original data: "
        f"{len(original)} items"
    )

    # ========================================================
    # 2. Load Augmented
    # ========================================================

    augmented = []

    if os.path.exists(
        PHASE3_INFORMALIZED_OUTPUT
    ):

        with open(
            PHASE3_INFORMALIZED_OUTPUT,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    rec = json.loads(line)

                    if (
                        rec.get("status")
                        in ("ok", "ok_unverified")
                        and rec.get("problem")
                    ):

                        output_rec = {

                            "index": len(augmented),

                            "subject": rec.get(
                                "subject",
                                "unknown"
                            ),

                            "level": rec.get(
                                "level",
                                1
                            ),

                            "problem": rec.get(
                                "problem",
                                ""
                            ),

                            "solution": rec.get(
                                "solution",
                                ""
                            ),

                            "answer": rec.get(
                                "ground_truth",
                                ""
                            ),

                            "smt_code": rec.get(
                                "smt_code",
                                ""
                            ),

                            "source": "augmented",

                            "mutation_strategy": rec.get(
                                "mutation_strategy",
                                ""
                            ),

                            "status": rec.get(
                                "status",
                                ""
                            ),
                        }

                        augmented.append(
                            output_rec
                        )

                except json.JSONDecodeError:
                    continue

    print(
        f"  Augmented data: "
        f"{len(augmented)} items"
    )

    # ========================================================
    # 3. Combine
    # ========================================================

    all_data = original + augmented

    print(
        f"\n  Combined: "
        f"{len(all_data)} items"
    )

    # ========================================================
    # 4. Distribution by Subject
    # ========================================================

    print(f"\n  Distribution by Subject:")

    print(
        f"  {'Subject':30s} "
        f"{'Original':>8s} "
        f"{'Augmented':>10s} "
        f"{'Total':>8s} "
        f"{'%':>6s}"
    )

    print(f"  {'-'*70}")

    by_subject = defaultdict(
        lambda: {
            "original": 0,
            "augmented": 0,
        }
    )

    for item in all_data:

        subject = item.get(
            "subject",
            "unknown"
        )

        by_subject[subject][
            item["source"]
        ] += 1

    total_items = len(all_data)

    for subject in sorted(
        by_subject.keys()
    ):

        o = by_subject[subject][
            "original"
        ]

        a = by_subject[subject][
            "augmented"
        ]

        t = o + a

        pct = (
            t / total_items * 100
            if total_items > 0
            else 0
        )

        print(
            f"  {subject:30s} "
            f"{o:8d} "
            f"{a:10d} "
            f"{t:8d} "
            f"{pct:5.1f}%"
        )

    # ========================================================
    # 5. Distribution by Level
    # ========================================================

    print(f"\n  Distribution by Level:")

    print(
        f"  {'Level':8s} "
        f"{'Original':>8s} "
        f"{'Augmented':>10s} "
        f"{'Total':>8s} "
        f"{'%':>6s}"
    )

    print(f"  {'-'*50}")

    by_level = defaultdict(
        lambda: {
            "original": 0,
            "augmented": 0,
        }
    )

    for item in all_data:

        level = item.get(
            "level",
            0
        )

        by_level[level][
            item["source"]
        ] += 1

    for level in sorted(
        by_level.keys()
    ):

        o = by_level[level][
            "original"
        ]

        a = by_level[level][
            "augmented"
        ]

        t = o + a

        pct = (
            t / total_items * 100
            if total_items > 0
            else 0
        )

        print(
            f"  Level {level}  "
            f"{o:8d} "
            f"{a:10d} "
            f"{t:8d} "
            f"{pct:5.1f}%"
        )

    # ========================================================
    # 6. Shuffle
    # ========================================================

    random.shuffle(all_data)

    # IMPORTANT:
    # reindex AFTER shuffle

    for idx, item in enumerate(all_data):
        item["index"] = idx

    # ========================================================
    # 7. Write
    # ========================================================

    with open(
        PHASE4_FINAL_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for item in all_data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\n  ✓ Written to: "
        f"{PHASE4_FINAL_OUTPUT}"
    )

    print(f"\n  Summary:")

    original_pct = (
        len(original) / len(all_data) * 100
        if all_data else 0
    )

    augmented_pct = (
        len(augmented) / len(all_data) * 100
        if all_data else 0
    )

    print(
        f"    Total samples: "
        f"{len(all_data)}"
    )

    print(
        f"    Original: "
        f"{len(original)} "
        f"({original_pct:.1f}%)"
    )

    print(
        f"    Augmented: "
        f"{len(augmented)} "
        f"({augmented_pct:.1f}%)"
    )

    print(f"\n  {'=' * 70}\n")


if __name__ == "__main__":
    run_phase4()