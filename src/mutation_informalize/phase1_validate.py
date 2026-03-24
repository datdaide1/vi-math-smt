"""
Phase 1: Validate + Classify + Agentic Fix SMT

Optimized Version:
- ONLY validate GOOD samples first
- WEAK/TRIVIAL/BROKEN go directly to repair
- GOOD + valid => pass through
- GOOD + invalid => repair as BROKEN
- Resumeable
- Agentic repair loop
- XML prompting

Usage:
    python phase1_validate_agentic.py
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_FILE,
    PHASE1_VALIDATED_OUTPUT,
    OUTPUT_DIR,
)

from z3_validator import validate_smt

from llm_client import (
    AsyncLLMClient,
    load_checkpoint,
    append_checkpoint,
    print_progress,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Classification
# ============================================================

def classify_smt(smt_code: str) -> str:
    """
    Classify SMT as:
        GOOD
        WEAK
        TRIVIAL
        BROKEN
    """

    if not smt_code or len(smt_code.strip()) < 10:
        return "TRIVIAL"

    lines = smt_code.strip().split("\n")

    asserts = [
        l.strip()
        for l in lines
        if l.strip().startswith("(assert")
    ]

    declares = [
        l.strip()
        for l in lines
        if l.strip().startswith("(declare")
    ]

    if len(declares) == 0:
        return "BROKEN"

    all_text = " ".join(asserts)

    arithmetic_ops = len(
        re.findall(r"\(\s*[+\-*/]", all_text)
    )

    comparisons = len(
        re.findall(r"\(\s*(?:>|<|>=|<=)", all_text)
    )

    logic_ops = len(
        re.findall(r"\(\s*(?:and|or|not|=>)", all_text)
    )

    ite_ops = len(
        re.findall(r"\(\s*ite", all_text)
    )

    total_complexity = (
        arithmetic_ops
        + comparisons
        + logic_ops
        + ite_ops
    )

    trivial_asserts = 0

    for a in asserts:

        if re.match(
            r"\(assert\s+\(=\s+\w+\s+[\d.\-]+\)\)",
            a,
        ):
            trivial_asserts += 1

        elif re.match(
            r"\(assert\s+\(=\s+\w+\s+\w+\)\)",
            a,
        ):
            trivial_asserts += 1

    non_trivial = len(asserts) - trivial_asserts

    if total_complexity >= 4 and non_trivial >= 3:
        return "GOOD"

    if total_complexity >= 2 and non_trivial >= 2:
        return "WEAK"

    return "TRIVIAL"


# ============================================================
# XML Prompt
# ============================================================

FIXER_XML_PROMPT = """
<system>
You are an expert SMT-LIB repair and optimization agent.

Goals:
1. Repair invalid SMT-LIB
2. Preserve semantics
3. Increase reasoning depth
4. Add intermediate variables
5. Add meaningful constraints
6. Keep SMT satisfiable
7. Ensure Z3 solvable

Rules:
- Return ONLY SMT-LIB
- No explanations
- No markdown
- No comments
</system>

<user>

Current SMT-LIB:

<![CDATA[
{smt_code}
]]>

Z3 Feedback:
{z3_feedback}

Task:
- Repair syntax
- Repair semantics
- Improve reasoning complexity
- Keep final answer consistent

Return ONLY SMT-LIB.

</user>
"""


# ============================================================
# SMT Extraction
# ============================================================

def extract_smt(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:smt2|lisp)?",
        "",
        text,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    m = re.search(
        r"```(?:smt2|lisp)?(.*?)```",
        text,
        re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return text.strip()


# ============================================================
# Agentic Repair
# ============================================================

async def repair_smt_agentic(
    llm,
    smt_code,
    max_attempts=3,
):

    current_smt = smt_code

    last_feedback = "Initial repair"

    for attempt in range(max_attempts):

        print(
            f"      Repair Attempt {attempt+1}/{max_attempts}"
        )

        prompt = FIXER_XML_PROMPT.format(
            smt_code=current_smt,
            z3_feedback=last_feedback,
        )

        response = await llm.call(
            "",
            prompt,
        )

        fixed_smt = extract_smt(response)

        if not fixed_smt:
            last_feedback = "Empty output"
            continue

        is_valid, answer, info = validate_smt(
            fixed_smt
        )

        if is_valid:

            return {
                "success": True,
                "fixed_smt": fixed_smt,
                "answer": answer,
                "attempts": attempt + 1,
            }

        last_feedback = f"""
Z3 validation failed.

Error:
{info}

Please repair again.
"""

        current_smt = fixed_smt

    return {
        "success": False,
        "fixed_smt": current_smt,
        "answer": None,
        "attempts": max_attempts,
    }


# ============================================================
# Main
# ============================================================

async def run_phase1():

    print("=" * 80)
    print(" PHASE 1: Validate + Agentic Repair SMT ")
    print("=" * 80)

    processed = load_checkpoint(
        PHASE1_VALIDATED_OUTPUT
    )

    print(
        f"\nCheckpoint Loaded: {len(processed)} items"
    )

    input_data = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                input_data.append(
                    json.loads(line)
                )

            except:
                continue

    total = len(input_data)

    print(f"Total Input: {total}")

    llm = AsyncLLMClient()

    stats = defaultdict(int)

    start_time = time.time()

    for idx, item in enumerate(input_data):

        if idx in processed:

            old = processed[idx]

            stats[
                old.get(
                    "validation_status",
                    "UNKNOWN",
                )
            ] += 1

            continue

        print_progress(
            idx,
            total,
            "processing SMT...",
        )

        smt_code = item.get(
            "smt",
            ""
        ).strip()

        if not smt_code:

            output_item = {
                **item,
                "validation_status": "UNFIXABLE",
                "classification": "EMPTY",
                "smt_code": "",
            }

            append_checkpoint(
                PHASE1_VALIDATED_OUTPUT,
                idx,
                output_item,
            )

            stats["UNFIXABLE"] += 1

            continue

        # ====================================================
        # Classification FIRST
        # ====================================================

        classification = classify_smt(
            smt_code
        )

        # ====================================================
        # GOOD → validate only
        # ====================================================

        if classification == "GOOD":

            is_valid, answer, info = validate_smt(
                smt_code
            )

            if is_valid:

                output_item = {
                    **item,
                    "validation_status": "GOOD",
                    "classification": classification,
                    "smt_code": smt_code,
                    "z3_answer": answer,
                }

                append_checkpoint(
                    PHASE1_VALIDATED_OUTPUT,
                    idx,
                    output_item,
                )

                stats["GOOD"] += 1

                continue

            else:
                classification = "BROKEN"

        # ====================================================
        # Repair Needed
        # ====================================================

        print(
            f"\n[{idx}/{total}] "
            f"Classification={classification} "
            f"→ Running Agentic Repair"
        )

        repair_result = await repair_smt_agentic(
            llm=llm,
            smt_code=smt_code,
            max_attempts=3,
        )

        if repair_result["success"]:

            output_item = {
                **item,
                "validation_status": "FIXED",
                "classification": classification,
                "repair_attempts": repair_result[
                    "attempts"
                ],
                "smt_code": repair_result[
                    "fixed_smt"
                ],
                "z3_answer": repair_result[
                    "answer"
                ],
            }

            append_checkpoint(
                PHASE1_VALIDATED_OUTPUT,
                idx,
                output_item,
            )

            stats["FIXED"] += 1

        else:

            output_item = {
                **item,
                "validation_status": "UNFIXABLE",
                "classification": classification,
                "repair_attempts": repair_result[
                    "attempts"
                ],
                "smt_code": smt_code,
            }

            append_checkpoint(
                PHASE1_VALIDATED_OUTPUT,
                idx,
                output_item,
            )

            stats["UNFIXABLE"] += 1

    # ========================================================
    # Summary
    # ========================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(" PHASE 1 COMPLETE ")
    print("=" * 80)

    print(f"\nElapsed Time: {elapsed:.2f}s")

    print("\nValidation Statistics:")

    for status in [
        "GOOD",
        "FIXED",
        "UNFIXABLE",
    ]:

        count = stats[status]

        pct = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {status:15s}: "
            f"{count:6d} "
            f"({pct:5.1f}%)"
        )

    print(f"\nOutput File:")
    print(f"  {PHASE1_VALIDATED_OUTPUT}")

    print("=" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_phase1())