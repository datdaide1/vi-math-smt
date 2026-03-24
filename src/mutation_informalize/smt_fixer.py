"""
Phase 1: Fix TRIVIAL + WEAK SMT codes.
Re-formalize using NVIDIA API, validate with Z3.
Fully resumeable — safe to interrupt and restart.

Usage:
    cd src/mutation_informalize
    python smt_fixer.py
"""
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INPUT_SMT, FIXED_SMT_OUTPUT, OUTPUT_DIR
from z3_validator import validate_smt, answers_match
from llm_client import AsyncLLMClient, load_checkpoint, append_checkpoint, print_progress

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Classification ───

def classify_smt(smt_code: str) -> str:
    """Classify SMT code as GOOD, WEAK, or TRIVIAL."""
    if not smt_code or len(smt_code.strip()) < 10:
        return "TRIVIAL"
    
    lines = smt_code.strip().split('\n')
    asserts = [l.strip() for l in lines if l.strip().startswith('(assert')]
    all_a = ' '.join(asserts)
    
    ops = len(re.findall(r'\(\s*[+\-*/]', all_a))
    comps = len(re.findall(r'\(\s*(?:>|<|>=|<=)', all_a))
    total_ops = ops + comps
    
    trivial_c = 0
    for a in asserts:
        if re.match(r'\(assert\s+\(=\s+\w+\s+[\d.\-]+\)\)', a):
            trivial_c += 1
        elif re.match(r'\(assert\s+\(=\s+\w+\s+\w+\)\)', a):
            trivial_c += 1
    non_trivial = len(asserts) - trivial_c
    
    if total_ops >= 2 and non_trivial >= 2:
        return "GOOD"
    elif total_ops >= 1 and non_trivial >= 1:
        return "WEAK"
    return "TRIVIAL"


def extract_smt_from_text(text: str) -> str:
    """Extract SMT-LIB block from training data text."""
    m = re.search(r'## SMT-LIB\n```smt2?\n(.*?)\n```', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_problem_from_text(text: str) -> str:
    """Extract the Vietnamese problem text."""
    # The problem is typically between [INST] and ## Lời giải or ## SMT-LIB
    m = re.search(r'\[INST\](.*?)(?:## Lời giải|## SMT-LIB|\[/INST\])', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: first paragraph
    lines = text.split('\n')
    return '\n'.join(lines[:10])


def extract_answer_from_text(text: str) -> str:
    """Extract ground truth answer from text."""
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'Đáp án cuối cùng là:\s*(.+)', text)
    if m:
        return m.group(1).strip()
    return ""


# ─── Prompts ───

REFORMALIZE_PROMPT = """You are a formal verification expert specializing in SMT-LIB 2.6.
The current SMT-LIB code for the following math problem is too trivial — it just assigns
the answer directly without modeling the problem's logic.

Rewrite the SMT-LIB code to properly model the FULL SOLUTION LOGIC:
- Each quantity must be a declared variable
- Each relationship/computation must be a separate assert
- The answer must be DERIVED from constraints, NOT hardcoded
- Must have ≥2 non-trivial asserts with arithmetic operators (+, -, *, /)

Rules:
- Use (set-logic) and declare all variables as Real
- Must include (check-sat) and (get-value (answer))
- The answer MUST equal: {expected_answer}

Output ONLY the SMT-LIB code wrapped in [SMT-CODE]...[/SMT-CODE].

## Problem:
{problem}

## Current (trivial) SMT code:
```smt2
{current_smt}
```

## Ground truth answer: {expected_answer}"""


ENRICH_PROMPT = """You are a formal verification expert specializing in SMT-LIB 2.6.
The current SMT-LIB code for this math problem is too simple — it only has 1 computation step.
Rewrite it with MORE intermediate steps that model the full solution logic.

Rules:
- Break compound computations into multiple steps with intermediate variables
- Each step should be a separate (assert (= var expr))
- Must have ≥3 assert statements with arithmetic operators
- Keep the same final answer
- Use (set-logic QF_LRA), all variables Real
- Output ONLY SMT-LIB in [SMT-CODE]...[/SMT-CODE]

## Problem:
{problem}

## Current (too simple) SMT code:
```smt2
{current_smt}
```

## Expected answer: {expected_answer}"""


def extract_smt_from_response(text: str) -> str:
    """Extract SMT code from LLM response."""
    if not text:
        return ""
    m = re.search(r'\[SMT-CODE\](.*?)\[/SMT-CODE\]', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'```(?:smtlib|lisp|smt2|smt)?\s*(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "(assert" in text and "(check-sat" in text:
        return text.strip()
    return ""


# ─── Main Phase 1 ───

async def run_phase1():
    """Fix TRIVIAL and WEAK SMT codes."""
    print("=" * 70)
    print("  PHASE 1: Fix TRIVIAL + WEAK SMT codes")
    print("=" * 70)
    
    # Load input data
    with open(INPUT_SMT, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    # Classify
    to_fix = []
    stats = defaultdict(int)
    
    for item in all_data:
        smt = extract_smt_from_text(item["text"])
        quality = classify_smt(smt)
        stats[quality] += 1
        
        if quality in ("TRIVIAL", "WEAK"):
            to_fix.append({
                "index": item.get("index", 0),
                "subject": item.get("subject", ""),
                "level": item.get("level", 0),
                "text": item["text"],
                "current_smt": smt,
                "quality": quality,
                "problem": extract_problem_from_text(item["text"]),
                "expected_answer": extract_answer_from_text(item["text"]),
            })
    
    print(f"\n  Classification:")
    for q in ["GOOD", "WEAK", "TRIVIAL"]:
        print(f"    {q:10s}: {stats[q]:6d}")
    print(f"    To fix   : {len(to_fix):6d}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(FIXED_SMT_OUTPUT)
    already_done = len(checkpoint)
    print(f"  Checkpoint : {already_done} already processed")
    
    # Filter out already done
    remaining = [item for item in to_fix 
                 if str(item["index"]) + "_" + item["quality"] not in checkpoint]
    
    print(f"  Remaining  : {len(remaining)}")
    
    if not remaining:
        print("\n  All done! ✓")
        return
    
    # Process
    client = AsyncLLMClient()
    total = len(remaining)
    done = 0
    success = 0
    failed = 0
    
    for item in remaining:
        uid = str(item["index"]) + "_" + item["quality"]
        done += 1
        
        # Choose prompt based on quality
        if item["quality"] == "TRIVIAL":
            prompt = REFORMALIZE_PROMPT.format(
                problem=item["problem"],
                current_smt=item["current_smt"],
                expected_answer=item["expected_answer"],
            )
        else:
            prompt = ENRICH_PROMPT.format(
                problem=item["problem"],
                current_smt=item["current_smt"],
                expected_answer=item["expected_answer"],
            )
        
        # Call API with retries
        best_smt = item["current_smt"]  # fallback to original
        fix_status = "unchanged"
        
        for attempt in range(3):
            response = await client.call(
                system_prompt="You are a formal verification expert.",
                user_prompt=prompt,
                temperature=0.3 if attempt == 0 else 0.7,
            )
            
            if not response:
                continue
            
            new_smt = extract_smt_from_response(response)
            if not new_smt:
                continue
            
            # Validate with Z3
            valid, answer, info = validate_smt(new_smt)
            if valid and item["expected_answer"]:
                if answers_match(answer, item["expected_answer"]):
                    best_smt = new_smt
                    fix_status = "fixed"
                    break
                else:
                    fix_status = f"wrong_answer_attempt{attempt}"
            elif valid:
                best_smt = new_smt
                fix_status = "fixed_no_verify"
                break
        
        record = {
            "uid": uid,
            "source_index": item["index"],
            "subject": item["subject"],
            "level": item["level"],
            "quality": item["quality"],
            "fix_status": fix_status,
            "smt_code": best_smt,
        }
        
        append_checkpoint(FIXED_SMT_OUTPUT, record)
        
        if fix_status.startswith("fixed"):
            success += 1
        else:
            failed += 1
        
        print_progress(done, total, "Phase 1", f"✓{success} ✗{failed}")
    
    print(f"\n\n  Phase 1 Complete: {success} fixed, {failed} unchanged out of {total}")


if __name__ == "__main__":
    asyncio.run(run_phase1())
