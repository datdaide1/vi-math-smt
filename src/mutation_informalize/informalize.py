"""
Phase 3: Informalize mutated SMT codes → Vietnamese math problems.
Uses NVIDIA API (gpt-oss-120b) with 3 keys rotating.
Fully resumeable via JSONL checkpoint.

Usage:
    cd src/mutation_informalize
    python informalize.py
"""
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MUTATED_SMT_OUTPUT, INFORMALIZED_OUTPUT, OUTPUT_DIR
from llm_client import AsyncLLMClient, load_checkpoint, append_checkpoint, print_progress

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Prompts ───

INFORMALIZE_SYSTEM = """Bạn là chuyên gia toán học và sư phạm. Nhiệm vụ của bạn là chuyển đổi 
mã SMT-LIB (mã logic hình thức) thành một bài toán bằng Tiếng Việt tự nhiên, 
kèm lời giải chi tiết."""

INFORMALIZE_USER = """Dựa trên mã SMT-LIB dưới đây, hãy:

1. **Bài toán**: Viết một bài toán bằng Tiếng Việt ở dạng word problem (câu chuyện thực tế).
   - Gắn các số với ngữ cảnh thực tế (ví dụ: 300 → "300 nghìn đồng", 7 → "7 ngày")
   - KHÔNG viết phương trình trực tiếp, phải là dạng câu chuyện
   - Phù hợp với chủ đề: {subject}
   - Độ khó: Level {level}/5

2. **Lời giải**: Viết lời giải từng bước bằng Tiếng Việt, rõ ràng và chi tiết.

3. **Đáp án**: Kết thúc bằng dòng:
   Đáp án cuối cùng là: $\\boxed{{{answer}}}$

## SMT-LIB Code:
```smt2
{smt_code}
```

## Đáp án đúng (từ solver): {answer}

Hãy viết bài toán và lời giải:"""


VERIFY_SYSTEM = """Bạn là chuyên gia toán học. Giải bài toán sau và chỉ đưa ra đáp án số."""

VERIFY_USER = """Giải bài toán sau. Chỉ trả lời đáp án số cuối cùng, không cần giải thích.

{problem}

Đáp án:"""


def extract_problem_and_solution(response: str) -> dict:
    """Parse LLM response into problem + solution."""
    if not response:
        return {"problem": "", "solution": "", "answer": ""}
    
    # Try to split by common headers
    problem = ""
    solution = ""
    answer = ""
    
    # Extract answer
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', response)
    if m:
        answer = m.group(1).strip()
    else:
        m = re.search(r'Đáp án cuối cùng là:\s*(.+)', response)
        if m:
            answer = m.group(1).strip()
    
    # Split problem and solution
    # Look for "Lời giải" or "Giải" header
    parts = re.split(r'(?:##?\s*)?(?:Lời giải|Giải|Solution)[:\s]*\n', response, maxsplit=1)
    if len(parts) == 2:
        problem = parts[0].strip()
        solution = parts[1].strip()
    else:
        # Try splitting by "Bài toán" header
        parts = re.split(r'(?:##?\s*)?(?:Bài toán|Problem)[:\s]*\n', response, maxsplit=1)
        if len(parts) == 2:
            remaining = parts[1]
            sol_parts = re.split(r'(?:##?\s*)?(?:Lời giải|Giải)[:\s]*\n', remaining, maxsplit=1)
            if len(sol_parts) == 2:
                problem = sol_parts[0].strip()
                solution = sol_parts[1].strip()
            else:
                problem = remaining.strip()
        else:
            problem = response.strip()
    
    # Clean problem header artifacts
    problem = re.sub(r'^#+\s*Bài toán[:\s]*', '', problem).strip()
    
    return {"problem": problem, "solution": solution, "answer": answer}


# ─── Main Phase 3 ───

async def run_phase3():
    """Informalize all mutated SMT codes → Vietnamese."""
    print("=" * 70)
    print("  PHASE 3: Informalize SMT → Vietnamese")
    print("=" * 70)
    
    # Load mutations
    if not os.path.exists(MUTATED_SMT_OUTPUT):
        print("  [ERROR] No mutations found. Run run_mutation.py first.")
        return
    
    mutations = []
    with open(MUTATED_SMT_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                mutations.append(json.loads(line))
    
    print(f"  Total mutations: {len(mutations)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(INFORMALIZED_OUTPUT)
    remaining = [m for m in mutations if m["uid"] not in checkpoint]
    print(f"  Checkpoint: {len(checkpoint)} done, {len(remaining)} remaining")
    
    if not remaining:
        print("\n  All done! ✓")
        return
    
    client = AsyncLLMClient()
    total = len(remaining)
    done = 0
    success = 0
    failed = 0
    
    # Process in batches for better concurrency
    BATCH_SIZE = 3  # Process 3 at a time (1 per API key)
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        tasks = []
        
        for item in batch:
            tasks.append(process_one(client, item))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for item, result in zip(batch, results):
            done += 1
            
            if isinstance(result, Exception):
                record = {
                    "uid": item["uid"],
                    "source_index": item["source_index"],
                    "subject": item["subject"],
                    "level": item["level"],
                    "status": "error",
                    "error": str(result)[:100],
                }
                append_checkpoint(INFORMALIZED_OUTPUT, record)
                failed += 1
            elif result:
                append_checkpoint(INFORMALIZED_OUTPUT, result)
                if result.get("status") == "ok":
                    success += 1
                else:
                    failed += 1
            else:
                failed += 1
            
            print_progress(done, total, "Phase 3", f"✓{success} ✗{failed}")
    
    print(f"\n\n{'=' * 70}")
    print(f"  Phase 3 Complete: {success} informalized, {failed} failed out of {total}")
    print(f"  Output: {INFORMALIZED_OUTPUT}")
    print(f"{'=' * 70}")


async def process_one(client: AsyncLLMClient, item: dict) -> dict:
    """Informalize a single mutation."""
    uid = item["uid"]
    smt = item["mutated_smt"]
    answer = item["z3_answer"]
    subject = item["subject"]
    level = item["level"]
    
    # Step 1: Informalize
    prompt = INFORMALIZE_USER.format(
        subject=subject,
        level=level,
        answer=answer,
        smt_code=smt,
    )
    
    response = await client.call(INFORMALIZE_SYSTEM, prompt, temperature=0.7)
    if not response:
        return {
            "uid": uid, "source_index": item["source_index"],
            "subject": subject, "level": level,
            "status": "api_fail",
        }
    
    parsed = extract_problem_and_solution(response)
    
    if not parsed["problem"]:
        return {
            "uid": uid, "source_index": item["source_index"],
            "subject": subject, "level": level,
            "status": "parse_fail",
        }
    
    # Step 2: Verify (optional — check consistency)
    # Ask LLM to solve the problem and compare answers
    verified = False
    verify_response = await client.call(
        VERIFY_SYSTEM,
        VERIFY_USER.format(problem=parsed["problem"]),
        temperature=0,
    )
    
    if verify_response:
        # Extract number from verify response
        numbers = re.findall(r'[-+]?\d*\.?\d+', verify_response)
        if numbers:
            try:
                verify_answer = float(numbers[-1])
                expected = float(answer)
                if abs(verify_answer - expected) < max(1e-2, 1e-4 * abs(expected)):
                    verified = True
            except (ValueError, TypeError):
                pass
    
    # Build training text in same format as existing data
    training_text = build_training_text(
        parsed["problem"], parsed["solution"], answer, smt, subject
    )
    
    return {
        "uid": uid,
        "source_index": item["source_index"],
        "subject": subject,
        "level": level,
        "algorithm": item.get("algorithm", ""),
        "status": "ok" if verified else "ok_unverified",
        "verified": verified,
        "text": training_text,
        "problem_vi": parsed["problem"],
        "solution_vi": parsed["solution"],
        "z3_answer": answer,
    }


def build_training_text(problem: str, solution: str, answer: str, 
                         smt_code: str, subject: str) -> str:
    """Build training text in the same format as existing clean data."""
    text = f"""[INST] Hãy giải bài toán sau và trình bày lời giải chi tiết.

{problem} [/INST]

## Lời giải

{solution}

Đáp án cuối cùng là: $\\boxed{{{answer}}}$

## SMT-LIB
```smt2
{smt_code}
```"""
    return text


if __name__ == "__main__":
    asyncio.run(run_phase3())
