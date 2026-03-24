import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3_informalize import validate_generation

DATA_FILE = os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized.jsonl")

def fix_data():
    if not os.path.exists(DATA_FILE):
        print("Data file not found!")
        return

    lines = []
    fixed_count = 0
    total_failed = 0

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            if item.get("status") == "failed":
                total_failed += 1
                parsed = {
                    "problem": item.get("problem", ""),
                    "solution": item.get("solution", ""),
                    "answer": item.get("answer", "")
                }
                
                # Re-validate with the new logic
                ok, feedback = validate_generation(parsed, item.get("ground_truth", ""))
                
                if ok:
                    item["status"] = "ok"
                    item["feedback"] = "fixed_by_script"
                    fixed_count += 1
                    print(f"Fixed Index {item['index']}: Z3={item['ground_truth']} | Solver={item['answer']}")
            
            lines.append(json.dumps(item, ensure_ascii=False))

    # Overwrite the file with the fixed data
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"\nSuccessfully fixed {fixed_count} / {total_failed} previously failed items.")

if __name__ == "__main__":
    fix_data()
