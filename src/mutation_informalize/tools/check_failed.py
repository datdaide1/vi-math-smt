import json
import os

with open(os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        if item["status"] == "failed":
            gt = item["ground_truth"]
            ans = item["answer"]
            print(f"Index {item['index']}: Z3: {gt} | Solver: {ans}")
