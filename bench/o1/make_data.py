"""Build the O1 micro-benchmark bundle from TRAIN splits only (no test items leave the laptop)."""
import json, random, re, sys
sys.path.insert(0, ".")
from vi_math_verified_aug.train import template as T

def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def ans_gsm(sol):
    m = re.search(r"####\s*(.+)\s*$", sol.strip())
    return m.group(1).strip() if m else None

rnd = random.Random(0)
g = rows("jobs/build/s2-modal-data/train.jsonl")
m = rows("data/finetune/math/math_train_vietnamese_final.jsonl")
rnd.shuffle(g); rnd.shuffle(m)
def to_item(r, src):
    sol = r.get("solution") or ""
    a = ans_gsm(sol) if src == "gsm" else None
    try:
        c = T.completion(sol, a)
    except ValueError:
        return None
    return {"src": src, "prompt": T.prompt(r["problem"]), "completion": c}
train, evalp = [], []
for src, rs in (("gsm", g), ("math", m)):
    its = [x for x in (to_item(r, src) for r in rs[:1600]) if x]
    train += its[:1024]
    evalp += [{"src": src, "question": rs[i]["problem"]} for i in range(1400, 1528)]
rnd.shuffle(train)
with open("bench/o1/data/train_mix.jsonl", "w", encoding="utf-8") as f:
    for x in train: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open("bench/o1/data/eval_prompts.jsonl", "w", encoding="utf-8") as f:
    for x in evalp: f.write(json.dumps(x, ensure_ascii=False) + "\n")
shots = json.load(open("jobs/build/s2-modal-data/fewshot4.json", encoding="utf-8"))
json.dump(shots, open("bench/o1/data/fewshot4.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(train), len(evalp), list(shots[0].keys()))
