
import time, sys, json
from vllm import LLM, SamplingParams
m = sys.argv[1]
t = time.time()
llm = LLM(model=m, dtype="float16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0, enforce_eager=False)
load = time.time() - t
ps = ["Bài toán:\nAn có 3 quả táo, mua thêm 5 quả. Hỏi An có bao nhiêu quả táo?\n\nLời giải:\n"] * 64
t = time.time()
outs = llm.generate(ps, SamplingParams(temperature=0, max_tokens=128))
dt = time.time() - t
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
print("RESULT", json.dumps({"model": m, "load_s": round(load,1), "gen_s": round(dt,2), "gen_tokens": ntok,
      "tok_per_s": round(ntok/dt,1), "sample": outs[0].outputs[0].text[:300]}, ensure_ascii=False))
