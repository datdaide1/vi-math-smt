"""O1 eval micro-benchmark worker (vLLM).  python eval_worker.py <mode> <model> <dtype> <out.json>

Workload per adapter = the runner's: 256 prompts (4-shot, train-split questions), greedy n=1 + sampling n=4
(T=0.6, top_p=0.95, top_k=20, per-request seed), max_tokens=1024, stop on the next "Bài toán:".
Modes:
  mkadapters  create 4 random-but-small LoRA adapters (r=32, all-linear) under /tmp/ad{k}
  seq         one vLLM engine per adapter (what evalrun/runner.py does today), adapters 0..N_SEQ-1
  multi       one engine, enable_lora max_loras=4, all 4 adapters' requests in one generate() call
  base        one engine, no LoRA (= the cost of evaluating a merged checkpoint, excl. merge/load)
"""
import json
import os
import sys
import time
import traceback

MODE, MODEL, DT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DATA = os.environ.get("O1_DATA", "/data")
N_ITEMS = int(os.environ.get("O1_N_ITEMS", "256"))
N_SEQ = int(os.environ.get("O1_N_SEQ", "2"))
MAXTOK = int(os.environ.get("O1_MAXTOK", "1024"))
STOPS = ["\n\nBài toán:", "\nBài toán:"]
res = {"mode": MODE, "model": MODEL, "dtype": DT, "n_items": N_ITEMS, "max_tokens": MAXTOK}


def dump():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def prompts():
    shots = json.load(open(f"{DATA}/fewshot4.json", encoding="utf-8"))
    pre = "".join(f"Bài toán:\n{s['problem'].strip()}\n\nLời giải:\n{s['solution'].strip()}\n\n" for s in shots)
    qs = [json.loads(l)["question"] for l in open(f"{DATA}/eval_prompts.jsonl", encoding="utf-8")][:N_ITEMS]
    return [pre + f"Bài toán:\n{q.strip()}\n\nLời giải:\n" for q in qs]


if MODE.endswith("_flex"):
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
    MODE = MODE[:-5]
try:
    if MODE == "mkadapters":
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        for k in range(4):
            torch.manual_seed(100 + k)
            m = get_peft_model(base, LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "lora_B" in n:
                        p.normal_(0, 0.01)
            m.save_pretrained(f"/tmp/ad{k}")
            base = m.unload()
        res["ok"] = True
        dump()
        sys.exit(0)

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    res["vllm"] = vllm.__version__
    P = prompts()
    dtype = {"bf16": "bfloat16", "fp16": "float16"}[DT]

    NOSEED = MODE.endswith("_noseed")

    def sps(seed0):
        g = SamplingParams(n=1, temperature=0.0, max_tokens=MAXTOK, stop=STOPS)
        s = [SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=20, max_tokens=MAXTOK, stop=STOPS,
                            seed=None if NOSEED else seed0 + i) for i in range(len(P))]
        return g, s

    def stats(outs):
        toks = [len(c.token_ids) for o in outs for c in o.outputs]
        trunc = sum(1 for o in outs for c in o.outputs if c.finish_reason == "length")
        return sum(toks), len(toks), trunc

    def engine(lora):
        kw = dict(model=MODEL, dtype=dtype, seed=0, gpu_memory_utilization=0.85, max_model_len=4096)
        if lora:
            kw.update(enable_lora=True, max_lora_rank=32, max_loras=lora)
        t = time.time()
        llm = LLM(**kw)
        return llm, time.time() - t

    runs = []
    if MODE in ("seq", "base", "base_noseed"):
        n_eng = N_SEQ if MODE == "seq" else 1
        for k in range(n_eng):
            llm, load_s = engine(1 if MODE == "seq" else 0)
            lr = LoRARequest(f"ad{k}", k + 1, f"/tmp/ad{k}") if MODE == "seq" else None
            g, s = sps(1000 * k)
            t = time.time()
            og = llm.generate(P, g, lora_request=lr, use_tqdm=False)
            osm = llm.generate(P, s, lora_request=lr, use_tqdm=False)
            gen_s = time.time() - t
            ntok, nseq, trunc = stats(og + osm)
            runs.append({"adapter": k, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "gen_tokens": ntok,
                         "n_seqs": nseq, "trunc": trunc, "tok_per_s": round(ntok / gen_s, 1),
                         "greedy_text": [o.outputs[0].text for o in og]})
            del llm
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    elif MODE == "multi":
        llm, load_s = engine(4)
        allp, allsp, alllr = [], [], []
        for k in range(4):
            lr = LoRARequest(f"ad{k}", k + 1, f"/tmp/ad{k}")
            g, s = sps(1000 * k)
            allp += P + P
            allsp += [g] * len(P) + s
            alllr += [lr] * (2 * len(P))
        t = time.time()
        outs = llm.generate(allp, allsp, lora_request=alllr, use_tqdm=False)
        gen_s = time.time() - t
        ntok, nseq, trunc = stats(outs)
        runs.append({"adapters": 4, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "gen_tokens": ntok,
                     "n_seqs": nseq, "trunc": trunc, "tok_per_s": round(ntok / gen_s, 1),
                     "greedy_text_ad0": [outs[i].outputs[0].text for i in range(len(P))],
                     "greedy_text_ad1": [outs[2 * len(P) + i].outputs[0].text for i in range(len(P))]})
    res["runs"] = runs
    res["ok"] = True
except Exception as e:  # noqa
    res.update(ok=False, error=repr(e)[:1500], tb=traceback.format_exc()[-3000:])
dump()
print("RESULT", MODE, res.get("ok"), [{k: v for k, v in r.items() if "text" not in k} for r in res.get("runs", [])],
      res.get("error"), flush=True)
