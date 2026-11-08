"""S2 student pilot v2 — runs ON a private Kaggle T4x2 kernel (one lane per GPU).

The laptop prepends one line `JOB = {...}` (see submit_pilot.py --src s2_pilot.py). Without it a default JOB is used.

What it does (all fp16 on T4, sm_75):
  env        install vllm (same pin as W2 v1), REMOVE torchao (peft>=0.18 refuses torchao<0.16 at LoRA dispatch),
             then a GPU smoke test: tiny Qwen3 + peft LoRA fwd/bwd in fp16. Writes results/env.json.
  eval       4-shot Vi-GSM8K on the fixed 300-item subset; vLLM first (each attempt in its own process), else HF
             generate. Scored with the frozen math_verify.py snapshot shipped in the dataset.
  packcheck  per-token logits of 4 items run separately vs packed in one row, in fp32 AND fp16, for 4 variants:
             position_ids only (W2 v1 path, use_cache default) / position_ids + use_cache=False / explicit 4-D
             block-diagonal mask / right-padded batch. Pass rule (from the brief): max |dlogit| < 1e-2.
  sft        100-step fp16 LoRA SFT, T3 sec.3.1 recipe: r=32, alpha=32, all-linear text decoder, fp32 adapters, AdamW fp32,
             LR 2e-4 (3% warm-up, then constant), 32 seq/step, completion-only token-mean NLL (fp32 CE), dynamic
             GradScaler, grad-clip 1.0. Batching = PADDED + LENGTH-GROUPED inside each 32-seq step (the step's items
             are sorted by length, then cut into micro-batches; same items per step as random order). Micro-batch is
             auto-reduced on OOM. Records real (non-pad) tokens/s, skipped steps, loss curve, and the max |activation|
             (inputs and outputs of every decoder Linear + every decoder-layer output) on every 4th step; the
             throughput is timed only on the un-hooked steps.
  sft_packed_hf  the same run with padding-free rows, ONLY if a packcheck variant passed in fp32.
  unsloth    Unsloth installed into an isolated --target dir (no deps, so torch/vllm are untouched). Logits-equality
             check of Unsloth's own padding-free collator path (trainer.compute_loss on the packed batch) vs separate
             forwards, fp32 then fp16; then 100-step Unsloth SFT throughput, padding_free on and off.

Outputs: /kaggle/working/results/{env.json, meta.json, <model_tag>/{eval.json, preds.jsonl, packcheck.json, sft.json,
         sft_packed_hf.json, unsloth.json, *.log, *.error.json, *.skipped.json}}
"""
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback

try:
    JOB  # noqa: F821  (injected by submit_pilot.py)
except NameError:
    JOB = {"lanes": [[["Qwen/Qwen3-0.6B-Base", ["packcheck", "sft"]]], [["Qwen/Qwen3-1.7B-Base", ["sft"]]]],
           "install": "pip install -q 'vllm==0.33.0'", "sft_steps": 100, "eval_n": 300}

WORK = os.environ.get("S2_WORK", "/kaggle/working")
RES = os.environ.get("S2_RES", f"{WORK}/results")
DTYPE_NAME = os.environ.get("S2_DTYPE", "fp16")  # fp16 (T4: autocast + GradScaler) | bf16 (L4/L40S/H100: no scaler)
UNS_SITE = "/tmp/uns_site"
UNS_MARK = "/tmp/uns_site.done"
SEED = 20261107
TEMPLATE_Q = "Bài toán:\n{q}\n\nLời giải:\n"
STOPS = ["\n\nBài toán", "Bài toán:"]
FP16_MAX = 65504.0


def tag(m):
    return m.split("/")[-1]


def find_data():
    if os.environ.get("S2_DATA"):
        return os.environ["S2_DATA"]
    hits = glob.glob("/kaggle/input/**/test300.jsonl", recursive=True)
    if not hits:
        raise FileNotFoundError("test300.jsonl not mounted")
    return os.path.dirname(hits[0])


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def clean_solution(sol, gt):
    s = re.sub(r"<<[^<>]*>>", "", sol).rsplit("####", 1)[0].rstrip()
    return f"{s}\nĐáp số: \\boxed{{{gt}}}"


def gpu_info():
    import torch
    return {"name": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def lib_versions():
    import importlib.metadata as md
    out = {}
    for d in ("torch", "transformers", "peft", "trl", "unsloth", "unsloth_zoo", "vllm", "accelerate", "torchao",
              "triton", "bitsandbytes", "flash-linear-attention", "causal-conv1d", "xformers", "datasets",
              "cut_cross_entropy"):
        try:
            out[d] = md.version(d)
        except Exception:
            out[d] = None
    try:
        import torch
        out["torch_full"] = torch.__version__
    except Exception:
        pass
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def hf_access(model):
    """Return None if the repo's config is downloadable, else an error string (gated / missing)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model, "config.json")
        return None
    except Exception as e:  # noqa
        return f"{type(e).__name__}: {str(e)[:400]}"


# =========================================================================== model loading
def _missing_ok(info):
    miss = [k for k in (info or {}).get("missing_keys", []) if not k.endswith("lm_head.weight")]
    return miss


def load_model(model, dtype, attn="sdpa"):
    """Text-only where possible. Returns the model on cuda with ._s2_loader describing how it was loaded."""
    import torch  # noqa
    import transformers as T
    cfg = T.AutoConfig.from_pretrained(model)
    errs = []
    multimodal = any(getattr(cfg, k, None) is not None for k in ("audio_config", "vision_config"))
    if multimodal:
        # 1) text-only class with the text sub-config (drops the audio/vision towers and their SDPA problem)
        tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", None)
        arch = (cfg.architectures or [""])[0].replace("ForConditionalGeneration", "ForCausalLM")
        cls = getattr(T, arch, None)
        if cls is not None and tcfg is not None:
            for km in (None, {r"^model\.language_model\.": "model."}):
                try:
                    kw = dict(config=tcfg, dtype=dtype, attn_implementation=attn, output_loading_info=True)
                    if km:
                        kw["key_mapping"] = km
                    m, info = cls.from_pretrained(model, **kw)
                    miss = _missing_ok(info)
                    if len(miss) > 0:
                        errs.append(f"{arch} key_mapping={km}: {len(miss)} missing e.g. {miss[:3]}")
                        del m
                        continue
                    m = m.to("cuda")
                    m._s2_loader = f"{arch}(text-only, key_mapping={bool(km)}, attn={attn})"
                    return m
                except Exception as e:  # noqa
                    errs.append(f"{arch} key_mapping={km}: {type(e).__name__}: {str(e)[:300]}")
        # 2) full model, per-sub-config attention (audio tower eager)
        for impl in ({"text_config": attn, "vision_config": attn, "audio_config": "eager"}, "eager"):
            for clsname in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
                try:
                    m = getattr(T, clsname).from_pretrained(model, dtype=dtype, attn_implementation=impl).to("cuda")
                    m._s2_loader = f"{clsname}(full, attn={impl})"
                    m._s2_load_errors = errs
                    return m
                except Exception as e:  # noqa
                    errs.append(f"{clsname} attn={impl}: {type(e).__name__}: {str(e)[:300]}")
        raise RuntimeError("all multimodal load paths failed:\n" + "\n".join(errs))
    for impl in (attn, "eager"):
        try:
            m = T.AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation=impl).to("cuda")
            m._s2_loader = f"AutoModelForCausalLM(attn={impl})"
            m._s2_load_errors = errs
            return m
        except Exception as e:  # noqa
            errs.append(f"AutoModelForCausalLM attn={impl}: {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("load failed:\n" + "\n".join(errs))


SKIP_RE = re.compile(r"(vision|visual|audio|multi_modal|embed_vision|embed_audio|lm_head|mtp|patch_embed|merger)", re.I)


def text_linear_targets(m):
    import torch.nn as nn
    names, suffixes = [], set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and not SKIP_RE.search(n):
            names.append(n)
            suffixes.add(n.split(".")[-1])
    return names, sorted(suffixes)


def decoder_layers(m):
    best = None
    for n, mod in m.named_modules():
        if n.endswith("layers") and hasattr(mod, "__len__") and len(mod) > 4 and not SKIP_RE.search(n):
            if best is None or len(mod) > len(best[1]):
                best = (n, mod)
    return best


def fwd(m, inp):
    """Forward; adds token_type_ids when a multimodal wrapper demands it during training."""
    import torch
    kw = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()}
    kw.setdefault("use_cache", False)
    try:
        return m(**kw)
    except (TypeError, ValueError) as e:
        if "token_type_ids" not in str(e):
            raise
        kw["token_type_ids"] = torch.zeros_like(kw["input_ids"])
        return m(**kw)


def nll_sum(logits, labels):
    import torch.nn.functional as F
    lb = labels[:, 1:].to(logits.device)
    mask = lb != -100
    lg = logits[:, :-1][mask].float()
    return F.cross_entropy(lg, lb[mask], reduction="sum")


# =========================================================================== data
def load_train(data_dir, tok, n_items, seed=SEED, max_len=512):
    rows = [json.loads(l) for l in open(f"{data_dir}/train.jsonl", encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out, dropped = [], 0
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, "add_bos_token", False)) else []
    for r in rows:
        p = tok(TEMPLATE_Q.format(q=r["problem"]), add_special_tokens=False)["input_ids"]
        c = tok(clean_solution(r["solution"], r["ground_truth"]), add_special_tokens=False)["input_ids"] + [
            tok.eos_token_id]
        ids = bos + p + c
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append({"ids": ids, "labels": [-100] * (len(bos) + len(p)) + c, "problem": r["problem"],
                    "completion": clean_solution(r["solution"], r["ground_truth"]), "gt": r["ground_truth"],
                    "train_index": r.get("index")})
        if len(out) >= n_items:
            break
    return out, dropped, bool(bos)


def batch_padded(items, pad_id):
    import torch
    L = max(len(it["ids"]) for it in items)
    ids = [it["ids"] + [pad_id] * (L - len(it["ids"])) for it in items]
    lab = [it["labels"] + [-100] * (L - len(it["labels"])) for it in items]
    att = [[1] * len(it["ids"]) + [0] * (L - len(it["ids"])) for it in items]
    return ({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}, torch.tensor(lab),
            sum(len(it["ids"]) for it in items), L * len(items))


def block_mask(lens, dtype):
    import torch
    L = sum(lens)
    allow = torch.zeros(L, L, dtype=torch.bool)
    o = 0
    for n in lens:
        allow[o:o + n, o:o + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
        o += n
    m = torch.zeros(L, L, dtype=dtype)
    m.masked_fill_(~allow, torch.finfo(dtype).min)
    return m[None, None]


def batch_packed(items, variant, dtype):
    """variant: pos_ids_default_cache | pos_ids_no_cache | block_mask_4d"""
    import torch
    ids, labels, pos = [], [], []
    for it in items:
        ids += it["ids"]
        labels += [-100] + it["labels"][1:]  # never predict across a boundary
        pos += list(range(len(it["ids"])))
    inp = {"input_ids": torch.tensor([ids]), "position_ids": torch.tensor([pos])}
    if variant == "pos_ids_default_cache":
        inp["use_cache"] = None  # -> model default (True for these configs): the W2 v1 call
    if variant == "block_mask_4d":
        inp["attention_mask"] = block_mask([len(it["ids"]) for it in items], dtype)
    return inp, torch.tensor([labels]), len(ids), len(ids)


# =========================================================================== eval
def build_prompts(data_dir, n):
    shots = json.load(open(f"{data_dir}/fewshot4.json", encoding="utf-8"))
    prefix = "\n\n".join(f"Bài toán:\n{s['problem']}\n\nLời giải:\n{s['solution']}" for s in shots) + "\n\n"
    items = [json.loads(l) for l in open(f"{data_dir}/test300.jsonl", encoding="utf-8")][:n]
    return items, [prefix + TEMPLATE_Q.format(q=it["question"]) for it in items]


def cut_stop(text):
    for s in STOPS:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text


def score_and_dump(model, data_dir, items, texts, ntoks, res):
    sys.path.insert(0, data_dir)
    from math_verify import extract_answer, is_correct
    correct, preds = 0, []
    for it, tx, nt in zip(items, texts, ntoks):
        body = cut_stop(tx)
        pred = extract_answer(body)
        ok = bool(is_correct(pred, it["ground_truth"])) if pred is not None else False
        correct += ok
        preds.append({"test_index": it["test_index"], "gold": it["ground_truth"], "pred": pred, "correct": ok,
                      "gen_tokens": nt, "boxed": "\\boxed" in body, "output": body})
    n = len(items)
    res.update({"correct": correct, "accuracy": round(correct / n, 4), "ci95_wilson": wilson(correct, n),
                "gen_tokens_total": sum(ntoks), "gen_tok_per_s": round(sum(ntoks) / res["gen_s"], 1),
                "boxed_rate": round(sum(p["boxed"] for p in preds) / n, 3),
                "hit_max_tokens": sum(1 for x in ntoks if x >= 400)})
    out_dir = f"{RES}/{tag(model)}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dump(f"{out_dir}/eval.json", res)
    print("EVAL", model, res["engine"], res["accuracy"], res["gen_tok_per_s"], flush=True)


def task_eval_vllm(model, data_dir, attempt):
    spec = (JOB.get("vllm_attempts", {}).get(tag(model)) or [{}])[attempt]
    for k, v in (spec.get("env") or {}).items():
        os.environ[k] = v
    items, prompts = build_prompts(data_dir, JOB.get("eval_n", 300))
    res = {"model": model, "n": len(items), "gpu": gpu_info(), "libs": lib_versions(), "engine": "vllm",
           "vllm_attempt": attempt, "vllm_spec": spec,
           "decoding": {"temperature": 0, "max_tokens": 400, "stop": STOPS, "shots": 4}}
    from vllm import LLM, SamplingParams
    kw = dict(model=model, dtype="float16", max_model_len=3072, gpu_memory_utilization=0.85, seed=0)
    kw.update(spec.get("kw") or {})
    t = time.time()
    llm = None
    for drop in ([], ["attention_config"], ["attention_config", "limit_mm_per_prompt"]):
        k2 = {k: v for k, v in kw.items() if k not in drop}
        if "limit_mm_per_prompt" not in drop and spec.get("mm_limit", True):
            k2.setdefault("limit_mm_per_prompt", {"image": 0, "video": 0, "audio": 0})
        try:
            llm = LLM(**k2)
            res["vllm_kwargs"] = {k: v for k, v in k2.items() if k != "model"}
            break
        except TypeError as e:
            res.setdefault("vllm_kw_errors", []).append(repr(e)[:300])
        except Exception as e:  # noqa
            msg = repr(e)
            if "limit_mm_per_prompt" in msg or "multimodal" in msg.lower() and "limit" in msg.lower():
                res.setdefault("vllm_kw_errors", []).append(msg[:300])
                continue
            raise
    if llm is None:
        raise RuntimeError("vLLM: no kwarg combination accepted")
    res["load_s"] = round(time.time() - t, 1)
    sp = SamplingParams(temperature=0, max_tokens=400, stop=STOPS, seed=0)
    t = time.time()
    outs = llm.generate(prompts, sp)
    res["gen_s"] = round(time.time() - t, 2)
    texts = [o.outputs[0].text for o in outs]
    ntoks = [len(o.outputs[0].token_ids) for o in outs]
    res["prompt_tokens_mean"] = round(sum(len(o.prompt_token_ids) for o in outs) / len(outs), 1)
    score_and_dump(model, data_dir, items, texts, ntoks, res)


def task_eval_hf(model, data_dir):
    import torch
    from transformers import AutoTokenizer
    items, prompts = build_prompts(data_dir, JOB.get("eval_n", 300))
    res = {"model": model, "n": len(items), "gpu": gpu_info(), "libs": lib_versions(), "engine": "hf_generate",
           "decoding": {"temperature": 0, "max_tokens": 400, "stop": STOPS, "shots": 4}}
    vl = sorted(glob.glob(f"{RES}/{tag(model)}/eval_vllm*.error.json"))
    res["vllm_errors"] = [(load(p) or {}).get("error", "")[-1500:] for p in vl]
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t = time.time()
    m = load_model(model, torch.float16)
    res["load_s"] = round(time.time() - t, 1)
    res["loader"] = m._s2_loader
    m.eval()
    bs = JOB.get("hf_eval_bs", 16)
    texts, ntoks = [None] * len(prompts), [0] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    t = time.time()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=400, do_sample=False, stop_strings=STOPS, tokenizer=tok,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            g = gen[j]
            g = g[g != tok.pad_token_id]
            ntoks[i] = int(g.numel())
            texts[i] = tok.decode(g, skip_special_tokens=True)
    res["gen_s"] = round(time.time() - t, 2)
    res["hf_eval_bs"] = bs
    score_and_dump(model, data_dir, items, texts, ntoks, res)


def task_eval_dev(model, data_dir, path):
    """0-shot greedy eval of a merged SFT model on dev300 = the 300 train items right after the SFT items."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, data_dir)
    from math_verify import extract_answer, is_correct
    tok = AutoTokenizer.from_pretrained(path)
    n_sft = JOB.get("sft_steps", 100) * 32
    allit, _, _ = load_train(data_dir, tok, n_sft + 300)
    dev = allit[n_sft:]
    prompts = [TEMPLATE_Q.format(q=it["problem"]) for it in dev]
    vdt = "bfloat16" if DTYPE_NAME == "bf16" else "float16"
    t = time.time()
    llm = LLM(model=path, dtype=vdt, max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    load_s = time.time() - t
    t = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=400, stop=STOPS, seed=0))
    gen_s = time.time() - t
    preds, correct = [], 0
    for it, o in zip(dev, outs):
        body = cut_stop(o.outputs[0].text)
        pred = extract_answer(body)
        ok = bool(is_correct(pred, it["gt"])) if pred is not None else False
        correct += ok
        preds.append({"train_index": it["train_index"], "gold": it["gt"], "pred": pred, "correct": ok})
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"model": model, "precision": DTYPE_NAME, "platform": os.environ.get("S2_PLATFORM", "kaggle-T4"),
           "n": len(dev), "correct": correct, "accuracy": round(correct / len(dev), 4),
           "ci95_wilson": wilson(correct, len(dev)), "vllm_dtype": vdt, "load_s": round(load_s, 1),
           "gen_s": round(gen_s, 1), "gen_tok_per_s": round(ntok / gen_s, 1), "libs": lib_versions(),
           "gpu": gpu_info(), "preds": preds}
    dump(f"{RES}/{tag(model)}/eval_dev.json", res)
    print("EVAL_DEV", model, DTYPE_NAME, res["accuracy"], flush=True)


# =========================================================================== packcheck
VARIANTS = ["pos_ids_default_cache", "pos_ids_no_cache", "block_mask_4d", "padded"]


def task_packcheck(model, data_dir):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items, _, _ = load_train(data_dir, tok, 4, seed=1)
    res = {"model": model, "libs": lib_versions(), "lens": [len(it["ids"]) for it in items], "threshold": 1e-2,
           "by_dtype": {}}
    for dname, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        m = load_model(model, dtype)
        m.eval()
        res["loader"] = m._s2_loader
        r = {}
        with torch.no_grad():
            sep = [fwd(m, {"input_ids": torch.tensor([it["ids"]])}).logits[0].float() for it in items]
            for v in VARIANTS:
                if v == "padded":
                    b, _, _, _ = batch_padded(items, pad_id)
                    lg = fwd(m, b).logits.float()
                    segs = [lg[k, :len(it["ids"])] for k, it in enumerate(items)]
                else:
                    b, _, _, _ = batch_packed(items, v, dtype)
                    if b.get("use_cache", 0) is None:
                        b.pop("use_cache")
                        kw = {k: x.to("cuda") for k, x in b.items()}
                        lg = m(**kw).logits[0].float()  # model-default use_cache (W2 v1 call)
                    else:
                        lg = fwd(m, b).logits[0].float()
                    segs, o = [], 0
                    for it in items:
                        segs.append(lg[o:o + len(it["ids"])])
                        o += len(it["ids"])
                per = []
                for k, it in enumerate(items):
                    a, p = sep[k], segs[k]
                    lab = torch.tensor([it["labels"]], device="cuda")
                    nc = max(1, (lab[:, 1:] != -100).sum().item())
                    per.append({"len": len(it["ids"]), "max_abs": round((a - p).abs().max().item(), 6),
                                "mean_abs": round((a - p).abs().mean().item(), 7),
                                "nll_sep": round(nll_sum(a[None], lab).item() / nc, 5),
                                "nll_var": round(nll_sum(p[None], lab).item() / nc, 5),
                                "argmax_agree": round((a.argmax(-1) == p.argmax(-1)).float().mean().item(), 4)})
                mx = max(x["max_abs"] for x in per)
                r[v] = {"per_item": per, "max_abs_all": mx, "max_abs_items_2_4": max(x["max_abs"] for x in per[1:]),
                        "pass_1e-2": mx < 1e-2}
        res["by_dtype"][dname] = r
        del m
        torch.cuda.empty_cache()
    res["fp32_passing_variants"] = [v for v in VARIANTS if v != "padded" and res["by_dtype"]["fp32"][v]["pass_1e-2"]]
    dump(f"{RES}/{tag(model)}/packcheck.json", res)
    print("PACKCHECK", model, {d: {v: x["max_abs_all"] for v, x in r.items()} for d, r in res["by_dtype"].items()},
          flush=True)


# =========================================================================== SFT (own loop)
class ActMon:
    """GPU-side running max of |x| for decoder Linear inputs/outputs and decoder-layer outputs (no per-hook sync)."""

    def __init__(self, m):
        import torch
        import torch.nn as nn
        self.on = False
        self.keys, self.hooks = [], []
        idx = {}

        def kid(k):
            if k not in idx:
                idx[k] = len(self.keys)
                self.keys.append(k)
            return idx[k]

        dl = decoder_layers(m)
        self.layer_path = dl[0] if dl else None
        specs = []
        if dl:
            for i, layer in enumerate(dl[1]):
                specs.append((layer, None, kid(f"layer{i:02d}.out")))
                for n, mod in layer.named_modules():
                    if isinstance(mod, nn.Linear) and "lora_" not in n:
                        suf = n.split(".")[-1] if not n.endswith("base_layer") else n.split(".")[-2]
                        specs.append((mod, kid(f"{suf}.in"), kid(f"{suf}.out")))
        self.buf = torch.zeros(max(1, len(self.keys)), device="cuda")
        self.run_max = {}

        def mk(i_in, i_out):
            def h(_m, inp, out):
                if not self.on:
                    return
                if i_in is not None and inp and torch.is_tensor(inp[0]):
                    self.buf[i_in] = torch.maximum(self.buf[i_in], inp[0].detach().abs().amax().float())
                o = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(o):
                    self.buf[i_out] = torch.maximum(self.buf[i_out], o.detach().abs().amax().float())
            return h

        for mod, i_in, i_out in specs:
            self.hooks.append(mod.register_forward_hook(mk(i_in, i_out)))

    def collect(self):
        vals = self.buf.tolist()
        self.buf.zero_()
        step_max, step_key = -1.0, None
        for k, v in zip(self.keys, vals):
            if v != v:  # NaN counts as overflow
                v = float("inf")
            self.run_max[k] = max(self.run_max.get(k, 0.0), v)
            if v > step_max:
                step_max, step_key = v, k
        return step_max, step_key

    def remove(self):
        for h in self.hooks:
            h.remove()


def task_sft(model, data_dir, packed_variant=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    random.seed(SEED)
    steps, seq_per_step = JOB.get("sft_steps", 100), 32
    tok = AutoTokenizer.from_pretrained(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data, dropped, has_bos = load_train(data_dir, tok, steps * seq_per_step)
    cdt = torch.bfloat16 if DTYPE_NAME == "bf16" else torch.float16
    t_load = time.time()
    m = load_model(model, cdt)
    t_load = time.time() - t_load
    loader = m._s2_loader
    names, suffixes = text_linear_targets(m)
    n_params = sum(p.numel() for p in m.parameters())
    n_params_text = sum(p.numel() for n, p in m.named_parameters() if not SKIP_RE.search(n) or "lm_head" in n)
    gc = n_params_text > float(os.environ.get("S2_GC_ABOVE", JOB.get("gc_above_params", 2.5e9))) or \
        tag(model) in JOB.get("force_gc", [])
    cfg = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=names, bias="none")
    m = get_peft_model(m, cfg)  # autocast_adapter_dtype=True -> LoRA weights fp32
    if gc:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    m.train()
    trainable = [p for p in m.parameters() if p.requires_grad]
    lora_dtypes = sorted({str(p.dtype) for p in trainable})
    opt = torch.optim.AdamW(trainable, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, enabled=(cdt == torch.float16))

    def step_micro(step_items, micro):
        srt = sorted(step_items, key=lambda it: -len(it["ids"]))  # length-grouped inside the step
        return [srt[i:i + micro] for i in range(0, len(srt), micro)]

    def make(mb):
        if packed_variant:
            return batch_packed(mb, packed_variant, cdt)
        return batch_padded(mb, pad_id)

    # --- memory probe on the worst micro-batch; halve micro on OOM
    micro_try = JOB.get("micro_by_model", {}).get(tag(model), JOB.get("micro", 8))
    longest = sorted(data, key=lambda it: -len(it["ids"]))
    oom_log = []

    # BUG (S2 v1): the probe ran inline, so after an OOM the failed attempt's `out`/`loss` (with the whole
    # autograd graph) stayed bound in this frame while the smaller micro-batch was tried -> cascading OOMs
    # (Qwen3-0.6B fell 8->4->2->1 on a 15 GB T4 with 3.4 GB actually needed). The probe now runs in its own
    # function; its locals die on return, then gc + empty_cache before the next size.
    def probe(n):
        try:
            inp, lab, _, _ = make(longest[:n])
            with torch.autocast("cuda", dtype=cdt):
                out = fwd(m, inp)
            loss = nll_sum(out.logits, lab) / max(1, (lab[:, 1:] != -100).sum().item())
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            return None
        except torch.cuda.OutOfMemoryError as e:
            return str(e)[:200]

    import gc as _gc
    while True:
        err = probe(micro_try)
        opt.zero_grad(set_to_none=True)
        _gc.collect()
        torch.cuda.empty_cache()
        if err is None:
            break
        oom_log.append({"micro": micro_try, "err": err})
        if micro_try == 1:
            raise torch.cuda.OutOfMemoryError(err)
        micro_try //= 2
    micro = micro_try
    accum = math.ceil(seq_per_step / micro)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, enabled=(cdt == torch.float16))  # fresh scaler after the probe
    mon = ActMon(m)
    hook_every = JOB.get("hook_every", 4)
    log, skipped, nonfinite = [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    real_tot, pad_tot, comp_tot = 0, 0, 0
    t_timed, tok_timed, comp_timed, n_timed = 0.0, 0, 0, 0
    t_all = time.time()
    budget = float(os.environ.get("S2_TASK_BUDGET_S", 1e9)) - 60 - (time.time() - T_TASK)
    partial = f"{RES}/{tag(model)}/{'sft_packed_hf' if packed_variant else 'sft'}.partial.json"
    stopped_early = None
    for step in range(steps):
        if time.time() - t_all > budget:
            stopped_early = f"in-task budget reached at step {step}"
            break
        hooked = step < 3 or step % hook_every == 0
        mon.on = hooked
        torch.cuda.synchronize()
        t0 = time.time()
        items = data[step * seq_per_step:(step + 1) * seq_per_step]
        n_comp = sum(sum(1 for x in it["labels"][1:] if x != -100) for it in items)
        loss_acc, s_tok, s_pad = 0.0, 0, 0
        for mb in step_micro(items, micro):
            inp, lab, ntok, npad = make(mb)
            s_tok += ntok
            s_pad += npad
            with torch.autocast("cuda", dtype=cdt):
                out = fwd(m, inp)
            loss = nll_sum(out.logits, lab) / n_comp
            if not torch.isfinite(loss):
                nonfinite += 1
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            del out, loss
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        s_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        was_skipped = (scaler.get_scale() < s_before) if scaler.is_enabled() else not math.isfinite(gnorm)
        skipped += int(was_skipped)
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        amax, akey = mon.collect() if hooked else (None, None)
        real_tot += s_tok
        pad_tot += s_pad
        comp_tot += n_comp
        if not hooked:
            t_timed += dt
            tok_timed += s_tok
            comp_timed += n_comp
            n_timed += 1
        log.append({"step": step, "loss": round(loss_acc, 5),
                    "gnorm": round(gnorm, 4) if math.isfinite(gnorm) else str(gnorm), "scale": s_before,
                    "skipped": was_skipped, "s": round(dt, 3), "tokens": s_tok, "padded_tokens": s_pad,
                    "hooked": hooked, "act_absmax": amax, "act_absmax_key": akey})
        if step % 10 == 9:
            dump(partial, {"model": model, "partial": True, "steps_done": step + 1, "micro_batch": micro,
                           "grad_ckpt": gc, "skipped_steps": skipped, "log": log,
                           "real_tokens_per_s_so_far": round(tok_timed / t_timed, 1) if t_timed else None})
        if step % 10 == 0 or step == steps - 1:
            print(f"SFT {tag(model)} {packed_variant or 'padded'} step {step} loss {loss_acc:.4f} gn {gnorm:.3f} "
                  f"{dt:.2f}s amax {amax}", flush=True)
    wall = time.time() - t_all
    mon.on = False
    mon.remove()
    steps = len(log)  # < requested if the in-task budget stopped the run
    losses = [x["loss"] for x in log]
    amaxes = [x["act_absmax"] for x in log if x["act_absmax"] is not None]
    finite_amax = [a for a in amaxes if a != float("inf")]
    top_keys = sorted(((v, k) for k, v in mon.run_max.items()), reverse=True)[:8]
    tps = tok_timed / t_timed if t_timed else None
    # what naive (random-order, un-grouped) padding would have cost for the same steps
    naive_pad = 0
    for s in range(len(log)):
        its = data[s * seq_per_step:(s + 1) * seq_per_step]
        for i in range(0, len(its), micro):
            mb = its[i:i + micro]
            naive_pad += max(len(x["ids"]) for x in mb) * len(mb)
    res = {"model": model, "mode": f"packed:{packed_variant}" if packed_variant else "padded_length_grouped",
           "oom_probe_fixed": True,
           "precision": DTYPE_NAME, "platform": os.environ.get("S2_PLATFORM", "kaggle-T4"),
           "loader": loader, "libs": lib_versions(), "gpu": gpu_info(),
           "n_params_total": n_params, "n_params_text": n_params_text, "grad_ckpt": gc,
           "lora_target_suffixes": suffixes, "lora_n_modules": len(names),
           "lora_trainable_params": sum(p.numel() for p in trainable), "lora_param_dtypes": lora_dtypes,
           "bos_added": has_bos, "items_used": len(data), "items_dropped_gt512": dropped,
           "mean_item_tokens": round(real_tot / max(1, len(data)), 1),
           "steps": steps, "seq_per_step": seq_per_step, "micro_batch": micro, "accum": accum, "oom_probe": oom_log,
           "lr": 2e-4, "load_s": round(t_load, 1), "train_wall_s": round(wall, 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "timed_steps": n_timed, "real_tokens_per_s": round(tps, 1) if tps else None,
           "completion_tokens_per_s": round(comp_timed / t_timed, 1) if t_timed else None,
           "s_per_step_timed": round(t_timed / max(1, n_timed), 3),
           "t4_hours_per_9M_tokens": round(9e6 / tps / 3600, 3) if tps else None,
           "real_tokens_total": real_tot, "padded_tokens_total": pad_tot,
           "pad_fraction": round(1 - real_tot / pad_tot, 4),
           "pad_fraction_naive_random_order": round(1 - real_tot / naive_pad, 4),
           "skipped_steps": skipped, "skipped_frac": round(skipped / steps, 4),
           "skipped_steps_after_10": sum(1 for x in log[10:] if x["skipped"]),
           "final_scale": scaler.get_scale(), "nonfinite_microbatch_losses": nonfinite,
           "loss_first5_mean": round(sum(losses[:5]) / len(losses[:5]), 4),
           "loss_last5_mean": round(sum(losses[-5:]) / len(losses[-5:]), 4),
           "loss_by_10": [round(sum(losses[i:i + 10]) / len(losses[i:i + 10]), 4) for i in range(0, steps, 10)],
           "loss_min": min(losses), "act_hooked_steps": len(amaxes),
           "act_absmax_max": max(finite_amax) if finite_amax else None,
           "act_any_nonfinite": len(finite_amax) != len(amaxes),
           "act_fp16_headroom_x": round(FP16_MAX / max(finite_amax), 2) if finite_amax else None,
           "act_top_keys": [[k, round(v, 1)] for v, k in top_keys], "layer_path": mon.layer_path,
           "stopped_early": stopped_early, "log": log}
    name = f"sft_packed_hf" if packed_variant else "sft"
    dump(f"{RES}/{tag(model)}/{name}.json", res)
    if os.environ.get("S2_SAVE_MERGED"):
        t = time.time()
        merged = m.merge_and_unload()
        merged.save_pretrained(os.environ["S2_SAVE_MERGED"], safe_serialization=True)
        tok.save_pretrained(os.environ["S2_SAVE_MERGED"])
        res["merged_saved_s"] = round(time.time() - t, 1)
        dump(f"{RES}/{tag(model)}/{name}.json", res)
    print("SFT_DONE", model, res["mode"], res["real_tokens_per_s"], res["loss_first5_mean"], res["loss_last5_mean"],
          res["skipped_steps"], res["act_absmax_max"], flush=True)


# =========================================================================== Unsloth (isolated site dir)
def task_unsloth(model, data_dir):
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    os.chdir("/tmp")  # unsloth_compiled_cache/ lands here, not in the kernel output
    out_path = f"{RES}/{tag(model)}/unsloth.json"
    res = load(out_path) or {"model": model, "stage": "import"}
    try:
        import unsloth  # noqa: F401  (must precede transformers/trl)
        from unsloth import FastLanguageModel
    except ImportError as e:
        name = (getattr(e, "name", None) or "").split(".")[0]
        mm = re.search(r"pip install ([A-Za-z0-9_.\-]+)", str(e))
        if mm:
            name = mm.group(1).strip("`'\".")
        if not name:
            raise
        res.setdefault("auto_installed", []).append(name)
        dump(out_path, res)
        if name and len(res["auto_installed"]) <= 8:
            p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", UNS_SITE,
                                name.replace("_", "-") if name != "cut_cross_entropy" else name],
                               capture_output=True, text=True)
            res.setdefault("auto_install_rc", []).append([name, p.returncode, (p.stdout + p.stderr)[-300:]])
            dump(out_path, res)
            sys.exit(7)  # lane retries
        raise
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    res["libs"] = lib_versions()
    res["stage"] = "check"
    dump(out_path, res)
    tok0 = None
    checks = {}
    for dname, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        try:
            mdl, tok = FastLanguageModel.from_pretrained(model_name=model, max_seq_length=1024, dtype=dtype,
                                                         load_in_4bit=False, load_in_8bit=False)
            tok0 = tok
            mdl = FastLanguageModel.get_peft_model(mdl, r=32, lora_alpha=32, lora_dropout=0, bias="none",
                                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                   "gate_proj", "up_proj", "down_proj"],
                                                   use_gradient_checkpointing=False, random_state=SEED)
            items, _, _ = load_train(data_dir, tok, 4, seed=1)
            ds = Dataset.from_list([{"prompt": TEMPLATE_Q.format(q=it["problem"]), "completion": it["completion"]}
                                    for it in items])
            cfg = SFTConfig(output_dir="/tmp/uns_chk", per_device_train_batch_size=4, gradient_accumulation_steps=1,
                            max_steps=1, learning_rate=0.0, padding_free=True, max_length=512, report_to="none",
                            fp16=(dtype == torch.float16), bf16=False, logging_steps=1, save_strategy="no",
                            seed=SEED, dataset_num_proc=1)
            tr = SFTTrainer(model=mdl, processing_class=tok, train_dataset=ds, args=cfg)
            batch = next(iter(tr.get_train_dataloader()))
            batch = tr._prepare_inputs(batch)
            c = {"param_dtype": str(next(mdl.parameters()).dtype),
                 "batch_keys": {k: list(v.shape) if torch.is_tensor(v) else str(v)[:80] for k, v in batch.items()}}
            FastLanguageModel.for_training(mdl)
            with torch.no_grad():
                loss, outputs = tr.compute_loss(mdl, batch, return_outputs=True)
                lg = getattr(outputs, "logits", None)
                c["packed_loss"] = float(loss)
                if lg is None or not torch.is_tensor(lg) or lg.numel() == 0:
                    c["error"] = "no logits returned on the packed path"
                else:
                    lg = lg.float()
                    c["packed_logits_shape"] = list(lg.shape)
                    ids = batch["input_ids"]
                    if ids.shape[0] != 1:
                        c["error"] = f"batch not padding-free (shape {list(ids.shape)})"
                    else:
                        if "position_ids" in batch:
                            pos = batch["position_ids"][0]
                            starts = (pos == 0).nonzero().flatten().tolist()
                        else:
                            starts = [0]
                        ends = starts[1:] + [ids.shape[1]]
                        per = []
                        for s, e in zip(starts, ends):
                            seg = ids[:, s:e]
                            sl = mdl(input_ids=seg).logits[0].float()
                            d = (sl - lg[0, s:e]).abs()
                            per.append({"len": e - s, "max_abs": round(d.max().item(), 6),
                                        "mean_abs": round(d.mean().item(), 7),
                                        "argmax_agree": round((sl.argmax(-1) == lg[0, s:e].argmax(-1)).float()
                                                              .mean().item(), 4)})
                        c["per_item"] = per
                        c["n_segments"] = len(per)
                        c["max_abs_all"] = max(x["max_abs"] for x in per)
                        c["pass_1e-2"] = c["max_abs_all"] < 1e-2
            checks[dname] = c
            del tr, mdl
            torch.cuda.empty_cache()
        except Exception:
            checks[dname] = {"error": traceback.format_exc()[-2500:]}
        res["checks"] = checks
        dump(out_path, res)
    fp32 = checks.get("fp32", {})
    real32 = fp32.get("param_dtype") == "torch.float32"
    accepted = bool(fp32.get("pass_1e-2")) and real32
    res["padding_free_accepted"] = accepted
    res["accept_rule"] = "fp32 Unsloth packed-vs-separate max |dlogit| < 1e-2 over all segments"
    dump(out_path, res)
    print("UNSLOTH CHECK", {k: v.get("max_abs_all", v.get("error", "")[-200:]) for k, v in checks.items()}, flush=True)

    # ---- 100-step Unsloth SFT throughput (fp16), padding-free on and off (same data, same 32 seq/step)
    steps = JOB.get("sft_steps", 100)
    res["sft"] = {}
    for pf in JOB.get("unsloth_sft_modes", [True, False]):
        key = "padding_free" if pf else "padded"
        try:
            mdl, tok = FastLanguageModel.from_pretrained(model_name=model, max_seq_length=1024, dtype=torch.float16,
                                                         load_in_4bit=False, load_in_8bit=False)
            mdl = FastLanguageModel.get_peft_model(mdl, r=32, lora_alpha=32, lora_dropout=0, bias="none",
                                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                   "gate_proj", "up_proj", "down_proj"],
                                                   use_gradient_checkpointing=False, random_state=SEED)
            items, _, _ = load_train(data_dir, tok, steps * 32)
            ntok = sum(len(it["ids"]) for it in items)
            ds = Dataset.from_list([{"prompt": TEMPLATE_Q.format(q=it["problem"]), "completion": it["completion"]}
                                    for it in items])
            cfg = SFTConfig(output_dir="/tmp/uns_sft", per_device_train_batch_size=8, gradient_accumulation_steps=4,
                            max_steps=steps, learning_rate=2e-4, warmup_steps=3,
                            lr_scheduler_type="constant_with_warmup", optim="adamw_torch", weight_decay=0.0,
                            max_grad_norm=1.0, padding_free=pf, max_length=512, report_to="none", fp16=True,
                            bf16=False, logging_steps=1, save_strategy="no", seed=SEED, dataset_num_proc=1)
            tr = SFTTrainer(model=mdl, processing_class=tok, train_dataset=ds, args=cfg)
            st = {"t": [], "skipped": 0}

            class CB(TrainerCallback):
                def on_step_end(self, args, state, control, **kw):
                    torch.cuda.synchronize()
                    st["t"].append(time.time())
                    if getattr(tr.accelerator, "optimizer_step_was_skipped", False):
                        st["skipped"] += 1

            tr.add_callback(CB())
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = tr.train()
            wall = time.time() - t0
            ts = st["t"]
            timed = (ts[-1] - ts[2]) if len(ts) > 3 else None
            tps = ntok * (len(ts) - 3) / len(ts) / timed if timed else None
            hist = [h for h in tr.state.log_history if "loss" in h]
            res["sft"][key] = {"padding_free": pf, "train_runtime_s": round(out.metrics.get("train_runtime", wall), 1),
                               "tokens_total": ntok, "real_tokens_per_s_steps3plus": round(tps, 1) if tps else None,
                               "real_tokens_per_s_overall": round(ntok / wall, 1),
                               "t4_hours_per_9M_tokens": round(9e6 / tps / 3600, 3) if tps else None,
                               "skipped_steps": st["skipped"],
                               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                               "loss_first5": [round(h["loss"], 4) for h in hist[:5]],
                               "loss_last5": [round(h["loss"], 4) for h in hist[-5:]],
                               "loss_by_10": [round(sum(h["loss"] for h in hist[i:i + 10]) / len(hist[i:i + 10]), 4)
                                              for i in range(0, len(hist), 10)],
                               "grad_norm_last5": [round(float(h.get("grad_norm", float("nan"))), 4)
                                                   for h in hist[-5:]]}
            del tr, mdl
            torch.cuda.empty_cache()
        except Exception:
            res["sft"][key] = {"error": traceback.format_exc()[-2500:]}
        dump(out_path, res)
    res["stage"] = "done"
    dump(out_path, res)
    print("UNSLOTH DONE", json.dumps({k: v.get("real_tokens_per_s_steps3plus") for k, v in res["sft"].items()}),
          flush=True)


# =========================================================================== env check
def task_envcheck():
    import torch
    out = {"libs": lib_versions(), "gpu": gpu_info()}
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32)
        m = Qwen3ForCausalLM(cfg).half().cuda()
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=8, target_modules="all-linear"))
        x = torch.randint(0, 1000, (2, 16), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = m(input_ids=x, labels=x).loss
        loss.backward()
        out["peft_lora_fp16_smoke"] = {"ok": True, "loss": float(loss),
                                       "lora_dtypes": sorted({str(p.dtype) for p in m.parameters()
                                                              if p.requires_grad})}
    except Exception:
        out["peft_lora_fp16_smoke"] = {"ok": False, "error": traceback.format_exc()[-2000:]}
    dump(f"{RES}/env_check.json", out)
    print("ENVCHECK", out["peft_lora_fp16_smoke"].get("ok"), flush=True)
    sys.exit(0 if out["peft_lora_fp16_smoke"]["ok"] else 4)


# =========================================================================== orchestration
T_TASK = time.time()


def run_task(task, model, data_dir, arg):
    try:
        if task == "eval_vllm":
            task_eval_vllm(model, data_dir, int(arg or 0))
        elif task == "eval_hf":
            task_eval_hf(model, data_dir)
        elif task == "packcheck":
            task_packcheck(model, data_dir)
        elif task == "sft":
            task_sft(model, data_dir)
        elif task == "sft_packed_hf":
            task_sft(model, data_dir, packed_variant=arg)
        elif task == "unsloth":
            task_unsloth(model, data_dir)
        elif task == "envcheck":
            task_envcheck()
        elif task == "eval_dev":
            task_eval_dev(model, data_dir, arg)
    except SystemExit:
        raise
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        name = task if task != "eval_vllm" else f"eval_vllm{arg or 0}"
        dump(f"{RES}/{tag(model)}/{name}.error.json", {"model": model, "task": task, "error": err[-6000:]})
        sys.exit(3)


T0 = time.time()


def sub(gpu, task, model, data_dir, arg="", extra_env=None, timeout=None, logname=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    env.update(extra_env or {})
    logf = f"{RES}/{tag(model)}/{logname or task}.log"
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    t = time.time()
    budget = timeout or JOB.get("task_timeout_s", 2400)
    left = JOB.get("deadline_s", 7800) + JOB.get("deadline_grace_s", 600) - (time.time() - T0)
    budget = max(60, min(budget, left))  # never run past the kernel deadline
    env["S2_TASK_BUDGET_S"] = str(int(budget))
    with open(logf, "a") as lf:
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--task", task, "--model", model,
                              "--data", data_dir, "--arg", str(arg)], env=env, stdout=lf,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            # kill the whole group: vLLM's EngineCore is a grandchild and would keep the GPU otherwise
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa
                p.kill()
            p.wait()
            rc = "timeout"
            dump(f"{RES}/{tag(model)}/{logname or task}.timeout.json",
                 {"model": model, "task": task, "budget_s": int(budget)})
    print(f"[lane {gpu}] +{(time.time()-T0)/60:.1f}m {tag(model)} {task}{(' ' + str(arg)) if arg else ''} rc={rc} "
          f"{time.time()-t:.0f}s", flush=True)
    with open(f"{RES}/tasks.jsonl", "a") as f:  # incremental task ledger; survives a killed kernel
        f.write(json.dumps({"lane": gpu, "model": model, "task": task, "arg": str(arg), "rc": rc,
                            "start_min": round((t - T0) / 60, 2), "s": round(time.time() - t, 1)}) + "\n")
    return rc


def skip(model, task, why):
    dump(f"{RES}/{tag(model)}/{task}.skipped.json", {"model": model, "task": task, "why": why})
    print(f"SKIP {tag(model)} {task}: {why}", flush=True)


def clear_hf_cache(model):
    d = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}")
    shutil.rmtree(d, ignore_errors=True)


def lane(gpu, plan, data_dir):
    for entry in plan:
        model, tasks = entry[0], entry[1]
        opts = entry[2] if len(entry) > 2 else {}
        d = f"{RES}/{tag(model)}"
        os.makedirs(d, exist_ok=True)
        access_err = hf_access(model)
        if access_err:
            for t in tasks:
                skip(model, t, f"no HF access: {access_err}")
            continue
        for task in tasks:
            if time.time() - T0 > JOB.get("deadline_s", 7800):
                skip(model, task, "kernel deadline reached")
                continue
            if task == "eval":
                n_att = len(JOB.get("vllm_attempts", {}).get(tag(model)) or [{}]) if JOB.get("use_vllm", True) else 0
                for a in range(n_att):
                    sub(gpu, "eval_vllm", model, data_dir, a, logname=f"eval_vllm{a}",
                        timeout=JOB.get("vllm_timeout_s", 1200))
                    if os.path.exists(f"{d}/eval.json"):
                        break
                if not os.path.exists(f"{d}/eval.json"):
                    sub(gpu, "eval_hf", model, data_dir, timeout=JOB.get("hf_eval_timeout_s", 2400))
            elif task in ("sft",):
                thr = opts.get("sft_if_acc")
                if thr is not None:
                    ev = load(f"{d}/eval.json")
                    if not ev or ev.get("accuracy", 0) < thr:
                        skip(model, task, f"4-shot accuracy {ev and ev.get('accuracy')} < {thr} (or eval failed)")
                        continue
                sub(gpu, "sft", model, data_dir)
            elif task == "sft_packed_hf":
                pc = load(f"{d}/packcheck.json") or {}
                ok = [v for v in ("pos_ids_no_cache", "block_mask_4d") if v in pc.get("fp32_passing_variants", [])]
                if not ok:
                    skip(model, task, "no packing variant passed the fp32 logits check")
                    continue
                sub(gpu, "sft_packed_hf", model, data_dir, ok[0])
            elif task == "unsloth":
                t = time.time()
                while not os.path.exists(UNS_MARK) and time.time() - t < 900:
                    time.sleep(10)
                for attempt in range(9):
                    rc = sub(gpu, "unsloth", model, data_dir, extra_env={
                        "PYTHONPATH": UNS_SITE + os.pathsep + os.environ.get("PYTHONPATH", ""),
                        "UNSLOTH_RETURN_LOGITS": "1"}, timeout=JOB.get("unsloth_timeout_s", 2400))
                    if rc != 7:
                        break
            else:
                sub(gpu, task, model, data_dir)
        if opts.get("cleanup"):
            clear_hf_cache(model)


def envcheck_block():
    """GPU, driver, python and key library versions (package metadata only; no heavy imports)."""
    smi = subprocess.run("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout.strip().splitlines()
    v = lib_versions()
    v.pop("torch_full", None)
    tc = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__, torch.version.cuda)"],
                        capture_output=True, text=True).stdout.strip()
    return {"gpus": smi, "python": sys.version.split()[0], "versions": v, "torch_cuda": tc}


def main():
    os.makedirs(RES, exist_ok=True)
    meta = {"job": JOB, "t0": T0}
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout
    meta["nvidia_smi"] = smi
    if "T4" not in smi:
        meta["fatal"] = "expected Tesla T4"
        dump(f"{RES}/meta.json", meta)
        sys.exit(2)
    # Unsloth into an isolated target dir, in the background (never touches the main site-packages)
    uns_cmd = JOB.get("unsloth_install")
    uns_proc = None
    if uns_cmd:
        uns_proc = subprocess.Popen(f"({uns_cmd}) > /tmp/uns_install.log 2>&1; touch {UNS_MARK}", shell=True)
    if JOB.get("install"):
        t = time.time()
        p = subprocess.run(JOB["install"], shell=True, capture_output=True, text=True)
        meta["install"] = {"cmd": JOB["install"], "rc": p.returncode, "s": round(time.time() - t, 1),
                           "tail": (p.stdout + p.stderr)[-1500:]}
    # ---- the W2 v1 blocker: peft refuses torchao<0.16 -> remove torchao (nothing in this stack needs it)
    fixes = []
    for cmd in JOB.get("env_fix", ["pip uninstall -y -q torchao"]):
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        fixes.append({"cmd": cmd, "rc": p.returncode, "tail": (p.stdout + p.stderr)[-500:]})
    meta["env_fix"] = fixes
    dump(f"{RES}/meta.json", meta)
    rc = sub(0, "envcheck", "env/check", "", timeout=600)
    if rc != 0:
        p = subprocess.run("pip install -q -U 'torchao>=0.16'", shell=True, capture_output=True, text=True)
        meta["env_fix"].append({"cmd": "pip install -U torchao>=0.16 (fallback)", "rc": p.returncode,
                                "tail": (p.stdout + p.stderr)[-500:]})
        rc = sub(0, "envcheck", "env/check", "", timeout=600)
    meta["envcheck_rc"] = rc
    meta["pip"] = subprocess.run("pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|trl|vllm|triton|torchao|"
                                 "accelerate|bitsandbytes|xformers|datasets|causal-conv1d|flash-linear-attention) '",
                                 shell=True, capture_output=True, text=True).stdout
    meta["envcheck"] = envcheck_block()
    print("ENVCHECK", json.dumps(meta["envcheck"]), flush=True)
    dump(f"{RES}/meta.json", meta)
    bad = []
    if rc != 0:
        bad.append("peft LoRA fp16 smoke test failed after torchao removal and torchao upgrade")
    if meta.get("install", {}).get("rc", 0) != 0:
        bad.append(f"install failed rc={meta['install']['rc']}: {meta['install']['tail'][-300:]}")
    if JOB.get("python") and not sys.version.startswith(JOB["python"]):
        bad.append(f"python {sys.version.split()[0]} != expected {JOB['python']} (Kaggle image changed)")
    for pkg, want in (JOB.get("pins") or {}).items():
        got = meta["envcheck"]["versions"].get(pkg)
        if got is None or not str(got).startswith(want):
            bad.append(f"version pin violated: {pkg} want {want} got {got}")
    if bad:
        meta["fatal"] = bad
        dump(f"{RES}/meta.json", meta)
        print("FATAL (fail fast, no GPU tasks run):\n  " + "\n  ".join(bad), flush=True)
        if JOB.get("fail_fast", True):
            sys.exit(5)
    data_dir = find_data()
    import threading
    th = [threading.Thread(target=lane, args=(g, plan, data_dir)) for g, plan in enumerate(JOB["lanes"])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if uns_proc is not None:
        try:
            meta["unsloth_install"] = {"cmd": uns_cmd, "rc": uns_proc.wait(timeout=5),
                                       "tail": open("/tmp/uns_install.log").read()[-1500:]}
        except Exception as e:  # noqa
            meta["unsloth_install"] = {"cmd": uns_cmd, "error": repr(e)}
        meta["unsloth_site_pkgs"] = sorted(os.listdir(UNS_SITE))[:200] if os.path.isdir(UNS_SITE) else None
    meta["wall_s"] = round(time.time() - T0, 1)
    dump(f"{RES}/meta.json", meta)
    print("ALL DONE", meta["wall_s"], flush=True)


if __name__ == "__main__":
    if "--task" in sys.argv:
        a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        if a["--task"] == "unsloth":
            sys.path.insert(0, UNS_SITE)
        run_task(a["--task"], a["--model"], a["--data"], a.get("--arg", ""))
    else:
        main()
