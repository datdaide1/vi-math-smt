"""O1 SFT micro-benchmark worker: one variant per process (isolation from OOM / compile failures).

python sft_worker.py <variant> <model> <dtype> <out.json>

Variants (all LoRA r=32 a=32 all-linear, fp32 adapters, 32 seq/step, identical items per micro-batch):
  pad_rand        padded, random order, micro-batch 8, SDPA, grad-ckpt on
  pad_group       padded, sorted by length inside each 32-seq step (Y1 default), mb 8, SDPA, ckpt on
  pad_group_mb16  as pad_group, mb 16
  pad_group_nock  as pad_group, grad-ckpt off
  pack_sdpa       padding-free: the same 8 rows concatenated, position_ids reset, use_cache=False, no attention_mask
  pack_flex       as pack_sdpa with attn_implementation=flex_attention
  pack_fa2        as pack_sdpa with attn_implementation=kernels-community/flash-attn2 (sm>=80 only)
  unsloth_pad     Unsloth FastLanguageModel, padded grouped, mb 8, Unsloth grad-ckpt
  unsloth_pack    Unsloth, packed rows with position_ids (+ packed_seq_lengths if supported)
Every packed variant first runs the logits-equality check (4 items: separate vs packed).
"""
import json
import math
import os
import random
import sys
import time
import traceback

V, MODEL, DT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DATA = os.environ.get("O1_DATA", "/data")
STEPS = int(os.environ.get("O1_STEPS", "7"))  # first step = warm-up (not timed)
SEQ_PER_STEP, MAXLEN = 32, 1024
res = {"variant": V, "model": MODEL, "dtype": DT}


def dump():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


try:
    import torch

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[DT]
    res["gpu"] = torch.cuda.get_device_name(0)
    unsloth = V.startswith("unsloth")
    if unsloth:
        import unsloth  # noqa: F401  (must be imported before transformers)
        from unsloth import FastLanguageModel
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    from peft import LoraConfig, get_peft_model

    res["libs"] = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    if unsloth:
        res["libs"]["unsloth"] = getattr(sys.modules["unsloth"], "__version__", "?")

    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = tok.eos_token_id
    items = []
    for line in open(f"{DATA}/train_mix.jsonl", encoding="utf-8"):
        r = json.loads(line)
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
        ids = p + c
        if len(ids) > MAXLEN:
            continue
        items.append({"ids": ids, "labels": [-100] * len(p) + c})
    lens = sorted(len(x["ids"]) for x in items)
    res["data"] = {"n": len(items), "mean_len": round(sum(lens) / len(lens), 1), "p50": lens[len(lens) // 2],
                   "p99": lens[int(0.99 * len(lens))], "max": lens[-1]}

    import re as _re
    _m = _re.search(r"_mb(\d+)$", V)
    VB = _re.sub(r"_mb\d+$", "", V)
    attn = {"pack_flex": "flex_attention", "pack_fa2": "kernels-community/flash-attn2"}.get(VB, "sdpa")
    ckpt = VB != "pad_group_nock"
    mb = int(_m.group(1)) if _m else 8
    packed = VB.startswith("pack") or VB == "unsloth_pack"
    group = V != "pad_rand"
    res.update(attn=attn, grad_ckpt=ckpt, micro_batch=mb, packed=packed, length_group=group)

    t0 = time.time()
    if unsloth:
        model, _ = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAXLEN * 8, dtype=dtype,
                                                     load_in_4bit=False, load_in_8bit=False, full_finetuning=False)
        model = FastLanguageModel.get_peft_model(
            model, r=32, lora_alpha=32, lora_dropout=0.0, bias="none", random_state=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation=attn).to("cuda")
        model.config.use_cache = False
        torch.manual_seed(0)
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    res["load_s"] = round(time.time() - t0, 1)

    def batch(rows):
        if packed:
            ids, lab, pos = [], [], []
            for r in rows:
                ids += r["ids"]
                lab += [-100] + r["labels"][1:]  # never predict across a boundary
                pos += list(range(len(r["ids"])))
            b = {"input_ids": torch.tensor([ids]), "position_ids": torch.tensor([pos]), "labels": torch.tensor([lab])}
            if V == "unsloth_pack":
                b["packed_seq_lengths"] = torch.tensor([len(r["ids"]) for r in rows], dtype=torch.int32)
            return {k: v.cuda() for k, v in b.items()}, len(ids)
        L = max(len(r["ids"]) for r in rows)
        pid = tok.pad_token_id if tok.pad_token_id is not None else eos
        ids = torch.tensor([r["ids"] + [pid] * (L - len(r["ids"])) for r in rows])
        am = torch.tensor([[1] * len(r["ids"]) + [0] * (L - len(r["ids"])) for r in rows])
        lab = torch.tensor([r["labels"] + [-100] * (L - len(r["labels"])) for r in rows])
        return {"input_ids": ids.cuda(), "attention_mask": am.cuda(), "labels": lab.cuda()}, int(am.sum())

    def fwd(b):
        kw = dict(b)
        kw["use_cache"] = False
        try:
            return model(**kw)
        except TypeError:
            kw.pop("packed_seq_lengths", None)
            res["packed_seq_lengths_unsupported"] = True
            return model(**kw)

    # ---------------- logits-equality check (packed vs separate), eval mode, no grad
    if packed:
        model.eval()
        rng = random.Random(1)
        four = rng.sample(items, 4)
        chk = []
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            sep = [model(input_ids=torch.tensor([r["ids"]]).cuda(), use_cache=False).logits[0].float() for r in four]
            b, _ = batch(four)
            b.pop("labels")
            pk = fwd(b).logits[0].float()
            o = 0
            for k, r in enumerate(four):
                L = len(r["ids"])
                a, p = sep[k], pk[o:o + L]
                o += L
                lab = torch.tensor(r["labels"][1:], device="cuda")
                m = lab != -100
                nll_a = torch.nn.functional.cross_entropy(a[:-1][m], lab[m]).item()
                nll_p = torch.nn.functional.cross_entropy(p[:-1][m], lab[m]).item()
                chk.append({"len": L, "max_abs": round((a - p).abs().max().item(), 4),
                            "mean_abs": round((a - p).abs().mean().item(), 5), "nll_sep": round(nll_a, 5),
                            "nll_packed": round(nll_p, 5),
                            "argmax_agree": round((a.argmax(-1) == p.argmax(-1)).float().mean().item(), 4)})
        res["logit_check"] = chk
        res["logit_check_max_nll_diff"] = max(abs(c["nll_sep"] - c["nll_packed"]) for c in chk)
        res["packing_correct"] = res["logit_check_max_nll_diff"] < 0.02 and min(c["argmax_agree"] for c in chk) > 0.97
        dump()
        model.train()

    # ---------------- throughput
    rng = random.Random(0)
    order = items[:]
    rng.shuffle(order)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(DT == "fp16"))
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times, toks, losses, padded_toks = [], [], [], []
    for s in range(STEPS):
        step = order[s * SEQ_PER_STEP:(s + 1) * SEQ_PER_STEP]
        if group:
            step = sorted(step, key=lambda r: len(r["ids"]))
        torch.cuda.synchronize()
        t = time.time()
        nt = 0
        npad = 0
        for i in range(0, SEQ_PER_STEP, mb):
            b, n = batch(step[i:i + mb])
            nt += n
            npad += b["input_ids"].numel()
            with torch.autocast("cuda", dtype=dtype):
                loss = fwd(b).loss
            scaler.scale(loss * (len(step[i:i + mb]) / SEQ_PER_STEP)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        toks.append(nt)
        padded_toks.append(npad)
        losses.append(round(loss.item(), 4))
        res["progress_steps"] = s + 1
    tt, tk = sum(times[1:]), sum(toks[1:])
    res.update(step_s=[round(x, 2) for x in times], tok_per_s=round(tk / tt, 1),
               pad_frac=round(1 - tk / sum(padded_toks[1:]), 4), losses=losses,
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               s_per_step=round(tt / (STEPS - 1), 2), ok=True)
except Exception as e:  # noqa
    res.update(ok=False, error=repr(e)[:1500], tb=traceback.format_exc()[-3000:])
dump()
print("RESULT", json.dumps({k: res.get(k) for k in ("variant", "ok", "tok_per_s", "peak_mem_gb", "packing_correct",
                                                    "logit_check_max_nll_diff", "error")}), flush=True)
