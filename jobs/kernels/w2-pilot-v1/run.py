import json as _json
JOB = _json.loads('{"lanes": [["google/gemma-4-E2B", "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3-0.6B-Base"], ["Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3-1.7B-Base"]], "tasks": ["eval", "packcheck", "sft"], "install": "pip install -q \'vllm==0.33.0\' \'flash-linear-attention==0.6.0\'", "sft_steps": 50, "eval_n": 300, "use_vllm": true, "hf_eval_bs": 16, "task_timeout_s": 2700, "kernel_timeout_s": 10800}')
"""W2 day-1 student pilot — runs ON a private Kaggle T4x2 kernel.

The laptop prepends one line `JOB = {...}` (see submit_pilot.py). Without it a default JOB is used.

Per model, each task runs in its own subprocess pinned to one GPU (2 lanes = 2 T4s):
  eval      4-shot Vi-GSM8K on the fixed 300-item subset (vLLM fp16 if it works on T4, else HF generate),
            scored with the frozen math_verify.py snapshot. Records generated tokens/s.
  sft       50-step fp16 LoRA SFT smoke (r=32, a=32, all-linear text decoder, fp32 adapters, AdamW fp32,
            LR 2e-4, 32 seq/step = micro-batch 4 x accum 8, completion-only token-mean NLL, GradScaler,
            grad-clip 1.0). packing=off: padded rows; packing=on: the SAME 4 items per micro-batch
            concatenated into one padding-free row (position_ids reset, no attention_mask). Same data order,
            so the two loss curves should match if packing is exact. Records real tokens/s, skipped
            (overflow) steps, max |activation| per decoder layer (fp16 overflow check).
  packcheck G2: per-token logits of 4 items run separately vs packed in one row (state-leak check).

Outputs: /kaggle/working/results/<model_tag>/{eval.json,preds.jsonl,sft_off.json,sft_on.json,packcheck.json}
"""
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback

try:
    JOB  # noqa: F821  (injected by submit_pilot.py)
except NameError:
    JOB = {"lanes": [["Qwen/Qwen3-0.6B-Base"], ["Qwen/Qwen3.5-0.8B-Base"]], "tasks": ["eval", "packcheck", "sft"],
           "install": "pip install -q 'vllm==0.33.0' peft flash-linear-attention", "sft_steps": 50,
           "eval_n": 300}

WORK = "/kaggle/working"
RES = f"{WORK}/results"
SEED = 20261107
TEMPLATE_Q = "Bài toán:\n{q}\n\nLời giải:\n"


def tag(m):
    return m.split("/")[-1]


def find_data():
    hits = glob.glob("/kaggle/input/**/test300.jsonl", recursive=True)
    if not hits:
        raise FileNotFoundError("test300.jsonl not mounted")
    return os.path.dirname(hits[0])


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def clean_solution(sol, gt):
    s = re.sub(r"<<[^<>]*>>", "", sol).rsplit("####", 1)[0].rstrip()
    return f"{s}\nĐáp số: \\boxed{{{gt}}}"


def gpu_info():
    import torch
    return {"name": torch.cuda.get_device_name(0), "cap": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def lib_versions():
    out = {}
    for mod in ("torch", "transformers", "peft", "vllm", "fla", "triton", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:  # noqa
            out[mod] = f"n/a ({type(e).__name__})"
    return out


# =========================================================================== eval
def build_prompts(data_dir, n):
    shots = json.load(open(f"{data_dir}/fewshot4.json", encoding="utf-8"))
    prefix = "\n\n".join(f"Bài toán:\n{s['problem']}\n\nLời giải:\n{s['solution']}" for s in shots) + "\n\n"
    items = [json.loads(l) for l in open(f"{data_dir}/test300.jsonl", encoding="utf-8")][:n]
    return items, [prefix + TEMPLATE_Q.format(q=it["question"]) for it in items]


STOPS = ["\n\nBài toán", "Bài toán:"]


def cut_stop(text):
    for s in STOPS:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text


def task_eval(model, data_dir):
    sys.path.insert(0, data_dir)
    from math_verify import extract_answer, is_correct  # frozen snapshot shipped in the dataset
    items, prompts = build_prompts(data_dir, JOB.get("eval_n", 300))
    res = {"model": model, "n": len(items), "gpu": gpu_info(), "libs": lib_versions(), "decoding":
           {"temperature": 0, "max_tokens": 400, "stop": STOPS, "shots": 4}}
    texts, ntoks, engine, err = None, None, None, None
    if JOB.get("use_vllm", True):
        try:
            from vllm import LLM, SamplingParams
            t = time.time()
            kw = dict(model=model, dtype="float16", max_model_len=3072, gpu_memory_utilization=0.85, seed=0)
            try:
                llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
            except Exception as e1:  # noqa
                res["vllm_mm_limit_error"] = repr(e1)[-500:]
                llm = LLM(**kw)
            res["load_s"] = round(time.time() - t, 1)
            sp = SamplingParams(temperature=0, max_tokens=400, stop=STOPS, seed=0)
            t = time.time()
            outs = llm.generate(prompts, sp)
            res["gen_s"] = round(time.time() - t, 2)
            texts = [o.outputs[0].text for o in outs]
            ntoks = [len(o.outputs[0].token_ids) for o in outs]
            res["prompt_tokens_mean"] = round(sum(len(o.prompt_token_ids) for o in outs) / len(outs), 1)
            engine = "vllm"
            try:
                res["vllm_attn_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
            except Exception:
                pass
        except Exception:
            err = traceback.format_exc()[-3000:]
            res["vllm_error"] = err
    if texts is None:
        import torch
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = load_model(model, torch.float16)
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
        engine = "hf_generate"
        res["hf_eval_bs"] = bs
    res["engine"] = engine
    correct, preds = 0, []
    for it, tx, nt in zip(items, texts, ntoks):
        body = cut_stop(tx)
        pred = extract_answer(body)
        ok = bool(is_correct(pred, it["ground_truth"])) if pred is not None else False
        correct += ok
        preds.append({"test_index": it["test_index"], "gold": it["ground_truth"], "pred": pred, "correct": ok,
                      "gen_tokens": nt, "boxed": "\\boxed" in body, "output": body})
    n = len(items)
    acc = correct / n
    res.update({"correct": correct, "accuracy": round(acc, 4),
                "ci95_wilson": wilson(correct, n), "gen_tokens_total": sum(ntoks),
                "gen_tok_per_s": round(sum(ntoks) / res["gen_s"], 1),
                "boxed_rate": round(sum(p["boxed"] for p in preds) / n, 3),
                "hit_max_tokens": sum(1 for x in ntoks if x >= 400)})
    out_dir = f"{RES}/{tag(model)}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/preds.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dump(f"{out_dir}/eval.json", res)
    print("EVAL", model, engine, res["accuracy"], res["gen_tok_per_s"], flush=True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


# =========================================================================== model loading
def load_model(model, dtype):
    import torch
    import transformers as T
    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        try:
            m = getattr(T, cls).from_pretrained(model, dtype=dtype, attn_implementation="sdpa").to("cuda")
            m._w2_loader = cls
            return m
        except Exception as e:  # noqa
            last = e
    raise last


def text_linear_targets(m):
    import torch.nn as nn
    skip = re.compile(r"(vision|visual|audio|multi_modal|embed_vision|embed_audio|lm_head|mtp|patch_embed|merger)",
                      re.I)
    names, suffixes = [], set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and not skip.search(n):
            names.append(n)
            suffixes.add(n.split(".")[-1])
    return names, sorted(suffixes)


def decoder_layers(m):
    best = None
    for n, mod in m.named_modules():
        if n.endswith("layers") and hasattr(mod, "__len__") and len(mod) > 4 and not re.search(
                r"vision|visual|audio", n, re.I):
            if best is None or len(mod) > len(best[1]):
                best = (n, mod)
    return best


def gdn_path(m):
    """Which gated-delta-rule implementation the Qwen3.5 linear-attention layers use."""
    for n, mod in m.named_modules():
        f = getattr(mod, "chunk_gated_delta_rule", None)
        if f is not None:
            return {"module": n, "impl": f"{getattr(f, '__module__', '?')}.{getattr(f, '__name__', '?')}",
                    "causal_conv1d_fn": str(getattr(mod, "causal_conv1d_fn", None))[:120]}
    return None


# =========================================================================== data for SFT / packcheck
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
        out.append({"ids": ids, "labels": [-100] * (len(bos) + len(p)) + c})
        if len(out) >= n_items:
            break
    return out, dropped, bool(bos)


def make_batch(items, packing, pad_id):
    import torch
    if packing:
        ids, labels, pos = [], [], []
        for it in items:
            ids += it["ids"]
            labels += [-100] + it["labels"][1:]  # never predict across a boundary
            pos += list(range(len(it["ids"])))
        return {"input_ids": torch.tensor([ids]), "position_ids": torch.tensor([pos])}, torch.tensor([labels]), len(ids)
    L = max(len(it["ids"]) for it in items)
    ids = [it["ids"] + [pad_id] * (L - len(it["ids"])) for it in items]
    lab = [it["labels"] + [-100] * (L - len(it["labels"])) for it in items]
    att = [[1] * len(it["ids"]) + [0] * (L - len(it["ids"])) for it in items]
    return ({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att)}, torch.tensor(lab),
            sum(len(it["ids"]) for it in items))


def fwd(m, inp):
    """Forward with the extra inputs some multimodal wrappers demand during training (Gemma: token_type_ids)."""
    import torch
    kw = {k: v.to("cuda") for k, v in inp.items()}
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
    lg = logits[:, :-1][mask].float()  # upcast only the completion positions (fp32 CE)
    return F.cross_entropy(lg, lb[mask], reduction="sum")


# =========================================================================== packcheck (G2)
def task_packcheck(model, data_dir):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    m = load_model(model, torch.float16)
    m.eval()
    items, _, has_bos = load_train(data_dir, tok, 4, seed=1)
    res = {"model": model, "loader": m._w2_loader, "gdn": gdn_path(m), "libs": lib_versions(), "per_item": []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        sep = []
        for it in items:
            out = fwd(m, {"input_ids": torch.tensor([it["ids"]])})
            sep.append(out.logits[0].float())
        b, _, _ = make_batch(items, True, 0)
        packed = fwd(m, b).logits[0].float()
        # same 4 items, padded batch (right padding) -> padding sanity
        b2, _, _ = make_batch(items, False, tok.pad_token_id if tok.pad_token_id is not None else 0)
        padded = fwd(m, b2).logits.float()
    off = 0
    for k, it in enumerate(items):
        L = len(it["ids"])
        a, p, q = sep[k], packed[off:off + L], padded[k, :L]
        lab = torch.tensor([it["labels"]], device="cuda")
        res["per_item"].append({
            "len": L, "packed_vs_sep_max_abs_logit_diff": round((a - p).abs().max().item(), 5),
            "packed_vs_sep_mean_abs_logit_diff": round((a - p).abs().mean().item(), 6),
            "padded_vs_sep_max_abs_logit_diff": round((a - q).abs().max().item(), 5),
            "nll_sep": round(nll_sum(a[None], lab).item() / max(1, (lab != -100).sum().item() - 1), 5),
            "nll_packed": round(nll_sum(p[None], lab).item() / max(1, (lab != -100).sum().item() - 1), 5),
            "argmax_agree_packed": round((a.argmax(-1) == p.argmax(-1)).float().mean().item(), 4)})
        off += L
    later = res["per_item"][1:]
    res["max_diff_items_2_4"] = max(x["packed_vs_sep_max_abs_logit_diff"] for x in later)
    res["leak_suspected"] = res["max_diff_items_2_4"] > 0.05
    dump(f"{RES}/{tag(model)}/packcheck.json", res)
    print("PACKCHECK", model, res["max_diff_items_2_4"], flush=True)


# =========================================================================== SFT smoke
def task_sft(model, data_dir, packing):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    random.seed(SEED)
    steps, micro, accum = JOB.get("sft_steps", 50), 4, 8
    tok = AutoTokenizer.from_pretrained(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    data, dropped, has_bos = load_train(data_dir, tok, steps * micro * accum)
    t_load = time.time()
    m = load_model(model, torch.float16)
    t_load = time.time() - t_load
    names, suffixes = text_linear_targets(m)
    n_params = sum(p.numel() for p in m.parameters())
    gc = n_params > 1.2e9 or JOB.get("force_gc", False)
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
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    # overflow monitor on decoder layers
    layer_max = {}
    dl = decoder_layers(m)
    hooks = []
    if dl:
        def mk(i):
            def h(_m, _inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if torch.is_tensor(o):
                    v = o.detach().abs().max().item()
                    layer_max[i] = max(layer_max.get(i, 0.0), v if math.isfinite(v) else float("inf"))
            return h
        for i, layer in enumerate(dl[1]):
            hooks.append(layer.register_forward_hook(mk(i)))
    log, skipped, nonfinite_loss = [], 0, 0
    torch.cuda.reset_peak_memory_stats()
    real_tok_total, pad_tok_total, t_timed, tok_timed = 0, 0, 0.0, 0
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.time()
        batch_items = data[step * micro * accum:(step + 1) * micro * accum]
        n_comp = sum(sum(1 for x in it["labels"][1:] if x != -100) for it in batch_items)
        loss_acc, step_tokens, step_padded = 0.0, 0, 0
        layer_max.clear()
        for a in range(accum):
            mb = batch_items[a * micro:(a + 1) * micro]
            inp, lab, ntok = make_batch(mb, packing, pad_id)
            step_tokens += ntok
            step_padded += inp["input_ids"].numel()
            with torch.autocast("cuda", dtype=torch.float16):
                out = fwd(m, inp)
            loss = nll_sum(out.logits, lab) / n_comp
            if not torch.isfinite(loss):
                nonfinite_loss += 1
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
        s_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        was_skipped = scaler.get_scale() < s_before
        skipped += int(was_skipped)
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        real_tok_total += step_tokens
        pad_tok_total += step_padded
        if step >= 3:
            t_timed += dt
            tok_timed += step_tokens
        amax = max(layer_max.values()) if layer_max else None
        log.append({"step": step, "loss": round(loss_acc, 5), "gnorm": round(gnorm, 4) if math.isfinite(gnorm) else
                    str(gnorm), "scale": s_before, "skipped": was_skipped, "s": round(dt, 3), "tokens": step_tokens,
                    "padded_tokens": step_padded, "act_absmax": amax,
                    "act_absmax_layer": (max(layer_max, key=layer_max.get) if layer_max else None)})
        if step % 10 == 0 or step == steps - 1:
            print(f"SFT {tag(model)} pack={packing} step {step} loss {loss_acc:.4f} gn {gnorm:.3f} {dt:.2f}s", flush=True)
    for h in hooks:
        h.remove()
    losses = [x["loss"] for x in log]
    amaxes = [x["act_absmax"] for x in log if x["act_absmax"] is not None]
    res = {"model": model, "packing": packing, "loader": m.base_model.model._w2_loader if hasattr(
        m.base_model.model, "_w2_loader") else None, "libs": lib_versions(), "gpu": gpu_info(),
        "n_params_total": n_params, "grad_ckpt": gc, "lora_target_suffixes": suffixes,
        "lora_trainable_params": sum(p.numel() for p in trainable), "lora_param_dtypes": lora_dtypes,
        "bos_added": has_bos, "items_used": len(data), "items_dropped_gt512": dropped,
        "steps": steps, "seq_per_step": micro * accum, "micro_batch": micro, "lr": 2e-4,
        "load_s": round(t_load, 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "real_tokens_per_s_steps3plus": round(tok_timed / t_timed, 1) if t_timed else None,
        "s_per_step_steps3plus": round(t_timed / max(1, steps - 3), 3),
        "real_tokens_total": real_tok_total, "padded_tokens_total": pad_tok_total,
        "pad_fraction": round(1 - real_tok_total / pad_tok_total, 4),
        "skipped_steps": skipped, "skipped_frac": round(skipped / steps, 4), "nonfinite_microbatch_losses": nonfinite_loss,
        "loss_first5_mean": round(sum(losses[:5]) / 5, 4), "loss_last5_mean": round(sum(losses[-5:]) / 5, 4),
        "loss_min": min(losses), "act_absmax_max": max(amaxes) if amaxes else None,
        "act_over_fp16_max": any((a == float("inf") or a > 65504) for a in amaxes),
        "gdn": gdn_path(m), "log": log}
    dump(f"{RES}/{tag(model)}/sft_{'on' if packing else 'off'}.json", res)
    print("SFT_DONE", model, packing, res["real_tokens_per_s_steps3plus"], res["loss_first5_mean"],
          res["loss_last5_mean"], flush=True)


# =========================================================================== orchestration
def run_task(task, model, data_dir, packing=None):
    try:
        if task == "eval":
            task_eval(model, data_dir)
        elif task == "packcheck":
            task_packcheck(model, data_dir)
        elif task == "sft":
            task_sft(model, data_dir, packing)
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        name = task if packing is None else f"sft_{'on' if packing else 'off'}"
        dump(f"{RES}/{tag(model)}/{name}.error.json", {"model": model, "task": task, "error": err[-6000:]})
        sys.exit(3)


def lane(gpu, models, data_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false")
    for model in models:
        for task in JOB["tasks"]:
            variants = [("off", "0"), ("on", "1")] if task == "sft" else [(None, "")]
            for name, flag in variants:
                logf = f"{RES}/{tag(model)}/{task}{'_' + name if name else ''}.log"
                os.makedirs(os.path.dirname(logf), exist_ok=True)
                t = time.time()
                with open(logf, "w") as lf:
                    try:
                        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--task", task, "--model",
                                              model, "--data", data_dir, "--packing", flag], env=env, stdout=lf,
                                             stderr=subprocess.STDOUT, timeout=JOB.get("task_timeout_s", 2700))
                    except subprocess.TimeoutExpired:
                        rc = "timeout"
                print(f"[lane {gpu}] {model} {task} {name or ''} rc={rc} {time.time()-t:.0f}s", flush=True)


def main():
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    meta = {"job": JOB, "t0": t0}
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                         shell=True, capture_output=True, text=True).stdout
    meta["nvidia_smi"] = smi
    if "T4" not in smi:
        meta["fatal"] = "expected Tesla T4"
        dump(f"{RES}/meta.json", meta)
        sys.exit(2)
    if JOB.get("install"):
        t = time.time()
        p = subprocess.run(JOB["install"], shell=True, capture_output=True, text=True)
        meta["install"] = {"cmd": JOB["install"], "rc": p.returncode, "s": round(time.time() - t, 1),
                           "tail": (p.stdout + p.stderr)[-2000:]}
    meta["pip"] = subprocess.run("pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|vllm|triton|"
                                 "flash-linear-attention|fla-core|causal-conv1d|accelerate) '", shell=True,
                                 capture_output=True, text=True).stdout
    dump(f"{RES}/meta.json", meta)
    data_dir = find_data()
    import threading
    th = [threading.Thread(target=lane, args=(g, ms, data_dir)) for g, ms in enumerate(JOB["lanes"])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    meta["wall_s"] = round(time.time() - t0, 1)
    dump(f"{RES}/meta.json", meta)
    print("ALL DONE", meta["wall_s"], flush=True)


if __name__ == "__main__":
    if "--task" in sys.argv:
        a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
        run_task(a["--task"], a["--model"], a["--data"], packing=(a.get("--packing") == "1")
                 if a["--task"] == "sft" else None)
    else:
        main()
