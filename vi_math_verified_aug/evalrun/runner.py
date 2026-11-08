"""Evaluation runner: generate (vLLM or HF), grade with ``eval.math_verify``, write item-level JSONL incrementally.

Determinism (T5 §4.1): every (run, language-neutral dataset, item) gets a sampling seed = sha256 hash (so a subset
file re-uses the seeds of the full set); vLLM uses it per request
(``SamplingParams(seed=..., n=K)``).  HF ``generate`` cannot seed per row, so the HF backend seeds each batch with
the first item's seed (documented weaker guarantee; the primary platform is vLLM).

Platform for reported numbers (O1 §3.2, T1r): Modal L4, bf16, vLLM 0.33 with ``VLLM_USE_FLASHINFER_SAMPLER=0``
(``evalrun/modal_eval.py``); K=4 sampled (T=0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024
generated tokens.  One engine can serve several evaluations (base rows and LoRA adapters): pass ``backend=`` to
:func:`run`; each adapter gets its own ``LoRARequest`` and adapters are never mixed inside one ``generate`` call.
Rows follow the Y2 schema (``analysis/schema.py``): ``dataset`` is the language-neutral set name taken from the
item's own ``dataset`` field when present (the eval-file name is kept as ``eval_set``), ``decode`` is
``sample``/``greedy`` and ``model`` is a short family-size tag (``qwen3-0.6b``).

Resumable: rows already in ``preds.jsonl`` are skipped; rows are appended (and flushed) after every chunk of
``chunk_items`` items, so a killed kernel keeps everything finished so far.  On wall-clock budget expiry the runner
stops between chunks and returns 75.  Raw responses are stored (``save_text``) so scoring changes are re-scored,
never re-generated (:func:`rescore`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from . import GREEDY_K
from ..train import EXIT_NEEDS_RESUME, EXIT_OK
from ..train import config as C
from ..train import data as D
from ..train import modeling as M
from ..train import reward as R
from ..train import template as T


# ---------------------------------------------------------------------------------------------------- items
def load_set(spec: dict, max_items: int | None = None, default_lang: str = "vi") -> list[dict]:
    fields = spec.get("fields") or {}
    out = []
    for i, r in enumerate(D.read_jsonl(spec["path"])):
        q = D.get_field(r, "question", fields)
        gold = D.get_field(r, "gold", fields)
        if gold is None:
            gold = D.item_answer(r, fields)
        if not q or gold is None:
            raise ValueError(f"{spec['path']} row {i}: needs a question and a gold")
        ds_field = spec.get("dataset_from_row", "dataset")
        out.append({"item_id": str(D.get_field(r, "id", fields) or f"{spec['name']}/{i}"), "question": q,
                    "gold": gold, "lang": spec.get("lang") or r.get("lang") or default_lang,
                    "dataset": str(r.get(ds_field) or spec["name"]) if ds_field else spec["name"]})
        if max_items and len(out) >= max_items:
            break
    ids = [it["item_id"] for it in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{spec['path']}: duplicate item ids")
    return out


def sampling_seed(run_id: str, dataset: str, item_id: str) -> int:
    h = hashlib.sha256(f"{run_id}|{dataset}|{item_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


_WS = re.compile(r"\S+")


def repetition_loop(text: str, n: int = 20, repeats: int = 3) -> bool:
    """T5 §4.2: an n-gram (n=20 whitespace tokens) repeated >= 3 times."""
    toks = _WS.findall(text)
    if len(toks) < n * repeats:
        return False
    c = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(c.values()) >= repeats


# ---------------------------------------------------------------------------------------------------- backends
class HFBackend:
    name = "hf"

    def __init__(self, cfg: dict, model_path: str, adapter: str | None):
        import torch
        e = cfg["eval"]
        self.device = M.pick_device(cfg["device"])
        prec = cfg["precision"] if self.device.startswith("cuda") else "fp32"
        self.tok = M.load_tokenizer(adapter if adapter and Path(adapter, "tokenizer_config.json").exists()
                                    else model_path)
        self.model = M.load_model(model_path, prec, self.device, cfg["attn_implementation"])
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.bs = e["batch_size"]
        self.torch = torch
        self.bos = M.bos_prefix(self.tok)

    def generate(self, prompts: list[str], *, n: int, greedy: bool, temperature: float, top_p: float, top_k: int,
                 min_p: float, max_new_tokens: int, seeds: list[int]) -> list[list[dict]]:
        torch, tok = self.torch, self.tok
        res: list[list[dict]] = []
        pad = tok.pad_token_id
        for b0 in range(0, len(prompts), self.bs):
            batch = prompts[b0:b0 + self.bs]
            enc = [self.bos + tok(p, add_special_tokens=False)["input_ids"] for p in batch]
            L = max(len(x) for x in enc)
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in enc], device=self.device)
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in enc], device=self.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=tok.eos_token_id,
                      stop_strings=T.STOP_STRINGS, tokenizer=tok)
            if greedy:
                kw.update(do_sample=False, num_return_sequences=1, temperature=None, top_p=None, top_k=None)
            else:
                kw.update(do_sample=True, num_return_sequences=n, temperature=temperature, top_p=top_p,
                          top_k=top_k or None, min_p=min_p or None)
            torch.manual_seed(seeds[b0])
            with torch.no_grad():
                out = self.model.generate(input_ids=ids, attention_mask=att, **kw)
            gen = out[:, L:].tolist()
            k = 1 if greedy else n
            for i in range(len(batch)):
                row = []
                for j in range(k):
                    g = gen[i * k + j]
                    end = g.index(tok.eos_token_id) + 1 if tok.eos_token_id in g else len(g)
                    while end > 0 and g[end - 1] == pad and pad != tok.eos_token_id:
                        end -= 1
                    raw = tok.decode(g[:end], skip_special_tokens=True)
                    text = T.cut_at_stop(raw)
                    if tok.eos_token_id in g[:end] or text != raw:
                        fr = "stop"
                    else:
                        fr = "length" if end >= max_new_tokens else "stop"
                    row.append({"text": text, "tokens": end, "finish_reason": fr})
                res.append(row)
        return res


class VLLMBackend:
    name = "vllm"

    def __init__(self, cfg: dict, model_path: str, adapter: str | None, enable_lora: bool | None = None):
        from vllm import LLM
        e = cfg["eval"]
        dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[cfg["precision"]]
        kw = dict(model=model_path, dtype=dtype, seed=0, gpu_memory_utilization=e["vllm_gpu_memory_utilization"],
                  max_model_len=e["vllm_max_model_len"], enable_prefix_caching=e["vllm_enable_prefix_caching"])
        self.model_path = model_path
        self.lora, self._lora_ids = None, {}
        self.lora_enabled = bool(adapter) if enable_lora is None else enable_lora
        if self.lora_enabled:
            kw.update(enable_lora=True, max_loras=1, max_lora_rank=max(cfg["lora"]["r"], e["vllm_max_lora_rank"]))
        self.llm = LLM(**kw)
        self.set_adapter(adapter)
        try:
            self.attn_backend = str(self.llm.llm_engine.model_config.__dict__.get("attention_backend", "auto"))
        except Exception:  # noqa: BLE001
            self.attn_backend = "unknown"

    def set_adapter(self, adapter: str | None) -> None:
        """Switch the LoRA used by later ``generate`` calls (None = the base weights).  One engine, many adapters."""
        if not adapter:
            self.lora = None
            return
        if not self.lora_enabled:
            raise RuntimeError("this vLLM engine was built without enable_lora")
        from vllm.lora.request import LoRARequest
        i = self._lora_ids.setdefault(adapter, len(self._lora_ids) + 1)
        self.lora = LoRARequest(f"adapter{i}", i, adapter)

    def generate(self, prompts, *, n, greedy, temperature, top_p, top_k, min_p, max_new_tokens, seeds):
        from vllm import SamplingParams
        sps = []
        for s in seeds:
            if greedy:
                sps.append(SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens, stop=T.STOP_STRINGS))
            else:
                sps.append(SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k or -1,
                                          min_p=min_p, max_tokens=max_new_tokens, seed=s, stop=T.STOP_STRINGS))
        outs = self.llm.generate(prompts, sps, lora_request=self.lora, use_tqdm=False)
        return [[{"text": c.text, "tokens": len(c.token_ids), "finish_reason": c.finish_reason or "stop"}
                 for c in o.outputs] for o in outs]


def make_backend(cfg: dict, model_path: str, adapter: str | None):
    want = cfg["eval"]["backend"]
    if want in ("auto", "vllm"):
        try:
            import torch
            import vllm  # noqa: F401
            if torch.cuda.is_available():
                return VLLMBackend(cfg, model_path, adapter)
            if want == "vllm":
                raise RuntimeError("backend vllm requested but CUDA is not available")
        except ImportError:
            if want == "vllm":
                raise
    return HFBackend(cfg, model_path, adapter)


# ---------------------------------------------------------------------------------------------------- run
def _repair_tail(path: Path) -> None:
    """A killed kernel can leave a torn last line; cut it so the next append starts on a fresh line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def _done_keys(path: Path) -> set:
    done = set()
    _repair_tail(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a torn last line from a killed kernel
                done.add((r["dataset"], r["item_id"], r["sample_k"]))
    return done


def _identity_row(cfg: dict) -> dict:
    return {"run_id": cfg["eval"]["source_run_id"] or cfg["run_id"], "arm": cfg["arm"],
            "model": cfg["eval"]["model_tag"] or model_tag(cfg["model"]), "size": str(cfg["size"]),
            "seed": cfg["seed"]}


def model_tag(model: str) -> str:
    """Y2 ``model`` factor: ``Qwen/Qwen3-0.6B-Base`` -> ``qwen3-0.6b``; ``LiquidAI/LFM2.5-2.6B-Base`` -> ``lfm2.5-2.6b``."""
    t = str(model).rstrip("/").split("/")[-1].lower()
    return re.sub(r"-(base|instruct|it)$", "", t)


def run_info() -> dict:
    """Engine/hardware facts for the manifest (batch composition and hardware change greedy text; O1 §3.2)."""
    import importlib.metadata as md
    out = {}
    for d in ("vllm", "torch", "transformers"):
        try:
            out[d] = md.version(d)
        except Exception:  # noqa: BLE001
            out[d] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    out["VLLM_USE_FLASHINFER_SAMPLER"] = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    return out


def run(cfg: dict, t_start: float | None = None, backend=None, on_chunk=None) -> int:
    """Evaluate one config.  ``backend``: an already-built backend to reuse (its adapter is switched to this
    config's ``eval.adapter_path``).  ``on_chunk(out_dir)`` is called after every fsynced chunk (Modal Volume
    commit).  Returns 0 / 75 (budget)."""
    t_start = time.time() if t_start is None else t_start
    e = cfg["eval"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    C.dump(cfg, out / "config.yaml")
    preds_path = out / "preds.jsonl"
    manifest = {**C.identity(cfg), "eval_of": _identity_row(cfg)["run_id"], "status": "STARTING",
                "code": cfg.get("code"), "platform": cfg.get("platform"),
                "grader": R.grader_version()}
    b = cfg["budget"]
    deadline = (t_start + b["wall_s"] - b["reserve_s"]) if b["wall_s"] else math.inf
    try:
        model_path = M.resolve_model(e["model_path"] or cfg["model"], cache_root=out.parent)
        if backend is None:
            backend = make_backend(cfg, model_path, e["adapter_path"])
        elif hasattr(backend, "set_adapter"):
            if getattr(backend, "model_path", model_path) != model_path:
                raise ValueError(f"shared backend serves {backend.model_path}, config wants {model_path}")
            backend.set_adapter(e["adapter_path"])
        t_eval0 = time.time()
        manifest.update(backend=backend.name, model_path=model_path, adapter_path=e["adapter_path"],
                        decoding={k: e[k] for k in ("temperature", "top_p", "top_k", "min_p", "k_samples", "greedy",
                                                    "max_new_tokens", "shots")},
                        attn_backend=getattr(backend, "attn_backend", None), chunk_items=e["chunk_items"],
                        precision=cfg["precision"], info=run_info())
        _write(out / "manifest.json", manifest)
        prefix = ""
        if e["shots"]:
            shots = D.read_jsonl(cfg["template"]["fewshot_path"])[:e["shots"]]
            prefix = T.fewshot_prefix(shots, cfg["lang"])
        ident = _identity_row(cfg)
        done = _done_keys(preds_path)
        K = e["k_samples"]
        want_k = list(range(K)) + ([GREEDY_K] if e["greedy"] else [])
        gv = R.grader_version()
        for spec in e["sets"]:
            items = load_set(spec, e["max_items"], cfg["lang"])
            todo = [it for it in items if any((it["dataset"], it["item_id"], k) not in done for k in want_k)]
            print(f"[eval] {spec['name']}: {len(items)} items, {len(todo)} to do ({backend.name})", flush=True)
            gen_tokens = 0
            for c0 in range(0, len(todo), e["chunk_items"]):
                if time.time() > deadline:
                    manifest.update(status="NEEDS_RESUME", reason="wall-clock budget")
                    _write(out / "manifest.json", manifest)
                    return EXIT_NEEDS_RESUME
                chunk = todo[c0:c0 + e["chunk_items"]]
                prompts = [prefix + T.prompt(it["question"], cfg["lang"]) for it in chunk]
                seeds = [sampling_seed(ident["run_id"], it["dataset"], it["item_id"]) for it in chunk]
                gen_kw = dict(temperature=e["temperature"], top_p=e["top_p"], top_k=e["top_k"], min_p=e["min_p"],
                              max_new_tokens=e["max_new_tokens"], seeds=seeds)
                outs = {}
                if K:
                    outs["sample"] = backend.generate(prompts, n=K, greedy=False, **gen_kw)
                if e["greedy"]:
                    outs["greedy"] = backend.generate(prompts, n=1, greedy=True, **gen_kw)
                rows = []
                for i, it in enumerate(chunk):
                    for mode, res in outs.items():
                        for j, g in enumerate(res[i]):
                            k = GREEDY_K if mode == "greedy" else j
                            gen_tokens += int(g["tokens"])
                            if (it["dataset"], it["item_id"], k) in done:
                                continue
                            v = R.score(g["text"], it["gold"], spec.get("locale"))
                            trunc = g["finish_reason"] == "length"
                            row = {**ident, "lang": it["lang"], "dataset": it["dataset"], "item_id": it["item_id"],
                                   "sample_k": k, "correct": int(v["correct"]), "tokens": int(g["tokens"]),
                                   "decode": "greedy" if mode == "greedy" else "sample", "eval_set": spec["name"],
                                   "decoding": mode, "temperature": 0.0 if mode == "greedy" else e["temperature"],
                                   "top_p": 1.0 if mode == "greedy" else e["top_p"], "sampling_seed": seeds[i],
                                   "finish_reason": g["finish_reason"], "truncated": trunc,
                                   "repetition_loop": bool(trunc and repetition_loop(g["text"])),
                                   "pred_span": v["span"], "grader_flag": v["flag"], "grader_version": gv,
                                   "backend": backend.name}
                            if e["save_text"]:
                                row["response"] = g["text"]
                            rows.append(row)
                with open(preds_path, "a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                done |= {(r["dataset"], r["item_id"], r["sample_k"]) for r in rows}
                if on_chunk:
                    on_chunk(out)
            manifest.setdefault("throughput", {})[spec["name"]] = {
                "gen_tokens": gen_tokens, "s": round(time.time() - t_eval0, 1),
                "gen_tok_per_s": round(gen_tokens / max(1e-9, time.time() - t_eval0), 1)}
            t_eval0 = time.time()
        metrics = summarize(read_preds(preds_path), K)
        _write(out / "metrics.json", {**ident, "per_dataset": metrics})
        manifest.update(status="COMPLETE", metrics=metrics)
        _write(out / "manifest.json", manifest)
        (out / "DONE").write_text("ok", encoding="utf-8")
        if on_chunk:
            on_chunk(out)
        return EXIT_OK
    except Exception:
        manifest.update(status="FAILED", error=traceback.format_exc()[-6000:])
        _write(out / "manifest.json", manifest)
        raise


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_preds(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def summarize(rows: list[dict], K: int) -> dict:
    """Per dataset: mean pass@1 over K samples, greedy accuracy, maj@K, truncation / loop / no-answer rates."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["dataset"]][r["item_id"]].append(r)
    out = {}
    for ds, items in by.items():
        p1, gr, maj, tr_s, tr_g, loops, noans, toks = [], [], [], [], [], [], [], []
        for rs in items.values():
            smp = [r for r in rs if r["sample_k"] != GREEDY_K]
            grd = [r for r in rs if r["sample_k"] == GREEDY_K]
            if smp:
                p1.append(sum(r["correct"] for r in smp) / len(smp))
                tr_s += [r["truncated"] for r in smp]
                spans = Counter(r["pred_span"] for r in smp if r.get("pred_span"))
                if spans:
                    # deterministic tie-break (QA-T E-maj): among the most frequent spans prefer the greedy answer,
                    # else the span of the lowest sample index -- independent of row order
                    best = max(spans.values())
                    tied = {sp for sp, c in spans.items() if c == best}
                    g_span = grd[0].get("pred_span") if grd else None
                    top = g_span if g_span in tied else min(
                        (r for r in smp if r.get("pred_span") in tied), key=lambda r: r["sample_k"])["pred_span"]
                    maj.append(int(any(r["correct"] for r in smp if r.get("pred_span") == top)))
                else:
                    maj.append(0)
            if grd:
                gr.append(grd[0]["correct"])
                tr_g.append(grd[0]["truncated"])
            loops += [r.get("repetition_loop", False) for r in rs]
            noans += [r.get("grader_flag") == "no-answer" for r in rs]
            toks += [r["tokens"] for r in rs]

        def mean(x):
            return round(sum(x) / len(x), 5) if x else None

        out[ds] = {"n_items": len(items), "mean_pass1": mean(p1), f"maj@{K}": mean(maj), "greedy_acc": mean(gr),
                   "truncation_rate_sampled": mean(tr_s), "truncation_rate_greedy": mean(tr_g),
                   "repetition_loop_rate": mean(loops), "no_answer_rate": mean(noans), "tokens_mean": mean(toks)}
    return out


def rescore(preds_path: str | Path, sets: list[dict], out_path: str | Path) -> int:
    """Re-grade stored responses with the current grader (never re-generate).  Returns the number of flips."""
    gold = {}
    for spec in sets:
        for it in load_set(spec):
            gold[(it["dataset"], it["item_id"])] = (it["gold"], spec.get("locale"))
    flips, gv = 0, R.grader_version()
    with open(out_path, "w", encoding="utf-8") as f:
        for r in read_preds(preds_path):
            g, loc = gold[(r["dataset"], r["item_id"])]
            v = R.score(r["response"], g, loc)
            flips += int(int(v["correct"]) != r["correct"])
            r.update(correct=int(v["correct"]), pred_span=v["span"], grader_flag=v["flag"], grader_version=gv)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return flips
