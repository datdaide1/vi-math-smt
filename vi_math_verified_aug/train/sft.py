"""SFT trainer (T3 §3.1).  A plain PyTorch loop; no HF Trainer, so every recipe detail is visible here.

Recipe
------
* completion-only, token-mean NLL: per optimiser step, sum of NLL over all completion tokens (incl. EOS) of the
  step's 32 sequences divided by their count; CE in fp32.
* fp16 base weights, LoRA r=32 / alpha=32 / dropout 0 on all linear layers of the text decoder, fp32 adapters,
  torch AdamW with fp32 states, weight decay 0.
* fp16 autocast + dynamic GradScaler; grad-clip 1.0 after unscale; skipped steps counted; abort (exit 4) when more
  than ``max_skip_frac`` (1%) of the executed optimiser steps were skipped.
* 32 sequences per step = micro-batch x accumulation (micro-batch auto-halved on OOM at a probe on the longest
  items); padded, length-grouped micro-batches.  engine=unsloth (O1 §3.1) may pack each micro-batch into one row
  (``sft.packing``); every Unsloth run first passes a logits gate (packed/padded vs separate per-item NLL, and
  Unsloth's fused loss vs the logits path) or exits 6 before training.
* ``use_cache=False`` in every training forward.
* Optional dev loss (``sft.dev_path``): token-mean completion NLL of held-out items at step 0, every
  ``dev_every_steps`` and at the end, through the same engine/batching (``dev_log.jsonl``).
* WSD schedule with epoch branches (schedule.py); one adapter per branch epoch plus the final one.
* Resumable: adapter + optimiser + scaler + RNG + step + branch bookkeeping every ``ckpt_every_min`` minutes, at
  every fork, and when the wall-clock budget runs out (then exit 75 = NEEDS_RESUME).
* One seed: LoRA init (torch seed set right before the adapter is created), data order and the subsample.

Outputs in ``out_dir``: config.yaml, manifest.json, data_stats.json, train_log.jsonl (one row per optimiser step,
flushed immediately), ckpt/, adapters/epoch<e>/ (PEFT adapter dirs), DONE.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

from . import (EXIT_ABORT_OVERFLOW, EXIT_CELL_STOPPED, EXIT_DATA_ERROR, EXIT_GATE_FAILED, EXIT_GUARD_ABORT,
               EXIT_NEEDS_RESUME, EXIT_OK)
from . import guards as G
from . import template as T
from . import config as C
from . import data as D
from . import modeling as M
from . import schedule as S


class TooManySkippedSteps(RuntimeError):
    pass


class BudgetReached(RuntimeError):
    pass


class GateFailed(RuntimeError):
    pass


class DataError(RuntimeError):
    """Data/setup problem (missing file, shortfall, malformed pool): never retried on another engine."""


def size_params(size) -> float | None:
    """'0.6B' -> 6e8 (used to decide grad-ckpt before an Unsloth load)."""
    try:
        t = str(size).strip().upper()
        return float(t[:-1]) * {"B": 1e9, "M": 1e6}[t[-1]]
    except (ValueError, KeyError, IndexError):
        return None


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass


def env_info() -> dict:
    import importlib.metadata as md
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for d in ("torch", "transformers", "peft", "trl", "accelerate", "vllm"):
        try:
            out[d] = md.version(d)
        except Exception:  # noqa: BLE001
            out[d] = None
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_cap"] = list(torch.cuda.get_device_capability(0))
            out["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return out


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


class SFTRun:
    def __init__(self, cfg: dict, t_start: float | None = None, shared: dict | None = None):
        if cfg["kind"] != "sft":
            raise C.ConfigError("SFTRun needs kind: sft")
        self.cfg = cfg
        self.s = cfg["sft"]
        self.t_start = time.time() if t_start is None else t_start
        self.out = Path(cfg["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt_root = self.out / "ckpt"
        self.engine, self.packed, self.dev_items, self.tok_check = cfg["engine"], bool(cfg["sft"]["packing"]), [], None
        self.g = cfg["guards"]
        self.shared = shared   # run_cell: {"base": (unsloth base model, tokenizer)} reused across the cell's arms
        self.manifest: dict = {**C.identity(cfg), "platform": cfg.get("platform"), "precision": cfg["precision"],
                               "code": cfg.get("code"),
                               "status": "STARTING", "env": env_info(),
                               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        C.dump(cfg, self.out / "config.yaml")

    # ------------------------------------------------------------------------------------------ setup
    def _save_manifest(self, **kw) -> None:
        self.manifest.update(kw)
        self.manifest["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(self.out / "manifest.json", self.manifest)

    def setup(self) -> None:
        cfg, s = self.cfg, self.s
        self.device = M.pick_device(cfg["device"])
        prec = cfg["precision"]
        if self.device == "cpu" and prec == "fp16":
            raise C.ConfigError("precision fp16 needs CUDA; use precision: fp32 for CPU smoke runs")
        import torch
        set_seed(cfg["seed"])
        self.model_path = M.resolve_model(cfg["model"], cache_root=self.out.parent)
        self.engine = cfg["engine"]
        self.packed = bool(s["packing"])
        if self.engine == "unsloth":
            os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
            import unsloth  # noqa: F401  (must precede transformers; train.cli imports it first as well)
        self.tok = M.load_tokenizer(self.model_path)
        self.pad_id = self.tok.pad_token_id

        # data: drop over-long items on the pool, then the seed's subsample
        try:
            rows = D.read_jsonl(cfg["data"]["train"])
        except (OSError, ValueError) as e:
            raise DataError(f"cannot read {cfg['data']['train']}: {e!r}") from e
        pool, pool_stats = D.prepare_pool(rows, self.tok, s["max_length"], cfg["data"]["fields"], cfg["lang"])
        items = D.subsample(pool, cfg["budget"]["n_train"], cfg["seed"])
        n_req = cfg["budget"]["n_train"]
        short = (n_req - len(items)) if n_req else 0
        pool_stats.update(n_train_requested=n_req, n_train_eff=len(items), shortfall=short,
                          shortfall_frac=round(short / n_req, 5) if n_req else 0.0)
        if n_req and short / n_req > cfg["guards"].get("max_shortfall", 0.01):
            raise DataError(f"n_train {n_req} requested, only {len(items)} eligible "
                            f"(shortfall {short / n_req:.2%} > {cfg['guards'].get('max_shortfall', 0.01):.0%}); "
                            f"drop rate {pool_stats['drop_rate']}, malformed {pool_stats['n_malformed']}")
        self.items = items
        self.fp = D.fingerprint(items)
        self.data_stats = {**pool_stats, **D.length_stats(items), "fingerprint": self.fp, "arm": cfg["arm"],
                           "train_path": cfg["data"]["train"]}
        _write_json(self.out / "data_stats.json", self.data_stats)
        self.g = cfg["guards"]
        if self.g["enabled"]:
            self.preflight_data(len(rows), [r for r in rows])
        print(f"[sft] pool {pool_stats['n_pool']} eligible {pool_stats['n_eligible']} dropped>max_len "
              f"{pool_stats['n_dropped_too_long']} (rate {pool_stats['drop_rate']}) -> train {len(items)}", flush=True)

        self.splan = D.StepPlan(len(items), s["seq_per_step"], cfg["seed"])
        self.plan = S.plan(self.splan.steps_per_epoch, cfg["budget"]["epochs"], s["branch_epochs"],
                           s["warmup_frac"], s["decay_frac"])
        if cfg["budget"]["max_steps"]:  # smoke cap: shorten the main run (branches beyond the cap are dropped)
            cap = int(cfg["budget"]["max_steps"])
            if cap < self.plan["total"]:
                self.plan = {**self.plan, "total": cap, "decay": S.decay_len(cap, s["decay_frac"]),
                             "branches": [b for b in self.plan["branches"] if b["end_step"] <= cap - 1]}
        self.dev_items = []
        if s["dev_path"]:
            dev_pool, dev_stats = D.prepare_pool(D.read_jsonl(s["dev_path"]), self.tok, s["max_length"],
                                                 cfg["data"]["fields"], cfg["lang"])
            self.dev_items = dev_pool[:s["dev_max_items"]]
            self.data_stats["dev"] = {**dev_stats, "n_used": len(self.dev_items), "path": s["dev_path"]}
            _write_json(self.out / "data_stats.json", self.data_stats)
        # model + adapter (LoRA init is seeded right here)
        torch.manual_seed(cfg["seed"])
        lc = cfg["lora"]
        gc = s["grad_ckpt"]
        if self.engine == "unsloth":
            est = size_params(cfg["size"]) or 0
            self.grad_ckpt = (est >= 1.5e9) if gc == "auto" else bool(gc)
            msl = max(s["max_length"] * s["micro_batch"], 2048)
            base = None
            if self.shared is not None:
                if self.shared.get("base") is None or self.shared.get("key") != (self.model_path, prec, msl):
                    self.shared.update(base=M.load_unsloth_base(self.model_path, prec, max_seq_length=msl),
                                       key=(self.model_path, prec, msl))
                base = self.shared["base"]
            self.model, utok = M.load_unsloth(
                self.model_path, prec, max_seq_length=msl, r=lc["r"], alpha=lc["alpha"], dropout=lc["dropout"],
                seed=cfg["seed"], grad_ckpt=self.grad_ckpt, targets=lc["targets"], base=base)
            self.lora_init = M.lora_init_sha(self.model)
            self.n_params = M.n_params(self.model.get_base_model())
            self.lora_summary = {"targets": "unsloth", "modules": sorted({n.split(".")[-4] for n, _ in
                                                                         self.model.named_parameters()
                                                                         if "lora_A" in n})}
            self.tok_check = {"eos_same": utok.eos_token_id == self.tok.eos_token_id,
                              "vocab_same": len(utok) == len(self.tok), "unsloth_pad": utok.pad_token_id,
                              "pad": self.pad_id}
            if not self.tok_check["eos_same"]:
                raise GateFailed(f"Unsloth tokenizer differs from AutoTokenizer: {self.tok_check}")
        else:
            base = M.load_model(self.model_path, prec, self.device, cfg["attn_implementation"])
            self.n_params = M.n_params(base)
            self.lora_summary = M.lora_targets_summary(base)
            torch.manual_seed(cfg["seed"])
            self.model = M.add_lora(base, lc["r"], lc["alpha"], lc["dropout"], lc["targets"])
            self.grad_ckpt = (self.n_params >= 1.5e9) if gc == "auto" else bool(gc)
            if self.grad_ckpt:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.model.enable_input_require_grads()
            self.tok_check = None
        self.model.train()
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.trainable, lr=s["lr"], betas=tuple(s["betas"]),
                                     weight_decay=s["weight_decay"])
        self.dev_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.amp_dtype = M.torch_dtype(prec)
        self.amp = prec in ("fp16", "bf16")
        self.scaler = self._new_scaler()
        self.micro = s["micro_batch"]
        self.st = {"global_step": 0, "branches_done": [], "exec_steps": 0, "exec_skipped": 0,
                   "exec_skipped_first10": 0, "nonfinite_losses": 0, "real_tokens": 0, "padded_tokens": 0,
                   "completion_tokens": 0, "train_s": 0.0, "resumed_from": [], "outputs": {}}
        self._save_manifest(status="SETUP", model_path=self.model_path, n_params=self.n_params,
                            lora=self.lora_summary, grad_ckpt=self.grad_ckpt, plan=self.plan,
                            data_stats=self.data_stats, engine=self.engine, packed=self.packed,
                            tok_check=self.tok_check, lora_init_sha=getattr(self, "lora_init", None),
                            shared_process=self.shared is not None, lora_param_dtypes=sorted({str(p.dtype) for p in
                                                                                   self.trainable}),
                            trainable_params=sum(p.numel() for p in self.trainable))

    def _peak_mem(self):
        try:
            import torch
            return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if self.dev_type == "cuda" else None
        except Exception:  # noqa: BLE001
            return None

    def _new_scaler(self):
        import torch
        return torch.amp.GradScaler(self.dev_type, init_scale=self.s["init_scale"],
                                    enabled=self.cfg["precision"] == "fp16")

    # ------------------------------------------------------------------------------------------ one step
    def _tensors(self, mb: list[dict], packed: bool | None = None):
        import torch
        packed = self.packed if packed is None else packed
        if packed:
            b = D.collate_packed(mb)
            t = {k: torch.tensor(b[k], device=self.device) for k in ("input_ids", "labels", "position_ids")}
            t["packed_seq_lengths"] = torch.tensor(b["packed_seq_lengths"], dtype=torch.int32, device=self.device)
        else:
            b = D.collate(mb, self.pad_id)
            t = {k: torch.tensor(b[k], device=self.device) for k in ("input_ids", "attention_mask", "labels")}
        return t, b["real_tokens"], b["padded_tokens"]

    def _forward_loss(self, t: dict):
        import torch
        with torch.autocast(self.dev_type, dtype=self.amp_dtype, enabled=self.amp):
            if self.engine == "unsloth":
                return M.unsloth_nll_sum(self.model, t)
            return M.completion_nll_sum(self.model, t["input_ids"], t.get("attention_mask"), t["labels"],
                                        position_ids=t.get("position_ids"))

    # ------------------------------------------------------------------------------------------ gate + dev
    def _logits(self, t: dict):
        import torch
        kw = {k: t[k] for k in ("input_ids", "attention_mask", "position_ids", "packed_seq_lengths") if k in t}
        lg = None
        try:
            with torch.no_grad(), torch.autocast(self.dev_type, dtype=self.amp_dtype, enabled=self.amp):
                lg = self.model(use_cache=False, **kw).logits
        except Exception as e:  # noqa: BLE001
            print(f"[sft] logits without labels failed: {e!r}", flush=True)
        if lg is None or not torch.is_tensor(lg) or lg.dim() != 3 or lg.shape[1] != t["input_ids"].shape[1]:
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
            try:
                with torch.no_grad(), torch.autocast(self.dev_type, dtype=self.amp_dtype, enabled=self.amp):
                    lg = self.model(use_cache=False, labels=t["labels"], **kw).logits
            finally:
                os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        return lg.float()

    def pack_gate(self) -> dict:
        """Start-of-job gate for engine=unsloth (O1 §3.1): on 4 seeded items, per-item completion NLL of
        (a) each item alone, (b) the training batch layout (packed row, or padded rows) must agree within
        ``pack_gate_max_dnll`` nats/token with argmax agreement > ``pack_gate_min_argmax``, and (c) Unsloth's fused
        training loss (x count) must equal the summed per-item NLL of the same batch.  Fails -> exit 6."""
        import torch
        s = self.s
        four = random.Random(f"gate:{self.cfg['seed']}").sample(self.items, 4)
        rows, ok = [], True
        sep = []
        for it in four:
            t, _, _ = self._tensors([it], packed=False)
            sep.append(self._logits(t)[0])
        # argmax noise floor of THIS precision: each item alone vs the same items as right-padded rows (another
        # micro-batch layout of the same math).  R-GPU: bf16 on L40S gives 0.964 on a 69-target item, fp16 ~1.0.
        tp, _, _ = self._tensors(four, packed=False)
        lgp = self._logits(tp)
        floor = min(float((sep[k][:len(it["ids"])].argmax(-1) == lgp[k, :len(it["ids"])].argmax(-1)).float().mean())
                    for k, it in enumerate(four))
        del lgp
        min_arg = min(s["pack_gate_min_argmax"], floor - s["pack_gate_argmax_margin"])
        t, _, _ = self._tensors(four)
        lg = self._logits(t)
        off = 0
        tot_b, n_b = 0.0, 0
        for k, it in enumerate(four):
            L = len(it["ids"])
            lab = torch.tensor(it["labels"], device=self.device)
            a = sep[k]
            b = lg[0, off:off + L] if self.packed else lg[k, :L]
            off += L
            na, cnt = M.item_nll_from_logits(a, lab)
            nb, _ = M.item_nll_from_logits(b, lab)
            tot_b, n_b = tot_b + nb, n_b + cnt
            agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
            d = abs(na - nb) / max(1, cnt)
            rows.append({"len": L, "n_targets": cnt, "nll_sep": round(na / cnt, 5), "nll_batch": round(nb / cnt, 5),
                         "dnll_per_tok": round(d, 6), "argmax_agree": round(agree, 4),
                         "max_abs_dlogit": round(float((a - b).abs().max()), 4)})
            ok &= d < s["pack_gate_max_dnll"] and agree >= min_arg
        with torch.no_grad():
            loss_sum = float(self._forward_loss(t))
        d_loss = abs(loss_sum - tot_b) / max(1, n_b)
        ok &= d_loss < s["pack_gate_max_dnll"]
        del sep, lg
        if self.dev_type == "cuda":
            torch.cuda.empty_cache()
        res = {"ok": bool(ok), "packed": self.packed, "engine": self.engine, "precision": self.cfg["precision"],
               "items": rows, "fused_loss_sum": round(loss_sum, 4), "logits_sum": round(tot_b, 4),
               "fused_vs_logits_per_tok": round(d_loss, 6), "max_dnll": s["pack_gate_max_dnll"],
               "min_argmax_cfg": s["pack_gate_min_argmax"], "argmax_floor_padded_vs_sep": round(floor, 4),
               "argmax_margin": s["pack_gate_argmax_margin"], "min_argmax_effective": round(min_arg, 4)}
        _write_json(self.out / "pack_gate.json", res)
        self._save_manifest(pack_gate=res)
        print(f"[sft] pack gate ok={ok} items={[r['dnll_per_tok'] for r in rows]} fused={d_loss:.2e}", flush=True)
        if not ok:
            raise GateFailed(f"Unsloth logits gate failed: {res}")
        return res

    # ------------------------------------------------------------------------------------------ guards
    def preflight_data(self, n_rows: int, rows: list[dict]) -> None:
        """Dataset identity, eval overlap, decoded labels + template round-trip (guards.py)."""
        cfg, g = self.cfg, self.g
        lang = cfg["lang"]
        res = {"dataset": G.check_dataset(cfg["data"]["train"], n_rows, g)}
        if g.get("eval_hashes_path"):
            hs = set(json.loads(Path(g["eval_hashes_path"]).read_text(encoding="utf-8")))
            res["eval_overlap"] = G.overlap_with_eval([it["q"] for it in self.items], hs)

        def dec(ids):
            return self.tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        res["labels"] = G.preflight_items(self.items, dec, prompt_fn=lambda q: T.prompt(q, lang),
                                          completion_fn=lambda sol, a: T.completion(sol, a, lang),
                                          eos_id=self.tok.eos_token_id, pad_id=self.pad_id, g=g)
        self.preflight = res
        _write_json(self.out / "preflight.json", res)
        for e in res["labels"]["decoded"]:
            print(f"[preflight] {e['id']} trained {e['n_trained']}/{e['n_total']}: {e['trained_text'][:400]!r}",
                  flush=True)

    def quick_eval(self, step: int) -> dict | None:
        """Greedy decode of ``quick_n`` fixed dev items (batch 1, no KV cache: the training forward itself, so it
        is valid for every engine), graded with the project grader.  Rates of correct / no-answer / repetition /
        truncation go to dev_log.jsonl."""
        g = self.g
        if not (g["enabled"] and self.dev_items and g["quick_n"]):
            return None
        import torch
        from . import reward as R
        from ..evalrun.runner import repetition_loop
        t0 = time.time()
        eos = self.tok.eos_token_id
        bos = M.bos_prefix(self.tok)
        rows = []
        dv = self.dev_items  # a FIXED seeded sample across the dev set's datasets (GSM8K + MATH seeds), same every check
        its = [dv[i] for i in sorted(random.Random("quick-eval").sample(range(len(dv)), min(g["quick_n"], len(dv))))]
        prompts = [bos + self.tok(T.prompt(it["q"], self.cfg["lang"]), add_special_tokens=False)["input_ids"]
                   for it in its]
        try:
            gens, path = self._generate_batched(prompts), "generate"
        except Exception as e:  # noqa: BLE001
            print(f"[sft] quick-eval batched generate failed ({e!r}); no-cache greedy loop", flush=True)
            gens, path = [self._greedy_nocache(p) for p in prompts], "nocache"
        for it, out in zip(its, gens):
            if eos in out:
                out = out[:out.index(eos)]
            text = T.cut_at_stop(self.tok.decode(out, skip_special_tokens=True))
            v = R.score(text, it["ans"])  # legacy-string gold: typed once by the grader (infer_gold)
            trunc = len(out) >= g["quick_max_new_tokens"]
            rows.append({"correct": v["correct"], "no_answer": v.get("flag") == "no-answer" or not v.get("span"),
                         "repetition": repetition_loop(text) or (trunc and repetition_loop(text, n=8)),
                         "truncated": trunc})
        q = G.quick_stats(rows)
        q.update(step=step, s=round(time.time() - t0, 1), path=path)
        print(f"[sft] quick-eval step {step}: {q}", flush=True)
        return q

    def _last_logits(self, x):
        """Next-token logits [V] of a 1-row batch; Unsloth may return no logits without labels unless
        UNSLOTH_RETURN_LOGITS=1 (same fallback as ``_logits``)."""
        import torch
        with torch.autocast(self.dev_type, dtype=self.amp_dtype, enabled=self.amp):
            try:
                lg = self.model(input_ids=x, use_cache=False, logits_to_keep=1).logits
            except TypeError:
                lg = self.model(input_ids=x, use_cache=False).logits
            if lg is None or not torch.is_tensor(lg) or lg.dim() != 3 or lg.shape[-1] < 1000:
                os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
                try:
                    lg = self.model(input_ids=x, use_cache=False, labels=x).logits
                finally:
                    os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
        return lg[0, -1].float()

    def _greedy_nocache(self, ids: list[int]) -> list[int]:
        import torch
        x = torch.tensor([ids], device=self.device)
        out = []
        with torch.no_grad():
            for _ in range(self.g["quick_max_new_tokens"]):
                nxt = int(self._last_logits(x).argmax())
                out.append(nxt)
                if nxt == self.tok.eos_token_id:
                    break
                x = torch.cat([x, torch.tensor([[nxt]], device=self.device)], dim=1)
        return out

    def _generate_batched(self, prompts: list[list[int]], bs: int = 16) -> list[list[int]]:
        """Greedy ``generate`` with KV cache, left-padded batches of 16.  Unsloth: for_inference / for_training
        around it (its documented in-training eval path).  LFM2 (short-conv state): batch 1, no padding."""
        import torch
        fam = M.family_of(self.model_path)
        if fam == "lfm2":
            bs = 1
        uns = self.engine == "unsloth"
        if uns:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(self.model)
        pad = self.pad_id if self.pad_id is not None else self.tok.eos_token_id
        outs = []
        try:
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    ch = prompts[i:i + bs]
                    L = max(len(p) for p in ch)
                    ids = torch.tensor([[pad] * (L - len(p)) + p for p in ch], device=self.device)
                    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in ch], device=self.device)
                    with torch.autocast(self.dev_type, dtype=self.amp_dtype, enabled=self.amp):
                        gen = self.model.generate(input_ids=ids, attention_mask=att, do_sample=False,
                                                  max_new_tokens=self.g["quick_max_new_tokens"], pad_token_id=pad,
                                                  eos_token_id=self.tok.eos_token_id, use_cache=True)
                    outs += [row[L:].tolist() for row in gen]
        finally:
            if uns:
                try:
                    FastLanguageModel.for_training(self.model, use_gradient_checkpointing=(
                        "unsloth" if self.grad_ckpt else False))
                except TypeError:
                    FastLanguageModel.for_training(self.model)
            self.model.train()
        return outs

    def midrun_check(self, step: int, final: bool = False) -> None:
        dv = self.dev_loss(step)
        q = self.quick_eval(step)
        tr = self.guard.recent_mean() if getattr(self, "guard", None) else None
        if step == 0:
            self.base_quick = q
            self.dev0 = dv and dv["dev_loss"]
        reasons, warns = ([], []) if step == 0 else G._midrun(dv and dv["dev_loss"], tr, q,
                                                               getattr(self, "base_quick", None), self.g,
                                                               getattr(self, "dev0", None))
        if step > 0:
            self.quick_hist = getattr(self, "quick_hist", []) + [q]
            qc = G.quick_collapse(self.quick_hist, getattr(self, "base_quick", None), self.g)
            if qc:
                reasons = reasons + [qc]
        if final:  # QA-T #2: the end-of-run check records, never aborts (the adapter is already saved)
            warns, reasons = warns + reasons, []
        if q is None and self.g["enabled"] and self.dev_items and self.g["quick_n"]:
            reasons = reasons + ["QUICK_EVAL_MISSING"]
        rec = {"step": step, "kind": "midrun", "final": final, "train_loss_recent": tr,
               "dev_loss": dv and dv["dev_loss"], "quick": q, "base_quick": getattr(self, "base_quick", None),
               "abort": reasons, "warnings": warns}
        with open(self.out / "dev_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._save_manifest(last_midrun=rec)
        if reasons and self.g["enabled"]:
            raise G.GuardAbort(reasons[0].split()[0], "; ".join(reasons))

    def dev_loss(self, step: int) -> dict | None:
        """Token-mean completion NLL of the dev items (same engine and batch layout as training, no grad)."""
        if not self.dev_items:
            return None
        import torch
        t0 = time.time()
        tot, n = 0.0, 0
        its = sorted(self.dev_items, key=lambda it: -len(it["ids"]))
        with torch.no_grad():
            for i in range(0, len(its), self.micro):
                mb = its[i:i + self.micro]
                t, _, _ = self._tensors(mb)
                tot += float(self._forward_loss(t))
                n += D.n_completion_tokens(mb)
        row = {"step": step, "dev_loss": round(tot / max(1, n), 6), "n_items": len(its), "n_tokens": n,
               "s": round(time.time() - t0, 2)}
        with open(self.out / "dev_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[sft] dev step {step}: {row['dev_loss']:.5f}", flush=True)
        self.last_dev = row
        return row

    def _act_hooks(self):
        """Max |output| of every Linear in the text decoder (fp16 overflow headroom, S2 §1); returns (store, rm)."""
        import torch.nn as nn
        store = {"max": 0.0}

        def h(_m, _i, out):
            try:
                store["max"] = max(store["max"], float(out.detach().abs().max()))
            except Exception:  # noqa: BLE001
                pass
        hs = [m.register_forward_hook(h) for n, m in self.model.named_modules()
              if isinstance(m, nn.Linear) and not M.SKIP_RE.search(n) and "lora_" not in n]
        return store, lambda: [x.remove() for x in hs]

    def _probe_once(self, batch: list[dict]) -> None:
        """One fwd+bwd in its own frame: on OOM the exception's traceback is the only reference to the graph, and it
        dies with this frame (S2 §7.9: the inline probe kept the failed graph alive and cascaded 8 -> 4 -> 2 -> 1;
        that is exactly what t1r-gate-hf did on Kaggle, 810 tok/s at micro 1)."""
        import torch
        t, _, _ = self._tensors(batch)
        loss = self._forward_loss(t)
        self.scaler.scale(loss / max(1, D.n_completion_tokens(batch))).backward()
        torch.cuda.synchronize()

    def probe_micro(self) -> None:
        """Forward+backward on the longest items; halve the micro-batch on CUDA OOM (memory freed between sizes)."""
        if self.dev_type != "cuda":
            return
        import gc
        import torch
        longest = sorted(self.items, key=lambda it: -len(it["ids"]))
        log = []
        while True:
            ok, err = False, None
            try:
                self._probe_once(longest[:self.micro])
                ok = True
            except torch.cuda.OutOfMemoryError as e:
                err = str(e)[:200]
            self.opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if ok:
                break
            log.append({"micro": self.micro, "err": err})
            if self.micro == 1:
                raise torch.cuda.OutOfMemoryError("micro-batch 1 does not fit: " + (err or ""))
            self.micro //= 2
        self.scaler = self._new_scaler()  # the probe must not move the loss scale
        self._save_manifest(micro_batch=self.micro, accum=math.ceil(self.s["seq_per_step"] / self.micro),
                            oom_probe=log)

    def train_step(self, step: int, mult: float, tag: str) -> dict:
        import torch
        mon_every = self.s["act_monitor_every"]
        mon = self._act_hooks() if (mon_every and step % mon_every == 0) else None
        t0 = time.time()
        step_items = [self.items[i] for i in self.splan.step_indices(step)]
        n_comp = D.n_completion_tokens(step_items)
        loss_tot, real, padded, nonfinite = 0.0, 0, 0, 0
        for mb in D.micro_batches(step_items, self.micro, self.s["length_grouping"]):
            t, r, p = self._tensors(mb)
            real, padded = real + r, padded + p
            loss_sum = self._forward_loss(t)
            loss = loss_sum / n_comp
            if not torch.isfinite(loss):
                nonfinite += 1
            self.scaler.scale(loss).backward()
            loss_tot += float(loss.detach())
        lr = self.s["lr"] * mult
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.scaler.unscale_(self.opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.s["grad_clip"]))
        before = self.scaler.get_scale()
        if self.scaler.is_enabled():
            self.scaler.step(self.opt)
            self.scaler.update()
            skipped = self.scaler.get_scale() < before
        else:  # fp32/bf16: no scaler; never apply a non-finite update
            skipped = not math.isfinite(gnorm)
            if not skipped:
                self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        if self.dev_type == "cuda":
            torch.cuda.synchronize()
        st = self.st
        st["exec_steps"] += 1
        st["exec_skipped"] += int(skipped)
        if st["exec_steps"] <= 10:
            st["exec_skipped_first10"] += int(skipped)
        st["nonfinite_losses"] += nonfinite
        st["real_tokens"] += real
        st["padded_tokens"] += padded
        st["completion_tokens"] += n_comp
        dt = time.time() - t0
        st["train_s"] += dt
        row = {"step": step, "epoch": round(step / self.splan.steps_per_epoch, 4), "tag": tag, "lr": lr,
               "loss": round(loss_tot, 6), "gnorm": round(gnorm, 5) if math.isfinite(gnorm) else str(gnorm),
               "scale": before, "skipped": bool(skipped), "n_completion_tokens": n_comp, "real_tokens": real,
               "padded_tokens": padded, "s": round(dt, 3), "t": round(time.time() - self.t_start, 1)}
        if mon is not None:
            mon[1]()
            row["max_abs_act"] = round(mon[0]["max"], 2)
            st["max_abs_act"] = max(st.get("max_abs_act", 0.0), mon[0]["max"])
        if self.dev_type == "cuda":
            row["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        with open(self.out / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._check_skips(final=False)
        return row

    def _check_skips(self, final: bool) -> None:
        """Abort rule.  Evaluated after every step once ``skip_check_after`` steps ran (the GradScaler's first
        calibration skips would otherwise trip it on short runs) and again at the end (``final``)."""
        st, s = self.st, self.s
        n = st["exec_steps"]
        if n < s["skip_check_after"]:
            if final and n:
                print(f"[sft] skip rule not applicable: only {n} steps (< {s['skip_check_after']})", flush=True)
            return
        frac = st["exec_skipped"] / n
        if frac > s["max_skip_frac"]:
            raise TooManySkippedSteps(f"{st['exec_skipped']}/{n} optimiser steps skipped "
                                      f"({frac:.2%} > {s['max_skip_frac']:.0%}) - fp16 overflow")

    # ------------------------------------------------------------------------------------------ checkpoints
    def _ckpt_state(self) -> dict:
        return {**self.st, "config_hash": self.cfg["config_hash"], "fingerprint": self.fp, "micro": self.micro,
                "run_id": self.cfg["run_id"], "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save_ckpt(self, tag: str = "") -> Path:
        import torch
        from peft import get_peft_model_state_dict
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        name = f"step_{self.st['global_step']:07d}_b{len(self.st['branches_done'])}{('_' + tag) if tag else ''}"
        tmp = self.ckpt_root / f".tmp_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        torch.save(get_peft_model_state_dict(self.model), tmp / "adapter.pt")
        torch.save(self.opt.state_dict(), tmp / "optimizer.pt")
        torch.save(self.scaler.state_dict(), tmp / "scaler.pt")
        rng = {"torch": torch.get_rng_state(), "python": random.getstate()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, tmp / "rng.pt")
        _write_json(tmp / "state.json", self._ckpt_state())
        final = self.ckpt_root / name
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        (self.ckpt_root / "LATEST").write_text(name, encoding="utf-8")
        self._prune()
        self._last_ckpt_t = time.time()
        return final

    def _prune(self) -> None:
        pend = {f"fork{b['epoch']}" for b in self.plan["branches"] if b["epoch"] not in self.st["branches_done"]}
        regular = sorted(p for p in self.ckpt_root.glob("step_*") if p.is_dir())
        keep = set(regular[-self.s["keep_ckpts"]:])
        for p in regular:
            if p not in keep and not any(p.name.endswith("_" + t) for t in pend):
                shutil.rmtree(p, ignore_errors=True)

    @staticmethod
    def find_ckpts(roots: list[Path]) -> list[Path]:
        found = []
        for r in roots:
            r = Path(r)
            for base in (r, r / "ckpt"):
                if base.is_dir():
                    found += [p for p in base.glob("step_*") if (p / "state.json").exists()]
        return found

    def load_ckpt(self, path: Path) -> None:
        import torch
        from peft import set_peft_model_state_dict
        st = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if st["config_hash"] != self.cfg["config_hash"]:
            raise RuntimeError(f"checkpoint {path} has config_hash {st['config_hash']} != {self.cfg['config_hash']}")
        if st["fingerprint"] != self.fp:
            raise RuntimeError(f"checkpoint {path} was trained on different data ({st['fingerprint']} != {self.fp})")
        set_peft_model_state_dict(self.model, torch.load(path / "adapter.pt", map_location=self.device))
        self.opt.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.device))
        self.scaler.load_state_dict(torch.load(path / "scaler.pt"))
        rng = torch.load(path / "rng.pt", weights_only=False)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        keep = ("global_step", "branches_done", "exec_steps", "exec_skipped", "exec_skipped_first10",
                "nonfinite_losses", "real_tokens", "padded_tokens", "completion_tokens", "train_s", "resumed_from",
                "outputs")
        for k in keep:
            if k in st:
                self.st[k] = st[k]
        self.micro = st.get("micro", self.micro)

    def maybe_resume(self) -> None:
        mode = self.cfg["resume"]
        if mode == "never":
            return
        roots = [self.ckpt_root] + [Path(p) for p in self.cfg["resume_dirs"]]
        if mode not in ("auto", "never"):
            roots = [Path(mode)]
        cands = self.find_ckpts(roots)
        if not cands:
            return
        def key(p):
            s = json.loads((p / "state.json").read_text(encoding="utf-8"))
            return (s["global_step"], len(s["branches_done"]), s["exec_steps"])
        best = max(cands, key=key)
        self.load_ckpt(best)
        self.st["resumed_from"] = self.st["resumed_from"] + [str(best)]
        print(f"[sft] resumed from {best} at step {self.st['global_step']}", flush=True)

    # ------------------------------------------------------------------------------------------ loop
    def _budget_left(self) -> float:
        b = self.cfg["budget"]
        if not b["wall_s"]:
            return math.inf
        return b["wall_s"] - b["reserve_s"] - (time.time() - self.t_start)

    def _save_adapter(self, epoch_label: str, extra: dict) -> str:
        path = self.out / "adapters" / epoch_label
        M.save_adapter(self.model, path, self.tok, meta={**C.identity(self.cfg), **extra})
        self.st["outputs"][epoch_label] = str(path)
        return str(path)

    def _run_branch(self, b: dict) -> None:
        """Decay branch from the current (fork) state; save its adapter; restore the fork state."""
        fork_ckpt = self.save_ckpt(tag=f"fork{b['epoch']}")
        for step in range(b["fork_step"], b["end_step"]):
            self.train_step(step, S.branch_mult(step, b, self.plan["warmup"]), tag=f"branch{b['epoch']}")
            if self._budget_left() <= 0:
                raise BudgetReached(f"budget reached inside branch {b['epoch']}; resume redoes it from {fork_ckpt}")
        self._save_adapter(f"epoch{b['epoch']}", {"branch": b, "steps_trained": b["end_step"]})
        exec_counters = {k: self.st[k] for k in ("exec_steps", "exec_skipped", "exec_skipped_first10",
                                                 "nonfinite_losses", "real_tokens", "padded_tokens",
                                                 "completion_tokens", "train_s", "outputs")}
        self.load_ckpt(fork_ckpt)
        self.st.update(exec_counters)  # executed-work counters are cumulative; the trajectory is restored
        self.st["branches_done"] = sorted(set(self.st["branches_done"]) | {b["epoch"]})
        self.save_ckpt()
        shutil.rmtree(fork_ckpt, ignore_errors=True)

    def train(self) -> dict:
        self.maybe_resume()
        if self.engine == "unsloth":
            self.pack_gate()
        P = self.plan
        self.guard = G.LossGuard(self.g, P["total"], self.scaler.is_enabled())
        mid = sorted({max(1, int(f * P["total"])) for f in self.g["midrun_fracs"]}) if self.g["enabled"] else []
        if not self.st["resumed_from"]:
            self.probe_micro()
            if self.g["enabled"]:
                self.midrun_check(0)
            else:
                self.dev_loss(0)
        self._last_ckpt_t = time.time()
        every_s = (self.s["ckpt_every_min"] or 0) * 60
        every_n = self.s["ckpt_every_steps"]
        self._save_manifest(status="RUNNING", micro_batch=self.micro)
        while self.st["global_step"] < P["total"]:
            g = self.st["global_step"]
            for b in P["branches"]:
                if b["fork_step"] == g and b["epoch"] not in self.st["branches_done"]:
                    self._run_branch(b)
            row = self.train_step(g, S.lr_mult(g, P["total"], P["warmup"], P["decay"]), tag="main")
            self.st["global_step"] = g + 1
            if self.g["enabled"]:
                if g == 0 and not self.st["resumed_from"]:
                    G.check_step0(row["loss"], self.g)
                tps = None if "max_abs_act" in row else row["real_tokens"] / max(1e-9, row["s"])
                self.guard.update(g, row["loss"], row["gnorm"], row["skipped"], tps)
                if g + 1 in mid and g + 1 < P["total"]:
                    self.midrun_check(g + 1)
            if g % max(1, self.s["log_every"] * 10) == 0:
                print(f"[sft] step {g}/{P['total']} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                      f"gn {row['gnorm']} skip {self.st['exec_skipped']}", flush=True)
            due = ((every_s and time.time() - self._last_ckpt_t >= every_s) or (every_n and (g + 1) % every_n == 0)
                   or (g + 1) % P["steps_per_epoch"] == 0)  # also at every epoch boundary (user rule 2026-11-08)
            if self._budget_left() <= 0:
                self.save_ckpt()
                raise BudgetReached(f"wall-clock budget reached at step {g + 1}/{P['total']}")
            de = self.s["dev_every_steps"]
            if de and (g + 1) % de == 0 and g + 1 < P["total"]:
                self.dev_loss(g + 1)
            if due and self.st["global_step"] < P["total"]:
                self.save_ckpt()
        self._check_skips(final=True)
        path = self._save_adapter(f"epoch{P['epochs']}" if P["total"] == P["steps_per_epoch"] * P["epochs"]
                                  else f"step{P['total']}", {"steps_trained": P["total"]})
        self.save_ckpt(tag="final")
        if self.g["enabled"]:  # QA-T #2: after the adapter + final checkpoint are on disk
            self.midrun_check(P["total"], final=True)
            dev_final = self.last_dev
        else:
            dev_final = self.dev_loss(P["total"])
        n = max(1, self.st["exec_steps"])
        summary = {"steps_total": P["total"], "exec_steps": self.st["exec_steps"],
                   "skipped_steps": self.st["exec_skipped"], "skipped_frac": round(self.st["exec_skipped"] / n, 5),
                   "skipped_first10": self.st["exec_skipped_first10"], "nonfinite_losses": self.st["nonfinite_losses"],
                   "final_scale": self.scaler.get_scale() if self.scaler.is_enabled() else None,
                   "pad_fraction": round(1 - self.st["real_tokens"] / max(1, self.st["padded_tokens"]), 5),
                   "real_tokens_per_s": round(self.st["real_tokens"] / max(1e-9, self.st["train_s"]), 1),
                   "train_s": round(self.st["train_s"], 1), "final_adapter": path, "outputs": self.st["outputs"],
                   "engine": self.engine, "packed": self.packed, "micro_batch": self.micro,
                   "dev_loss_final": dev_final and dev_final["dev_loss"], "max_abs_act": self.st.get("max_abs_act"),
                   "peak_mem_gb": self._peak_mem(),
                   "resumed_from": self.st["resumed_from"]}
        self._save_manifest(status="COMPLETE", summary=summary)
        (self.out / "DONE").write_text(json.dumps(summary["outputs"]), encoding="utf-8")
        return summary


def release(r: "SFTRun") -> None:
    """Free a finished run of a cell and hand its base back LoRA-free: unload the adapter (PEFT ``unload`` restores
    the original Linear modules), drop optimizer / scaler / tensors, gc + empty_cache."""
    import gc
    try:
        import torch
        if getattr(r, "shared", None) is not None and r.shared.get("base") is not None and getattr(r, "model", None):
            base_model = r.model.unload()
            for p in base_model.parameters():
                p.requires_grad_(False)
            if hasattr(base_model, "peft_config"):
                delattr(base_model, "peft_config")
            r.shared["base"] = (base_model, r.shared["base"][1])
        for a in ("model", "opt", "scaler", "items", "dev_items"):
            if hasattr(r, a):
                setattr(r, a, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001  (a broken shared base is dropped; the next arm reloads it)
        print(f"[cell] release failed ({e!r}); the next run reloads the base", flush=True)
        if getattr(r, "shared", None) is not None:
            r.shared["base"] = None


def run_cell(cfgs: list[dict], stop_on_fail: bool = False) -> list[int]:
    """All arms of one (size, lang, seed) cell in ONE process (lead 2026-11-08): the Unsloth import and the base
    load happen once; every arm gets a freshly seeded LoRA, a new optimizer / scaler / schedule and its own RNG
    state (set_seed in setup).  Parity with a fresh process is a test: tests/test_train_cell.py (mocks) and the
    GPU parity check (jobs/t2/cell_parity.json)."""
    shared: dict = {"base": None}
    rcs = []
    for i, cfg in enumerate(cfgs):
        rc = run(cfg, shared=shared)
        rcs.append(rc)
        if stop_on_fail and rc not in (EXIT_OK, EXIT_NEEDS_RESUME):
            # lead: a guard failure in a seed-0 (canary) cell stops its remaining arms -- never burn the rest of the
            # cell on a broken config; the launcher then blocks the language
            for c in cfgs[i + 1:]:
                out = Path(c["out_dir"])
                out.mkdir(parents=True, exist_ok=True)
                _write_json(out / "manifest.json", {**C.identity(c), "status": "CELL_STOPPED",
                                                    "reason": f"{cfg['run_id']} exited {rc}"})
                rcs.append(EXIT_CELL_STOPPED)
            print(f"[cell] STOPPED after {cfg['run_id']} rc={rc}", flush=True)
            break
    return rcs


def run(cfg: dict, t_start: float | None = None, shared: dict | None = None) -> int:
    """Run an SFT config; returns the process exit code (0 / 75 needs-resume / 4 overflow abort / 3 error)."""
    r = SFTRun(cfg, t_start, shared=shared)
    try:
        return _run(r, cfg)
    finally:
        if shared is not None:
            release(r)


def _run(r: "SFTRun", cfg: dict) -> int:
    if (r.out / "DONE").exists() and cfg["resume"] != "never":
        print(f"[sft] {cfg['run_id']} already DONE", flush=True)
        return EXIT_OK
    try:
        r.setup()
        r.train()
        return EXIT_OK
    except BudgetReached as e:
        r._save_manifest(status="NEEDS_RESUME", reason=str(e), counters=r.st)
        print(f"[sft] NEEDS_RESUME: {e}", flush=True)
        return EXIT_NEEDS_RESUME
    except DataError as e:
        r._save_manifest(status="FAILED_DATA", reason=str(e)[:4000])
        print(f"[sft] FAILED_DATA: {e}", flush=True)
        return EXIT_DATA_ERROR
    except G.GuardAbort as e:
        r._save_manifest(status=f"ABORTED_{e.reason}", reason=str(e)[:4000], counters=getattr(r, "st", None))
        print(f"[sft] ABORTED_{e.reason}: {e}", flush=True)
        return EXIT_GUARD_ABORT
    except GateFailed as e:
        r._save_manifest(status="GATE_FAILED", reason=str(e)[:4000])
        print(f"[sft] GATE_FAILED: {e}", flush=True)
        return EXIT_GATE_FAILED
    except TooManySkippedSteps as e:
        r._save_manifest(status="ABORTED_FP16_OVERFLOW", reason=str(e), counters=r.st)
        print(f"[sft] ABORT: {e}", flush=True)
        return EXIT_ABORT_OVERFLOW
    except Exception:
        r._save_manifest(status="FAILED", error=traceback.format_exc()[-6000:])
        if r.shared is not None:   # in a cell: record, drop the (possibly broken) base, continue with the next arm
            r.shared["base"] = None
            traceback.print_exc()
            return 3
        raise


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run(C.load(sys.argv[1])))
