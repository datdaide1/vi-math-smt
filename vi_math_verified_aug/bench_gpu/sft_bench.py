"""In-container SFT benchmark driver.  Runs ``vi_math_verified_aug.train.sft.run`` UNCHANGED (same template, masking,
loss, packing, schedule) and applies one optional, recorded variant patch in-process:

    none            the T1r recipe as configured (engine hf | unsloth, bf16)
    unsloth_fp8     FastLanguageModel.from_pretrained(..., load_in_fp8=<RGPU_FP8_MODE or True>) (Unsloth FP8: base
                    weights FP8, LoRA bf16/fp32)
    ao_fp8          engine hf: torchao ``convert_to_float8_training`` on every LoRA ``base_layer`` (dynamic scaling,
                    recipe RGPU_AO_RECIPE = tensorwise | rowwise); LoRA A/B stay fp32/bf16
    ao_fp8_compile  ao_fp8 + ``torch.compile`` of every decoder layer (torchao: float8 needs compile to be fast)
    hf_compile      engine hf, bf16, ``torch.compile`` of every decoder layer (the control for ao_fp8_compile)

    python -m vi_math_verified_aug.bench_gpu.sft_bench --config cfg.json --variant unsloth_fp8 --budget-s 1500

Writes ``<out_dir>/rgpu_variant.json`` (variant, versions, dtype histogram of the frozen base weights).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _dtype_hist(model) -> dict:
    h: dict = {}
    for n, p in model.named_parameters():
        k = f"{'lora' if 'lora_' in n else 'base'}:{p.dtype}"
        h[k] = h.get(k, 0) + p.numel()
    for n, b in model.named_buffers():
        k = f"buf:{b.dtype}"
        h[k] = h.get(k, 0) + b.numel()
    return h


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "transformers", "peft", "unsloth", "unsloth_zoo", "torchao", "triton", "vllm"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def _decoder_layers(model):
    for _, mod in model.named_modules():
        if mod.__class__.__name__.endswith("DecoderLayer"):
            yield mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default="none",
                    choices=["none", "unsloth_fp8", "ao_fp8", "ao_fp8_compile", "hf_compile"])
    ap.add_argument("--budget-s", type=float, default=1800)
    a = ap.parse_args(argv)
    t0 = time.time()
    raw = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if raw.get("engine") == "unsloth":
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")
        import unsloth  # noqa: F401  (before transformers)
    from vi_math_verified_aug.train import config as C
    from vi_math_verified_aug.train import modeling as M
    cfg = C.build(raw, {"budget": {"wall_s": a.budget_s}})
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {"variant": a.variant, "versions": _versions(), "env": {k: v for k, v in os.environ.items()
                                                                          if k.startswith("RGPU_")}}

    def record(**kw):
        info.update(kw)
        (out / "rgpu_variant.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")

    record()
    if os.environ.get("RGPU_PAD16") == "1":  # float8 GEMMs need every dim % 16 == 0 (incl. tokens in grad_weight)
        from vi_math_verified_aug.train import data as D
        orig_collate = D.collate

        def collate16(batch, pad_id):
            L = max(len(it["ids"]) for it in batch)
            L16 = -(-L // 16) * 16
            extra = [{"ids": it["ids"] + [pad_id] * (L16 - len(it["ids"])) if it is batch[0] else it["ids"],
                      "labels": it["labels"] + [-100] * (L16 - len(it["ids"])) if it is batch[0] else it["labels"]}
                     for it in batch]
            b = orig_collate(extra, pad_id)
            if L16 != L:  # the first row carries the extra pad: mask it and do not count it as real
                n0 = len(batch[0]["ids"])
                b["attention_mask"][0] = [1] * n0 + [0] * (L16 - n0)
                b["real_tokens"] = sum(len(it["ids"]) for it in batch)
            return b
        D.collate = collate16
        record(pad16=True)
    if a.variant == "unsloth_fp8":
        from unsloth import FastLanguageModel
        orig = FastLanguageModel.from_pretrained
        mode = os.environ.get("RGPU_FP8_MODE", "true")
        mode_v = True if mode == "true" else mode

        def fp8_from_pretrained(*args, **kw):
            kw["load_in_fp8"] = mode_v
            return orig(*args, **kw)
        FastLanguageModel.from_pretrained = fp8_from_pretrained
        orig_lu = M.load_unsloth

        def lu(*args, **kw):
            m, tok = orig_lu(*args, **kw)
            record(dtype_hist=_dtype_hist(m), fp8_mode=str(mode_v),
                   sample_base_layer=repr(next((mod for n, mod in m.named_modules() if n.endswith("q_proj")), None))[:600])
            return m, tok
        M.load_unsloth = lu
    elif a.variant in ("ao_fp8", "ao_fp8_compile", "hf_compile"):
        orig_add = M.add_lora

        def add(model, *args, **kw):
            import torch
            m = orig_add(model, *args, **kw)
            n_conv = 0
            if a.variant.startswith("ao_fp8"):
                from torchao.float8 import Float8LinearConfig, convert_to_float8_training
                recipe = os.environ.get("RGPU_AO_RECIPE", "tensorwise")
                fcfg = Float8LinearConfig.from_recipe_name(recipe)
                convert_to_float8_training(m, config=fcfg,
                                           module_filter_fn=lambda mod, fqn: fqn.endswith("base_layer"))
                n_conv = sum(1 for _, mod in m.named_modules() if mod.__class__.__name__ == "Float8Linear")
            n_comp = 0
            if a.variant.endswith("compile"):
                torch._dynamo.config.cache_size_limit = 256
                for layer in _decoder_layers(m):
                    layer.compile(dynamic=True)
                    n_comp += 1
            record(dtype_hist=_dtype_hist(m), float8_modules=n_conv, compiled_layers=n_comp)
            return m
        M.add_lora = add
    from vi_math_verified_aug.train.sft import run
    try:
        rc = run(cfg, t0)
    except Exception:  # noqa: BLE001
        record(error=traceback.format_exc()[-4000:])
        traceback.print_exc()
        return 3
    record(rc=rc, wall_s=round(time.time() - t0, 1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
