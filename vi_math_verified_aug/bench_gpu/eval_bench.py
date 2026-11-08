"""In-container vLLM eval driver: ``evalrun.runner.run`` UNCHANGED, one engine for a block of configs, with an
optional quantisation of the base weights (``--quant fp8`` = vLLM dynamic FP8 W8A8, per-tensor weight scales,
dynamic per-token activation scales; LoRA adapters stay bf16).

    python -m vi_math_verified_aug.bench_gpu.eval_bench --spec spec.json [--quant fp8]

spec = {"base": HF id, "tag": str, "configs": [eval config dicts with /paths already resolved]}.
Writes ``<tag>.json`` next to the spec: gpu, engine load s, per-config metrics + throughput + wall s.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    import vllm
    orig = vllm.LLM
    if a.quant:
        def LLM(*args, **kw):  # noqa: N802
            kw["quantization"] = a.quant
            return orig(*args, **kw)
        vllm.LLM = LLM
    from vi_math_verified_aug.evalrun import runner as RN
    from vi_math_verified_aug.train import config as C
    gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True,
                         text=True).stdout.strip()
    cfgs = [C.build(raw) for raw in spec["configs"]]
    any_lora = any(c["eval"]["adapter_path"] for c in cfgs)
    t = time.time()
    backend = RN.VLLMBackend(cfgs[0], spec["base"], None, enable_lora=any_lora)
    res = {"gpu": gpu, "quant": a.quant, "engine_load_s": round(time.time() - t, 1), "vllm": vllm.__version__,
           "attn_backend": backend.attn_backend, "runs": {}}
    try:
        res["quant_method"] = str(backend.llm.llm_engine.model_config.quantization)
    except Exception:  # noqa: BLE001
        pass
    for cfg in cfgs:
        t = time.time()
        rc = RN.run(cfg, backend=backend)
        man = json.loads(Path(cfg["out_dir"], "manifest.json").read_text(encoding="utf-8"))
        res["runs"][cfg["run_id"]] = {"rc": rc, "s": round(time.time() - t, 1), "out_dir": cfg["out_dir"],
                                      "metrics": man.get("metrics"), "throughput": man.get("throughput")}
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
