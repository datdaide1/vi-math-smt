# Tranche adapters — verification

_Generated 2026-11-08 09:26 by `handoff/tools/verify_adapters.py`. Full sha256 of both files per adapter: `handoff/ADAPTERS.sha256`._

Location: `jobs/outputs/t2-modal/<run_id>/adapters/epoch3/` (pulled from the Modal Volume `t1r-eval`, `/train/<run_id>/adapters/epoch3`; git-ignored; shipped in the `runs` bundle). Checks: safetensors header parses and the file size matches it; `adapter_config.json` base model = run manifest model, r = α = 32, LoRA, target modules = manifest. The 11-08 manifests record no final-adapter hash, so these sha256 are the reference from now on.

**25/25 adapters pass.**

| Run | MiB | tensors / dtype | sha256 (first 16) | Verdict |
|---|---|---|---|---|
| `t2-sft-Nm-en-0p6B-s0` | 77.0 | 392 F32 | `d1038586c40f5ad1` | OK (no training-time hash recorded) |
| `t2-sft-Nm-vi-0p6B-s0` | 77.0 | 392 F32 | `d648fb83fccfa48b` | OK (no training-time hash recorded) |
| `t2-sft-Nm-vi-0p6B-s1` | 77.0 | 392 F32 | `e9c61e21cc10dd12` | OK (no training-time hash recorded) |
| `t2-sft-Nm-vi-0p6B-s2` | 77.0 | 392 F32 | `1831b5c98bf94836` | OK (no training-time hash recorded) |
| `t2-sft-Nm-vi-0p6B-s3` | 77.0 | 392 F32 | `00b1f22a9013a166` | OK (no training-time hash recorded) |
| `t2-sft-Np-en-0p6B-s0` | 77.0 | 392 F32 | `be51036a945c4fb7` | OK (no training-time hash recorded) |
| `t2-sft-Np-vi-0p6B-s0` | 77.0 | 392 F32 | `4f437ab8d4f2bc98` | OK (no training-time hash recorded) |
| `t2-sft-Np-vi-0p6B-s1` | 77.0 | 392 F32 | `845681d03916354c` | OK (no training-time hash recorded) |
| `t2-sft-Np-vi-0p6B-s2` | 77.0 | 392 F32 | `bc1feb7c587a8f3a` | OK (no training-time hash recorded) |
| `t2-sft-Np-vi-0p6B-s3` | 77.0 | 392 F32 | `60ac59a013ff9b76` | OK (no training-time hash recorded) |
| `t2-sft-S0-en-0p6B-s0` | 77.0 | 392 F32 | `88f9712067c0a8ff` | OK (no training-time hash recorded) |
| `t2-sft-S0-vi-0p6B-s0` | 77.0 | 392 F32 | `8d96e231d7e7b553` | OK (no training-time hash recorded) |
| `t2-sft-S0-vi-0p6B-s1` | 77.0 | 392 F32 | `0b4c803b5a6028f0` | OK (no training-time hash recorded) |
| `t2-sft-S0-vi-0p6B-s2` | 77.0 | 392 F32 | `1436cd459dad61aa` | OK (no training-time hash recorded) |
| `t2-sft-S0-vi-0p6B-s3` | 77.0 | 392 F32 | `5886be7b90d3b4bb` | OK (no training-time hash recorded) |
| `t2-sft-Ym-en-0p6B-s0` | 77.0 | 392 F32 | `f41c5729806daf4a` | OK (no training-time hash recorded) |
| `t2-sft-Ym-vi-0p6B-s0` | 77.0 | 392 F32 | `344b2cebc8e16a6a` | OK (no training-time hash recorded) |
| `t2-sft-Ym-vi-0p6B-s1` | 77.0 | 392 F32 | `e932d94daa686fd7` | OK (no training-time hash recorded) |
| `t2-sft-Ym-vi-0p6B-s2` | 77.0 | 392 F32 | `6a12c13140160d0d` | OK (no training-time hash recorded) |
| `t2-sft-Ym-vi-0p6B-s3` | 77.0 | 392 F32 | `5c9880b1f82ecdfb` | OK (no training-time hash recorded) |
| `t2-sft-Yp-en-0p6B-s0` | 77.0 | 392 F32 | `566d978d71920915` | OK (no training-time hash recorded) |
| `t2-sft-Yp-vi-0p6B-s0` | 77.0 | 392 F32 | `8a6cea0b27468ccf` | OK (no training-time hash recorded) |
| `t2-sft-Yp-vi-0p6B-s1` | 77.0 | 392 F32 | `1023f7c56945afba` | OK (no training-time hash recorded) |
| `t2-sft-Yp-vi-0p6B-s2` | 77.0 | 392 F32 | `6b394cb4ea9ebba9` | OK (no training-time hash recorded) |
| `t2-sft-Yp-vi-0p6B-s3` | 77.0 | 392 F32 | `543f9c7b27f67b61` | OK (no training-time hash recorded) |
