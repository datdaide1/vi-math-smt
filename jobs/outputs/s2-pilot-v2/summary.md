## Environment

```
accelerate                               1.15.0
datasets                                 4.8.5
peft                                     0.21.2
torch                                    2.14.0
transformers                             5.20.2
triton                                   3.8.0
vllm                                     0.33.0
```
env_fix: [('pip uninstall -y -q torchao', 0)]; envcheck rc=0; peft smoke=True; wall_s=1618.3
unsloth_install rc=0

## Per model

| model | 4-shot acc (300) | 95% CI | engine / attn | eval gen tok/s | SFT real tok/s | T4-h / 9M tok | micro×accum, GC | peak GB | pad frac (grouped / naive) | loss first5→last5 | skipped (after step 10) | act absmax (headroom×) | top act key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | — | — | — | — | 2352.4 | 1.063 | 4×8, off | 8.4 | 0.0693 / 0.2732 | 0.695→0.5286 | 0 (0) | 6484.0 (10.1) | layer25.out |
| Qwen3-1.7B-Base | — | — | — | — | 1357.9 | 1.841 | 4×8, off | 13.13 | 0.0693 / 0.2732 | 0.616→0.4617 | 0 (0) | 14360.0 (4.56) | down_proj.in |

## Packing check — Qwen3-0.6B-Base (max |Δlogit| vs separate, all 4 items; items 2–4 in brackets)

| variant | fp32 | fp16 | NLL item2 sep→variant (fp16) |
|---|---|---|---|
| pos_ids_default_cache | 22.954269 [22.954269] fail | 22.984375 [22.984375] | 0.87475→3.70336 |
| pos_ids_no_cache | 0.000342 [0.000342] PASS | 0.119141 [0.119141] | 0.87475→0.8748 |
| block_mask_4d | 0.000342 [0.000342] PASS | 0.119141 [0.119141] | 0.87475→0.8748 |
| padded | 0.000355 [0.000355] PASS | 0.18457 [0.18457] | 0.87475→0.87516 |

## Unsloth — Qwen3-0.6B-Base

libs: {'torch': '2.14.0', 'transformers': '5.20.2', 'peft': '0.21.2', 'trl': '1.17.1', 'unsloth': '2026.11.3', 'unsloth_zoo': '2026.11.3', 'vllm': '0.33.0', 'accelerate': '1.15.0', 'triton': '3.8.0', 'bitsandbytes': '0.50.2', 'datasets': '4.8.5', 'cut_cross_entropy': '25.1.1', 'torch_full': '2.14.0+cu130'}
auto_installed: None; accepted: False
- check fp32: ERROR batch not padding-free (shape [4, 337])
- check fp16: ERROR batch not padding-free (shape [4, 337])
- sft padding_free: tok/s=3054.5 T4-h/9M=0.818 skipped=0 peak=10.86 loss10=[0.661, 0.5928, 0.5592, 0.5626, 0.548, 0.5304, 0.5349, 0.5241, 0.5163, 0.5033]
- sft padded: tok/s=3024.8 T4-h/9M=0.827 skipped=0 peak=10.86 loss10=[0.6609, 0.5927, 0.5592, 0.5627, 0.548, 0.5304, 0.5347, 0.524, 0.5163, 0.5034]

## Grid re-estimate (runs × 9M tokens / measured padded tok/s)

| block | runs | measured tok/s | T4-h/run | T4-h |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 30 | 2352.4 | 1.06 | 32 |
| Qwen3-1.7B-Base | 92 | 1357.9 | 1.84 | 169 |
| **Qwen3 subtotal** | | | | **201** |
