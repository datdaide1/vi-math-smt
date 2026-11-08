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
env_fix: [('pip uninstall -y -q torchao', 0)]; envcheck rc=0; peft smoke=True; wall_s=4935.5
unsloth_install rc=0

## Per model

| model | 4-shot acc (300) | 95% CI | engine / attn | eval gen tok/s | SFT real tok/s | T4-h / 9M tok | micro×accum, GC | peak GB | pad frac (grouped / naive) | loss first5→last5 | skipped (after step 10) | act absmax (headroom×) | top act key |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Base | 0.1733 | [0.1347, 0.2202] | vllm/TRITON_ATTN | 1126.0 | 2355.2 | 1.061 | 8×4, off | 13.07 | 0.1038 / 0.2897 | 1.0373→0.3697 | 0 (0) | 12.5 (5253.45) | in_proj.in |
| LFM2.5-2.6B-Base | 0.6033 | [0.547, 0.6571] | vllm/TRITON_ATTN | 952.9 | 692.2 | 3.612 | 8×4, on | 9.85 | 0.1585 / 0.3611 | 0.7841→0.5086 | 1 (0) | 26.9 (2433.11) | in_proj.in |
| Llama-3.2-1B | — | — | no HF access: GatedRepoError: 401 Client Error. (Request ID: Root=1-6aef0225-15504f5e4f3833fe1b69d478;aadf3a37-d3b9-404e | — | no HF access: GatedRepoError: 401 Client Error. (Request ID: Root=1-6aef0225-15504f5e4f3833fe1b69d478;aadf3a37-d3b9-404e | | | | | | | | |
| Qwen3-0.6B-Base | 0.3833 | [0.3301, 0.4395] | vllm/TRITON_ATTN (W2-v1) | 398.0 | 735.2 | 3.4 | 1×32, off | 3.41 | 0.0 / 0.0 | 0.695→0.5283 | 0 (0) | 6488.0 (10.1) | layer25.out |
| Qwen3-1.7B-Base | 0.5867 | [0.5302, 0.641] | vllm/TRITON_ATTN (W2-v1) | 364.7 | 1383.5 | 1.807 | 4×8, off | 13.25 | 0.0693 / 0.2732 | 0.6159→0.4619 | 0 (0) | 14360.0 (4.56) | down_proj.in |
| Qwen3-4B-Base | 0.7833 | [0.7333, 0.8262] | vllm/TRITON_ATTN | 222.4 | 425.7 | 5.873 | 8×4, on | 14.24 | 0.1557 / 0.358 | 0.5394→0.3998 | 0 (0) | 7112.0 (9.21) | layer19.out |
| SmolLM3-3B-Base | 0.52 | [0.4636, 0.5759] | vllm/TRITON_ATTN | 416.3 | 598.8 | 4.175 | 8×4, on | 10.87 | 0.1616 / 0.3634 | 0.8359→0.5473 | 0 (0) | 593.0 (110.46) | down_proj.in |
| gemma-4-E2B | 0.2233 | [0.1799, 0.2738] | hf_generate | 52.0 | 437.7 | 5.712 | 2×16, on | 13.23 | 0.0242 / 0.1586 | 1.7571→0.512 | 4 (0) | 1554.0 (42.15) | per_layer_input_gate.in |
<!-- gemma-4-E2B eval_vllm0: RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} -->

## Packing check — Qwen3-0.6B-Base (max |Δlogit| vs separate, all 4 items; items 2–4 in brackets)

| variant | fp32 | fp16 | NLL item2 sep→variant (fp16) |
|---|---|---|---|
| pos_ids_default_cache | 22.954269 [22.954269] fail | 22.984375 [22.984375] | 0.87475→3.70336 |
| pos_ids_no_cache | 0.000342 [0.000342] PASS | 0.119141 [0.119141] | 0.87475→0.8748 |
| block_mask_4d | 0.000342 [0.000342] PASS | 0.119141 [0.119141] | 0.87475→0.8748 |
| padded | 0.000355 [0.000355] PASS | 0.18457 [0.18457] | 0.87475→0.87516 |

## Unsloth — Qwen3-0.6B-Base

libs: {'torch': '2.14.0', 'transformers': '5.20.2', 'peft': '0.21.2', 'trl': '1.17.1', 'unsloth': '2026.11.3', 'unsloth_zoo': '2026.11.3', 'vllm': '0.33.0', 'accelerate': '1.15.0', 'triton': '3.8.0', 'datasets': '4.8.5', 'cut_cross_entropy': '25.1.1', 'torch_full': '2.14.0+cu130'}
auto_installed: None; accepted: False
- check fp32: ERROR ImportError: Unsloth: Please install bitsandbytes via `pip install bitsandbytes`
- check fp16: ERROR ImportError: Unsloth: Please install bitsandbytes via `pip install bitsandbytes`
- sft padding_free: ERROR ImportError: Unsloth: Please install bitsandbytes via `pip install bitsandbytes`
- sft padded: ERROR ImportError: Unsloth: Please install bitsandbytes via `pip install bitsandbytes`

## Grid re-estimate (runs × 9M tokens / measured padded tok/s)

| block | runs | measured tok/s | T4-h/run | T4-h |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 30 | 735.2 | 3.40 | 102 |
| Qwen3-1.7B-Base | 92 | 1383.5 | 1.81 | 166 |
| Qwen3-4B-Base | 30 | 425.7 | 5.87 | 176 |
| **Qwen3 subtotal** | | | | **444** |
| 2nd family = LFM2.5-1.2B-Base | 78 | 2355.2 | 1.06 | 83 (total 527) |
| 2nd family = LFM2.5-2.6B-Base | 78 | 692.2 | 3.61 | 282 (total 726) |
| 2nd family = SmolLM3-3B-Base | 78 | 598.8 | 4.18 | 326 (total 770) |
| 2nd family = gemma-4-E2B | 78 | 437.7 | 5.71 | 446 (total 890) |
