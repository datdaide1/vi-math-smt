# QA-T — Independent audit of the training and evaluation stack (2026-11-07, 22:50 → 23:20 +07)

Owner: QA-T (independent reviewer; I did not write the code under review). Code owner: T1r/T2. Scope: everything
from data to tokens, the SFT recipe, the loss, inference/eval, GRPO, guards and reproducibility, as listed in the brief.
Rules followed: read-only on `train/` and `evalrun/`. New files are only `vi_math_verified_aug/qa/*`,
`tests/test_qa_training_audit.py` and the outputs under `jobs/outputs/QA_T/`. No commits. Modal spend ≈ $0.10 [EST]:
one CPU tokenizer run and one T4 NLL probe, retries=0.

Verdicts: **OK**, **RISK** (works, but can bias results or waste compute), **BUG** (wrong), **BLOCKING** (must be fixed
before the 06:00 arm runs). The Status column shows the state at 23:20. T1r and G4 fixed several items while this audit
was running; each fix was re-verified with `tests/test_qa_training_audit.py`.

## 0. Bottom line

1. **The core of the SFT path is correct, and I verified it with the real tokenizers.**
   - Masking, BOS/EOS, the template, the token-mean loss under gradient accumulation and the packed-loss equivalence all
     check out (§1, §3).
   - I decoded real labels with the Qwen3-0.6B-Base and LFM2.5-2.6B-Base tokenizers on a Modal CPU container. The trained
     text is exactly the completion plus EOS, and the prompt is masked.
   - Tokenizing prompt and completion separately gives the same ids as tokenizing them jointly on all 4 sources.
2. **I found three blocking problems and sent each to the lead as soon as I found it. All three are now fixed.**
   - B1 was fixed by G4 and the lead. B2 and B3 were fixed by T1r.
   - B1: the arm-Y files could not be trained (wrong path, no `solution` field, and a different answer rule from N).
   - B2: the 16-item quick-eval guard aborted healthy runs about 43% of the time. Its final check ran *before* the adapter
     was saved, so a run that aborted at the end lost its adapter.
   - B3: a single dropped row in an exactly-K-row arm file crashed setup, and the HF fallback then crashed the same way.
3. **Two high-priority items are still open and the lead must decide them before 06:00.** Both are in §8.
   - (H1) dev300 problems are also generation seeds for N/Y. About 4% of N/Y items are mutations of dev300, so dev-loss
     and quick-eval are biased by arm.
   - (H2) The G6 sweep kernels were built for n = 7,157. The arms now train on n = 4,000, which gives 250 instead of 447
     steps at 2 epochs.
4. **GRPO is not ready.** Its gaps do not affect the 06:00 SFT tranche, but they must be closed before P4/P11 (§5).
   - It is planned in bf16, while T3's own evidence says to use fp16.
   - It has no `vllm_max_model_length`.
   - It has no difficulty-filtered prompt pool and no dev eval during training.
   - The eval of GRPO adapters must use the merged SFT base.

## 1. Data → tokens

| # | Item | Verdict | Evidence | Fix / note | Owner | Status |
|---|---|---|---|---|---|---|
| 1.1 | Template for base models, no chat template | **OK** | `train/template.py:19-23`. VI `Bài toán:\n{q}\n\nLời giải:\n`, EN `Problem:\n{q}\n\nSolution:\n`. The answer line is `Đáp số: \boxed{a}` / `Final answer: \boxed{a}` | — | — | — |
| 1.2 | BOS/EOS, Qwen3 | **OK** | Real tokenizer (`jobs/outputs/QA_T/tokcheck.json`): no BOS. eos = pad = `<|endoftext|>` 151643 = `generation_config.eos_token_id`. EOS is the last trained label. Padding labels are set to -100 explicitly (`data.collate`), so the pad==eos case does not mask the EOS. Unsloth swaps the pad to `<|vision_pad|>` on its own tokenizer (log in `jobs/t1r/train_modal_v1.result.json`), but training uses the AutoTokenizer, so there is no effect | — | — | — |
| 1.3 | BOS/EOS, LFM2.5-2.6B | **RISK** | `bos_token=<|startoftext|>` exists, but `tok("x")` → `[96]`: the pinned stack (transformers 5.20.2) does **not** prepend BOS, so `bos_prefix()` = []. Training and vLLM are consistent with each other, since both get no BOS. The comment in `modeling.py:101` ("tokenizer prepends <|startoftext|>") is wrong for this stack. eos = `<|im_end|>` 124900 = generation config | Before P10, measure LFM2.5 dev NLL with and without BOS. If BOS helps, prepend it explicitly in `encode`, in the eval prompts and in TRL | T1r | open (P10 only) |
| 1.4 | Completion-only masking on real examples | **OK** | Real Qwen3 and LFM2 tokenizers; `guards.preflight_items` passes on S0-VI, S0-EN, dev300 and the N-pilot traces. First trained token = first solution token ("N", "C", "Gi"). Prompt tail = `…\n\nLời giải:\n`. Label fraction 0.23–0.89. Decoded tail: `…Đáp số: \boxed{72}<|endoftext|>`. Joint vs separate tokenisation identical on all 4 sources (`boundary_joint_equals_separate: true`) | — | — | — |
| 1.5 | One final-answer format across SFT targets, few-shot exemplars and eval, VI and EN | **OK after fix** | `completion()` rewrites trailing `####` / `Đáp số:` / `Final answer:` lines into the one answer line. Few-shot exemplars go through the same `completion()`. Before the fix, `**Đáp số:** \boxed{}`, `$\boxed{}$.` and a bare `\boxed{}` tail produced 2 answer lines | Fixed by T1r. Test `test_completion_has_exactly_one_answer_line…` passes. Remaining cosmetic issue: N traces after `clean_trace` end with a bare number line (`…\n\n16000\nĐáp số: \boxed{16000}`, tokcheck) | T1r / D1 | fixed |
| 1.6 | Grader extracts what SFT emits (VI/EN) | **OK** | `qa/grader_edges.py`: 27/27 cases correct, including decimal comma `2,5` and `2{,}5`, grouping `1.500`/`1,500` (flagged ambiguous), `\$18`, `18 đồng`, `\text{ km}`, `20\%`, `\frac`/`\dfrac`/`0.75` ↔ `3/4`, `\sqrt{12}` = `2\sqrt3`, tuples, intervals, U+2212 minus. One open case: a runaway continuation after the answer (next fake "Bài toán:") is graded on the *last* box unless `cut_at_stop` is applied. Eval applies it; GRPO now does too | — | T1r | fixed |
| 1.7 | `max_length` and drop rule; drop-rate difference between arms | **OK / RISK** | 1024 tokens, drop (never truncate), on the pool before the subsample. Measured lengths (Qwen3 tokens, completion p99 / max sequence): S0-VI 263 / 496, S0-EN 228 / 434, N-VI pilot 394 / 646 → 0 drops in every source. **The real confound is length, not drops.** N-pilot completions average **190 tokens vs 115 for S0 (1.65×)** (`nll_probe.json`). With matched items (T3 §3.2 A), N gets about 1.65× more supervised tokens | Report completion-token totals per arm (data_stats has them). The matched-tokens control (T3 §3.2 B) is **not in the queue**: add one run-set (S0 vs best vs worst arm, 3 seeds) | T2 | open |
| 1.8 | n_train > eligible crashes | **BLOCKING → fixed** | `data.subsample` raised when one row was dropped. The run went to FAILED, `on_fail` started the HF fallback, and the fallback crashed the same way. The launcher's `_CharTok` check can't see >1024-token drops | Fixed by T1r: `test_subsample_tolerates_a_few_dropped_rows` passes | T1r | fixed |
| 1.9 | Dedup and train/eval disjointness enforced in code | **OK / RISK** | Exact normalised-question hash vs 11,245 eval hashes (`guards.overlap_with_eval`). Both primary eval files are 1000/1000 covered by the hash file. S0, S0-EN, dev300 and few-shot have 0 overlap; dev300 ∩ S0 = 0; S0 has 0 duplicate questions. Near-duplicates are X5's job (13-gram + embedding). **There is no within-arm duplicate check in the training code** | The contract test `test_arm_training_file_contract` asserts unique questions and ≥ N_TRAIN usable rows per arm file. Run it at 06:00 | QA-T / T2 | test ready |
| 1.10 | dev300 vs generation seeds | **RISK (high)** | dev300 = `vi_gsm8k_train/<i>`. `seed_id(...)` puts **0/300** of these items in the DEV bucket (`% 20 == 0`), the only bucket excluded from generation. **4/203** N-pilot items have `seed_id ∈ dev300`. S0 excludes dev300, but N/Y contain about 4% mutations of it | §8 H1 | lead / D1 / G4 | open |
| 1.11 | Arm-Y data contract | **BLOCKING → fixed** | (a) Y export wrote `plus/train_10k.jsonl`, but the grid read `train_10k_plus.jsonl`. (b) Y rows had **no `solution` field**. (c) N used the trace's own boxed answer, while Y used the program target, so Y− targets could contradict their own trace | G4 rewrote `arm_y/export.py` (shared solver rows, trace answer, N layout). The lead then moved every arm to D5's selector output `data/train/final/{Y,N}/{lang}/train_{plus,minus}.jsonl`, 4,000 rows. **Nothing writes `data/train/final` yet**, so `test_arm_training_file_contract` (8 cases) skips until it lands | G4 / D5 | fixed in code; data pending |

## 2. SFT recipe against the evidence

| # | Item | Verdict | Evidence | Note |
|---|---|---|---|---|
| 2.1 | LR 2e-4, LoRA r=32 α=32 (α/r = 1), all-linear, dropout 0 | **OK as a prior; not yet tuned** | Thinking Machines "LoRA Without Regret": optimal LoRA LR is "consistently 10x the one used for FullFT". It is approximately independent of rank under their 1/r parametrization with **α = 32**. Short runs prefer about **15×**. They found "attention-only LoRA significantly underperforms MLP-only", so all-linear is right. Unsloth guide: start at **2e-4**, r 16 or 32, α = r or 2r, dropout 0, 1–3 epochs. Our setting matches their parametrization exactly (α=32, r=32), so 2e-4 ≈ 10× a 2e-5 full-FT LR | The sweep {1e-4, 2e-4, 4e-4} brackets the prior. **The LR is swept only at 0.6B**, but the FullFT optimum falls with model size → 1.7B (P5/P6) inherits an untuned LR. See §7 |
| 2.2 | WSD 3% warm-up, 10% linear decay to 0 | **OK** | `schedule.py`. Decay multipliers are D/(D+1)…1/(D+1). Branch plan checked by hand for the new shape (n=4,000, 125 steps/epoch, 2 epochs, branch [1]): W=8, fork 113, end 125, main decay 225–249. LoRA Without Regret used a constant LR, so WSD is a defensible choice, not a borrowed default | Branch fidelity is G6 (§7) |
| 2.3 | 32 sequences per step | **OK** | LoRA w/o Regret: LoRA is "less tolerant of large batch sizes". 32 is moderate. Unsloth's suggested effective batch is 16 | — |
| 2.4 | Epochs | **RISK → mitigated** | Fixed at 2 with no sweep result yet. T1r adopted my suggestion `branch_epochs: [1]` (`launcher.py:166`): every arm run now also yields an epoch-1 adapter for about +5% compute | The launcher evaluates only `adapters/epoch2` (`ad[-1]`). Evaluating epoch1 is a separate, cheap ($0.10) job, needed only if G6 picks 1 epoch |
| 2.5 | Grad-clip 1.0 after unscale; weight decay 0; AdamW fp32 states | **OK** | `sft.py:train_step`. Unsloth suggests wd 0.01, but wd 0 is the T3 choice (removes a confound). Smoke step-0 gnorm 0.76 → clip rarely active | — |
| 2.6 | fp16 + GradScaler | **OK / RISK** | init scale 65,536, skip = scale decreased, abort at >1% skipped after 50 steps. S2 measured: 0 skips after step 10, max \|act\| 6,484 (0.6B) / 14,360 (1.7B, 4.6× headroom). **The activation monitor is blind on the Unsloth engine** (`max_abs_act: 0.0` in `t1r-smoke-v1/smoke-uns-*/train_log.jsonl`, because the hooks never fire on fused layers). Overflow is still caught through the non-finite loss / skip guards | For 1.7B Unsloth, accept this, or run one HF fp16 step with `act_monitor_every` in the canary |
| 2.7 | Matched items vs matched tokens (T3 §3.2) | **RISK** | Option A (matched items) is implemented and gives an identical step count. N traces are 1.65× longer than S0 (1.7). The A-vs-B control run-set is not in `t2_queue.jsonl` | Add the control. Report tokens per arm |
| 2.8 | Seeds: LoRA init, data order, subsample | **OK / RISK** | One seed drives `torch.manual_seed` before `add_lora` (HF) or `random_state` (Unsloth), `Random("order:seed:epoch")` and `Random("subsample:seed")`. **Asymmetry:** S0 draws 4,000 of 7,157 rows per seed (subsample variance), while the arm files have exactly 4,000 rows (no subsample variance). The HF fallback gets a different LoRA init than Unsloth for the same seed | Report this in the GLMM. If a fallback fires, re-run that run on Unsloth rather than mixing engines inside an arm comparison |

## 3. Loss computation

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 3.1 | Token-mean NLL under gradient accumulation | **OK** | Every micro-batch divides its NLL *sum* by the step's total trained-token count `n_completion_tokens(step_items)` (`sft.py:train_step`, `data.py:226`). The step loss is the exact token mean over 32 sequences, so the classic grad-accumulation normalisation bug is absent. Test `test_step_loss_denominator…`. CE runs in fp32 on completion positions only (`modeling.completion_nll_sum`) |
| 3.2 | Packing / padding | **OK** | Padded: labels -100 on pads. Packed (Unsloth): positions reset per item, every item starts with a masked prompt token, so no target crosses a boundary. The start-of-job gate passed in the smoke and the dry runs: per-item ΔNLL 4e-5–6e-4, fused vs logits 1e-5–3.5e-4 per token. Step-0 loss HF 0.733367 vs Unsloth packed 0.733366 vs padded 0.733322 |
| 3.3 | fp16 loss scaling | **OK** | `scaler.scale(loss/n_comp)` per micro-batch, then `unscale_`, `clip`, `step`, `update`. The OOM probe re-creates the scaler, so the probe never moves the scale |
| 3.4 | Dev-loss definition | **OK** | Token-mean completion NLL of up to 300 dev items through the same engine and batching (`sft.dev_loss`). Base values (my T4 probe, 0.6B fp16): dev300-VI 0.752, dev300-EN 0.746, S0-VI 0.795, S0-EN 0.794, **N-pilot 0.747**; 1.7B: 0.655 / 0.640 / 0.701 / 0.691 / 0.619 |
| 3.5 | Step-0 band and DEV_DIVERGED (dev > 2× train) for LLM-written arms | **OK** | I feared that LLM-written N traces would start with a very low loss and then trip the band or the ratio. Measured: no. N-pilot base NLL is 0.747, the same as S0, so step-0 bands [0.3,1.6] (0.6B) and [0.25,1.5] (1.7B) hold. A 2× dev/train ratio would need train ≈ 0.35, which is below anything S2 saw (0.53 after 100 steps) | 

## 4. Inference / evaluation

| # | Item | Verdict | Evidence | Fix / note | Owner |
|---|---|---|---|---|---|
| 4.1 | vLLM config | **OK** | T=0.6, top-p 0.95, top-k 20, K=4 + greedy, 1,024 cap, stop `\n(\n)Bài toán:` / `\n(\n)Problem:`, 0-shot after SFT, 4-shot (same template) for base rows, `max_model_len` 4096 | — | — |
| 4.2 | 1,024 cap vs T5 D8 (2048, or 4096 if truncation > 1%) | **OK (measured)** | Spot set, same 200 items and seeds at 2048: only **2/800 sampled and 1/200 greedy** answers truncated at 1024 were correct at 2048. Sampled accuracy 0.343 at 1024 vs 0.334 at 2048 (noise). The cap is justified for the base model | Re-check once on an N-arm adapter (longest traces) | T1r |
| 4.3 | LoRA in vLLM | **OK / RISK** | `enable_lora`, `max_lora_rank = max(32, 32)`, one engine with adapters switched by `LoRARequest`. Qwen3 q/k/v/o/gate/up/down are packed-supported. vLLM `Lfm2ForCausalLM` declares `SupportsLoRA` and remaps `.conv.` → `.short_conv.` ([vLLM docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/lfm2/)). The 6-step smoke adapters evaluated in vLLM give outputs different from base (`t1r-smoke-v1/result_v2.json`), but **nobody has checked adapter-vs-PEFT parity**. A silently ignored module would quietly evaluate something close to base | On the first P1 adapter: 100 dev items greedy with HF+PEFT (`HFBackend`) vs vLLM LoRA, with answer agreement ≥ 95%. For LFM2.5 (P10), do the same check or evaluate merged weights | T1r |
| 4.4 | Per-request seeds and determinism | **RISK** | Seeds are sha256(run, dataset, item) and identical across runs. Measured on the same A10, same seeds, 1000-item vs 200-item batch: **only 532/800 sampled and 142/200 greedy responses match in the first 200 characters.** vLLM is not batch-invariant, so seeds do not make runs reproducible, and greedy is not stable across batch composition. The base rows ran on **A10**, while the T2 plan says L4 first | Treat eval as stochastic (fine for mean pass@1 over K=4 × 3 seeds). Pin one GPU type per comparison block, or keep `gpu` as a covariate. Never claim bit-reproducible greedy | T1r / Y2 |
| 4.5 | Repetition loops / truncation | **OK** | `finish_reason=length` → truncated. Loop = a 20-gram repeated ≥ 3× (only checked on truncated rows). Base rates: greedy truncation 4.8% (GSM8K) and 12.3% (MATH), loops 1.7% / 5.7% | — | — |
| 4.6 | Metrics recomputed from preds.jsonl | **OK / BUG (minor)** | `qa/recompute_metrics.py` reproduces `metrics.json` exactly: GSM8K pass@1 0.35000, greedy 0.38167, maj@4 0.42000; MATH 0.33500 / 0.37500 / 0.42000. 5 rows per item, no duplicate `sample_k`. **maj@K tie-break depends on sample order:** 241/600 GSM8K and 182/400 MATH items have a tied modal span, and `Counter.most_common` picks the first inserted. Spans are also voted by surface form (`0.5` ≠ `1/2`) | Score ties fractionally, or cluster spans by grader equivalence. maj@4 is secondary in T5, so this does not block anything. `test_maj_at_k_is_order_invariant` (xfail) | T1r |
| 4.7 | Grader backend consistency | **RISK (low)** | Modal eval uses `math-verify-0.9.0`. The Kaggle quick-eval uses the SymPy fallback (no math-verify in the Kaggle pins). Numeric and int answers go through the exact-Fraction path in both, so only expression golds can differ | The guard only — acceptable | — |
| 4.8 | Eval template language | **OK** | `eval_config` sets `lang = r["lang"]` and the eval set by language. Few-shot prefix and stop strings follow the language | — | — |

## 5. GRPO (not in the 06:00 tranche; must be fixed before P4/P11)

| # | Item | Verdict | Evidence | Fix | Owner |
|---|---|---|---|---|---|
| 5.1 | Precision | **BUG (vs design)** | T2 §4 plans P4 on Modal L4 **bf16**, and `modal_eval.py:113` merges in bf16 by default. T3 §4.1 (citing 2510.26788) requires **fp16 trainer + fp16 vLLM** because "bf16 LoRA-GRPO collapses" | precision fp16 for the merge and for GRPO on every GPU | T1r |
| 5.2 | vLLM colocate memory | **RISK** | `grpo_args` does not set `vllm_max_model_length`. TRL's doc tells you to set it to at least prompt + completion. At the model maximum (about 40k for Qwen3), one sequence needs about 114 KB/token × 40,960 ≈ 4.7 GB of KV (0.6B, fp16). At 0.35 × 16 GB that does not fit beside a 1.7B on T4. `gradient_checkpointing` defaults to True in GRPOConfig (TRL doc), which is fine | `vllm_max_model_length = max_prompt_length + max_completion_length` (1,024) | T1r |
| 5.3 | Loss config | **OK** | `dr_grpo`, β=0, `scale_rewards="none"`, ε 0.2 / 0.28, G=8, 32 prompts/step, `mask_truncated_completions`. TRL defaults differ (dapo / group / mask False), so each is set explicitly. Unknown GRPOConfig fields are refused | — | — |
| 5.4 | Reward | **OK after fix** | `math_verify` reward now grades `cut_at_stop(completion)`, the same view as eval (`reward.py:64`). The random-reward control is Bernoulli(0.5) keyed by (seed, step, idx) and logs `true_accuracy` | — | T1r |
| 5.5 | LR 2e-5 with fresh LoRA α=64 (α/r = 2) | **RISK** | That is effectively about 4e-5 in α=r terms. Unsloth suggests 5e-6 for GRPO. LoRA w/o Regret suggests ≈10× FullFT RL LR (FullFT ~1e-6 → ~1e-5). T3 plans a {1e-5, 4e-5} pilot | Run the 100-step LR pilot before P11 | T2 |
| 5.6 | Prompt pool / signal at 0.6B | **RISK** | P4 smoke uses S0 *training* prompts, which the SFT model has already seen, so there will be many all-correct groups. T3 §4.1 requires a fixed ~4k pool filtered by S0-SFT pass@8 ∈ [1/8, 7/8]. **No code implements that filter**, and nothing logs the zero-variance-group fraction | Build the pool and log TRL's `frac_reward_zero_std`. For P4 the smoke is acceptable | T2 |
| 5.7 | 512 cap (train) vs 1,024 (eval) | **RISK** | N-arm SFT traces are longer (p99 394 at SFT, more at T=1.0), so `mask_truncated` removes more of N's rollouts than S0's. That makes the effective batch arm-dependent | Log truncation rate per arm, or use 768 | T2 |
| 5.8 | Dev eval during GRPO | **missing** | `grpo.dev_eval_every` is in the config but `grpo.py` never uses it. T3 wants dev eval every 25 steps | Add a callback or evaluate the saved checkpoints | T1r |
| 5.9 | Eval of GRPO adapters | **BUG (latent)** | A GRPO adapter sits on the **merged SFT** model, but `launcher.eval_config` sets only `adapter_path`, so `model_path` falls back to the base | `eval.model_path = merged SFT dir` for kind=grpo | T1r |
| 5.10 | LFM2 + TRL tokenisation | **OK** | There is no BOS anyway on the pinned stack (1.3), so TRL's plain-text tokenisation matches SFT | — | — |

## 6. Guards (lead's fail-fast list)

| Guard | Exists | Works | Evidence / gap |
|---|---|---|---|
| Pre-flight decoded labels + template round-trip + label fraction + EOS + answer present | ✅ | ✅ | Ran it myself with the real tokenizers (1.4) |
| Data sha256 / rows vs SHA256SUMS; exact eval overlap | ✅ | ✅ | `guards.check_dataset`, `overlap_with_eval`; hashes cover 100% of both primary sets |
| Step-0 loss band | ✅ | ✅ | Bands contain every measured base NLL (3.4–3.5) |
| In-loop: non-finite, spike 3× median, collapse < 0.05, no-progress at step 100, throughput < 0.5× | ✅ | ✅ | Note: the 80-step canary never reaches the step-100 no-progress check |
| Mid-run dev + quick eval | ✅ | **was BLOCKING → fixed** | 16 items with a 0.10 drop rule had P(trip) = 0.25 per check and **0.43 per run** for two equal-accuracy decodes (`qa/quick_eval_noise.py` on real preds). It is now 64 items, accuracy drop is a warning, aborts happen only on format collapse (repetition / no-answer / 0 correct), and the end check runs after the save |
| Modal canary per new config (80 steps, fp16 T4) | ✅ | ✅ | `launcher.launch_canaries`; Kaggle runs are released only after a clean canary |
| Post-run checker that blocks the queue | ✅ | ✅ | `check_runs` → `launcher` sets `blocking` until `ack` |
| **Missing / to add** | — | — | (a) Duplicate questions inside an arm file → `test_arm_training_file_contract`. (b) Code-bundle hash in the manifests (§7). (c) An "adapter actually applied" check in eval (4.3). (d) **Unsloth `for_inference` → `for_training` round trip after the step-0 quick eval:** verify in the canary that steps 0–5 reproduce the smoke trajectory (0.7334, 0.6388, 0.6195, 0.5553 … on the 256-item smoke config) and that the step after each mid-run check is not slower. (e) The HF engine generates in `train()` mode: with grad-ckpt on (1.7B HF fallback) the cache is disabled and quick-eval gets slow. Call `model.eval()` around `generate`. (f) GRPO has no guards at all (5.8) |

## 7. Reproducibility and hand-off

| Item | Verdict | Evidence / fix |
|---|---|---|
| Pinned envs | **OK** | `requirements-kaggle.txt` (torch 2.14.0 image, transformers 5.20.2, peft 0.21.2, vllm 0.33.0, Unsloth 2026.11.3 isolated, trl 1.17.1) plus ENVCHECK. The Modal image pins the same (`modal_eval.py:36-39`); `pyyaml` is unpinned (harmless) |
| Config hash | **OK** | sha1 of the canonical config without run_id/paths/wall budget. Checkpoints refuse a different hash or data fingerprint on resume |
| Data sha256 → run | **OK** | `expected_sha256` from the producer's SHA256SUMS, verified in-kernel (`PREFLIGHT_DATA_SHA`). Eval rows carry `run_id`, sampling seed, grader version, backend and GPU |
| Code identity | **RISK** | Kernels are built from the **uncommitted working tree** (`render.bundle_code`), and nothing records a git SHA or a bundle hash. Several fixes in this audit landed tonight as uncommitted edits. Fix: write `sha256(code bundle)` plus `git rev-parse HEAD` / `git status --porcelain` into `job.json` and each manifest |

## 8. Open items for the lead (decide before 06:00)

- **H1 dev300 contamination (1.10).**
  - Preferred fix: rebuild the guard/dev set from the DEV-seed bucket (`seed_id % 20 == 0`, ~373 GSM8K-train items that
    no generator ever used).
  - Alternatives: drop the ~4% of arm rows whose `seed_id ∈ dev300`, or use dev300 only for the S0 sweep and never to
    compare arms.
- **H2 sweep at n = 4,000.**
  - The built sweep (`jobs/build/t2-sweep-{a,b}`) has `n_train: null` (7,157).
  - Rebuild with `budget.n_train = 4000` if it is not pushed yet. Otherwise add one fresh 2-epoch run at n = 4,000 at
    the chosen LR before freezing.
- **Data contract at 06:00.**
  - Run `pytest tests/test_qa_training_audit.py -k contract` once `data/train/final/*` lands.
  - It checks that question/solution/answer are present with zero malformed rows, at least 4,000 usable rows, a
    SHA256SUMS entry, and no duplicate questions.

## 9. Top-5 decisions the LR × epoch sweep (G6) must settle

1. **LR, chosen on accuracy rather than loss, with the rule fixed in advance.**
   - Pick from {1e-4, 2e-4, 4e-4} by dev mean pass@1 (K=4) plus greedy, on a dev set that no generator used (H1). Never
     use `vi_primary_1000`, which is the test set.
   - Rule: take the best cell. If it is within the paired-bootstrap noise of a cheaper cell (lower LR, or fewer epochs),
     take the cheaper cell. Dev NLL is only a tie-break.
   - A single seed cannot separate cells closer than about 3 pt at n = 300 (S2: paired 95% CI ±3–4 pt). Use at least
     600 items, or 2 seeds for the top two cells.
2. **Epochs at the arms' real shape.**
   - Choose 1 vs 2 vs 3 at n = 4,000 and 32 seq/step (125 steps/epoch), not at 7,157 (H2).
   - Every arm run now carries an epoch-1 branch, so a "1 epoch" outcome needs only extra evals ($0.10 each).
3. **WSD branch fidelity, with a test that has enough power.**
   - The ±1 pt accuracy criterion in T3 §6 G6 is below the noise floor for one seed and 300 items.
   - Use dev NLL difference < 0.01 nats/token as the primary criterion (it is deterministic). Add accuracy with a paired CI
     that must include 0.
4. **Does the S0 optimum transfer to the arms?**
   - N traces are 1.65× longer and LLM-styled, and Y renders differ again.
   - At the chosen LR, compare dev accuracy of epoch-1 vs epoch-2 (both free from the branch) on one N and one Y arm
     (P1 seed 0). If the best epoch differs from S0's, say so in the paper rather than re-tuning per arm (T3 "frozen for
     every arm").
5. **Whether the LR transfers to 1.7B.**
   - LoRA w/o Regret ties the LoRA LR to about 10× the FullFT LR, and the FullFT optimum falls with model size.
   - Before P5, run {1e-4, 2e-4} on S0 at 1.7B with a 1-epoch branch, or record "LR tuned at 0.6B only" as a limitation
     in the hand-off.

## 10. Artifacts

- `vi_math_verified_aug/qa/recompute_metrics.py`: an independent recomputation of metrics from preds.jsonl.
- `vi_math_verified_aug/qa/grader_edges.py`: 27 VI/EN answer-format cases plus the runaway-continuation case.
- `vi_math_verified_aug/qa/quick_eval_noise.py`: the quick-eval false-abort simulator on real preds.
- `vi_math_verified_aug/qa/modal_tokcheck.py` → `jobs/outputs/QA_T/tokcheck.json`: real-tokenizer label decode, BOS/EOS
  and length quantiles.
- `vi_math_verified_aug/qa/modal_nll_probe.py` → `jobs/outputs/QA_T/nll_probe.json`: base completion NLL per source
  (0.6B and 1.7B, T4 fp16).
- `tests/test_qa_training_audit.py`: 7 pass, 1 xfail (maj@K), 8 data-contract cases that skip until
  `data/train/final` lands.

Sources opened: [LoRA Without Regret (Thinking Machines)](https://thinkingmachines.ai/blog/lora/) ·
[Unsloth LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) ·
[TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) ·
[vLLM lfm2 model API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/lfm2/)
