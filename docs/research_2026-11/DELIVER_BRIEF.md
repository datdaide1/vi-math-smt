# DELIVER_BRIEF — hand-off by 2026-11-08 ~20:00 (UTC+7)

Set by the user on 2026-11-07 22:25: deliver the internal hand-off package tomorrow evening. **Scope is not cut.** The
research design (P1–P3, C1–C4, H1–H4, the E1/O3 English mirror) stays as it is. Training compute is what we cannot
finish in time: the full SFT grid is about 634 T4-h and our resources are Kaggle (about 27 h left this week) plus
Modal (about $29). So the package will contain:
1. **Everything that is not compute-bound, finished and frozen:** data, benchmark, code, prompts, judges, docs.
2. **A first training tranche with real numbers:** 0.6B, all arms, both languages as far as compute allows.
3. **The rest of the grid as a runnable, ordered queue,** with measured cost per run. The successor runs it unchanged.

Read before acting: `STATUS.md` (state), `01_TECHNICAL_DESIGN.md` (design), `EXEC_BRIEF2.md` (rules: file ownership,
laptop vs cloud, secrets), `exec/E1_english_mirror.md` (EN mirror), `tech/T1..T6` (algorithms, thresholds).

## Milestones (UTC+7)
| When | What |
|---|---|
| 07/11 24:00 | Arm-Y v3 go/no-go (fix the absurdity rate). N-arm batch infrastructure working end-to-end on 200 items. T1r parity gate done (Kaggle, or Modal if Kaggle is still queued). |
| 08/11 06:00 | VI pools frozen: Y± and N± (20k generated pool per arm, 10k training subsample), decontaminated. Q1 `core_v1` and S3 `pool_v1` frozen. |
| 08/11 09:00 | EN pools frozen (same seeds, same slots). Joint 200-item gate audit per arm, same judge. |
| 08/11 08:00–18:00 | Training tranche: 0.6B VI, all arms; then extra seeds; then EN 0.6B; then 1.7B if quota remains. Eval runs on Modal. |
| 08/11 18:00 | Freeze. Write the hand-off docs, STATUS, ledgers. |
| 08/11 20:00 | Deliver. |

## Hard rules (unchanged)
- Never print key or token values.
- Test and benchmark items go to Claude only, never to Gemini or NIM. Gemini flash-lite is allowed for judging
  generated training data.
- Kaggle kernels and datasets stay private.
- Laptop: at most 300 MB per process, no torch or transformers. Heavy work runs on Modal (GPU) or Kaggle.
- Claude outputs used for training get provenance recorded in the datasheet, and API logs are kept under
  `logs/api/<WS>/`.
- No legacy code reuse.
- Commits are allowed (user, 22:45): logical commits of your own files only, with the trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>". Never commit .env*, data/ or secrets. Never push.
- Every claim in a report needs evidence: file paths, counts, CIs.

## Budgets (hard caps; stop and report if you would exceed one)
- **Claude:** about $43.3 spent as of 22:40 (console-consistent; the earlier $32.5 missed the X2 and X3 logs). Caps by workstream:

  | Workstream | Cap |
  |---|---|
  | G4 → Y v3 + scale | $12 |
  | D1 N-arm | $12 |
  | D5 audit + C3 | $10 |
  | Reserve | about $20 (the lead decides) |

  Use the **Message Batches API (−50%)** for any job with 100 or more calls.
- **Modal:** about $29 left.

  | Workstream | Cap |
  |---|---|
  | T1r/T2 training + eval | $18 |
  | D1 decontam | $1.5 |
  | G4 decontam | $1.5 |
  | Reserve | $8 |

  Use GPU fallback lists, `retries=0`.
- **Kaggle:** at most 20 GPU-h through tomorrow. T4×2 counts once against quota, so run 2 jobs per kernel (1 per GPU).

## Output locations
- Training pools: `data/train/{Y,N}/{vi,en}/` with `pool.jsonl`, `train_10k.jsonl` (for ±: `plus`/`minus`),
  `SHA256SUMS`, `manifest.json`.
- Reports: `exec/<ID>_*.md`.
- Hand-off package: `handoff/` (owned by D4).

## Changes 23:05
- Claude: D1 cap raised to $20 (EN +$8 from reserve), so the reserve is ≈ $12.
- Modal: R-GPU benchmark gets $5. Modal is now allocated: T1r/T2 $18, D1 $1.5, G4 $1.5, R-GPU $5, which leaves ≈ $3 reserve. The platform for comparison blocks will be decided after R-GPU (01:30) and QA-T (03:30).
- Quality gate added: T3 G6 LR × epoch sweep on S0 0.6B tonight before any arm run. QA-T independently audits the training/eval stack.
- Gemini: free tier is 500 requests/day/key **per model** (not per minute). gemini-3.1-flash-lite is partly exhausted until ~15:00 +07; 3.5-flash-lite has quota. Common LanePool now raises LaneStall instead of waiting forever (S3 fix).

## Changes 23:10 (lead decisions)
- Training-row rule, same for both arms: {question, solution, answer}. The solution comes from ONE shared writer
  (`scale_gen.solver`); the answer is the trace's own boxed answer. "+" keeps rows where trace answer == verified
  answer; "−" keeps the trace unchecked. Y files use the N layout.
- One common validity screen for Y+ and N+ (D1's flash-lite screen, or Haiku-v3 ∧ flash-lite if calibration says
  so). No Sonnet gate for Y only, because that would confound the arms.
- n_train per language = min over the 4 arms (≤ 10k). Block if it falls below 8k.
- Claude caps: G4 $18 (since 22:30), D1 $20, D5 $10, Q1 $15.5 total. Remaining margin ≈ $5.

## Changes 23:10 (data size, budget, resilience)
- **n_train = 4,000 per arm per language** (Y+, Y−, N+, N−, S0), the same for every arm. The user said we don't need
  much data, and the Claude budget binds. Train files carry ≥ 4,200 rows in priority order. The 10k/20k top-ups are
  costed for the successor.
- Claude ledger at 23:04: $49.48, plus about $9.1 of D1 batches pending. Caps:

  | Workstream | Cap |
  |---|---|
  | G4+G5 | $21 |
  | D1 | $19 |
  | D5 | $6 |
  | Q1 | $15.5 |

  Expected total ≈ $92, leaving about $8 margin.
- **Resilience (user requirement):** every stage appends results per item, every LLM call is cached, and Batch IDs
  are recorded and collected on restart. Training checkpoints every ≤15 min and at each epoch. Each run writes a
  complete output directory as soon as it finishes. Launcher state is written atomically. Eval preds are written
  per chunk. Nothing gets redone from scratch after an interruption.

## Changes 02:15
- Gemini free quota is exhausted on both flash-lite models and all 6 keys until about 15:00 +07 (per-day,
  per-project, per-model limit).
- Common screen is now **Haiku-v3 only** for both arms. Calibration: 8% [5–12] absurd among accepts, vs 6% [3–11]
  with lite, not significantly different. Lite is recorded as "not applied (quota)".
- The Y cycle check must stay cross-family. Primary is NVIDIA NIM (gemma-4-31b-it / glm-5.3-flash / nemotron),
  after validating ≥ 90% agreement with the Gemini verdicts on ~100 pilot items. Fallback is a self-hosted open model
  on Modal (cap $3). A Haiku cycle is an ablation label only.
- n_train = 3,000; D5 selects with independent S+/S− seed sets.

## Changes 03:30
- Claude ledger at 03:28: $76.4. Caps: D1 $21, G4+G5 $25, D5 $4 (audit via Batch, 100 per arm, VI 4 arms then EN
  plus), Q1 done at $15.36. Expected total ≈ $96.5.
- Minus arms draw only non-adaptive first items (Y wave-1, N j=1). D5's part 1 (S−) is at
  data/train/final/selection_vi_v1.json. S+ is part 2.
- Y cycle model: chunked Qwen3-32B-AWQ on Modal (+$2) if it validates at ≥ 90%, else GLM via NIM. One model for
  all Y items.
- Hard data cutoff 11:30: S+ = seeds with Y+ at that time, n = min(3,000, reachable).
- Platform final: Modal L40S bf16, Unsloth packed mb32, no GPU fallback. Eval also on L40S. 5 seeds per arm at
  0.6B, VI and EN.

## Changes 05:15 (Modal budget binds)
- Modal month-to-date is $21.04 of $30. G5's total Modal is capped at $9.0, with the EN Qwen cycle first.
  **VI wave 2 stopped**: no Modal money for its cycle check. Its renders are kept as a costed successor item.
- Training and eval are capped at $6.5:
  - base rows;
  - VI 0.6B 5 arms × 4 seeds (primary K=4 + greedy, Vi-ExamMath greedy);
  - EN 0.6B 5 arms × 2 seeds.
  VI seed 4 and EN seeds 2–4 are on HOLD for the successor, with costs.
- Measured rates:
  - VI Y+ wave 1: 2,712 seeds, +~110 from retry.
  - EN Y: screen pass 70.6%, cycle pass 97%.
  - p_EN(N | VI N+) = 90.3%.
  - Expected final n ≈ 1,600–2,100.
