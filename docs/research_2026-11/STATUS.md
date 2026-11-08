# STATUS — resume point (written 2026-11-07 ~17:50, before the lead's usage quota ran out)

Read this first when resuming. Design = `01_TECHNICAL_DESIGN.md`; rules = `EXEC_BRIEF2.md`; all reports in
`exec/`. Nothing is committed yet (branch `feat/sprint0`); all work is in the working tree.

## Decisions made by the user (2026-11-07)
- Topic: **symbolic data × Vietnamese math**; problems P1 faithfulness gap, P2 verification premium × language
  (English mirror = `exec/E1_english_mirror.md`, option O3), P3 coverage. Q2 target, no scope cut.
- No reuse of old code (legacy = reference only). Optimization only to make experiments runnable.
- Human raters → LLM judges. **NIM off the critical path.** Students = **Qwen3-Base** (pilot v1).
- Resources: Claude API $100 (+workspace header), Gemini pool 6 keys = 6 projects (flash-lite only — 3.x-flash
  quotas are tiny), Kaggle T4×2 30 h/week, Modal $30/month (GPU works; choose GPU by perf/$, avoid CPU-only).
- Laptop = code + light tests only (RAM hit 0.02 GB once).

## Workstreams
| WS | State | Output / next |
|---|---|---|
| G1 generator pilot | DONE | `exec/G1_generator_pilot.md`: Haiku 5.5 best for native gen / formalize / edit / solve; Opus 5.5 only reliable judge. No arm meets the quality gate yet; Opus-judge-everything ≈ $345 per 10k |
| G2 judge-bias + prompt iteration | RUNNING | self-preference check with non-Claude judge, cheap-judge cascade vs Opus, native prompt variants on 40 new seeds → `exec/G2_judge_prompt.md` (Claude cap $8) |
| E1 English mirror | DONE | O3 chosen (Y±/N± at 0.6B/1.7B, 5 seeds, 10k) |
| S2 student pilot v2 | RUNNING (Kaggle `datuni/s2-pilot-v1` RUNNING; quota ≈1.4/30 h) | its Modal app `s2-student-pilot` was crash-looping (`IndexError(2)`) — see Open issues |
| X1 grader | DONE v3.0.2 | `exec/X1_grader.md`; self-match 100% on all frozen sets; supports `accept_also` |
| X2 Vi-ExamMath collection | RUNNING | → `exec/X2_benchmark.md` (target Core ≥ 500) |
| X3 eval sets | DONE v1.1 | `data/eval/` + `exec/X3_eval_sets.md`; primary_1000 frozen; 11 items still flagged |
| X4 symbolic library | report written (`exec/X4_symbolic.md`), handback pending | typed DAG IR, M1–M6, answer-tail rule |
| X5 decontamination | RUNNING | n-gram/MinHash local; embeddings on Modal (bge-m3, A10) → `exec/X5_decontam.md` |
| Y1 train/eval code | DONE (mock-tested only) | `exec/Y1_train_eval.md`; needs S2 pins + first real GPU run |
| Y2 analysis pipeline | report written, Modal job pending | one old Modal app crash-looping — see Open issues |
| Y3 common infra | DONE | `exec/Y3_common.md`; `uv sync` not run yet (would rewrite shared .venv) |
| O1 throughput optimization | RUNNING (timeboxed ~90 min, Modal ≤ $4) | → `exec/O1_optimization.md` |

## Update 21:02 (after usage-limit reset)
- DONE since 17:50: X2 (1,585 Core, 385 three-key agree, QA queue frozen; 2024/HSG gap — user decision
  pending, recommended: don't fill now), X4 (210 tests; old operators 1008/1008 rejected), O1 (eval on Modal
  L4 ≈6× T4; Unsloth 1.74× T4 / 2.4× L4 with exact packing; pilot-v1 leak = `use_cache=True`; revised plan
  ≈8.7 Kaggle weeks + ≈$30 Modal with GRPO at 0.6B), X5 main (bge-m3 thresholds 0.85/0.80; `vinhnq29` holds a
  re-translated GSM8K test; 336 MetaMathQA-vi near-twins of Vi-MATH test; 33 Vi-ExamMath items flagged).
- Cut off by the usage limit and resumed at 21:02: S2 (Kaggle v2 COMPLETE, quota 2.52/30 h), X5 (final
  flags), G2, G3, Y2.

## Update ~22:00
- DONE: Y2 (H1–H4 proposed, needs user ratification), G2 (no Opus self-preference; cascade ≈$31/10k;
  native prompt v3 passes gate on n=40), S2 (pins; T4 fp16 tok/s; LFM2.5-2.6B 2nd family; SFT ≈634 T4-h),
  G3 (arm Y fails gate: faithful 66/66, teacher 35%, absurd 12%; seeds only 17% teacher-acceptable).
- RUNNING: G4 — arm-Y renderer v2 (edit+fluency pass vs free render from IR), paired on G3's 60 seeds.
  Q1 — Vi-ExamMath key adjudication (631 conflict/<2-key Core items, Opus, 150-item audit) → core_v1.
  T1r — apply S2/O1 findings to Y1 code + first real runs (Kaggle 0.6B parity gate + S0 SFT, LFM2.5 smoke,
  Modal L4 eval on primary_1000). S3 — clean frozen seed pool (repairs, typed gold, decontam, translationese).
- Adopted defaults (user said "continue"): H1–H4 ratified; 2024/HSG gap not filled now (01 §8b).
- NEXT: 200-item gate audit for both arms with the chosen renderer/prompt; then start generation at scale.
- Pending user answers: ratify H1–H4; don't fill 2024/HSG gap now (recommended).

## Open issues (need action on resume)
1. ~~Modal crash loop~~ RESOLVED: both crash-looping apps (`ap-9fjZI5…` s2-student-pilot,
   `ap-rFfLlU…` y2-analysis) were stopped at 17:49; S2/Y2 told to fix `IndexError(2)` and use `retries=0`
   before relaunching. `modal app stop` is now allowed in `.claude/settings.local.json` (user edit, which also
   fixed a stray `]` that made that file invalid JSON).
2. Modal L4 capacity is scarce ("waiting to be scheduled on a GPU_L4 worker") — use GPU fallback lists
   (e.g. `["L4","A10G","L40S"]`).
3. Modal spend so far: metered $0.72, billed $0.00 (2026-11-07 17:50).
4. Claude spend so far ≈ $13 of $100 (G1 7.55, X3 1.73, probes, X2/G2 ongoing).
5. Venv sprawl (`E:\venv_tools`, `E:\venv_mv`, `E:\venv_y1`, `E:\venv_dc`, …) → consolidate on Y3's `uv.lock`.
6. Pending user decisions: remove third-party `paper_text.txt` from the tree / history rewrite.

## Next steps after the running items report
1. G2 → fix generation prompt + judge cascade; re-run quality gate. 2. X4 → real edit lists → re-test arm-Y
editor (G1 got 21–26% acceptable with simplified lists). 3. S2 + O1 → pin Kaggle env, pick training configs,
first real SFT run with Y1 code. 4. X2 → benchmark QA (3-key protocol). 5. Commit everything in logical commits.

Monitoring: `jobs/watch_progress.ps1` (terminal tab), Kaggle CLI `E:\venv_tools\Scripts\kaggle.exe`,
Modal CLI `E:\venv_tools\Scripts\modal.exe`.

## Update 22:35 — deadline set: hand-off 2026-11-08 ~20:00 (+07)
- Plan + budgets + milestones: `DELIVER_BRIEF.md`. Claude spent ≈ $43.3 (G1 7.55, G2 6.5, G3 4.2, G4 4.15, Q1 9.65, S3 0.6, X2 8.84, X3 1.73; console ≈ $43).
- First real eval: Qwen3-0.6B-Base 4-shot, primary_1000 (Modal): GSM8K-Platinum pass@1 .35 / greedy .38; MATH alg+prealg pass@1 .335.
- G4 interim: R1 teacher 61% / absurd 21% (N 80% / 5%) → absurd story content is the blocker.
- Launched: D1 (N arm scale VI+EN), D4 (hand-off package), D5 (C3 attribution + joint 200-item gate audit).
  Follow-ups sent: G4 → Y v3 (semantic constraints M3/M4, drop M6, cheap absurd gate) + Y scale; T1r → Modal fallback if
  Kaggle still QUEUED at 22:45 + T2 training tranche launcher + full-grid queue.
- Still running: Q1 (core_v1), S3 (pool_v1). Monitor: task blplpsfyk.

## Update 23:25 — RESUME POINT (usage limit hit; agents left running on the grace allowance and may be cut off mid-step — all stages are resumable)
Decisions in force: see DELIVER_BRIEF.md "Changes" sections.
- n_train = **3,000** per arm per language.
- One common screen for both arms (Haiku-v3 ∧ flash-lite).
- Shared solution writer (`scale_gen.solver`).
- Final files come from D5's selector, with S+/S− seed sets (independent mode).
- Platform pending R-GPU: Modal **L40S bf16, Unsloth packed, mb32** (36.3k tok/s). FP8 measured as not viable. GRPO runs in fp16.
- G6 LR × epoch sweep at n=3,000 on L40S, selected on dev_v1 accuracy.
- Dev set comes from the DEV bucket.
- QA-T: 3 blockers fixed; lower-risk items were sent to T1r (eval GPU pin, maj tie-break, vLLM vs PEFT parity, git sha in manifests).

State of each workstream when stopped:
| WS | State | Resume |
|---|---|---|
| S3 | seed solve-check rerun 3 (12 Gemini lanes, hang fixed) | freeze `data/seeds/pool_v1.jsonl` + `dev_v1.jsonl` |
| D1 | N-VI generation, 13.5k slots, Batch (IDs in data/train/N/_work/batches.jsonl) | collect batches, screen to 7k passes, pool.jsonl; EN on D5's list |
| G5 | Y-VI scale, breadth-first, cap $25 | ≥ 3,150 distinct seeds with Y+; EN on D5's list |
| D5 | C3 done; selector select_v1 (S+/S−) being built | selection_vi_v1.json → D1/G5 EN; final files in data/train/final |
| Q1 | adjudicating low-confidence items (round 4), cap $15.5 | core_v1.1 + new 150-item audit |
| T1r | Kaggle t1r-runs-v1 RUNNING (S0 + parity, T4); QA fixes in progress | switch launcher to Modal L40S primary; sweep; P1 after R-GPU + data |
| R-GPU | benchmarks done except the mb32 vs mb8 quality check | write exec/R_GPU_platform.md |
| D4 | skeleton done | finalize at 18:00 freeze |
Background API batches and scripts may still be running; all are resumable (per-item append, Batch IDs recorded).
Claude ledger ≈ $49.5 at 23:04 plus pending batches; caps G4+G5 $25, D1 $19, D5 $6, Q1 $15.5.

## FINAL — delivered 2026-11-08 ~09:50 (+07)
- Bundle: `E:\MATH-REASONING-THESIS\handoff_bundle_20261108\`, 3 zips (data 232 MiB, runs 1.8 GiB incl. 25 adapters,
  private 312 MiB, which must NEVER be made public). `sha256sum -c MANIFEST.sha256` passes 3/3. Git HEAD on feat/sprint0
  (not pushed).
- Start here: `handoff/README.md`, then `handoff/RUNBOOK.md` §5 (successor queue with costs), `handoff/KNOWN_ISSUES.md`,
  and `exec/R1_results.md` (results).
- Results (0.6B, VI seeds 0–3, EN seed 0, n=1,808 per arm):
  - Generated data beats the S0 seeds everywhere.
  - H2 and H3 are equivalent within ±3 pp on Vi-ExamMath Core.
  - H1 is provisional (1 EN seed); H4 is underpowered.
- Spend: Claude $91.40 (console), Modal $29.43 of $30, Kaggle 3.43 h.
- Successor queue order: EN seed 1 → 4-shot eval of the adapters → length-matched retrain → 1.7B / second family →
  the rest (RUNBOOK §5).
