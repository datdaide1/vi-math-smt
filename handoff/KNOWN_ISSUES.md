# KNOWN_ISSUES — open issues, unfinished work, pending decisions

FINAL FREEZE 2026-11-08 (UTC+7, HEAD `ffa8ae6` + D4's last commit), by D4, from the exec reports of S3, Q1, D1,
G4/G5, D5, C3, R-GPU, QA-T, T1r, T2 and R1 (final). Each item: ID · what · evidence · owner/next step.

**Successor queue — start here (full table with costs and commands: `RUNBOOK.md` §5; order = R1 §6 / D5):**
1. **EN seed 1** (≈ $0.70) — makes H1 testable (U-2).
2. **4-shot evaluation of the 25 SFT adapters**, or a stop-sequence fix for the 0-shot base (U-3) — the only clean
   arm-vs-base comparison (≈ $3–4, or ≈ $0.30).
3. **Length-matched retrain** (U-9; ≈ $2.6 VI + $1.6 EN).
4. **1.7B and a second model family** (U-6).
5. Then: remaining HOLD seeds, H4 yes-vs-no lme4, VI Y wave 2 + cycle, 10k/20k top-ups, GRPO (fp16 + QA-T
   prerequisites), T5 baselines, EN R8 estimate (U-10), grader gaps (§B), X2 segmentation (§C), M-1 (§H).
All Modal items need new credit: November's $30 is spent ($29.43 billed month to date, LEDGER §3).

## A. Unfinished work at hand-off

| ID | What | Evidence | Next step / cost |
|---|---|---|---|
| U-1 | ~~VI seed 3 not evaluated~~ **DONE** at the freeze: all 25 adapters (VI s0–3, EN s0) are evaluated; adapters pulled locally and verified 25/25 (`handoff/ADAPTERS.md`) | `jobs/queue/t2_state.json`, `handoff/MODEL_CARDS/` | — |
| U-2 | **EN has 1 seed** → H1 is provisional (estimate +0.3 [−1.4, +2.0] pp; TOST "equivalent" not claimed). VI seed 4, EN seeds 1–4 and the S0-EN recipe check are HOLD rows | R1 §0 item 4; queue P11/P12 | **EN s1 first** (≈ $0.70), then the rest (≈ $3.53) |
| U-3 | **No clean arm-vs-base contrast.** The 4-shot base uses another prompt format (it beats every 0-shot arm on Core, 33.8 vs 30.3–30.9); the 0-shot base in the arms' template **does not stop** (56.9 % of its Core decodes hit the 1,024 cap; Core 17.8–18.1) | R1 §0 item 6; `jobs/t2/ev-base-qwen3-0.6b-0shot-vi.metrics.json` | 4-shot eval of the adapters (≈ $3–4) or a stop-sequence fix + 0-shot base re-run (≈ $0.30) |
| U-4 | **n = 1,808 per arm**, not 10k/4k/3k: Claude then Modal budget; VI Y wave 2 stopped at 05:13 for lack of Modal money for its cycle check | D5 §2, G5 §0, DELIVER_BRIEF 05:15 | wave 2 + retries → n ≈ 2,800 (≈ $5–7); 10k/20k top-ups costed in RUNBOOK §5 #8 |
| U-5 | **Arm-Y quality gate not met** (final sets, Opus): Y+ VI absurd 12 % [7–20], teacher 66 %; only EN N+ passes all four T1 §2.5 thresholds. Reported, not enforced | `exec/D5_gate_audit.md` §1 | report as a C3 result; the Sonnet screen leg (1 % absurd) was excluded so both "+" arms share one screen |
| U-6 | **Only Qwen3-0.6B, one family.** 1.7B, 4B, LFM2.5-2.6B, T5 baselines (T/T-off/M — data not built) and GRPO 1.7B are queued | R1 caveats; `jobs/queue/t2_queue.totals.json` | RUNBOOK §5 #4, #9, #10 |
| U-7 | ~~GLMM not run~~ **DONE**: lme4 on a private Kaggle CPU kernel (`datuni/d5-r1-lme4`); H4 yes-vs-no sensitivity still bootstrap-only | R1 §1, `data/audit/r1/glmm_out_final/` | re-export with `--binary yes_vs_no` (free) |
| U-8 | Seed pool **PENDING2**: 291 seeds await the stage-2 solve-check batch (`msgbatch_01M9aNn5D9Ny6Ss5sDt9u3VB`); `pool_v1.1` not built. The final sets allow PENDING2 seeds and flag them (19 in S+, 43 in S−) | S3 §3.2, D5 §1 | `seeds.build` collect2 → mta → `freeze --version v1.1` (≤ $0.10) |
| U-9 | **Length confound (S-LEN):** question-word SMD Y vs N 0.46–0.65, aliased with arm (a mediator, not a covariate) | D5 §0, R1 caveats | length-matched retrain (RUNBOOK §5 #3) |
| U-10 | **EN R8 estimate is biased** (EN screened only on S+ seeds, which were selected on VI Y+ passes) | D1 §7 | ≈ $0.40 Haiku-v3 Batch on 1,559 EN N− items |
| U-11 | PII scan of Vi-ExamMath (names, phones) not run | 06 §5.6 D4 | before any release |
| U-12 | Matched-tokens control (T3 §3.2 B) and the LFM2.5 BOS check (QA-T 1.3) not done | T2 §5.4, QA-T §1 | before P10 |
| U-13 | **Core truncation is high** (S0 30 %, arms 10–19 % of greedy decodes hit the cap) and verified arms truncate more than unverified ones at equal accuracy (Y+ 14.2 vs Y− 10.8 %; N+ 19.2 vs N− 10.5 %) | R1 caveats, §5 item 5 | descriptive; look at stored preds |

## B. Grader gaps (X1, grader 3.0.2; found by Q1 — they decide whether a correct answer gets credit on Vi-ExamMath)

| ID | Gap | Example |
|---|---|---|
| G-1 | **Named points**: only bare coordinates parse | `M(0;0), M(2;2)` graded wrong vs set gold `(0; 0); (2; 2)` |
| G-2 | **Compound conditions** → `unparsed-interval` | `0 \le x < 4, x 
e 1` vs `[0,1)\cup(1,4)`; `m 
e 1` vs `(-\infty,1)\cup(1,\infty)` |
| G-3 | **Prose with several labelled values** is read as its last value | "Đội I: 48 ngày; Đội II: 24 ngày" graded **correct** for gold `24` (FP) and **wrong** for tuple gold `48; 24` (FN); `\max A = 16 khi x = y = -1` correct for `-1` |
| G-4 | A runaway continuation after the answer (next fake "Bài toán:") is graded on the last box unless cut | QA-T 1.6 |
Fix in `eval/math_verify.py` (X1 owns it), add cases to `tests/grader_cases/`, then re-score the stored preds with
`python -m vi_math_verified_aug.evalrun.cli rescore …` (no GPU) and re-run R1.

## C. Data defects and provenance notes

| ID | What | Evidence | Next step |
|---|---|---|---|
| X2-SEG | **Sibling sub-questions:** an item's text holds "a) … b) …" while the key covers one part (both audit-#2 errors). Scan of core_v1.1: 6 items with ≥ 2 line-initial labels, all LOW_CONF, 0 VERIFIED; inline / "2.1" labels not caught | Q1 §0 | fix `vi_exam/segment.py`, rebuild, re-key affected items → `core_v1.2` |
| Q1-NV | 894 X2-consensus + 88 "keys agree" items were never validity-checked by an LLM (audit #2 found its 2 errors in that route, 2/82) | Q1 §0 | optional adjudication pass |
| P-1 | `final/manifest_vi.json` records the N-VI pool input as `cde1e8a3…`; D1 rewrote `N/vi/pool.jsonl` at 07:04 (now `73c466cd…`). D4 checked: all 3,616 selected N-VI rows exist in the current pool with identical question/solution/answer/verified | `data/train/final/manifest_vi.json` `inputs` | none; note for provenance |
| P-2 | 74 G5 EN2 rows unchecked by the cycle (Modal cap), `plus=None` | G5 §0 | ≈ $0.1 Modal |
| P-3 | dev300 ∩ generation seeds (QA-T H1) resolved by switching the dev set to `dev_v1` (DEV bucket, never a generation seed); the **Kaggle pilot adapters** saw ≈ 353 dev_v1 seeds and are pilot-only | T1r §1 | never use pilot numbers as comparison rows |
| P-4 | Kaggle T4 pilot runs are a PILOT (R-GPU: the same seed on another GPU flips 11–15 % of dev items) | T1r status box | comparison rows come only from Modal L40S |

## D. Bugs / failing checks found by D4

| ID | What | Evidence | Owner / fix |
|---|---|---|---|
| B-1 | `tests/test_common_prompts_config.py::test_repo_registry_has_all_role_templates` **fails**: it asserts every registry prompt is `language: vi`, but the registry now holds English prompts (`seed_solve_en`, `q1_*`, `validity_judge.v2/v3`, `*_en`) | `handoff/test_report.md` | Y3: relax the assertion to `language in {"vi","en"}`; the prompts are correct |
| B-2 | Seven `tests/test_train_*.py` files **skip entirely** in the repo `.venv` (pyyaml missing → `importorskip`); they pass (81/81, 2026-11-08) under `E:\venv_y1` | `test_report.md` (interpreter column) | fixed by K-ENV-1 (`uv sync` installs pyyaml from the lock) |
| B-3 | `vi_exam_qa/__init__.py` docstring says `keys grade-solve2`; the CLI only accepts `prepare` / `triage` | `vi_exam_qa/keys.py:370` | Q1: fix the docstring |
| B-4 | Light tests importing numpy exceed a 300 MB commit cap unless `OPENBLAS_NUM_THREADS=1` (OpenBLAS per-thread buffers ≈250 MB) | first run of `run_light_tests.py`: test_analysis 290.7 MB, test_decontam 289.7 MB → 95 / 43 MB with 1 thread | set `OPENBLAS_NUM_THREADS=1` for every laptop process (runner does it) |
| B-5 | G1's API ledger timestamps are local time without zone (others are UTC `Z`) | `logs/api/G1/*.jsonl` field `ts` | note only; LEDGER.md first/last columns for G1 are +07 |
| B-6 | STATUS.md's Claude totals (e.g. ≈ $32.5 at 22:35) omitted X2 ($8.84) and X3 ($1.73). **Reconciled at the freeze:** ledger $91.06 vs Anthropic console $91.40 (2026-11-08 09:45 +07; −$0.34, −0.4 %); at 22:40 on 11-07 ledger $43.22 vs console ≈ $43. The console is the source of truth | `handoff/LEDGER.md` §1 | none (within $1) |
| B-7 | Three G2 Gemini quota probes were sent without the logger | G2 report header | free tier, dev prompts only; note in LEDGER |
| B-8 | `logs/api/W5/dev_formalize_test.jsonl` (11 NIM calls, early probe, stage `A.formalize`, items hashed `formalizeA:<hash>`) has no report; its name contains "test" | file listing | owner of W5 (pre-brief probe): confirm the items were train/dev seeds, not test items (rule: test items never to NIM) |
| B-9 | Ledger prompt id `pilot_gen/native_v1` (41 calls) has no registry file (G1 legacy prompt in `pilot_gen/prompts/native_v1.txt`) | `PROMPTS.md` orphan table | note only |

## E. Environment

| ID | What | Next step |
|---|---|---|
| K-ENV-1 | Repo `.venv` drifted from `uv.lock`: `uv sync --locked --all-extras --dry-run` would uninstall 25 and install 33 packages (adds anthropic, math-verify, pyarrow, huggingface-hub, pyyaml; removes pandas, openai, dateutil). `uv lock --check` itself passes, and `requirements.lock.txt` equals `uv export`. Not synced tonight: other agents run from `.venv` | run `uv sync --locked --all-extras` once no agent is using `.venv` (after 11-08 20:00), then rerun `handoff/tools/run_light_tests.py` |
| K-ENV-2 | **Venvs to be removed** after consolidation (do not delete before all workstreams stop): `E:\venv_tools` (anthropic SDK, Kaggle + Modal CLIs, 77 pkgs), `E:\venv_y1` (60 pkgs, **torch 2.14.0+cpu**), `E:\venv_dc` (44 pkgs, **torch 2.14.1+cu126**), `E:\venv_mv` (14), `E:\venv_stats` (21) | move the CLIs to `uv tool install kaggle` / `uv tool install modal`; then delete the five venvs; update README §4 commands from `E:/venv_tools/...` to `.venv` |
| K-ENV-3 | Two transformers pins across Modal images (5.20.2 train/eval; 5.22.2 embedding apps) | unify when touching the images; re-run the eval parity check |
| K-ENV-4 | Unsloth exactness was measured on transformers 5.17.0 (O1); the lock uses 5.20.2 and relies on the per-run logits gate | read each run's `pack_gate.json` |
| K-ENV-5 | Kaggle image drifts weekly (torchao broke peft in v1) | the kernel's ENVCHECK fails fast (exit 5); re-pin weekly |

## F. Repository hygiene

| ID | What | Next step |
|---|---|---|
| R-1 | `src/mutation_informalize/reference/paper_text.txt` (third-party paper text) is now untracked and git-ignored (commit `faa7d8f`) but **remains in history** (first added in `0b6e492`). User decision: no history rewrite | keep in mind before making the repo public: either rewrite history (`git filter-repo --path ... --invert-paths`) or publish a fresh squashed repo |
| R-2 | ~~`.gitignore` gaps~~ **DONE** by the lead: `*.bak`, `*.pem`, `kaggle.json` and the large eval dumps (`jobs/outputs/t2-eval/`, `t2-g6-eval/`, `bench/r_gpu/results/preds/`) are ignored; adapters are ignored via `jobs/outputs/**/adapters/` and `*.safetensors` (checked with `git check-ignore`) | — |
| R-3 | Secrets scan (2026-11-08 08:52, 8,171 files): **0 hits in git-tracked files, 0 verbatim `.env` secrets anywhere** (16 values searched). 683 shape hits: 682 in 341 cached loigiaihay pages (`data/vi_exam/raw/`, ignored) that embed the website's own public browser key (1 distinct value, the site's, not ours); 1 in `logs/api/S3/claude.jsonl` line 1496, inside the base64 `raw_response` blob of a Haiku call (a random substring of an opaque signature, not a key; checked without printing it) | private bundle only (already); never publish raw HTML or raw API logs |
| R-4 | Legacy code (`src/`, `notebooks/`, `vi_math_verified_aug/{verify,ir,informalize,symbolic_aug,data_fix,scripts}`) still sits beside new code | move to `legacy/` in a later cleanup (06 §5.4); new code never imports it |

## G. Decisions pending (user / advisor)

| ID | Decision | Context |
|---|---|---|
| D-1 | **Claude provenance disclosure**: Y/N pools and the models trained on them are Claude-generated (Haiku 5.5; audits Opus 5.5); every release carries this provenance with model ids, prompt hashes and the API-log references | recorded in `DATASHEETS/README.md`, `train_pools.md`; needed before any public release of Y/N pools or models |
| D-2 | Vi-ExamMath license reading (state-issued papers unprotected under Luật SHTT Điều 15(2) + NĐ 17/2023 Điều 8(2)) is **UNVERIFIED** | ask the VNU legal/research office (06 §4.4) |
| D-3 | 2024 / HSG / chuyên coverage gap: default "do not fill now" (adopted 2026-11-07, reversible) | X2 Addendum B lists sources (OCR needed) |
| D-4 | H1–H4 adopted by default when the user said "continue"; H2/H4 wording must be fixed in the pre-registration | Y2 §2, 01 §8b |
| D-5 | Licenses of `madrylab/gsm8k-platinum` and `HuggingFaceH4/MATH-500` to confirm on their cards; origin of the legacy Vietnamese test/train translations (Gemini 2.5 Flash per 06 §4.3) to confirm from Sept-2026 logs | `DATASHEETS/eval_sets_v1.1.md` |
| D-6 | NIM-generated text exists only in the G1 pilot NIM outputs and the legacy SMT-LIB corpus, **not** in the current Y/N pools (Claude) | do not publish NIM-generated text |
| D-7 | How much of the queued grid the successor must run (RUNBOOK §5: ≈ $5 to finish the 0.6B design; ≈ $15 for the 1.7B block; ≈ $31 for 4B/LFM2.5 + 7.4 Kaggle quota-h; ≈ $56 for GRPO 1.7B) and with which budget (the November Modal credit is spent) | RUNBOOK.md §5 |
| D-8 | Deviations to ratify in the pre-registration/paper: Vi-ExamMath greedy-only (T5 asked K = 4); n = 1,808 instead of 10k; Haiku-v3-only "+" screen (Gemini quota); Qwen3-32B-AWQ as the Y cycle model; seed-matched (not slot-paired) selection; Y gate not met | T2 §5.2, D5 §1, G5 §0 | lead / advisor |

## H. Methodological notes (for the paper)

| ID | What | Evidence | Consequence |
|---|---|---|---|
| M-1 | **Dev loss on fixed-style references is not comparable across arms with different solution styles.** dev_v1's references are in the GSM8K/MATH seed style; Y+ (and the shared writer's style generally) trains "**Bước 1:** …" solutions. Training on Y+ RAISED dev_v1 NLL while quick-eval accuracy rose: VI Y+ s0 dev 0.79 → 0.90, accuracy 0.36 → 0.52; EN Y+ s0 dev 0.78 → 0.97, accuracy 0.61 → 0.63 (step 42 of ≈170). The original guard (dev > 2 × train loss) aborted both seed-0 cells (2026-11-08 07:10, ≈ $0.2) | `jobs/logs/T2/t2-sft-Yp-{vi,en}-0p6B-s0.dev_log.jsonl`, `jobs/logs/T2/t2-modal-*-s0.log`; T2 §1 | Arm-level monitoring uses ACCURACY (quick-eval, 64 fixed dev_v1 items), not dev NLL: dev-vs-train gap is a warning; aborts only on dev NLL > 2 × its own step-0 value or non-finite, quick-eval accuracy ≤ 50 % of its step-0 value at 2 consecutive checks, or format collapse (commits `3c32f35` + next). Never compare dev NLL across arms in the paper; report accuracy |
| M-2 | **R8: the ± manipulation is unequal.** First-attempt verification Y 54.5 % VI / 63.4 % EN vs N 80.1 % / 87.1 %; verification removes ~45 % of Y items but ~15 % of N items. H3 averages two very different manipulation strengths; H1's VP is diluted | D5_selection §0, D5_gate_audit §3 | report per arm (Y+ − Y−, N+ − N−) next to H3/H1 |
| M-3 | **S0 is GSM8K-only and hurts MATH and Vi-ExamMath** vs the 4-shot base (VI math −7.8 pp, Core −8.1 pp; format-confounded); the arms' seeds are GSM8K-only too | R1 §0, T2 §4 | always report per dataset; arm − S0 is the training effect of generated data |
| M-4 | **Verification keeps easier seeds** (t0 0.60 vs t2 0.46); matched quotas remove the shift, so H3 = "correctness at matched difficulty" | D5 §1 | state it with H3 |
| M-5 | **Vi-ExamMath greedy-only** in the tranche (K = 1 inside Y2's estimators); primary sets keep pass@1 over K = 4 | T2 §5.2 | deviation from T5 (D-8) |
