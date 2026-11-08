# vi-math-smt — hand-off package

Internal hand-off of the thesis project **"Symbolic data for Vietnamese math: verified symbolic vs LLM-generated
training data for small LLMs"** (branch `feat/sprint0`, delivered 2026-11-08). This folder is the entry point for the
successor. Read in this order:

1. this README: what the project is, the repo map, where everything lives, how to reproduce each stage;
2. [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md): what is unfinished, broken or awaiting a decision;
3. [`RUNBOOK.md`](RUNBOOK.md): state of the training grid, how to run the rest (Modal L40S), and the **successor queue in
   priority order with costs** (§5);
4. [`ENV.md`](ENV.md): environments (laptop, Kaggle, Modal);
5. [`DATASHEETS/`](DATASHEETS/), [`PROMPTS.md`](PROMPTS.md), [`MODEL_CARDS/`](MODEL_CARDS/INDEX.md),
   [`ADAPTERS.md`](ADAPTERS.md), [`LEDGER.md`](LEDGER.md), [`BUNDLE.md`](BUNDLE.md) (data that is not in git; the
   zips are in `E:\MATH-REASONING-THESIS\handoff_bundle_20261108\`).

Status markers used below: **FROZEN** (versioned, hashed, do not edit), **DONE** (code finished and tested),
**PARTIAL** (some of the planned runs exist), **QUEUED** (successor runs it from the queue; RUNBOOK §5).

---

## 1. The research in one page

Design of record: [`docs/research_2026-11/01_TECHNICAL_DESIGN.md`](../docs/research_2026-11/01_TECHNICAL_DESIGN.md).
State log: [`docs/research_2026-11/STATUS.md`](../docs/research_2026-11/STATUS.md). Plan for delivery day:
[`DELIVER_BRIEF.md`](../docs/research_2026-11/DELIVER_BRIEF.md).

### Problems
| # | Problem | Question |
|---|---|---|
| **P1** | The symbolic certificate covers the *number*, not the *problem*: the program → Vietnamese text step is unverified (old pipeline: 97.5 % answer agreement but 10.5 % teacher-acceptable). | How do we make symbolic Vietnamese data both verified **and** faithful/natural, and how big is the faithfulness gap? |
| **P2** | Is verification worth more in Vietnamese than in English? | Does the verification premium grow from English to Vietnamese (language × verification interaction)? English mirror design: [`exec/E1_english_mirror.md`](../docs/research_2026-11/exec/E1_english_mirror.md) (option O3). |
| **P3** | Coverage: certificates exist only for part of the curriculum. | What share of Vietnamese exam math is certifiable, and does training on certifiable types transfer to non-certifiable ones? |

### Contributions
| # | Contribution | Where |
|---|---|---|
| **C1** | **Vi-ExamMath**: native grade-10 entrance-exam benchmark (270 exams, 58 provinces, 2017–2026), solver-checked keys, "certifiable or not" tags, temporal split | `vi_math_verified_aug/vi_exam/`, `vi_exam_qa/`; [X2](../docs/research_2026-11/exec/X2_benchmark.md); datasheet [`vi_exam_core_v1.md`](DATASHEETS/vi_exam_core_v1.md) |
| **C2** | Controlled comparison: symbolic arm **Y** vs LLM-native arm **N**, each ± verification, × language (VI/EN) | `symbolic/`, `arm_y/`, `scale_gen/`, `train/`; datasheet [`train_pools.md`](DATASHEETS/train_pools.md) |
| **C3** | Faithfulness-gap attribution study + fixed pipeline (P1) | [G3](../docs/research_2026-11/exec/G3_symbolic_pilot.md), [G4](../docs/research_2026-11/exec/G4_symbolic_v2.md), [C3](../docs/research_2026-11/exec/C3_attribution.md), [D5 gate audit](../docs/research_2026-11/exec/D5_gate_audit.md), `analysis/c3_*.py` |
| **C4** | Does the advantage survive RLVR (GRPO)? | `train/grpo.py`; fp16 smoke DONE (T2 §6), P11 1.7B QUEUED |

### Arms (training data conditions; matched n = 1,808 per arm × language, matched seeds and strata)
| Arm | Meaning |
|---|---|
| `S0` | seed problems only (translated Vi-GSM8K train with upstream solutions; EN = GSM8K originals), n-matched subsample; anchor |
| `Y+` / `Y−` | symbolic: GSM8K seed → typed quantity-DAG IR (Haiku formalizer) → mutation M1–M5 → render (R2b; EN from the same IR) → shared solution writer; **+** = trace = program answer ∧ Haiku-v3 screen ∧ blind Qwen3-32B-AWQ cycle check; **−** = first item per seed, format gates only |
| `N+` / `N−` | LLM-native: Claude Haiku 5.5 writes a new problem from the seed + operator slot (`native_generator.v3`) → shared solution writer; **+** = trace = generator answer ∧ the same Haiku-v3 screen; **−** = first slot per seed, format gates only |
| `-EN` suffix | the same arms built in English from the same seeds/slots (E1 mirror) |
| `T`, `T-off`, `M` | T5 baselines (translated data, unfiltered translated, MetaMath-style) — QUEUED (P9; data not built) |

### Confirmatory hypotheses (Holm, m = 4) — ratified by the user 2026-11-07 ("continue")
Defined in [`exec/Y2_analysis.md`](../docs/research_2026-11/exec/Y2_analysis.md) §2, code `analysis/contrasts.py`.
| H | Contrast |
|---|---|
| **H1 (P2)** | I2 = VP_vi − VP_en on the 1,000 parallel items, VP_L = ½[(Y+ − Y−) + (N+ − N−)] |
| **H2 (C2)** | ½(Y+ + Y−) − ½(N+ + N−) on Vi-ExamMath Core |
| **H3** | ½(Y+ + N+) − ½(Y− + N−) on Core (verification effect) |
| **H4 (P3)** | (Y − N) on certifiable Core − (Y − N) on non-certifiable Core |

Decision rule: GLMM and cluster bootstrap must both reject after Holm and agree in sign; otherwise TOST ±3 pp.

---

## 2. Repo map

| Path | What | Owner WS / report |
|---|---|---|
| `vi_math_verified_aug/common/` | shared LLM client (Claude/Gemini/NIM, Batches), API ledger, cost cap, cache, rate limits, prompt registry, job queue | Y3 [report](../docs/research_2026-11/exec/Y3_common.md) |
| `vi_math_verified_aug/eval/math_verify.py` | **the** accuracy definition (typed Vietnamese grader 3.0.2) | X1 [report](../docs/research_2026-11/exec/X1_grader.md) |
| `vi_math_verified_aug/eval/decontam*.py` | n-gram/MinHash + bge-m3 (Modal) decontamination | X5 [report](../docs/research_2026-11/exec/X5_decontam.md) |
| `vi_math_verified_aug/eval_sets/` | frozen eval sets v1.1 (VI + EN mirror) | X3 [report](../docs/research_2026-11/exec/X3_eval_sets.md) |
| `vi_math_verified_aug/vi_exam/` | Vi-ExamMath crawl → segment → keys | X2 [report](../docs/research_2026-11/exec/X2_benchmark.md) |
| `vi_math_verified_aug/vi_exam_qa/` | Q1 key adjudication (LLM judges) → `core_v1.1` | [Q1](../docs/research_2026-11/exec/Q1_benchmark_qa.md) |
| `vi_math_verified_aug/seeds/` | S3 clean seed pool → `pool_v1` | [S3](../docs/research_2026-11/exec/S3_seed_pool.md) |
| `vi_math_verified_aug/symbolic/` | typed IR, numerals, operators M1–M6, answer-tail rule | X4 [report](../docs/research_2026-11/exec/X4_symbolic.md) |
| `vi_math_verified_aug/pilot_symbolic/`, `pilot_symbolic_v2/` | arm-Y pilots G3, G4 | [G3](../docs/research_2026-11/exec/G3_symbolic_pilot.md), [G4](../docs/research_2026-11/exec/G4_symbolic_v2.md) |
| `vi_math_verified_aug/arm_y/` | arm-Y v3 pipeline at scale (VI + EN from one IR), Qwen cycle on Modal, export | [G5](../docs/research_2026-11/exec/G5_Y_scale.md) |
| `vi_math_verified_aug/scale_gen/` | arm-N generation at scale (VI + EN); the shared solution writer and the common "+" screen | [D1](../docs/research_2026-11/exec/D1_N_scale.md) |
| `vi_math_verified_aug/pilot_gen/`, `llm_aug/` | generator pilot G1, judge/prompt study G2 | [G1](../docs/research_2026-11/exec/G1_generator_pilot.md), [G2](../docs/research_2026-11/exec/G2_judge_prompt.md) |
| `vi_math_verified_aug/train/` | SFT loop, GRPO, configs, Kaggle kernel template, **grid queue + launcher** | Y1 [report](../docs/research_2026-11/exec/Y1_train_eval.md), [T2](../docs/research_2026-11/exec/T2_training_tranche.md) |
| `vi_math_verified_aug/evalrun/` | vLLM eval runner (local CLI + Modal app `t1r-eval`: training cells, eval, GRPO) | Y1 / [T1r](../docs/research_2026-11/exec/T1r_first_runs.md) |
| `vi_math_verified_aug/qa/`, `vi_math_verified_aug/bench_gpu/`, `bench/r_gpu/` | independent stack audit; GPU/precision benchmarks | [QA-T](../docs/research_2026-11/exec/QA_T_training_audit.md), [R-GPU](../docs/research_2026-11/exec/R_GPU_platform.md) |
| `vi_math_verified_aug/analysis/` | GLMM export, cluster bootstrap, Holm/BH/TOST, IRT, power; `select_v1` (final training sets); `c3_*` (C3, gate audit, R1 results) | [Y2](../docs/research_2026-11/exec/Y2_analysis.md), [D5_selection](../docs/research_2026-11/exec/D5_selection.md), [R1](../docs/research_2026-11/exec/R1_results.md) |
| `vi_math_verified_aug/infra/` | Kaggle runner, Modal S2 app, student pilot | S2 [report](../docs/research_2026-11/exec/S2_student_pilot.md) |
| `vi_math_verified_aug/{verify,ir,informalize,symbolic_aug,data_fix,scripts}`, `src/`, `notebooks/` | **legacy** (Sept 2026 pipeline); reference only, never imported by new code | — |
| `prompts/` | versioned prompt registry (catalog: [`PROMPTS.md`](PROMPTS.md)) | all |
| `tests/` | light unit tests (laptop) + `test_analysis_heavy.py` (Modal) | all; latest run [`test_report.md`](test_report.md) |
| `jobs/` | Kaggle kernels, job specs, run outputs (manifests/metrics in git; adapters/preds in the bundle), queue `jobs/queue/t2_queue.jsonl`, logs | S2, T1r/T2 |
| `bench/o1/` | throughput benchmarks | O1 [report](../docs/research_2026-11/exec/O1_optimization.md) |
| `docs/research_2026-11/` | design (`00`–`06`, `01_TECHNICAL_DESIGN`), tech notes `tech/T1–T6`, content reviews, **exec reports** `exec/*.md` | lead |
| `data/` (git-ignored) | every dataset; delivered by the bundle ([`BUNDLE.md`](BUNDLE.md)) | per WS |
| `logs/api/<WS>/` (git-ignored) | append-only API ledgers (prompt + response per call, keys never logged); private bundle | per WS |
| `handoff/` | this package; generators in `handoff/tools/` | D4 |

## 3. Where every artifact lives

| Artifact | Path | Status | Datasheet / doc |
|---|---|---|---|
| Vi-ExamMath raw crawl + items (4,209) + Core candidates (1,585) | `data/vi_exam/` (`items.jsonl` public fields, `items_private.jsonl` third-party solutions — never release) | FROZEN (X2) | [vi_exam_core_v1.md](DATASHEETS/vi_exam_core_v1.md) |
| Vi-ExamMath **core_v1.1** (1,545 items: 1,367 VERIFIED, 178 LOW_CONF; audit label-error 1.3 % [0.2, 4.7]) | `data/vi_exam_qa/core_v1.1.jsonl`, sha256 `ceac95de…` (+ `invalid_v1.1.jsonl`; `core_v1.jsonl` kept) | FROZEN (Q1) | same |
| Eval sets **v1.1** (VI + EN parallel; `vi/en_primary_1000` = primary) | `data/eval/*.jsonl` + `*.datacard.json` + `SHA256SUMS` (18/18 OK) | FROZEN (X3) | [eval_sets_v1.1.md](DATASHEETS/eval_sets_v1.1.md) |
| Seed pool **pool_v1** (9,649) + `dev_v1` (512) | `data/seeds/`, sha256 `1b565411…` (`SHA256SUMS`) | FROZEN (S3; `pool_v1.1` for 291 PENDING2 seeds not built) | [seed_pool_v1.md](DATASHEETS/seed_pool_v1.md) |
| S0 anchor sets (6,788 VI + EN) + dev sets | `data/t1r/` | FROZEN (T1r) | seed_pool_v1 §4b |
| Generated pools Y/N × VI/EN | `data/train/{Y,N}/{vi,en}/` (`pool.jsonl`, provisional `train_10k_*`, `SHA256SUMS`, `manifest.json`) | FROZEN (G5, D1) | [train_pools.md](DATASHEETS/train_pools.md) |
| **Final matched training sets** (n = 1,808 per arm × lang) + secondary sets (`final_shared`, `final_natural`, `final_opmatched`) | `data/train/final*/{Y,N}/{vi,en}/train_{plus,minus}.jsonl`, `manifest_{vi,en}.json`, `SHA256SUMS` | FROZEN (D5 `select_v1`) | same |
| Gate audit of the final sets (Opus) | `data/audit/d5/` | DONE (D5) | [D5_gate_audit](../docs/research_2026-11/exec/D5_gate_audit.md) |
| Pilot data + judge labels (G1–G5) | `data/pilot/{generator,g2,symbolic,symbolic_v2,symbolic_v3}/` | FROZEN | exec reports |
| Contamination scans and flags | `data/contam/` | FROZEN (X5) | X5 report |
| Prompts | `prompts/*.v<k>.md` | versioned | [PROMPTS.md](PROMPTS.md) |
| API ledgers (all Claude/Gemini/NIM calls) | `logs/api/<WS>/*.jsonl` | append-only | [LEDGER.md](LEDGER.md) |
| **Trained adapters** (25 = Qwen3-0.6B × {S0, Y+, Y−, N+, N−} × {VI s0–3, EN s0}) | local `jobs/outputs/t2-modal/<run_id>/adapters/epoch3/` (git-ignored, `runs` bundle; verified, sha256 in [`ADAPTERS.md`](ADAPTERS.md)); originals on Modal Volume `t1r-eval`, `/train/<run_id>/adapters/epoch3` | DONE, all evaluated | [MODEL_CARDS/](MODEL_CARDS/INDEX.md), [BUNDLE.md](BUNDLE.md) §Adapters |
| Run manifests, configs, logs; post-run checks and loss curves | `jobs/outputs/t2-modal/<run_id>/`, `jobs/outputs/T2_checks/<run_id>/` | in git | MODEL_CARDS |
| Item-level predictions (Y2 schema) | `jobs/outputs/t2-eval/ev-<run_id>-{primary,exam-greedy}.jsonl` | bundle `runs` | R1 |
| Results (final for the tranche; lme4 output in `glmm_out_final/`) | `data/audit/r1/{tables.md,results.json,glmm_out_final/}` | DONE | [R1_results](../docs/research_2026-11/exec/R1_results.md) |
| Training queue (186 rows, costed) + launcher state | `jobs/queue/t2_queue.jsonl`, `t2_queue.totals.json`, `t2_state.json` | P1–P3 trained + evaluated; base 4-shot + 0-shot evaluated; P4 smoke done; P11 (EN s1, first successor) / P12 HOLD; rest QUEUED | [RUNBOOK.md](RUNBOOK.md) |
| Exec reports (evidence for every claim) | `docs/research_2026-11/exec/*.md` | — | — |

### Results at hand-off (final; source of truth: [`exec/R1_results.md`](../docs/research_2026-11/exec/R1_results.md))
Tranche: Qwen3-0.6B-Base LoRA SFT, n = 1,808 per arm × language, VI seeds 0–3 and EN seed 0, evaluated on Modal L40S;
tests: lme4 GLMM + two-way cluster bootstrap (items × seeds), Holm over H1–H4, TOST ±3 pp, BH for exploratory contrasts.

**Supported**
- **Generated data beats the translated seeds (arm − S0) on every dataset, both languages:** VI gsm8k +6.1 to +9.5 pp,
  math +10.2 to +11.2 pp, Vi-ExamMath Core +6.1 to +6.7 pp; EN (1 seed) +13 to +18 pp; all q < 0.001.
- **H2 (symbolic Y vs LLM-native N, Core) and H3 (verification, Core) are null and equivalent within ±3 pp** (TOST
  90 % CIs: H2 [−1.6, +1.2], H3 [−1.8, +0.9] pp; bootstrap and lme4 both non-significant after Holm). Effects of 1–2 pp
  cannot be excluded.
- Exploratory: on VI gsm8k_platinum N beats Y (H2-type −2.5 [−4.0, −0.9] pp, q = .005); not on VI math or Core.

**Not supported / provisional**
- **H1 (language × verification)** is provisional: +0.3 [−1.4, +2.0] pp with 1 EN seed; "equivalent" is not claimed.
  EN seed 1 is the next step.
- **H4 (certifiable transfer)** is null and underpowered (+0.3 [−3.1, +3.7] pp, MDE80 4.8 pp; TOST fails).
- **Arm vs base is not a clean contrast:** the 4-shot base (different format) beats every 0-shot arm on Core (33.8 vs
  30.3–30.9); the 0-shot base does not stop (56.9 % of Core decodes hit the cap; Core ≈ 18).

**Caveats on everything:** 0.6B and Qwen3 only; GSM8K seeds only (MATH unreachable for Y); question length differs Y
vs N (SMD 0.46–0.65, aliased with arm); the ± manipulation is much stronger in Y than in N (first-attempt
verification 54–63 % vs 80–87 %); Core is greedy-only (K = 1); Core truncation is high (S0 30 %, arms 10–19 %); the
Y+ training set fails the T1 quality gate (absurd 12 % VI). The successor queue that addresses these is
`KNOWN_ISSUES.md` (top) and `RUNBOOK.md` §5.

## 4. Reproduce each stage (exact commands)

All commands run from the repo root on Windows (Git Bash or PowerShell). `PY=.venv/Scripts/python.exe` is the light
project env; stages that call the Anthropic SDK or HF run under `E:/venv_tools/Scripts/python.exe` today (see
[`ENV.md`](ENV.md) — after `uv sync --locked --all-extras` the project `.venv` covers both). API stages need `.env`
(names in `.env.example`); everything else needs no key. **Paid stages are marked $.** Cached responses
(`.cache/vimath/llm_cache.sqlite`) make a re-run of a logged call free.

### 4.0 Setup
```
uv sync --locked --all-extras          # CPU env from uv.lock (no torch); see ENV.md before running on the shared laptop
$PY handoff/tools/run_light_tests.py   # light tests, 300 MB/process cap -> handoff/test_report.md
```

### 4.1 Eval sets v1.1 (X3) — deterministic except the $ Opus review stage
```
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.fetch        # pinned HF revisions -> data/eval/_sources
$PY -m vi_math_verified_aug.eval_sets.build_base
$PY -m vi_math_verified_aug.eval_sets.review_queue
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix run --stage propose|solve|judge|judge_orig   # $
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix decide
$PY -m vi_math_verified_aug.eval_sets.freeze                                     # byte-identical SHA256SUMS
cd data/eval && sha256sum -c SHA256SUMS                                          # verify (18/18 OK on 2026-11-07)
```

### 4.2 Vi-ExamMath (X2 → Q1)
```
$PY -m vi_math_verified_aug.vi_exam.discover          # hub + monthly sitemaps -> exam_index.jsonl (robots-checked, 2 s/request)
$PY -m vi_math_verified_aug.vi_exam.collect all       # exam pages (cached under data/vi_exam/raw/) + Wayback first-seen
$PY -m vi_math_verified_aug.vi_exam.build             # offline: segment, tag, keys -> items*.jsonl, qa_queue, summary.json
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.vi_exam.claude_key submit|collect   # $ blind Opus key (Batch)
# Q1 adjudication -> core_v1 (vi_math_verified_aug/vi_exam_qa/__init__.py lists the 6 steps)
$PY -m vi_math_verified_aug.vi_exam_qa.keys prepare
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.llm_stage run solve2      # $
$PY -m vi_math_verified_aug.vi_exam_qa.keys triage          # (package docstring calls it grade-solve2; the CLI choice is `triage`)
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.llm_stage run adjudicate  # $
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.llm_stage run audit       # $
$PY -m vi_math_verified_aug.vi_exam_qa.finalize audit-prep
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.vi_exam_qa.llm_stage run audit_judge # $
$PY -m vi_math_verified_aug.vi_exam_qa.finalize freeze                                         # core_v1(.1).jsonl + SHA256
```
(Sub-commands checked against the argparse choices; Q1 §1 has the exact flags per round. `llm_stage spend` prints
Q1's spend.)

### 4.3 Decontamination (X5)
```
$PY -m vi_math_verified_aug.eval.decontam scan|calibrate|gate|bench-index --help   # lexical scans + gate (numpy only)
E:/venv_tools/Scripts/modal.exe run vi_math_verified_aug/eval/decontam_modal.py   # $ bge-m3 embeddings on Modal L4
$PY -m vi_math_verified_aug.eval.decontam_report
```
Thresholds of record: bge-m3 cos < 0.85 (VI↔VI), < 0.80 (VI↔EN), templated 13-gram coverage < 0.5, MinHash J < 0.8.

### 4.4 Seed pool (S3)
```
# stages in order: align detect solvecheck($) mta($) gold decontam embed-inputs translationese freeze; `spend` = report
$PY -m vi_math_verified_aug.seeds.build align        # local stages under .venv
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.seeds.build solvecheck   # $ API stages under venv_tools (Gemini flash-lite on train seeds only; Haiku->Sonnet MTA)
```

### 4.5 Training pools and the final selection
```
# arm Y (symbolic, v3 operators, VI + EN from one IR; G5 lists its run scripts arm_y/run_scale.sh, run_rest.sh)
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.arm_y.pipeline <run_dir> init pool <n_seeds> R2b,EN batch
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.arm_y.pipeline <run_dir> formalize|mutate|render|cycle|gates|vgate   # $
#   blind cycle check: Qwen3-32B-AWQ on Modal, chunked (arm_y/vllm_cycle_modal2.py)                                         # $
$PY -m vi_math_verified_aug.arm_y.export <run_dir> <variant> <lang> items|write          # N-layout pool (+ decontam on Modal)
# arm N (LLM-native; D1 lists the exact sequence)
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.scale_gen.run slots
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.scale_gen.run gen --langs vi,en --n <N> --cfg A                    # $
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.scale_gen.run status
$PY -m vi_math_verified_aug.scale_gen.report gates|audit|costs|examples
# final matched sets (D5)
$PY -m vi_math_verified_aug.analysis.select_v1 part1 vi
$PY -m vi_math_verified_aug.analysis.select_v1 part2 vi,en --force     # final + shared + natural + opmatched
$PY -m vi_math_verified_aug.analysis.select_v1 funnel x | coverage x | project x
```

### 4.6 Training + evaluation (Modal L40S)
```
$PY -m vi_math_verified_aug.train.grid --out jobs/queue/t2_queue.jsonl       # rebuild the costed queue
$PY -m vi_math_verified_aug.train.launcher tick --dry-run                    # validate data + render cells, push nothing
$PY -m vi_math_verified_aug.train.launcher loop --every 120 --max-prio 12 --backend modal --platform modal_l40s \
    --modal-cap-usd <block $> --workspace-cap-usd <credit - margin>
$PY -m vi_math_verified_aug.train.launcher status          # also: tail <run_id>, ack <run_id|lang:xx>
```
Details, guards, caps and per-run costs: [`RUNBOOK.md`](RUNBOOK.md).

### 4.7 Analysis
```
$PY -m vi_math_verified_aug.analysis.c3_r1_results --pred "jobs/outputs/t2-eval/ev-*.jsonl"   # R1 tables (local, ~70 MB)
$PY -m vi_math_verified_aug.analysis.run export --pred <all preds>.jsonl --out glmm_export     # then lme4 (Kaggle CPU R
                                                                                              # kernel; or analysis/modal_jobs.py)
```

### 4.8 Refresh this package
```
$PY handoff/tools/ledger.py [--console-usd X --console-when "..."]   # LEDGER.md (Claude per WS, Modal MTD, Kaggle)
$PY handoff/tools/prompt_catalog.py         # PROMPTS.md + PROMPTS.sha256
$PY handoff/tools/fill_model_cards.py       # MODEL_CARDS/*.md from jobs/queue/t2_state.json + jobs/outputs/t2-modal/*/manifest.json
$PY handoff/tools/verify_adapters.py        # ADAPTERS.md + ADAPTERS.sha256 (25 local adapters vs manifests)
$PY handoff/tools/make_bundle.py [--hash]   # BUNDLE_MANIFEST.tsv (plan); --build --out <dir outside repo> for zips
$PY handoff/tools/secrets_scan.py           # SECRETS_SCAN.md (paths only)
$PY handoff/tools/run_light_tests.py        # test_report.md
```

## 5. Rules the successor inherits
- Test/benchmark items (Vi-GSM8K/Vi-MATH test, Vi-ExamMath) go to **Claude only**, never to Gemini or NIM
  (`split="benchmark"/"test"` guards in `common/llm.py`).
- Never print key or token values; keys live in `.env` only (never committed).
- Kaggle kernels and datasets stay **private**.
- Laptop: ≤ 300 MB per process, no torch/transformers; heavy work on Modal or Kaggle.
- Every Claude output used for training keeps its provenance (datasheet + `logs/api/<WS>/`).
- Every number in a report cites a file, a count and a CI.
