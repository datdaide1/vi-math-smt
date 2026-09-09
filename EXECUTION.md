# EXECUTION.md — task-by-task plan for a fresh executor (AI or human)

> **Read `RESEARCH_PLAN.md` first** for the *why* and the design. This file is the *how*: a sequenced,
> self-contained backlog. Work top to bottom, respecting each task's **Depends-on**. Independent tasks
> in the same sprint can run in parallel.

---

## For the executing agent — start here

1. Read `RESEARCH_PLAN.md` fully. The spine is the research question in §0; everything serves it.
2. **Hard rules (never violate):** — see `RESEARCH_PLAN.md` §6.0 for the full model table.
   - **Generation = `deepseek-ai/deepseek-v4-pro-0813` via NVIDIA NIM (`seed=42`, 40 RPM, no credit
     cap), nothing else.** One fixed model. Optional batching for speed.
   - **Fine-tuning = `Qwen3-0.6B-Base` (local, 4 GB RTX 3050 Ti, QLoRA) / `Qwen3-1.7B-Base` (Modal L4,
     ~$6/month) only.** No model >1.7B is ever fine-tuned. **Nothing is self-hosted on a GPU.** Every
     frozen larger model is reached through a hosted API: S2-alt weaker generator + round-trip solver +
     difficulty reference = `meta/llama-3.1-8b-instruct` via NIM; leaderboard = free endpoints / published numbers.
   - No DPO / RL. SFT only. No Kaggle.
   - **No paid services.** Modal $30/month is **1.7B-fine-tuning-only** (~$6–10/cycle) — generation and
     the 0.6B grid never touch it.
   - No `model.generate` loops for evaluation — always vLLM batched.
   - One accuracy definition everywhere: `eval/math_verify.py`.
   - Every result traces to a prediction file + run id in `RESULTS_PROVENANCE.md`.
   - Modal: never add a payment method (keeps the $30/month hard cap).
3. Work the backlog. After each task: run its **Acceptance** check, commit on a feature branch, update
   the task's checkbox here.
4. Tasks marked **[HUMAN]** cannot be done by the agent — flag them to the researcher and continue with
   unblocked work.
5. Tasks marked **[GATE]** are go/no-go decision points — stop and get a decision before proceeding.

**Repo layout to create (Sprint 0):**
```
vi-math-verified-aug/            # new top-level package (can live inside this repo)
  ir/            typed_dag.py                     # lightweight IR: typed quantities + operation DAG
  symbolic_aug/  constant_sub.py  structural.py   # S1 operators
  llm_aug/       clients.py  generate.py          # S2/S3 generation
  informalize/   generate.py  leakage_filter.py
  vi_exam/       scrape.py  segment.py  verify.py       # C1 benchmark build (scrape-first, no OCR)
  verify/        z3_utils.py  uniqueness.py  round_trip.py
  eval/          math_verify.py  run_eval.py  data_quality.py  decontam.py
  scripts/       sft.py  build_s0.py … build_s6.py  run_grid.py
  configs/       *.yaml
  modal_app.py
  data/          arms/  vi_exam/  eval/  INVENTORY.md
  results/       anchors.json  runs/  predictions/  data_quality/  analysis/  master_results.csv
RESULTS_PROVENANCE.md
```

---

## Capacity & critical path

- **Team:** 1 researcher + AI coding assistants.
- **AI does:** all implementation, data pipelines, training/inference orchestration, analysis, first
  drafts. Fast (hours–days per task).
- **Researcher is the bottleneck for:** API keys, obtaining exam PDFs, recruiting/managing human
  annotators (MOS + benchmark QA), advisor coordination, judgment calls at [GATE]s.
- **Wall-clock critical path:** API keys → LLM generation of all arms → data-quality panel → training
  grid → analysis. In parallel: exam scrape → normalize/segment → solver-verify → human QA → Vi-ExamMath.
- **Plan to ~75% capacity.** Expect iteration on Sprint 1–2 (the symbolic pipeline fixes and the
  benchmark scrape/segment are the least predictable).

**Sprint calendar (2 weeks each; ~3 months; GPU spend confined to Sprint 4's billing month):**

| Sprint | Theme | Exit criterion |
|---|---|---|
| S0 | Foundation | harness + templates + keys work; anchor numbers logged |
| S1 | Symbolic pipeline (S1 arm) + exam-source survey | S1 pipeline runs end-to-end on 100 problems |
| S2 | All 7 arms built + benchmark segment/verify | `data/arms/s{0..6}.jsonl` exist; `data/vi_exam/verified.jsonl` exists |
| S3 | Data-quality panel + human eval + benchmark QA | quality panel done; Vi-ExamMath Core `final.jsonl` (≥300) |
| S4 | Training grid + inference | `results/master_results.csv` complete |
| S5 | Analysis + C1 resource paper | RQ-A…E answered; C1 draft; advisor review #1 |
| S6 | Full paper + submit | paper submitted to an ARR cycle; artifacts released |

---

## SPRINT 0 — Foundation

### T0.1 — Repo skeleton
- **Goal:** create the package layout above with a stub in each module and a top `README` linking
  `RESEARCH_PLAN.md` + this file.
- **Depends-on:** —
- **Output:** committed skeleton.
- **Acceptance:** `python -c "import vi_math_verified_aug"` works; `pytest` collects 0 tests without error.

### T0.2 — `eval/math_verify.py`
- **Goal:** one function `is_correct(pred: str, gold: str) -> bool` = answer normalization + SymPy
  equivalence + numeric tolerance (abs 1e-6 or rel). Also `extract_answer(text) -> str` (`\boxed{}`,
  `Đáp án…`, `####`, last number — priority-ordered).
- **Reuse:** logic in `src/evaluate/base/*.py`, `src/evaluate/aug/aug_evaluate.py` (`normalize_answer`,
  `sympy_equiv`) — consolidate the three existing variants into one.
- **Depends-on:** T0.1
- **Output:** `eval/math_verify.py` + `tests/test_math_verify.py`.
- **Acceptance:** ≥30 hand-labelled `(pred, gold, expected)` cases pass, covering integers, decimals,
  simple fractions, `\frac`, radicals, tuples/coords, `\boxed` vs bare, LaTeX noise, unit suffixes
  ("51 đô la").

### T0.3 — `eval/run_eval.py` (harness)
- **Goal:** config-driven eval. Input: `{model: hf_path|api_spec, test_set: path.jsonl, n_shot,
  max_new_tokens, prompt_template}`. Runs vLLM for local models / an API client for hosted ones →
  writes `predictions.jsonl` (`{id, problem, gold, raw_output, pred_answer, correct}`) + `metrics.json`
  (pass@1 overall + by any `subject`/`level`/`answer_type` field).
- **Depends-on:** T0.2
- **Output:** `eval/run_eval.py`.
- **Acceptance:** runs on a 20-item toy set with a tiny local model and with one API model; metrics
  match a hand count.

### T0.4 — `scripts/sft.py` (training template)
- **Goal:** Unsloth LoRA SFT. Config: `{base_model, train_jsonl, N, seed, max_seq_len(auto=p99),
  lora_r=16, epochs=1, packing=True, completion_only=True, load_in_4bit, output_dir}`. No literal `<s>`
  in text. Runs locally for 0.6B (4-bit QLoRA on the 4 GB RTX 3050 Ti) and on Modal L4 for 1.7B.
- **Depends-on:** T0.1
- **Output:** `scripts/sft.py` + `configs/sft.example.yaml`.
- **Acceptance:** a 300-example smoke run on Qwen3-0.6B-Base finishes < 15 min **on the local laptop
  GPU** (peak VRAM < 4 GB) and produces a loadable adapter; loss decreases.

### T0.5 — `modal_app.py`
- **Goal:** a Modal app exposing (a) an SFT function wrapping `scripts/sft.py` on an L4 — **used for the
  ~25 Qwen3-1.7B-Base runs only** (0.6B trains locally), (b) a vLLM eval-inference function for ≤1.7B
  adapters on an L4. No self-hosted generator anywhere. Region default, preemptible.
- **Depends-on:** T0.4
- **Output:** `modal_app.py` + `docs/MODAL.md` (setup; **do not add a payment method**; usage-limit
  stays $30; **Modal is 1.7B-fine-tuning-only** — generation never touches it).
- **Acceptance:** `modal run modal_app.py::smoke` trains the 300-example smoke run for < $0.30.

### T0.6 — LLM clients + smoke test — **[HUMAN provides free keys]**
- **Goal:** `llm_aug/clients.py` — one wrapper, two roles:
  - `generate(...)` / `informalize(...)` → **`deepseek-ai/deepseek-v4-pro-0813` via NVIDIA NIM only**
    (`integrate.api.nvidia.com`, OpenAI-compatible), `seed=42` + pinned `temperature`/`top_p`, a 40 RPM
    limiter, resumable checkpoint. One fixed model. Fallback: DeepSeek direct API → Groq (all hosted).
  - `generate_weak(...)` → **`meta/llama-3.1-8b-instruct` via NIM** (same key), `seed=42` — the S2-alt
    weaker generator.
  - `solve(problem)` → callable against **each** of {deepseek-v4-pro (NIM), `meta/llama-3.1-8b-instruct`
    (NIM), optional Groq Llama-3.3-70B (free)} independently, plus SymPy, for the round-trip check.
- **[HUMAN] first:** the NIM account (`@hus.edu.vn`) is already rate-limit-only (40 RPM, no credit
  cap) — just generate an `NVIDIA_API_KEY`. Optionally request a 200 RPM increase. Also register a
  DeepSeek direct API key (fallback) and optionally a Groq key (3rd solver).
- **Depends-on:** T0.1; researcher puts **free** keys in `.env`: `NVIDIA_API_KEY` + `DEEPSEEK_API_KEY` + optional `GROQ_API_KEY
  or `GEMINI_API_KEYS`) + `DEEPSEEK_API_KEY`. No paid keys.
- **Output:** `llm_aug/clients.py` + `results/api_bench.json` (generation quality on 20 seeds; per
  solver accuracy/latency; **measured per-key RPD**).
- **Acceptance:** the generator produces well-formed Vietnamese problems on 20 seeds; each solver
  answers ≥17/20; the 40 RPM limiter works.

### T0.7 — Anchor numbers
- **Goal:** establish reference points under the unified harness. Run zero-shot `Qwen3-1.7B-Base`
  (4-shot) and `DeepSeek-R1-Distill-Qwen-1.5B` on Vi-GSM8K; re-score the existing
  `data/result/wizardMATH/` predictions through `math_verify`.
- **Depends-on:** T0.3
- **Output:** `results/anchors.json`.
- **Acceptance:** DeepSeek-R1-Distill-1.5B lands in a plausible band; WizardMath re-score near
  60.3 / 21.8.

### T0.8 — Data inventory
- **Goal:** checksum + document every reused asset (Vietnamese translations, GSM8K SMT, `metaMathQA-vi`,
  4 test sets). Note per-file schema and row count.
- **Depends-on:** —
- **Output:** `data/INVENTORY.md`.
- **Acceptance:** every path in `RESEARCH_PLAN.md` §2.1 resolves; counts recorded.

---

## SPRINT 1 — Symbolic pipeline (S1) + exam-source survey

### T1.1 — Uniqueness gate + audit  *(also a C3 result)*
- **Goal:** `verify/uniqueness.py::classify(smt) -> {"derived"|"hardcoded"|"unknown"}` — drop the
  `answer` assertion, ask Z3 whether the system still has a unique solution (block the first model, ask
  for a second; `unsat` ⇒ unique). For non-linear, try `QF_NRA`/`QF_NIA` then SymPy.
- **Depends-on:** T0.1
- **Output:** `verify/uniqueness.py`; `results/uniqueness_audit.json` (rate by subject over ≥500 base
  GSM8K + MATH SMT).
- **Acceptance:** reproduces the ~40% derived / 15.9% hardcoded ballpark from `RESEARCH_PLAN.md` §2.2
  N2; a hand check of 20 flagged "hardcoded" agrees.

### T1.2 — M0: constrained constant substitution (algorithm: RESEARCH_PLAN.md §6.1)
- **Goal:** `symbolic_aug/constant_sub.py` — replace the constants in a verified problem (not the
  answer term); constrain sampling so every intermediate value **and** the final answer stay
  integer / clean-fraction; recompute with Z3; reject physically nonsensical results. Delete the old
  `mutate_structure/expression/difficulty`.
- **Depends-on:** T1.1
- **Output:** module + `tests/`.
- **Acceptance:** on 100 GSM8K seeds, ≥80% yield a valid variant; 0% ugly-decimal answers; a hand
  read of 20 confirms the reasoning structure is preserved.

### T1.3 — Leakage-free informalization
- **Goal:** `informalize/generate.py` — build a semantic description (typed quantities, relations,
  query) from the mutated representation; the LLM prompt contains **only** that description + "rewrite
  as a natural Vietnamese problem, keep every number and the question". Never pass `(assert …)`,
  variable names, or the answer. Replaces `phase3_informalize.py::GENERATOR_PROMPT`.
- **Depends-on:** T0.6, T1.2
- **Output:** module + the new prompt in `configs/prompts/`.
- **Acceptance:** on 50 mutated items, a hand read finds no formal-artifact leakage.

### T1.4 — Leakage filter
- **Goal:** `informalize/leakage_filter.py::check(problem_text, answer) -> (ok, reason)` — regex for
  `h_\d+`, `sum_exponents`, `assert`, `declare-`, `=>`, `(check-sat)`; NER for entities absent from the
  spec; the answer string appearing verbatim in the problem body.
- **Depends-on:** T0.1
- **Output:** module + `tests/`.
- **Acceptance:** catches ≥95% of the 16.3% leakage cases in the current `phase3_informalized.jsonl`;
  < 2% false-positive on the clean base translations.

### T1.5 — Round-trip gate
- **Goal:** `verify/round_trip.py` — (a) re-formalize the generated Vietnamese problem (EN round-trip)
  → Z3 check the answer within tolerance; and/or (b) 3 independent LLM solves → self-consistency
  majority must equal the verified answer. Configurable; report retention.
- **Depends-on:** T0.6, T1.3
- **Output:** module.
- **Acceptance:** on 100 items, retention rate is reported; 20 rejected items hand-checked are indeed
  wrong/ill-posed.

### T1.6 — O1 + O2 structural operators (algorithm: RESEARCH_PLAN.md §6.1)
- **Goal:** `ir/typed_dag.py` (parse a verified problem into typed quantities + a reduced operation
  DAG with a query node) and `symbolic_aug/structural.py`:
  - **O1 step insertion:** add `op(v, w) -> v'` with divisibility/domain guards keeping `v'`'s type;
    rewire; re-verify SAT + uniqueness.
  - **O2 backward/FOBAR:** promote a clean input leaf to the query, demote the old query to a given;
    new answer = former input; Z3 checks unique solvability.
- **Depends-on:** T1.1, T1.2
- **Output:** modules + `tests/`.
- **Acceptance:** on 100 seeds each: ≥50% valid variants; 100% integer/clean answers; O1 variants have
  ≥1 more reasoning step than the seed (measured).

### T1.7 — S1 pipeline runner
- **Goal:** `scripts/build_s1.py` — seed → {constant_sub, O1, O2} per a config mix → informalize →
  leakage filter → round-trip → dedup → `data/arms/s1.jsonl` at a target N. Emits a build report
  (yield per stage).
- **Depends-on:** T1.2–T1.6
- **Output:** script + `data/arms/s1.jsonl` (small, N≈500 for now) + `results/s1_build_report.json`.
- **Acceptance:** end-to-end run completes; report shows per-stage retention.

### T1.8 — Exam scraper + source survey  *(prototype: `_scratch/vi_exam/scrape_loigiaihay.py` + `sample_core_triples.md`)*
- **Goal:** `vi_exam/scrape.py` — per-source adapters (loigiaihay HTML+LaTeX; toanmath Word→LaTeX via
  pandoc; VnExpress/VietnamNet dated articles; hanoi.edu.vn official PDFs). Output raw
  `(exam, source, year, province, exam_name, url, license, raw_latex)`. **No OCR** — flag scanned PDFs
  for a MathPix fallback pass. Hà Nội grade-10 2017–2026 first, then ~10–15 provinces, then HSG/chuyên.
- **Depends-on:** —
- **[HUMAN]:** ask the advisor about official Sở GD-ĐT Hà Nội channels; confirm the license line for
  public release.
- **Output:** `vi_exam/scrape.py` + `data/vi_exam/raw.jsonl` (≥60 exams) + `data/vi_exam/SOURCES.md`
  (per source: URL, coverage, format, license) + `results/scrape_yield.md`.
- **Acceptance:** ≥5 sources; ≥60 exams scraped; figure-free segment yield reported (prototype: ~62%);
  30 hand-checked triples are faithful.

---

## SPRINT 2 — All arms + benchmark segment/verify

### T2.1 — **[GATE]** Validate S1 quality on 100 problems
- **Goal:** run T1.7 on 100 seeds; compute answer-type distribution, leakage %, and hand-read 30.
- **Gate:** proceed only if answer-type distribution ≈ seed (≥55% integer) **and** leakage < 2%
  **and** 30/30 hand reads are well-posed with correct answers. Otherwise iterate T1.2–T1.5 before
  building the full grid.
- **Output:** `results/s1_gate.md` with the decision.

### T2.2 — Build S1 at all N
- **Goal:** `data/arms/s1.jsonl` sized so N∈{2k,5k,10k,20k} subsets can be drawn (build ≥20k).
- **Depends-on:** T2.1
- **Acceptance:** 20k rows, schema-valid, dedup'd against test sets.

### T2.3 — LLM augmentation generator
- **Goal:** `llm_aug/generate.py` — **`deepseek-v4-pro-0813` via NIM** (`seed=42`, 40 RPM, client from
  T0.6) generates, per seed, structurally-varied variants + step-by-step solutions + `\boxed{}`
  answers; diversity sampling; resumable checkpoint. Also generates the **S2-alt** subset (~1–2k) with
  a weaker generator (`meta/llama-3.1-8b-instruct` via NIM).
- **Depends-on:** T0.6
- **Output:** module + `data/arms/_llm_raw.jsonl` (+ `_llm_raw_strong.jsonl`).
- **Acceptance:** 200-seed dry run; hand-read 20 for variety and correctness; RPD budget check for the
  full run.

### T2.4 — Build S2 (LLM aug + independent check)
- **Goal:** `scripts/build_s2.py` — keep a candidate only if an **independent** SymPy evaluation of the
  final answer (and optional 3-way LLM self-consistency) confirms it. Match N to S1.
- **Depends-on:** T2.3
- **Output:** `data/arms/s2.jsonl` + retention rate.

### T2.5 — Build S3 (LLM aug, no check)
- **Depends-on:** T2.3 → `data/arms/s3.jsonl`.

### T2.6 — Build S4 (S1 minus verification)
- **Goal:** rerun the S1 pipeline with the SMT verification + round-trip gates disabled.
- **Depends-on:** T2.2 → `data/arms/s4.jsonl`.

### T2.7 — Build S5 (translate-train)
- **Goal:** format a slice of `data/test/metaMATH/metaMathQA-395K_json_vi.jsonl` (algebra/arithmetic,
  `_vi` fields) into the SFT schema at matched N.
- **Depends-on:** T0.8 → `data/arms/s5.jsonl`.

### T2.8 — Build S0 (seed only) and S6 (optional hybrid)
- **Goal:** S0 = the translated seed set formatted for SFT. S6 (optional) = structural symbolic
  mutation + LLM informalization + double check.
- **Depends-on:** T0.8 (S0); T1.6, T2.3 (S6) → `data/arms/s0.jsonl`, `data/arms/s6.jsonl`.

### T2.9 — Vi-ExamMath segmentation + scope filter
- **Goal:** `vi_exam/segment.py` — from `raw.jsonl`: normalize LaTeX; split to **sub-part** granularity
  (`Câu II.3`); extract `(problem, reference_solution, final_answer)`; classify grade band + topic +
  answer type; flag figure-dependent items. **Core filter:** numeric answer, arithmetic/algebra/
  functions, figure-free. **Hard bucket:** competition / expression-answer / figure. Structural dedup
  across years/provinces and vs the S1/S2/S5 training pools.
- **Depends-on:** T1.8
- **Output:** `data/vi_exam/segmented.jsonl` + `data/vi_exam/core_candidates.jsonl` (≥500) +
  `data/vi_exam/hard_candidates.jsonl`.
- **Acceptance:** ≥500 Core candidates; final-answer extracted for ≥90% of Core; 30 hand-checked.

### T2.10 — Vi-ExamMath Core solver verification
- **Goal:** `vi_exam/verify.py` — EN round-trip → formalize (SMT/SymPy) → check the official answer +
  uniqueness gate (T1.1). Label `verified` / `unverified`; flag likely scrape/OCR errors.
- **Depends-on:** T2.9, T1.1
- **Output:** `data/vi_exam/verified.jsonl` + `results/exam_verify_report.md` (rate + error taxonomy —
  a C1 result).

---

## SPRINT 3 — Quality panel + human eval + benchmark QA

### T3.1 — Data-quality panel
- **Goal:** `eval/data_quality.py` — per arm: answer-type JS divergence vs seed + magnitude histogram;
  leakage %; diversity (distinct-2, self-BLEU, embedding dispersion, k-seed sweep);
  independently-checked correctness %; real-vs-synthetic discriminator AUROC; contamination overlap vs
  every test set; % derived (symbolic arms, from T1.1).
- **Depends-on:** T2.4–T2.8, T1.1
- **Output:** `results/data_quality/<arm>.json` + a comparison table + plots.
- **Acceptance:** every arm has every metric; the table is figure-ready.

### T3.2 — Human MOS study — **[HUMAN: recruit ≥3 raters]**
- **Goal:** 200 problems sampled across arms, blinded; ≥3 raters score naturalness (1–5) and validity;
  a calibration round first; compute Krippendorff α.
- **Depends-on:** T2.4–T2.8
- **Output:** `results/mos.json` + rubric in `docs/MOS_RUBRIC.md`.
- **Acceptance:** α reported (target ≥ 0.4 after calibration); per-arm means with CIs.

### T3.3 — Vi-ExamMath human QA — **[HUMAN: 2 annotators]**
- **Goal:** 2 annotators independently re-derive each item's answer + rate well-posedness; report
  Cohen κ; adjudicate disagreements. Core (solver + human) first; Hard (human only) if time.
- **Depends-on:** T2.10
- **Output:** `data/vi_exam/final.jsonl` — **Core target 400 (min 300)**, splits `verified` (solver+human)
  / `human-only`; **Hard target 200 (optional)**; `data/vi_exam/DATACARD.md` (sources, per-item license,
  grade-band/topic/answer-type stratification, contamination argument).
- **Acceptance:** κ reported; no item structurally duplicates a training item; datacard complete;
  discriminativeness gate checked after T4.5 leaderboard.

### T3.4 — Template-perturbable subset  *(if time)*
- **Goal:** ~50 GSM-Symbolic-style templates (typed slots + constraints) from Vi-ExamMath items;
  generate perturbed instantiations.
- **Depends-on:** T3.3 → `data/vi_exam/template/`.

### T3.5 — Vietnamese robustness slice
- **Goal:** GSM-Plus-style perturbations (numeric substitution, distractor insertion, operation
  reversal) of a 300-item Vi-GSM8K subset.
- **Depends-on:** T0.6 → `data/eval/robustness_vi.jsonl`.

### T3.6 — Decontamination report
- **Goal:** `eval/decontam.py` — n-gram + embedding overlap of every arm's training data against every
  test set (Vi-ExamMath, Vi-GSM8K, robustness, template).
- **Depends-on:** T2.4–T2.8, T3.3 → `results/decontam.json`.

---

## SPRINT 4 — Training grid + inference  *(1.7B GPU spend confined to one Modal billing cycle)*

### T4.1 — Grid config
- **Goal:** `configs/grid.yaml` enumerating every run: `{arm, base, seed, N, testsets, where}`.
  Core: S0/S1/S2/S5 × {Qwen3-0.6B-Base (local), Qwen3-1.7B-Base (Modal L4)} × 3 seeds.
  Diagnostic: S3/S4/S6 × Qwen3-1.7B-Base × 2 seeds.
  Data-efficiency: S0/S1/S2 × N∈{2k,5k,10k} × Qwen3-1.7B-Base × 2 seeds.
  English control: S0/S1/S2/S3 × Qwen3-1.7B-Base × 2 seeds (English GSM8K).
  ≈ 55 runs (~30 local 0.6B, ~25 Modal L4 1.7B).
- **Depends-on:** T2.2–T2.8, T3.1
- **Output:** `configs/grid.yaml`.

### T4.2 — Run training grid
- **Goal:** `scripts/run_grid.py` — for each run: 20-sample smoke test (`format_ok` > 80%, no
  repetition) → full SFT → save adapter + `results/runs/<run_id>/` (config, loss curve, metadata).
  0.6B runs on the local laptop GPU; 1.7B runs on Modal L4. Resumable.
- **Depends-on:** T4.1, T0.4, T0.5
- **Acceptance:** all runs complete; total Modal spend < $12; each run logged.

### T4.3 — Inference sweeps
- **Goal:** merge each adapter → vLLM → predictions on Vi-ExamMath + Vi-GSM8K + robustness + template.
  Stratified 1k subset during the grid; full sets for the finalists (best config per arm).
- **Depends-on:** T4.2, T0.3
- **Output:** `results/predictions/<run_id>/`.

### T4.4 — Master results table
- **Goal:** score all predictions through `math_verify`; assemble
  `results/master_results.csv` = `{arm, base, seed, N, testset, split, metric, value, run_id,
  prediction_file}`; bootstrap 95% CI for headline cells.
- **Depends-on:** T4.3 → `results/master_results.csv` + `RESULTS_PROVENANCE.md` rows.

### T4.5 — C1 leaderboard
- **Goal:** score the ≤1.7B models we run + >1.7B models via free hosted endpoints / published numbers
  + the WizardMath re-score, all on Vi-ExamMath.
- **Depends-on:** T3.3, T0.3 → `results/leaderboard.csv`.

---

## SPRINT 5 — Analysis + C1 resource paper

### T5.1 — Analysis
- **Goal:** notebook / scripts answering RQ-A (S1 vs S2, every axis incl. generation cost), RQ-B
  (verification premium: S1 vs S4, S2 vs S3, difficulty-histogram-matched), RQ-C (quality→utility
  regression), RQ-D (data-efficiency curves), RQ-E (VN vs EN arm ranking). All figures from
  `master_results.csv` + `data_quality/`.
- **Depends-on:** T4.4, T3.1, T3.2, T3.6
- **Output:** `results/analysis/` (figures + a findings memo).

### T5.2 — Provenance
- **Goal:** finalize `RESULTS_PROVENANCE.md` — every number in the paper → prediction file + run id +
  data commit hash + date.
- **Depends-on:** T4.4, T4.5

### T5.3 — C1 resource paper draft
- **Goal:** Vi-ExamMath paper: motivation (contamination), construction pipeline, verification-rate
  finding, leaderboard, analysis. 4–8 pages.
- **Depends-on:** T3.3, T4.5, T2.10
- **Output:** `paper/vi_exammath/`.

### T5.4 — C1 release prep
- **Goal:** HF dataset repo, per-item license, datacard, eval script, anonymized-for-review variant.
- **Depends-on:** T3.3

### T5.5 — **[HUMAN][GATE]** Advisor review #1
- Advisor reviews the findings memo, `master_results.csv`, 20 predictions/arm, 20 Vi-ExamMath items.
- **Gate:** decide — (a) full paper C1+C2+C3 to an ARR cycle, or (b) C1 to a Q3 venue now + C2 later.

---

## SPRINT 6 — Full paper + submit

### T6.1 — Full paper draft (C1+C2+C3)
- Intro → related work → C3 (why the naive answer-expression approach degrades data; framed as
  motivation, not "our mistake") → C1 (Vi-ExamMath) → C2 (arms, protocol, comparison) → results →
  analysis → limitations (incl. §2.2 N2) → conclusion.
- **Depends-on:** T5.1, T5.3

### T6.2 — Figures + tables
- Regenerate every figure from `master_results.csv` / `data_quality/`. No hand-entered numbers.

### T6.3 — Artifact release
- Fixed augmentation pipeline, all `data/arms/*`, `data/vi_exam/*`, `eval/` harness. License notes.

### T6.4 — **[HUMAN][GATE]** Advisor review #2
- Results table, 20 random predictions, 20 samples/arm, 20 benchmark items. Sign-off.

### T6.5 — **[HUMAN]** Submit
- ARR cycle (Q1–Q2 2027) for the full paper; and/or the chosen Q3 venue for C1. Register with a
  deadline in the researcher's calendar.

---

## Human-only tasks & decision gates (collated)

| ID | What | When |
|---|---|---|
| T0.6 | Provide **free** `.env` keys: NVIDIA_API_KEY (generator) + DEEPSEEK_API_KEY (fallback+solver) + optional GROQ_API_KEY (3rd solver) | Sprint 0 |
| T1.8 | Ask advisor re official Sở GD-ĐT Hà Nội channels; confirm public-release license line | Sprint 1 |
| T2.1 | **[GATE]** Is fixed-S1 data quality good enough to build the grid? | Sprint 2 |
| T2.9 | Spot-check segmented triples (30) for fidelity | Sprint 2 |
| T3.2 | Recruit + manage ≥3 MOS raters | Sprint 3 |
| T3.3 | Serve as 1 of 2 Vi-ExamMath QA annotators; recruit the other | Sprint 3 |
| T3.1 | **[GATE]** Does the S1-vs-S2 finding hold on the S2-alt (weaker-generator) subset? If it flips, the finding is generator-dependent — report that. | Sprint 3 |
| T5.5 | **[GATE]** Advisor review #1 → venue decision | Sprint 5 |
| §10.6 | Pick the exact Q3 venue + deadline for C1 | Sprint 5 |
| T6.4 | **[GATE]** Advisor review #2 → submit | Sprint 6 |

## Definition of Done (every task)
- [ ] Code reviewed (self or advisor) and merged to a feature branch
- [ ] Acceptance check in the task passes
- [ ] Outputs written to the paths named in the task
- [ ] `RESULTS_PROVENANCE.md` updated if the task produced a number that may appear in the paper
- [ ] This file's checkbox ticked

## Risks (execution-level; see `RESEARCH_PLAN.md` §9 for research-level)

| Risk | Mitigation |
|---|---|
| Symbolic pipeline fixes (Sprint 1) take longer than 2 weeks | T2.1 gate; O1/O2 are optional for v1 — `constant_sub` + leakage filter + round-trip alone give a valid S1 |
| Scrape/segment quality too low | most sources are text/LaTeX/Word (prototype: ~62% figure-free, clean LaTeX); T2.10 solver-verify + T3.3 human QA are the filters; hand-transcribe a residual to hit the Core-300 minimum |
| Vi-ExamMath Core not discriminative | discriminativeness gate after T4.5; rebalance grade-band mix; lean paper on C1 + C3 |
| Can't recruit 3 MOS raters | drop to 2 + report κ with the caveat; MOS is one panel metric among many |
| Generation volume vs 40 RPM | ~15–30k items ≈ 1–3 h wall-clock at 40 RPM; resumable checkpoint; optional 5–10-item batching; fallback DeepSeek direct → Groq |
| Grid exceeds one Modal cycle | the ~30 0.6B runs are free on the laptop; only ~25 1.7B runs hit L4 (~$6–10); cut English control to 2 arms, non-primary seeds to 2, drop S6 |
| 0.6B QLoRA OOMs on 4 GB | drop `lora_r` to 8, `max_seq_len` to p95, batch 1 + grad-accum; last resort move that base to Modal L4 too (~$4 more) |
