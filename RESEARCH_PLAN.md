# Research Plan — Symbolic vs. LLM Math-Data Augmentation for Small Models, with a Contamination-Controlled Vietnamese Exam Benchmark

**Status:** approved direction, ready to execute.
**Owner:** Tran Hoang Dat (VNU University of Science). Advisor: Dr. Ha My Linh.
**Working paper title:** *Does Symbolic Verification Still Earn Its Keep? A 2026 Comparison of Symbolic
and LLM-based Math-Data Augmentation for Small Models in a Low-Resource Language.*
**Constraints:** single researcher; laptop (4 GB RTX 3050 Ti) + hosted API + Modal $30/month free tier;
**no paid services, no self-hosted GPU** — generator = DeepSeek V4-Pro on NVIDIA NIM (free, 40 RPM, no
credit cap); Modal ≈ $6/month for the 1.7B fine-tunes only.
**Targets:** a guaranteed Q3-tier resource paper (Contribution 1 alone) + an upside Q2-tier
(Findings/COLING) full paper (Contributions 1+2+3).

---

## 0. TL;DR (read this first)

**The question.** In 2024, generating new, correct math problems required a symbolic pipeline
(formalize → mutate in a solver-checkable representation → informalize) because LLMs were not reliable
enough — this is the Li et al. NeurIPS 2024 "neuro-symbolic data generation" line. In 2026, strong open
LLMs (DeepSeek V4, GLM, Gemini 2.5) generate correct problem variants + solutions cheaply and fluently,
including in Vietnamese. **So: does the symbolic machinery — its correctness certificate and its
controllable diversity — still justify its cost and its coverage limits (only linear arithmetic/algebra)
when the goal is to train a *small* model in a *low-resource* language?** Nobody has compared these two
augmentation strategies cleanly for that setting.

**Three contributions.**

| # | Contribution | Publication floor | Risk |
|---|---|---|---|
| **C1** | **Vi-ExamMath** — a contamination-controlled Vietnamese math benchmark from *real* school + competition exams (grade-10 entrance / HSG / chuyên), human-authored, solver-verified, human-QA'd, spanning primary → competition; plus a baseline leaderboard. Two splits: **Core** (numeric, solver+human, the required deliverable) and **Hard** (competition, human-only, optional). | **Q3 (high probability, conditional):** publishable as a resource paper (VLSP / LREC / RIVF / KSE) **if** Core ≥ 300, the solver-verification rate + error taxonomy is a reportable finding, and related work differentiates from V-Math / Vi-S1K. | low–medium |
| **C2** | **A controlled comparison** of symbolic-verified vs. strong-LLM vs. unverified data augmentation for supervised fine-tuning of small models, evaluated on the contamination-controlled benchmark. | **Q2 upside (Findings/COLING)** if the finding is clear or surprising. Downstream signal is fragile at sub-2B — the size-independent **data-quality panel + cost comparison** are the fallback headline. | medium — depends on results |
| **C3** | **A failure analysis** of the naive symbolic-mutation approach (answer-expression perturbation), diagnosing why it degrades data quality, plus the fixes used in C2's symbolic arm. | **The most robust piece** — all data in hand, zero new compute, no benchmark/training dependency. C1 + C3 alone is a coherent Findings-tier submission if C2 underdelivers. | low |

**Everything is free.** Data collection is web-scraping + normalization + human QA (OCR only as a
fallback for scanned PDFs). The augmentation generator is **`deepseek-ai/deepseek-v4-pro-0813`**
(pinned model + access date recorded in the datacard; `seed=42` + fixed sampling params — note hosted
endpoints are not bit-reproducible, so the **released dataset + prompts + commit hash** are the unit of
reproducibility, not the API call) on **NVIDIA NIM's free tier** — the researcher's account is
**rate-limit-only (40 RPM, no credit cap)**. The whole ~15–30k-item corpus is ~1–2 hours of wall-clock.
Fine-tuning is Qwen3-0.6B-Base on the researcher's own laptop (4 GB RTX 3050 Ti, QLoRA) + Qwen3-1.7B-Base
on Modal L4 for ~$6 of the free $30/month credit. No Kaggle.

> **HARD RULE — no model larger than 1.7B is ever *fine-tuned*, and nothing is self-hosted on a GPU.**
> The 7B Mistral + Online DPO + slow `model.generate` loop is exactly what broke the previous attempt's
> compute budget. Every SFT arm is Qwen3-0.6B-Base (laptop) or Qwen3-1.7B-Base (Modal L4); every eval
> sweep is a ≤1.7B adapter under vLLM. Every larger model is *frozen* and reached through a **hosted
> API**: the NIM generator + weaker-generator + solver + difficulty-reference models (§6.0 rows A/B/C/D/I),
> and leaderboard rows via free endpoints / published numbers.

**This is not a pivot.** The subject is unchanged — math-data augmentation for Vietnamese. The existing
project pipeline becomes one experimental arm (S1). The existing Vietnamese translations become a seed
pool and a secondary evaluation set. What changes is the *question* (2026-relevant) and the *rigor*
(unified protocol, contamination-controlled evaluation).

---

## 1. Background & motivation

### 1.1 Math-data augmentation, 2024 → 2026

The dominant recipe since late 2023: take GSM8K + MATH training seeds, use a strong model to rewrite
questions and/or sample many solutions, filter by answer-match, and SFT a base model. Representative
methods and their mechanisms:

- **MetaMath** (ICLR 2024, arXiv:2309.12284) — question bootstrapping: rephrasing, forward/backward
  (FOBAR) reformulation, answer augmentation. 395K examples. LLaMA-2-7B: GSM8K 66.5 / MATH 19.8.
- **WizardMath** (ICLR 2025, arXiv:2308.09583) — Evol-Instruct + reward model + PPO.
- **DART-Math** (NeurIPS 2024, arXiv:2407.13690) — difficulty-aware rejection tuning; observes that
  vanilla rejection sampling is biased toward easy problems.
- **MathGenie** (ACL 2024, arXiv:2402.16352) — perturb solutions, then *back-translate* to questions to
  avoid logical inconsistency.
- **RV-Syn** (2025, arXiv:2504.20426) — decompose seeds into a typed Python function library, compose
  new problems as **computation graphs (DAGs)**, back-translate to natural language, execute to verify.
  +6.3% over prior synthetic SOTA on LLaMA-3-8B, 2× data efficiency.
- **MathAgent** (ACL 2026 Findings, arXiv:2604.11188) — data synthesis as optimization over a
  **constraint graph**, then semantic instantiation decoupled from language.

Field consensus (DART-Math, "Common 7B LMs Already Possess Strong Math", MuggleMath): base models
already contain the capability; SFT augmentation mostly raises pass@1 *reliability*; gains are largely
**in-distribution** and **do not transfer** across domains (augmenting GSM8K does not help MATH).

### 1.2 The neuro-symbolic line and its assumption

**Li et al., "Neuro-Symbolic Data Generation for Math Reasoning"** (NeurIPS 2024, arXiv:2412.04857) is
the flagship. Mechanism: represent each problem's constraints in **SMT-LIB**; use an LLM to informalize
in both directions; use solvers (Z3, CVC4, MathSAT, SymPy, SciPy via PySMT) to guarantee the mutated
problem is satisfiable and to compute the ground-truth answer; walk the symbolic space with projected
MCMC so every accepted sample is valid. ~860K problems. Beats MetaMath / WizardMath at equal model size
(Mistral-7B: GSM8K 86.8 / MATH 37.3).

**The implicit assumption:** you need the symbolic layer because an LLM writing a problem + answer is not
trustworthy enough, and because mutating in symbolic space gives *controllable*, *guaranteed-valid*
diversity that free-form LLM rewriting does not.

### 1.3 Why 2026 challenges the assumption

- Frontier LLMs are reachable for free — e.g. DeepSeek V4 Pro on NVIDIA NIM's free tier — and generate
  fluent, usually-correct Vietnamese problem variants + solutions. Generating tens of thousands is a
  bounded job.
- A final-answer check with SymPy (or LLM self-consistency) catches most errors.
- So the *marginal* value of an SMT pipeline over "strong LLM + light check" is now unclear —
  especially given the SMT approach only faithfully covers linear arithmetic/algebra (see §3.3).

Meanwhile the community is *aware* of pitfalls in "verified" synthetic data: tautological /
trivially-verifiable specifications are a known problem (VeriGeo, arXiv:2606.14176 builds multi-stage
non-trivial verification); ill-posed synthetic questions need dedicated filtering (MathQ-Verify,
arXiv:2505.13903).

### 1.4 The low-resource angle

There is **no** verified/symbolic math-data-augmentation study for any low-resource language.
Multilingual math work (MathOctopus / MSVAMP, arXiv:2310.20246; African-languages math, arXiv:2505.19848)
is **translate-train** only. MGSM (arXiv:2210.03057) has **no Vietnamese split**.

**Vietnamese math benchmarks that already exist** — the space is no longer empty, so C1 must be positioned
against them, not claimed as "the first":

| Benchmark | Level / format | Gap it leaves |
|---|---|---|
| **VNHSGE** (arXiv:2305.12199) | national high-school-graduation (THPT QG), 9 subjects, ~19k **MCQ**, 2019–2023 | multiple-choice, static, **now contaminated** (3+ years old) |
| **V-Math / NHSGMEs** (arXiv:2509.12251, Sep 2025) | THPT QG math, MCQ + free-response, human worked solutions, agentic solver | THPT-only; **no solver certificate**; no per-item contamination provenance |
| **Vi-S1K / Vi-Elementary-Bench** (arXiv:2604.17794, VNU) | **elementary** math, 1,010 items, Qwen3-1.7B test-time-scaling | translation-derived reasoning traces; elementary-only; same group + same base model as this thesis (**scoop risk — monitor**) |
| **THPT-Ladder / "Partial-Credit Gap"** (arXiv:2608.18336, 2026) | 632 items, 21 official THPT exams, 11 subjects, graded on the 2025 marking scheme | true/false format, THPT-only |
| **VMLU**, **VMMU/ViExam** (arXiv:2508.13680) | MCQ under data agreement / multimodal exam QA | MCQ or multimodal, not free-form numeric math |

**What no Vietnamese benchmark does, and Vi-ExamMath will:** (i) build from **grade-10 entrance exams
(`đề thi vào 10`) and competition papers (HSG / trường chuyên)** — nobody has turned these into an LLM
benchmark; (ii) span **primary → lower-secondary → grade-10 → competition** in one difficulty ladder,
filling the gap *between* Vi-Elementary-Bench (primary) and V-Math (THPT); (iii) attach a **solver
verification certificate** and **per-item provenance** (year / province / exam / license) to argue
against pretraining inclusion. It stays **math-only** — "diversity" means topic and grade band, not
extra school subjects (physics/chemistry break solver verification and have no link to the augmentation
study).

### 1.5 The evaluation problem

GSM8K and MATH are in every model's pretraining. GSM1k (arXiv:2405.00332) showed accuracy drops up to
13 pp on a fresh, difficulty-matched set; Inference-Time Decontamination reduces GSM8K accuracy ~23%.
**When an augmentation method "improves GSM8K," part of that may be the augmented data acting as a
memorization trigger, not genuine learning.** A benchmark of *real, recent, human-authored* problems that
the models have not seen is the instrument that separates the two — and it must be curated, not
generated (a generated benchmark defeats its own purpose).

---

## 2. What already exists in this project

Repository: `E:\MATH-REASONING-THESIS\vi-math-smt`. Draft paper: `E:\CONFERENCE\conference_paper_IEEE_en.tex`.

### 2.1 Assets that are sound and reused as-is

| Asset | Location | Notes |
|---|---|---|
| Vietnamese translations of GSM8K + MATH train (14,870) and 4 test sets (Vi-GSM8K 1,319 / Vi-MATH 5,000 / Vi-SVAMP 1,000 / Vi-ASDiv 2,305) | `data/finetune/`, `data/test/` | Gemini 2.5 Flash, component-wise; translations verified clean by spot check. → seed pool + secondary eval. |
| SMT-LIB formalization of GSM8K train (7,473/7,473) via a LangGraph **ReAct agent** with Z3 as a tool | `src/formalize/gsm8k/` | Genuinely works; linear arithmetic in `QF_LRA` is faithful. → machinery for S1 and for verifying Vi-ExamMath. |
| The formalize–mutate–informalize pipeline code | `src/mutation_informalize/` | Architecture reused; operators replaced (§6.1). |
| `data/test/metaMATH/metaMathQA-395K_json_vi.jsonl` — full MetaMathQA-395K translated to Vietnamese (vi + en) | `data/test/metaMATH/` | Optional translate-train comparison arm. |
| WizardMath-7B-V1.1 predictions on the Vietnamese test sets (**60.3% Vi-GSM8K, 21.8% Vi-MATH** under the old scorer) | `data/result/wizardMATH/` | Deprioritized. A 2023 English 7B math model is not a meaningful reference at 1.5–1.7B scale in 2026. Keep only as an **optional, free** historical leaderboard row: re-score the *existing* prediction files through `math_verify` (CPU, no GPU). Do **not** re-run it. Never frame a 1.5B arm as "beating WizardMath-7B". |

### 2.2 Audit findings (measured — these become material for C3 and design decisions)

**N1. The shipped mutation engine only perturbs the answer expression.** `src/mutation_informalize/
mutation_engine.py` implements exactly three operators (`mutate_structure`, `mutate_expression`,
`mutate_difficulty`), all of which wrap the `answer` term in arithmetic (`answer := (expr+k)·k`,
`answer := expr + 2h`, `answer := expr·(expr+1)`). Measured on the 19,495 accepted augmented samples:

- **40.7%** have decimal/ugly-fraction answers (`366/25`, `2801.62`); the seed data has **1.6%**.
- **16.3%** leak formal artifacts into the problem text: raw variable names (`h_16296 = 4.61`),
  verbalized SMT constraints ("the result does not exceed 20"), or the answer stated in the prompt
  ("Expected result: answer = 15.5").
- One operator is a no-op: `(x + 3.03) − 3.03 = x`.

**N2. For MATH, the "verification" is often vacuous.** Auditing 14,870 base SMT programs: **15.9%
hard-code the answer** (`(assert (= x 1.4142135623730951)) (assert (= answer x))` — the solver checks a
tautology the model built by looking at the ground truth). Only ~40% genuinely *derive* the answer
through ≥2 relational constraints, and those are dominated by GSM8K's linear structure. `QF_LRA` cannot
faithfully express radicals, digit extraction, most number theory, geometry, or combinatorics. The
formalizer's anti-hard-coding gates (`is_suspicious_constant`, `check_answer_not_hardcoded`) are
insufficient.

**N3. Fine-tuning configuration bugs caused unreliable outputs** (repetition, format collapse) in the
initial pipeline, so its downstream numbers are not trustworthy and **all training/evaluation is redone
from scratch under a unified protocol** (§6.3–6.4). Root causes: `MAX_SEQ_LENGTH=2048` truncates MATH
solutions before the answer marker (model never learns to stop → loops); inference `repetition_penalty=
1.3` + `no_repeat_ngram_size=4` is far too aggressive for math (which must repeat digits/variables);
the training `text` field begins with a literal `<s>` while the tokenizer also adds BOS (double-BOS,
known to degrade Mistral); SFT loss is computed over the prompt tokens (no completion-only masking);
the base model (Mistral-7B-Instruct-v0.2) is weak at math.

**N4. Draft-paper wording must be re-aligned with the code** (for the rewrite): the draft says "5
mutation strategies M0–M4" (there are 3; the `mutation_strategy` field is a running index, not a label),
"Z3 / CVC5" (only Z3 is used), "generate–verify–repair loop" for MATH (it is a sequential loop; the
Phase-3 repair path never executes because `max_attempts` defaults to 1), and "SMT-LIB integrated into
the training data" (the SMT is dropped before training).

---

## 3. Research questions

- **RQ-A — Is symbolic still worth it?** For SFT of a small model in Vietnamese, does symbolic-verified
  augmentation (S1) beat strong-LLM-generated + answer-checked augmentation (S2)? On which axis —
  downstream accuracy, robustness, answer-distribution match, or generation cost?
- **RQ-B — Does verification matter, independent of who does it?** S1 vs. S4 (symbolic without the
  solver check) and S2 vs. S3 (LLM without the answer check), at matched sample budget and matched
  difficulty distribution.
- **RQ-C — Which data properties predict downstream utility?** Symbolic data is uglier (worse
  naturalness, worse answer cleanliness) but *not distilled from a teacher*. LLM data is fluent but
  carries teacher style/bias and diversity collapse. Regress the data-quality panel against downstream
  accuracy.
- **RQ-D — How much augmented data is enough?** Data-efficiency curves for S0/S1/S2 at
  N ∈ {2k, 5k, 10k, 20k}.
- **RQ-E — Is any of this low-resource-specific?** Repeat S0/S1/S2 on original English GSM8K; compare
  the arm ranking.

---

## 4. Contribution 1 — Vi-ExamMath (build first, in parallel with everything)

### 4.1 Design goals

A **native**, **free-form**, **solver-verified**, **contamination-controlled** Vietnamese math benchmark
built from real Vietnamese school and competition exams. "Contamination-controlled" = every problem is
human-authored, drawn from a dated public exam, and carries provenance (year / province / exam / license)
sufficient to argue it post-dates or is absent from a given model's pretraining. It is **math-only**;
"diversity" = topic (arithmetic, algebra, functions, number theory, combinatorics/probability,
computational geometry) and **grade band** (primary → lower-secondary → grade-10 entrance → competition).

### 4.2 Two tiers

The benchmark has two splits with **different verification standards and different roles**:

| Split | Content | Answer type | Verification | Role |
|---|---|---|---|---|
| **Core** | primary + lower-secondary + grade-10 entrance (`đề thi vào 10`), computational items | **numeric** (integer / clean fraction / short radical / decimal-with-unit) | **solver (Z3 / SymPy) + 2 human annotators** | the **C2 downstream instrument** + the guaranteed resource; scope matches the augmentation study (§5.2) |
| **Hard** *(stretch)* | HSG (`học sinh giỏi` grade 9), `trường chuyên` entrance, selected THPT items | numeric **or** closed-form expression | **2 human annotators** (no solver) | difficulty ceiling; leaderboard only; **not** in any C2 statistic |

Only **Core** is required for the paper. Hard is built if the schedule allows and strengthens C1 as a
standalone resource.

### 4.3 Sources — web-first, OCR only as fallback

Vietnamese exam content is published as **selectable text with MathJax `\(…\)` LaTeX** (news outlets,
`loigiaihay.com`) or as **Word/LaTeX files** (`toanmath.com`), so the pipeline is a **scraper +
normalizer**, not OCR. OCR (MathPix free tier / an open Vietnamese math-OCR model) runs only on
scanned-PDF sources with no text layer.

Priority order (Hà Nội first — the Hà Nội Department of Education publishes clean official
exam + answer-key + rubric PDFs every year and the press covers them densely, giving the sharpest
contamination timeline; other provinces are added for the cross-province de-duplication story and for
volume):

1. **Grade-10 entrance exams (`đề thi tuyển sinh lớp 10`)**, Hà Nội 2017–2026, then ~10–15 other
   provinces (TP.HCM, Đà Nẵng, Hải Phòng, Nam Định, Nghệ An, …), recent years. Official answer keys +
   rubrics from the provincial Departments of Education (state documents, usable with attribution).
2. **Competition papers** (Hard split): Hà Nội city-level HSG grade 9; `chuyên` entrance (Hà Nội –
   Amsterdam, Chuyên KHTN, Chuyên Sư phạm, Chuyên Nguyễn Huệ); selected THPT QG items.
3. **Primary / lower-secondary** computational items (Core, easy band): Violympic / provincial primary
   exams, textbook end-of-chapter problems with published solutions.

Concrete channels: `hanoi.edu.vn` (official); VnExpress / VietnamNet / Tuổi Trẻ exam+solution articles
(dated, HTML+LaTeX); `loigiaihay.com` (HTML+LaTeX, ~10 yr Hà Nội + ~40 provinces, free, no paywall);
`thcs.toanmath.com` / `toanmath.com` (PDF **+ Word**, hundreds of exams); `diendantoanhoc.org`
(community LaTeX of HSG/olympiad). See `data/vi_exam/SOURCES.md` for per-source coverage + license
notes.

### 4.4 Construction pipeline

1. **Scrape** each source → raw `(exam_html_or_docx, source, year, province, exam_name, url, license)`.
2. **Normalize** → text with LaTeX preserved; Word→LaTeX via `pandoc` / MathType export; flag every
   item whose statement contains a figure (`<img>`) — those go to a `figure` bucket (Hard, or dropped).
3. **Segment** into `(problem, reference_solution, final_answer)` triples at the **sub-part** level
   (`Câu II.3`, `Bài 3b`), not the `Câu` level.
4. **Scope filter** for Core: numeric final answer; arithmetic / algebra / functions; no figure
   dependency. (Geometry-with-figure, "chứng minh", open-ended → Hard or out.)
5. **Structural de-duplication**: across provinces/years, and against the S1/S2/S5 training pools
   (n-gram + embedding; §6.4).
6. **Solver verification** (Core): EN round-trip for the formal layer only (LLMs formalize more stably
   from English; the *content* stays Vietnamese-authored) → SMT/SymPy → check the official answer +
   the uniqueness gate (§6.1). This simultaneously catches scrape/OCR errors and yields the
   `solver-verified` label. Report the verification rate and an error taxonomy (**a C1 result**).
7. **Human QA** (both splits): 2 annotators (researcher + 1 other) independently re-derive the answer
   and rate well-posedness; report Cohen's κ; adjudicate disagreements.
8. *(if time)* **template-perturbable subset**: ~50 GSM-Symbolic-style templates (typed slots +
   constraints) built from Core items, for a robustness slice.

### 4.5 Measured feasibility (prototype, Sept 2026)

A prototype scraper over `loigiaihay.com` (one listing page) pulled **13 exams** (Hà Nội grade-10
2017–2026 + 4 provinces): **96** `Câu`-level segments, **72** with full solution text, **60** figure-free,
**58** with a regex-extractable final answer. At sub-part granularity that is ≈ 3× more items. One Hà Nội
grade-10 exam ≈ 5 `Câu` ≈ 15–20 sub-items ≈ 8–12 figure-free numeric ones; **10 years of Hà Nội alone
≈ 80–120 Core items**, and `loigiaihay.com` covers ~40 provinces. **Reaching the Core-400 target
requires the multi-province + multi-level spread — a single exam family is not enough.** Extraction
quality is high (LaTeX intact; word problems and linear systems — e.g. "ba lô + máy tính, tổng 885
nghìn, giảm 20%/25%, phải trả 682 nghìn" — come out solver-ready).

### 4.6 Deliverable

**Core: target 400 items** (minimum 300); **Hard: target 200** (optional). Core stratified by grade band
/ topic / answer type; splits `verified` (solver + human) and `human-only`. Per-item metadata: source,
year, province, exam name, URL, license, grade band, topic, answer type, solver-status, κ-status.
`data/vi_exam/DATACARD.md` records the full contamination argument.

**Baseline leaderboard** (every row labeled with parameter count — a leaderboard, not a controlled
comparison, so mixed sizes are fine; all scored through the unified `eval/math_verify.py`):
- *≤1.7B — we run these ourselves (0.6B on the laptop, ≥1B on Modal L4 / a free endpoint, vLLM):*
  **Qwen3-0.6B**, **Qwen3-1.7B**, **Qwen2.5-1.5B**, **Qwen2.5-Math-1.5B**,
  **DeepSeek-R1-Distill-Qwen-1.5B** (the meaningful strong small reference), **Sailor2-1B** (SEA-adapted).
- *larger open (>1.7B) — NOT run locally:* SeaLLMs-v3-7B, Vistral-7B, Qwen3-4B/8B, Qwen2.5-Math-7B — via
  a **free hosted endpoint** (OpenRouter / HF Inference Providers free tier / the model's own free API)
  or a **published number** with an explicit caveat. Optional, non-blocking.
- *API (free):* `deepseek-v4-pro` via NVIDIA NIM.
- *historical, optional, no GPU:* WizardMath-7B — re-score the **existing** `data/result/wizardMATH/`
  predictions through `math_verify` (CPU). Not re-run.

**Discriminativeness gate:** after the leaderboard, check that Core actually separates models — if every
≤1.7B model clusters within ~5 points, or the frontier API model scores > 90%, Core is too easy/hard to
serve as the C2 instrument → rebalance the grade-band mix before the training grid, and lean the paper
on C1 + C3.

**Vi-ExamMath Core on its own is a submittable Q3 resource paper.**

---

## 5. Contribution 2 — the controlled comparison

### 5.1 Held constant

Base model, optimizer, SFT recipe, training-example budget N, evaluation harness, seed problem set.
**No DPO / RL** (it was a confound in the original design and is fragile on free GPUs) — SFT only; RL is
future work.

### 5.2 Seed set

GSM8K (7,473) + the algebra + prealgebra subset of MATH (~2,900), already translated to Vietnamese.
Geometry / number-theory-digit / combinatorics / precalculus are **explicitly out of scope** (SMT is not
faithful there — see N2). This also matches Vi-ExamMath's scope.

### 5.3 Arms (same N, ≥3 seeds for the primary arms)

| Arm | Description |
|---|---|
| **S0** | no augmentation — seed only (floor) |
| **S1** | **symbolic-verified** — the project pipeline, fixed (§6.1): constrained mutation that preserves answer type + informalization that never sees SMT + a leakage filter + the uniqueness gate |
| **S2** | **strong-LLM augmentation + independent answer check** — `deepseek-v4-pro` (§6.2, NIM free tier) generates a variant problem + solution; kept only if an *independent* SymPy check (and/or LLM self-consistency) confirms the answer. "The obvious 2026 approach: ask a frontier LLM." |
| **S3** | strong-LLM augmentation, **no check** — baseline |
| **S4** | S1 **without the solver verification step** — isolates the value of verification in the symbolic branch |
| **S5** | **translate-train** — SFT on a Vietnamese-translated slice of a known-good English augmentation corpus (`metaMathQA-395K_json_vi`, which we already have), at matched N. The "existing best practice" reference — 2026 papers that touch low-resource math default to this. |
| **S6** *(optional)* | hybrid — structural symbolic mutation + LLM informalization + double check |

### 5.4 Measurements per arm

- **Generation cost:** API tokens / $ / GPU-hours / wall-clock / number of LLM calls.
- **Data-quality panel:** answer-distribution divergence vs. seed (Jensen–Shannon) + answer-magnitude
  histogram; leakage rate (regex + NER for formal tokens, SMT keywords, answer-in-prompt); diversity
  (distinct-2, self-BLEU, embedding dispersion, k-seed sweep); **independently-checked correctness rate**
  (a verifier *different* from the one used during generation — SymPy + LLM self-consistency); naturalness
  MOS on ~200 samples, ≥3 raters, Krippendorff α; real-vs-synthetic discriminator AUROC (near 0.5 =
  good); contamination overlap (n-gram + embedding) against every test set; for symbolic arms, the
  "genuinely derived" fraction from the uniqueness gate.
- **Downstream utility:** SFT Qwen3-1.7B-Base and Qwen3-0.6B-Base (see §6.3) → evaluate on
  **Vi-ExamMath Core** and **Vi-GSM8K** (co-primary — Vi-GSM8K is where sub-2B models are most likely to
  show separable signal and it links to the literature; Vi-ExamMath Core is the contamination-clean
  instrument), a GSM-Plus-style Vietnamese robustness slice, and the Vi-ExamMath `template` subset. The
  `Hard` split is leaderboard-only, never a C2 number.

### 5.5 Analysis

1. **RQ-A:** S1 vs. S2 across every axis. Explicitly separate "which produces a better model" from
   "which is cheaper/simpler/broader in coverage."
2. **RQ-B / verification premium:** S1 vs. S4 and S2 vs. S3, with S4/S3 resampled to match S1/S2's
   difficulty histogram (pre-empts the "the verifier is just a difficulty filter" objection).
3. **RQ-C:** quality → utility regression; report which properties predict gain.
4. **RQ-D:** data-efficiency curves.
5. **RQ-E:** English control; does the arm ranking differ?

**Any outcome is publishable:** "symbolic is obsolete for this regime" / "symbolic still wins on
robustness or hard items" / "both underperform no-aug at 1.5B" — each is a timely, citable finding.

---

## 6. Methods detail

### 6.0 Every model, and what it is for (canonical — resolve any ambiguity here)

| # | Role | Model | Notes |
|---|---|---|---|
| **A** | **Generation** — S1 informalization (symbolic → Vietnamese text) + S2/S3 variant+solution generation | **`deepseek-ai/deepseek-v4-pro-0813`** via **NVIDIA NIM**, `seed=42` — researcher's account is **40 RPM, no credit cap** (confirmed). | **The one fixed generator** — a frontier model, reproducible. Optional: batch 5–10 items/call for speed. Fallback chain: NIM → DeepSeek direct API (5M-token grant) → Groq (free). **All hosted API — no local GPU.** |
| **B** | **New formalization** — verify Vi-ExamMath, re-formalize the MATH algebra subset (write SMT/SymPy from a problem) | `deepseek-v4-pro` via NIM (same as A) | Z3 / SymPy + the uniqueness gate do the actual *verification*; the LLM only drafts the formal code. |
| **C** | **S2-alt subset** (~1–2k) — generator-strength robustness check: does the S1-vs-S2 finding hold with a *weaker* generator? | **`meta/llama-3.1-8b-instruct`** via **NVIDIA NIM** (free, same key as A), `seed=42`, frozen | A deliberately weaker hosted model. Any weaker LLM serves the design — no local GPU, no self-hosting. |
| **D** | **Round-trip solver check** — independently re-solve a generated problem, majority vote vs the symbolic answer | **SymPy exact** (primary) + `deepseek-v4-pro` (NIM) + `meta/llama-3.1-8b-instruct` (NIM) — different families, all hosted | SymPy is the real check; 2 independent hosted LLM solvers back it up. Optional 3rd: Groq Llama-3.3-70B (free). |
| **E** | **SFT base models** (fine-tuned — the C2 grid) | **`Qwen3-0.6B-Base`** (runs on the local 4 GB RTX 3050 Ti) + **`Qwen3-1.7B-Base`** (Modal L4, ~$6 of the free monthly credit) — optional `Sailor2-1B-Base` | `-Base`, not `-Instruct`. **Only ≤1.7B is ever fine-tuned. Two sizes are required for the C2 headline.** |
| **F** | **C1 leaderboard — small models we run ourselves** (zero-shot / few-shot) | `Qwen3-0.6B/1.7B`, `Qwen2.5-1.5B`, `Qwen2.5-Math-1.5B`, `DeepSeek-R1-Distill-Qwen-1.5B`, `Sailor2-1B` | local 0.6B; ≥1B on Modal L4 or a free hosted endpoint, vLLM. |
| **G** | **C1 leaderboard — larger models, NOT fine-tuned** | `SeaLLMs-v3-7B`, `Vistral-7B`, `Qwen3-4B/8B`, `deepseek-v4-pro` (NIM), `WizardMath-7B` (re-score existing predictions, CPU) | free hosted endpoints / published numbers. |
| **H** | **Anchor sanity numbers** | `Qwen3-1.7B-Base` + `DeepSeek-R1-Distill-Qwen-1.5B` | zero-shot Vi-GSM8K — checks the harness. |
| **I** | **Difficulty-metric reference solver** (data-quality panel) | **`meta/llama-3.1-8b-instruct`** via **NVIDIA NIM** (frozen, one fixed hosted model) | its solve-rate on an item = that item's difficulty label. Hosted — no local GPU. |
| **J** | **Embeddings** — diversity dispersion, contamination overlap, real-vs-synthetic discriminator | one multilingual sentence-embedding model | not generative. |

**Three things to remember:** (1) generation = `deepseek-v4-pro-0813` via NVIDIA NIM (40 RPM, no credit
cap), one fixed model, `seed=42`; (2) fine-tuning = Qwen3 `-Base` ≤1.7B only — 0.6B local on the 4 GB
RTX 3050 Ti, 1.7B on Modal L4 (~$6/month); (3) everything else is either a **hosted NIM API call** (no
local GPU: rows A/B/C/D/I) or CPU-only analysis. **Nothing is self-hosted on a GPU** — that constraint,
plus "no 7B fine-tuned," is what keeps this runnable on the researcher's laptop.

### 6.1 The fixed symbolic pipeline (arm S1) — the mutation algorithm, `src/mutation_informalize/`

**Per-seed setup.** Vietnamese text `q`, solution `s`, answer `a`; a Z3-verified SMT-LIB form `φ` (GSM8K
exists; the MATH algebra/prealgebra subset is re-formalized — B in §6.0). Parse `φ` → typed variables,
the `assert` set, the `answer` definition; for O1/O2 also a **lightweight typed DAG** (quantities = nodes,
ops = edges, one query node). Record the seed's **answer-type class** `T` (integer / clean fraction /
small radical / …). Logic: `QF_LRA` for linear (GSM8K + most algebra); `QF_NIA`/`QF_NRA` or SymPy for
the non-linear remainder.

**The three operators replace the old `mutate_structure/expression/difficulty` (which only wrapped the
answer term). None touches the answer expression.**

**M0 — constrained constant substitution** *(the bulk operator)*
1. Identify the free numeric *givens* `C = {c₁…cₖ}` in `φ` (leaf/input values, not derived vars; exclude
   structural 0/1).
2. Per `cᵢ`, derive an admissible range `Rᵢ` from its role (count → positive integers near `cᵢ`;
   divisor → constrained to preserve divisibility; rate/price → same-magnitude positive; …).
3. Sample `c′ ∈ ∏ Rᵢ` (Latin-hypercube / random).
4. Substitute → `φ′`; run Z3: must be **SAT**; extract `a′ = model(answer)`; require
   `a′ ∈ T`; require **every derived variable's model value** ∈ its clean class (no `7/3` mid-solution);
   require `a′ ≠ a` and `a′ ∉` already-emitted variants.
5. Fail → resample (≤ 30 attempts). Pass → `(φ′, a′)`.
6. Recompute the step-by-step solution `s′` **deterministically from the DAG + Z3's variable values**
   (not from an LLM): each step is `name = expr = value`.

**O1 — type-safe step insertion** *(adds one reasoning step)*
1. Pick a node `v` in the DAG (given or intermediate), type `τ(v)`.
2. Introduce a fresh quantity `w` with a sampled clean value.
3. Pick `op ∈ {+,−,×,÷}` such that `op(value(v), value(w)) ∈ T`: `÷` only if `value(w) | value(v)`; `−`
   only if the result stays positive (for counts); etc.
4. Replace `v → v′ = op(v, w)` downstream; add `w` as a new given.
5. Compile the DAG → `φ′`; Z3 must be **SAT**, the query value **unique** (uniqueness gate), `a′ ∈ T`.
6. Assert `depth(query)` increased by exactly 1 (a genuine extra step); recompute `s′`.

**O2 — backward / FOBAR** *(reverses an input ↔ the output)*
1. Pick a clean input leaf `c`; promote `c` to the query, demote the old query `v*` to a given with its
   verified value `a`.
2. New answer `= value(c)` — a former clean input ⇒ **clean by construction**, no decimal blow-up.
3. Compile → `φ′`; Z3 must show the system has a **unique** solution for `c` (reject if under-determined).
4. Recompute `s′` (now a backward derivation).

**Gates — applied to every candidate from M0/O1/O2 (S4 skips gates 1 and 4 — that is the ablation):**
1. **Uniqueness gate** *(also the N2/C3 measurement instrument)* — remove the `answer` assert from `φ′`;
   if Z3 still yields a unique model → "derived" ✓; else reject (would be hard-coded).
2. **Informalization** — `phase3_informalize.py::GENERATOR_PROMPT` rewritten to generate from a
   **semantic description / template** (entities, quantities with surface nouns, the query), never the
   raw `(assert …)` list. The generator (DeepSeek V4-Pro) never sees variable names, constraints, or
   `a′`.
3. **Leakage filter** — reject/repair outputs containing `h_\d+`-style tokens, `sum_exponents`, SMT
   keywords (`assert`, `declare-`, `(check-sat)`), or `a′` verbatim in the problem body; NER for
   entities absent from the template.
4. **Round-trip gate** — re-formalize the generated Vietnamese text → Z3 answer must equal `a′` within
   tolerance; **and/or** SymPy + 2 independent hosted LLM solves (DeepSeek V4-Pro + `llama-3.1-8b-instruct`,
   both via NIM), majority must equal `a′`.
5. **Difficulty guard** — the reference solver (`llama-3.1-8b-instruct` via NIM, I in §6.0) solve-rate must not
   collapse to ~0 (i.e. not accidentally unsolvable); spot-check flags.

### 6.2 The LLM augmentation arms (S2/S3)

**One fixed generative model — `deepseek-ai/deepseek-v4-pro-0813` via NVIDIA NIM.** S2 is "the
strong-LLM arm": a single, named frontier model, `seed=42`, pinned `temperature`/`top_p`. The
researcher's NIM account is **rate-limit-only: 40 RPM, no credit cap** (can request 200 RPM). Endpoints,
in priority order:
1. **NVIDIA NIM** (`integrate.api.nvidia.com`, model `deepseek-ai/deepseek-v4-pro-0813`, key
   `nvapi-…`) — free, **40 RPM, no credit cap** (confirmed on the researcher's `@hus.edu.vn` account;
   200 RPM on request). **This is the only key needed** — NIM *hosts* DeepSeek's model.
2. **DeepSeek's own API** (`api.deepseek.com`, a *separate company*, key `sk-…` from a separate signup
   at platform.deepseek.com) — 5M free tokens on signup, 60 RPM. **Optional**, only as a fallback if
   NIM is ever unavailable, or a parallel endpoint for extra throughput.

**Volume fit.** Smoke-tested Sept 2026 (`_scratch/nim_smoke.py`): the key works, `deepseek-ai/
deepseek-v4-pro-0813` resolves and is in the model list, `seed=42`+`temp=0` gave byte-identical output
across repeats, and it solves a Vietnamese grade-10 word problem correctly. **But latency is ~35 s /
call** (~350-token reasoning output) — so throughput is latency-bound, not RPM-bound: at 40 concurrent
requests, 10k items ≈ 2.5 h; serial it would be days. **Batching 5–10 items/call is mandatory, not
optional**, and concurrency must be pushed to the 40-RPM ceiling. `deepseek-v4-flash-0731` (also on NIM)
is the faster fallback for high-volume solver/round-trip checks. The data-efficiency curve runs at
2k/5k/10k.

- **Fallback chain** (documented switchover point): NIM (DeepSeek V4-Pro) → DeepSeek direct API
  (5M-token grant) → Groq (free). All hosted — no local GPU in the chain.
- Prompt: generate a variant of the seed at a specified difficulty + step-by-step solution + boxed
  answer. Encourage structural (not just numeric) variation; sample with diversity.
- **S2-alt — generator-dependence check:** a small-N arm (~1–2k) generated with a deliberately
  **weaker** model — **`meta/llama-3.1-8b-instruct` via NVIDIA NIM** (free, same key, `seed=42`). Show
  whether the S1-vs-S2 ranking holds when the LLM is weaker — if it flips, the finding is
  generator-dependent and we say so. The design only needs "a weaker LLM"; any weaker hosted model works.
- **S2 check** — an **independent** SymPy evaluation of the answer (primary), backed by LLM
  self-consistency across {DeepSeek V4-Pro (NIM), `meta/llama-3.1-8b-instruct` (NIM), optional Groq
  Llama-3.3-70B (free)} — majority vote. Solver-check volume is low.
- S3: keep everything (no check).

**No paid services, no self-hosted GPU.** NIM free tier (no credit cap) + DeepSeek 5M grant + Groq free
tier + the $30/month Modal credit (fine-tuning only — 0.6B is local).

### 6.3 Fine-tuning protocol (fixes N3; identical across all arms)

- **Unsloth** LoRA, rank 16, `use_gradient_checkpointing="unsloth"`, **`packing=True`** (sequence
  packing — the single biggest speedup for short math examples).
- `max_seq_len` set from the p99 of the training-text length (do not truncate solutions before the
  answer marker).
- **Completion-only loss** (mask the instruction/problem tokens).
- No literal `<s>` in the text field (let the tokenizer add BOS once).
- 1 epoch, cosine schedule, effective batch 16–32, bf16 on L4 / 4-bit QLoRA locally, seed logged.
- **Where each base trains:** **Qwen3-0.6B-Base** runs *locally* on the researcher's 4 GB RTX 3050 Ti
  (QLoRA r=16, `packing=True`, ~30–50 min/run — the whole 0.6B grid is a few laptop-nights).
  **Qwen3-1.7B-Base** does not fit 4 GB → its ~25 grid runs go to **Modal L4** (~20 min/run ≈ 8 GPU-h ≈
  **$6 of the free $30/month**). No Kaggle. Everything else (generation, verification, mutation, the
  data-quality panel, analysis) is laptop CPU + hosted API.
- **Base models — use `-Base` (not `-Instruct`)**: the SFT arm *is* the instruction tuning, so a base
  model removes the instruction-data confound; 2026 augmentation papers (e.g. AgentMath, ICLR 2026) do
  the same.
  - **Two sizes, same family (required for the C2 headline):** **Qwen3-0.6B-Base** and
    **Qwen3-1.7B-Base** (April 2025, 36T-token pretrain, 119 languages incl. Vietnamese). The pair lets
    the paper report whether the arm ranking is size-stable; a single size would not convince reviewers.
    The sub-2B / free-compute / low-resource regime is stated as the deliberate scope — the setting
    where the practical stakes are highest (precedent: *Can A Gamer Train A Math Reasoning Model?*,
    arXiv:2506.08935, 1.5B on one consumer GPU).
  - *Optional 3rd:* **Sailor2-1B-Base** (Qwen2.5-based, continually pretrained on SEA languages incl.
    Vietnamese) — "does a Vietnamese-adapted base change the arm ranking?"
  - *Optional robustness check:* Qwen2.5-Math-1.5B — a math-saturated base has less headroom for
    augmentation to help; a useful contrast if quota allows.
  - **Not** Mistral-7B (too large for the grid, weak at math). **Not** WizardMath (2023, English, 7B —
    not a meaningful reference at this scale; see §4.4).

### 6.4 Evaluation protocol (fixes N4's metric inconsistency)

- **One accuracy definition** for all arms and all models: `math_verify` (answer normalization + SymPy
  equivalence + numeric tolerance). Every arm and every leaderboard model is scored through this exact
  function. Anchor numbers: zero-shot Qwen3-1.7B-Base (few-shot) and DeepSeek-R1-Distill-Qwen-1.5B on
  Vi-GSM8K, checked against published English ranges.
- Harness: LightEval or lm-eval-harness; log prompt, few-shot count, and parser for every number.
- Inference: **vLLM**, LoRA merged into the base first, greedy decode, `repetition_penalty ≤ 1.1`,
  `max_new_tokens` 512–640.
- Report: pass@1 (overall / by topic / by answer type); a **robustness delta** (accuracy drop on the
  perturbed slice); maj@k where cheap; training efficiency (accuracy per training token). FCR and a
  reasoning-similarity score are kept only as secondary diagnostics, never as evidence of reasoning
  quality.
- **Variance:** ≥3 seeds for primary arms; bootstrap 95% CI over the test set for headline numbers.
- **Decontamination:** n-gram + embedding overlap of every training arm's data against every test set;
  reported.

### 6.5 Contribution 3 (failure analysis) — framing

Neutral, forward-looking. The section reads: *"The obvious way to port Li et al. to word problems —
perturbing the answer expression — produces data with these problems …"* It presents the N1 measurements
as a **motivating analysis of the naive approach** (the way DART-Math motivates its method by first
showing naive rejection sampling is biased), diagnoses the causes, and shows how S1's design removes
them. It also reports the N2 finding (only ~40% of formalizations genuinely derive the answer) as a
limitation of the whole symbolic paradigm. It does **not** frame anything as "our previous mistake";
arm S1 is "the symbolic pipeline (Li et al. lineage)."

---

## 7. Compute & tooling (entirely free — laptop + hosted API + ~$6 Modal)

*This section is purely instrumental — it exists to confirm the study in §§3–6 is runnable on the
researcher's own hardware. It is not a contribution, does not belong in the abstract, and no results
section is organized around it. The previous attempt failed on compute (Mistral-7B + Online DPO + a
per-example `model.generate` loop exceeded free quotas), so the plan removes every part of that: **no
model above 1.7B is fine-tuned, nothing is self-hosted on a GPU, and there is no Kaggle dependency.**
Beyond that, do not spend design effort here — spend it on the research questions.*

**The researcher's machine:** a laptop with an **NVIDIA RTX 3050 Ti, 4 GB VRAM** (verified). This fits a
0.6B model under 4-bit QLoRA + `packing`, and vLLM inference for a 0.6B adapter. It does **not** fit a
1.7B fine-tune or any 7B/8B model — those are the reason for the two escape hatches below (hosted NIM
API for frozen inference; Modal L4 for the 1.7B fine-tunes only).

| Task | Where | Cost |
|---|---|---|
| C1 scrape + normalize (Word/HTML→LaTeX) — OCR only for scanned PDFs | laptop + MathPix free tier (fallback) | $0 |
| Solver verification (Z3 / SymPy) + the whole data-quality panel + all analysis | **laptop CPU** | $0 |
| Generation (S1 informalization + S2/S3) — **one fixed model** | **`deepseek-v4-pro-0813`** (`seed=42`) via NVIDIA NIM — 40 RPM, **no credit cap**; N ≈ 8–10k/arm; ~1–3 h total | $0 |
| Round-trip / self-consistency solver check | **SymPy (laptop)** + `deepseek-v4-pro` (NIM) + `llama-3.1-8b-instruct` (NIM); low volume | $0 |
| S2-alt robustness subset (~1–2k, *weaker* generator) | `meta/llama-3.1-8b-instruct` via **NIM** (free) | $0 |
| Difficulty-reference solver (I) | `meta/llama-3.1-8b-instruct` via **NIM** (free) | $0 |
| **SFT — Qwen3-0.6B-Base** (~30 grid runs) | **laptop RTX 3050 Ti**, 4-bit QLoRA + `packing` | $0 |
| **SFT — Qwen3-1.7B-Base** (~25 grid runs) | **Modal L4** ($0.80/h), ~20 min/run ≈ 8 GPU-h | **~$6** (of the free $30/month) |
| Eval inference — 0.6B adapters | **laptop**, vLLM | $0 |
| Eval inference — 1.7B adapters + ≥1B leaderboard rows | **Modal L4** (bundled with the SFT runs) or a free hosted endpoint | ~$1 |
| Larger-model leaderboard rows (>1.7B) | free hosted endpoints / published numbers | $0 |

**Concrete feasibility check.** *0.6B, local:* Qwen3-0.6B-Base QLoRA r=16 with `packing=True` on ~20k
short math examples on a 4 GB card ≈ 30–50 min; a 0.6B vLLM sweep over ~1k problems ≈ 5–10 min. The
~30-run 0.6B grid is a few laptop-nights. *1.7B, Modal L4:* ≈ 15–25 min per SFT run, ≈ 10 min per
inference sweep; ~25 runs ≈ 8 GPU-h ≈ **$6**. *Generation:* DeepSeek V4-Pro on NIM at 40 RPM (no credit
cap) — the full ~15–30k-item corpus is ~1–3 h wall-clock, zero GPU.

**Modal account** `tran-hoang-dat-2312`, Starter plan: **$30 credit per month, resets each cycle, does
not roll over.** Keep the usage limit at $30 and **do not add a payment method** — this hard-caps spend
so the card can never be charged. L4 $0.80/h. Use the default region and preemptible execution. The 1.7B
grid uses ~$6–10 of one month's credit; it must fit a single cycle (do not spread across months).

**Optimization levers (apply all):** `packing=True` (up to 5× for short sequences) · Unsloth kernels
(≈2×) · **4-bit QLoRA** (fits 0.6B in 4 GB) · **≤1.7B, never 7B** · 1 epoch · LoRA r=16 · **vLLM** for
inference with the adapter merged (10–20× vs. a `model.generate` loop) · `max_new_tokens` 512–640 ·
evaluate on a stratified 1k subset during the grid, full sets only for finalists.
→ One 0.6B SFT run ≈ 30–50 min (laptop); one 1.7B SFT run ≈ 15–25 min (L4); one inference sweep ≈ 5–15 min.

**Grid size (~55 runs):** S0/S1/S2/S5 × 2 bases (0.6B-Base local, 1.7B-Base on L4) × 3 seeds = 24 ·
S3/S4/S6 × 1 base × 2 seeds = 6 · data-efficiency (S0/S1/S2 × {2k,5k,10k} × 1 base × 2 seeds) = 18 ·
English control (S0/S1/S2/S3 × 1 base × 2 seeds) = 8. The 0.6B half runs on the laptop for free; the
1.7B half is ~8–12 L4 GPU-hours ≈ **$6–10, one Modal cycle**.

---

## 8. Timeline (~2.5–4 months calendar; GPU spend within one month)

Engineering is fast (AI-assisted coding). Data collection, API generation under rate limits, human
evaluation, analysis, and writing do not compress. Phases overlap.

| Phase | Duration | Work |
|---|---|---|
| **P0** | ~1 week | New repo structure; `math_verify` + eval harness; Unsloth-SFT + vLLM templates; **wire the DeepSeek-V4-Pro / NIM generation client** (`seed=42`, 40 RPM limiter, resumable); anchor numbers: zero-shot Qwen3-1.7B-Base + DeepSeek-R1-Distill-Qwen-1.5B on Vi-GSM8K (+ free re-score of existing WizardMath predictions). |
| **P1** | 3–4 weeks *(parallel from P0)* | **C1: scrape + normalize + segment + scope-filter + solver-verify + human-QA Vi-ExamMath Core (Hà Nội-first, multi-province, multi-level); Hard split if time.** |
| **P2** | 1.5–2 weeks | Fix the symbolic pipeline (§6.1); build the S1 dataset. |
| **P3** | 1–1.5 weeks | Build S2/S3/S4 datasets (LLM generation, paced against rate limits); run the full data-quality panel. |
| **P4** | 1.5–2 weeks | Training grid (~55 runs): 0.6B half on the laptop, 1.7B half on Modal L4 (~$6–10, one cycle) + vLLM inference. |
| **P5** | ~1 week | Analysis (RQ-A … RQ-E). |
| **P6** | 2–3 weeks | Write; regenerate all figures from real numbers; `RESULTS_PROVENANCE.md`; advisor review; release artifacts; submit. |

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Method novelty is thin (symbolic synthesis is a crowded space in EN) | The paper is framed as a **comparison + resource + analysis**, not a new method. C1 is a standalone contribution. The 2026 question ("is symbolic still worth it") is timely and unaddressed. |
| C2's finding is boring ("LLM wins, as expected") | Still publishable as the first clean measurement; sharpen with cost/coverage/robustness axes where symbolic may still win. And C1 carries the submission regardless. |
| Scrape/normalization errors in Vietnamese math (LaTeX, MathType, figure loss) | Most sources are text/LaTeX/Word, not scans → OCR is the fallback, not the path. Solver verification + human QA are the filters; report the error rate as a C1 finding. |
| A neighbouring group (V-Math THPT, Vi-S1K elementary — **same VNU dept, same Qwen3-1.7B base**) publishes something overlapping | Vi-ExamMath's niche (grade-10 entrance + HSG, solver-verified, contamination provenance, primary→competition ladder) is distinct from both. Monitor arXiv weekly; differentiate explicitly in related work; the symbolic-vs-LLM comparison (C2) is theirs to lose, not ours. |
| Vi-ExamMath Core too small (single exam family ≈ 80–120 items) | Multi-province (~40 available on `loigiaihay.com`) + multi-level (primary → grade-10) spread; prototype confirms ~62% figure-free yield. Minimum 300; hand-transcribe a residual if needed. |
| Vi-ExamMath Core not discriminative (models cluster / saturate) | Discriminativeness gate after the leaderboard (§4.6); rebalance grade-band mix; fall back to C1 + C3. |
| Sub-2B bases → noisy downstream signal | The regime (sub-2B, low-resource, free compute) is stated as deliberate scope, not a limitation. Headline results are the **data-quality panel** (measured directly, size-independent) and the **cost comparison**; downstream is one axis (co-primary Vi-GSM8K + Vi-ExamMath Core), reported with CIs, across **two sizes** (0.6B + 1.7B) so the arm ranking's size-stability is itself a result. If all downstream CIs overlap, the paper leans on C1 + C3 + the panel. |
| $30/month Modal is not enough | The 0.6B half of the grid runs on the laptop for free; only the ~25 1.7B runs need L4 (~$6–10). Cut the English control to 2 arms and non-primary seeds to 2 if the cycle is tight. |
| NVIDIA changes the NIM free tier mid-project | run generation **early** (Sprint 1); fallback chain NIM → DeepSeek direct API (5M-token grant) → Groq (free). All hosted — no GPU fallback needed. |
| Exam-source licensing for the public release | Release only items from official public exams, with per-item provenance and a conservative non-commercial license; textbook-sourced items (if any) stay out of the release. |

---

## 10. Decisions (recommended — confirm or override)

1. **Base models:** Qwen3-0.6B-Base (laptop) + Qwen3-1.7B-Base (Modal L4, ~$6) — both required, same
   family, two sizes. Optionally Sailor2-1B-Base and/or Qwen2.5-Math-1.5B. Use `-Base`, not `-Instruct`.
   No Mistral-7B, no WizardMath re-run, no DPO, no self-hosted GPU.
2. **Seed scope:** GSM8K + algebra/prealgebra MATH only; other subjects out of scope (justified by N2).
3. **"Strong LLM" for S2/S3:** `deepseek-v4-pro-0813` via NVIDIA NIM (`seed=42`, 40 RPM, no credit cap).
   Fallback → DeepSeek direct → Groq (all hosted). S2-alt *weaker-generator* subset =
   `meta/llama-3.1-8b-instruct` via NIM (§6.2).
4. **Reference arms:** S5 = translate-train is a **baseline band, not a controlled arm** (its seed
   coverage + transformation family differ from S0–S4 — match by N only, report separately). Primary
   source `metaMathQA-vi` (already have); optional S5b = `OpenMathInstruct-1-50k-vi` (5CD-AI on HF,
   different solution family). DeepSeek-R1-Distill-Qwen-1.5B is a zero-shot leaderboard reference.
5. **Vi-ExamMath size:** Core target 400 (min 300); Hard target 200 (optional). Math-only; multi-level
   (primary → grade-10 → competition); Hà Nội-first, ~10–15 provinces.
6. **Venue:** submit **C1 alone** to a Q3 venue early (VLSP 2026, deadline ~Aug–Sep; or LREC / RIVF /
   KSE 2027). Submit the **full paper (C1+C2+C3)** to an ARR cycle in Q1–Q2 2027 → COLING / EMNLP
   Findings. Two outputs reduce risk. Do not use TACL unless it is the sole first submission (9-month
   ARR lockout).

---

## 11. Reproducibility & provenance requirements

- Every number in the paper must trace to a prediction file and a training/inference run ID, recorded
  in `RESULTS_PROVENANCE.md` (arm, base model, seed, N, data commit hash, run link, date).
- Per-adapter smoke test before any full run: 20 Vi-GSM8K samples, `format_ok` > 80%, no repetition.
- Sanity bounds: zero-shot < S0; small-model numbers within published English ranges (Qwen3-1.7B and
  DeepSeek-R1-Distill-1.5B have known GSM8K/MATH500 figures); the free WizardMath re-score should land
  near 60.3 / 21.8. No arm is ever framed as "beating" a differently-sized model.
- Round-trip check on every retained S1 variant: re-formalize → Z3 matches the answer within tolerance.
- Vi-ExamMath: two independent QA passes with reported κ; no benchmark item structurally duplicates a
  training item; year/province recorded to argue against pretraining inclusion.
- Advisor reviews the results table, 20 random predictions, 20 samples per augmentation arm, and 20
  Vi-ExamMath items before submission.
- Release: the fixed augmentation pipeline, all arm datasets, Vi-ExamMath, and the evaluation harness.

---

## 12. Honesty notes (internal)

- The initial pipeline's downstream evaluation is not trustworthy (N3) and is fully redone under the
  unified protocol; the new paper reports arm comparisons, not named models with headline numbers.
- The draft `E:\CONFERENCE\conference_paper_IEEE_en.tex` is not submitted anywhere; the new paper is
  written fresh.
- The advisor (Dr. Ha My Linh) is aware of the true state and this direction.
- Whether C2 concludes "symbolic no longer earns its keep" or "symbolic still wins for X", both are
  reported faithfully.
- **Publication-tier realism.** The "Q3 cannot fail" of the original framing is now "Q3 high
  probability, conditional" — V-Math (arXiv:2509.12251) and Vi-S1K (arXiv:2604.17794) have narrowed the
  gap. What still clears a Vietnamese resource venue: the **solver-verification certificate** (no VN
  benchmark has one), the **grade-10-entrance / HSG sources** (untouched), and **per-item contamination
  provenance**. The N2 problem applies to the benchmark too — grade-10 exam sub-parts are ~40–50%
  solver-numeric-friendly (rút gọn biểu thức, chứng minh, GTNN, hình học are not) → the `verified` split
  may be 200–300 and the rest `human-only`; that is acceptable for a resource paper but the
  solver-verified headline covers only part of Core. **Decision gate (P4/T5.5):** if the sub-2B training
  grid returns overlapping CIs, submit **C1 + C3** (benchmark + failure analysis + data-quality panel)
  as the paper and demote C2 to a measured sub-result — do not stretch a null C2 into a headline.
- **Speed is a risk mitigation.** The neighbouring VNU group could publish a grade-10 or entrance-exam
  benchmark. Submit C1 to VLSP 2026 as early as the benchmark allows.

---

## 13. Immediate next steps (Phase P0)

1. **Local SFT smoke test first:** confirm Qwen3-0.6B-Base QLoRA (r=16, `packing=True`, 4-bit) trains on
   the 4 GB RTX 3050 Ti — 200 steps on a Vi-GSM8K slice, watch VRAM. Then **Modal:** install the client,
   run `get_started.py`, write one L4 LoRA-SFT function (Unsloth) + one vLLM-inference function for the
   1.7B runs. Do **not** add a payment method.
2. **DONE (Sept 2026):** `NVIDIA_API_KEY` is in `.env` and smoke-tested — model resolves, seed
   deterministic, VN math correct, **~35 s/call → batching + concurrency mandatory** (§6.2). Optional
   `DEEPSEEK_API_KEY` / `GROQ_API_KEY` not needed yet. Still to do: full 20-seed generation-quality +
   solver bench (T0.6).
3. New repo skeleton separating: IR / symbolic-mutation / LLM-augmentation / informalization /
   verification / eval-harness. Stand up `math_verify`. Anchor numbers on Vi-GSM8K: zero-shot
   Qwen3-1.7B-Base (few-shot) + DeepSeek-R1-Distill-Qwen-1.5B; free re-score of the existing WizardMath
   predictions.
4. Prototype constrained constant substitution + the leakage filter on 100 already-formalized GSM8K-vi
   problems; measure the % integer-answer and % leakage vs. the current pipeline; read 30 outputs by
   hand.
5. Prototype the uniqueness gate on 500 base SMT programs; measure the genuinely-derived rate by
   subject.
6. **C1 prototype (done Sept 2026):** scraper over `loigiaihay.com` → 13 exams, 96 `Câu`-level segments,
   72 with solutions, 60 figure-free, 58 with regex final-answer; LaTeX extraction clean. Next: extend
   to sub-part granularity + Word sources (`toanmath.com`) + HSG/chuyên; wire `data/vi_exam/SOURCES.md`;
   ask the advisor about official Sở GD-ĐT Hà Nội channels; hand-check 30 extracted triples.

---

## 14. Key references

- Li et al., *Neuro-Symbolic Data Generation for Math Reasoning* — NeurIPS 2024 — arXiv:2412.04857 (S1 lineage)
- Zhang et al., *RV-Syn* — 2025 — arXiv:2504.20426 · *MathAgent* — ACL 2026 Findings — arXiv:2604.11188 (verified DAG synthesis, EN)
- Mirzadeh et al., *GSM-Symbolic* — ICLR 2025 — arXiv:2410.05229 (re-instantiation as training data: negative)
- Yu et al., *MetaMath* — ICLR 2024 — arXiv:2309.12284 · Lu et al., *MathGenie* — ACL 2024 — arXiv:2402.16352 (LLM-generated augmentation)
- Tong et al., *DART-Math* — NeurIPS 2024 — arXiv:2407.13690 (naive rejection sampling is biased; motivation-then-method structure)
- Zhang et al., *GSM1k* — NeurIPS 2024 — arXiv:2405.00332 · *rephrased-sample contamination* — arXiv:2311.04850 (clean evaluation)
- *MathQ-Verify / Let's Verify Math Questions Step by Step* — 2025 — arXiv:2505.13903 (question-validity filtering)
- *VeriGeo* — 2026 — arXiv:2606.14176 (non-trivial verification)
- Shi et al., *MGSM* — ICLR 2023 — arXiv:2210.03057 · Chen et al., *MathOctopus / MSVAMP* — EMNLP Findings 2024 — arXiv:2310.20246 (multilingual math)
- *Improving Multilingual Math Reasoning for African Languages* — 2025 — arXiv:2505.19848 (low-resource precedent, translate-train)
- **Vietnamese benchmark neighbours (differentiate in related work):** Dao et al., *VNHSGE* — arXiv:2305.12199 (THPT QG, 9-subject MCQ) · *V-Math / NHSGMEs* — arXiv:2509.12251 (THPT QG math, MCQ+free-response, human solutions, agentic) · *Vi-S1K / Vi-Elementary-Bench* — arXiv:2604.17794 (VNU; elementary; Qwen3-1.7B test-time-scaling — closest competitor) · *Partial-Credit Gap / THPT-Ladder* — arXiv:2608.18336 (632 items, 21 official exams, 2025 marking scheme) · *VMMU / ViExam* — arXiv:2508.13680 (multimodal exam QA)
- **Vietnamese math corpora (seed / S5):** 5CD-AI HF datasets — `Vietnamese-395k-meta-math-MetaMathQA`, `Vietnamese-nvidia-OpenMathInstruct-1-50k`, `Vietnamese-microsoft-orca-math-200k` (all machine-translated) · Zalo AI 2023 Elementary Math · `hllj/vi_grade_school_math_mcq`
- *VLSP 2025 Numerical Reasoning QA (ViNumQA)* — ACL Anthology 2025.vlsp-1.25 (Plan B venue)
- Yang et al., *Qwen2.5-Math* — 2024 — arXiv:2409.12122 · Dettmers et al., *QLoRA* — NeurIPS 2023 — arXiv:2305.14314
- *Qwen3 Technical Report* — 2025 — arXiv:2505.09388 (Qwen3-0.6B / 1.7B base models, 119 languages incl. Vietnamese) · *DeepSeek-R1* — Nature 2025 / arXiv:2501.12948 (R1-Distill-Qwen-1.5B)
- *Sailor2: Inclusive Multilingual LLMs for South-East Asia* — 2025 — arXiv:2502.12982 (Sailor2-1B, Vietnamese in pretraining)
- *VibeThinker-1.5B* — 2025 — arXiv:2511.06221 · *DeepScaleR-1.5B* / *FastCuRL-1.5B* (strong small-model math, 2025–2026)
- *AgentMath* — ICLR 2026 (uses Qwen3-1.7B-Base for SFT augmentation comparisons)
- Shin, *Can A Gamer Train A Mathematical Reasoning Model?* — 2025 — arXiv:2506.08935 (1.5B / single-GPU precedent)
- Unsloth documentation — sequence packing / kernel speedups (2026)
