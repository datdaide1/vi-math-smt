# Research Plan — Symbolic vs. LLM Math-Data Augmentation for Small Models, with a Contamination-Controlled Vietnamese Exam Benchmark

**Status:** approved direction, ready to execute.
**Owner:** Tran Hoang Dat (VNU University of Science). Advisor: Dr. Ha My Linh.
**Working paper title:** *Does Symbolic Verification Still Earn Its Keep? A 2026 Comparison of Symbolic
and LLM-based Math-Data Augmentation for Small Models in a Low-Resource Language.*
**Constraints:** single researcher; free compute only (Kaggle + Modal $30/month free tier); **no paid
services** — the generator runs on Gemini's free tier (multi-key), the Modal budget is training-only.
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
| **C1** | **Vi-ExamMath** — a contamination-controlled Vietnamese math-reasoning benchmark built from *real* recent entrance-exam problems (human-authored, solver-verified, human-QA'd), plus a baseline leaderboard of existing models. | **Q3 (cannot fail):** if the benchmark is built, it is publishable as a resource paper (VLSP / LREC / RIVF / KSE). | low |
| **C2** | **A controlled comparison** of symbolic-verified vs. strong-LLM vs. unverified data augmentation for supervised fine-tuning of small models, evaluated on the contamination-controlled benchmark. | **Q2 upside (Findings/COLING)** if the finding is clear or surprising. | medium — depends on results |
| **C3** | **A failure analysis** of the naive symbolic-mutation approach (answer-expression perturbation), diagnosing why it degrades data quality, plus the fixes used in C2's symbolic arm. | supporting section | low |

**Everything is free.** Data collection is OCR + human QA. The augmentation generator is
**`Gemini 3.1 Flash Lite`** on Google's free tier (4–5 keys rotated, ~1.5–2k requests/day, paced over
the generation window) — $0, and it keeps the Modal budget entirely for training. `DeepSeek` free tier
is a second solver for the round-trip check. Training uses free Kaggle GPUs + Modal's $30/month free
credit, on ≤1.7B models, with heavy optimization so the whole grid fits one month's credit (or ~2 weeks
of Kaggle's free weekly quota).

> **HARD RULE — no model larger than 1.7B is ever *fine-tuned*.** The 7B Mistral + Online DPO + slow
> `model.generate` loop is exactly what broke the previous attempt's compute budget. Every SFT arm is
> Qwen3-0.6B-Base or Qwen3-1.7B-Base; every eval sweep is a ≤1.7B adapter under vLLM. Larger models are
> used only *frozen, for inference*: the API generator (§6.2), the small self-hosted solver / S2-strong
> robustness subset on Kaggle free T4, and the leaderboard rows via free endpoints / published numbers.

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

- Strong LLMs — open (Qwen3, DeepSeek, GLM) or a free API tier (Gemini Flash Lite) — generate fluent,
  usually-correct Vietnamese problem variants + solutions. Generating 40K variants is a bounded job.
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
is **translate-train** only. MGSM (arXiv:2210.03057) has **no Vietnamese split**. The closest Vietnamese
neighbor, Vi-S1K / Vi-Elementary-Bench (arXiv:2604.17794, VNU-UET), is a **translation** of s1K.

Vietnamese also has **no native, free-form, contamination-controlled math benchmark**: VNHSGE
(arXiv:2305.12199) is multiple-choice and static; V-Math / NHSGMEs (arXiv:2509.12251) is exam-format;
VMLU is multiple-choice under a data agreement.

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

A **native**, **free-form**, **solver-verified**, **contamination-controlled** Vietnamese math benchmark.
"Contamination-controlled" = problems are human-authored, recent (2023–2025), from specific provincial
exams, and each item records enough provenance to argue it is unlikely to be in a model's pretraining.

### 4.2 Sources (priority order)

1. **Grade-10 entrance exams (`đề thi tuyển sinh lớp 10`), free-response, algebra/arithmetic sections**,
   years 2023–2025, specific provinces. Official answer keys + scoring rubrics are published by
   provincial Departments of Education (state documents — usable for research with attribution).
2. National high-school exam (`THPT QG`) algebra items, 2024–2025; specialized-school (`chuyên`) entrance
   exams.

### 4.3 Construction pipeline (free)

1. Collect PDFs from public provincial-department repositories. → normalize.
2. **OCR** (MathPix free tier, or an open Vietnamese math-OCR model) → LaTeX normalization.
3. Segment into `(problem, reference_solution, final_answer)` triples. Filter to: numeric final answer,
   algebra/arithmetic scope.
4. **Structural de-duplication** (across provinces/years, and against the S1/S2 training pool).
5. **Solver verification of a core subset**: translate the Vietnamese problem to English for the formal
   layer only (LLMs are more stable formalizing from English; the *content* stays Vietnamese-authored),
   formalize to SMT/SymPy, check the official answer. This simultaneously catches OCR errors and yields
   a `solver-verified` label. Report the verification rate and an error taxonomy.
6. **Human QA of the entire benchmark**: two annotators (researcher + one other — classmate or advisor),
   independent answer + well-posedness check, report inter-annotator agreement (Cohen's κ).
7. *(if time)* a **template-perturbable subset** (~50 GSM-Symbolic-style templates with typed slots and
   constraints) for a robustness slice.

### 4.4 Deliverable

Target **600 items** (minimum viable **400**), stratified by grade band / topic / difficulty. Splits:
`verified` (solver-checked) and `human-only`; plus the optional `template` subset. Per-item metadata:
source, year, province, license.

**Baseline leaderboard** (every row labeled with parameter count — this is a leaderboard, not a
controlled comparison, so mixed sizes are fine; all scored through the unified `math_verify` harness):
- *≤1.7B — we run these ourselves on Kaggle T4 + vLLM (cheap, part of the plan):* **Qwen3-0.6B**,
  **Qwen3-1.7B**, **Qwen2.5-1.5B**, **Qwen2.5-Math-1.5B**, **DeepSeek-R1-Distill-Qwen-1.5B** (the
  meaningful strong small reference), **Sailor2-1B** (SEA-adapted, Vietnamese in pretraining).
- *larger open (>1.7B) — NOT run locally:* SeaLLMs-v3-7B, Vistral-7B, Qwen3-4B/8B, Qwen2.5-Math-7B —
  obtained via a **free hosted endpoint** (OpenRouter free tier, HF Inference Providers free tier, the
  model's own free API) or, failing that, a **published number** on a comparable Vietnamese benchmark
  with an explicit caveat. Optional and non-blocking for the paper.
- *API (free):* Gemini 2.5 Flash, DeepSeek V4.
- *historical, optional, no GPU:* WizardMath-7B — re-score the **existing** prediction files in
  `data/result/wizardMATH/` through `math_verify` (CPU). Not re-run.

**Vi-ExamMath on its own is a submittable Q3 resource paper.**

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
| **S2** | **strong-LLM augmentation + independent answer check** — `Gemini 3.1 Flash Lite` (§6.2, free tier) generates a variant problem + solution; kept only if an *independent* SymPy check (and/or LLM self-consistency) confirms the answer. "The obvious 2026 approach: ask a modern LLM." |
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
  **Vi-ExamMath** (primary), Vi-GSM8K (secondary, links to literature), a GSM-Plus-style Vietnamese
  robustness slice, and the Vi-ExamMath `template` subset.

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
| **A** | **Generation** — S1 informalization (symbolic → Vietnamese text) + S2/S3 variant+solution generation | **`Gemini 3.1 Flash Lite`** — free tier, 4–5 API keys round-robin | **The one fixed generator.** Pin `model_version` + decode config. This *is* the "strong-LLM arm". |
| **B** | **New formalization** — verify Vi-ExamMath, re-formalize the MATH algebra subset (write SMT/SymPy from a problem) | `Gemini 3.1 Flash Lite` | Z3 / SymPy + the uniqueness gate do the actual *verification*; the LLM only drafts the formal code. |
| **C** | **S2-strong subset** (~1–2k) — generator-strength robustness check; also the Gemini fallback | `Qwen3-14B` self-hosted on **Kaggle free T4** (vLLM, frozen) | Inference only — not fine-tuned. |
| **D** | **Round-trip solver check** — independently re-solve a generated problem (3 models, majority vote vs the symbolic answer) | `Gemini 3.1 Flash Lite` + `DeepSeek` (free API) + `Qwen2.5-Math-1.5B` self-hosted | Multiple *independent* solvers is the point here. |
| **E** | **SFT base models** (fine-tuned — the C2 grid) | **`Qwen3-1.7B-Base`** + **`Qwen3-0.6B-Base`** (optional `Sailor2-1B-Base`) | `-Base`, not `-Instruct`. **Only ≤1.7B is ever fine-tuned.** |
| **F** | **C1 leaderboard — small models we run ourselves** (zero-shot / few-shot) | `Qwen3-0.6B/1.7B`, `Qwen2.5-1.5B`, `Qwen2.5-Math-1.5B`, `DeepSeek-R1-Distill-Qwen-1.5B`, `Sailor2-1B` | Kaggle T4 + vLLM. |
| **G** | **C1 leaderboard — larger models, NOT run locally** | `SeaLLMs-v3-7B`, `Vistral-7B`, `Qwen3-4B`, Gemini / DeepSeek API, `WizardMath-7B` (re-score existing predictions, CPU) | free hosted endpoints / published numbers. |
| **H** | **Anchor sanity numbers** | `Qwen3-1.7B-Base` + `DeepSeek-R1-Distill-Qwen-1.5B` | zero-shot Vi-GSM8K — checks the harness. |
| **I** | **Difficulty-metric reference solver** (data-quality panel) | `Qwen2.5-Math-1.5B` (frozen, one fixed model) | its solve-rate on an item = that item's difficulty label. |
| **J** | **Embeddings** — diversity dispersion, contamination overlap, real-vs-synthetic discriminator | one multilingual sentence-embedding model | not generative. |

**Three things to remember:** (1) generation = Gemini Flash Lite, nothing else; (2) fine-tuning = Qwen3
`-Base` ≤1.7B only; (3) everything else is frozen inference / evaluation.

### 6.1 The fixed symbolic pipeline (arm S1) — targeted changes, `src/mutation_informalize/`

- **`mutation_engine.py` — replace the three answer-term operators with:**
  - **Constrained constant substitution:** change the numeric constants in the problem (not the answer
    term); constrain the sampling so every intermediate value *and* the final answer stay
    integer/clean; recompute with Z3. Reject variants that become physically nonsensical.
  - **O1 — type-safe step insertion:** add one reasoning step `op(v, w) → v'` with domain/divisibility
    guards so `v'` keeps `v`'s type; rewire downstream; re-verify SAT **and** answer uniqueness.
  - **O2 — backward / FOBAR:** promote a clean input leaf to the query node, demote the old query to a
    given with its verified value; the new answer is a former input ⇒ clean by construction; Z3 checks
    unique solvability.
  - A lightweight intermediate representation is needed for O1/O2: `{typed quantities}` + a reduced
    operation DAG. Not a full IR.
- **`phase3_informalize.py::GENERATOR_PROMPT` — informalize from a semantic description, never from the
  raw `(assert ...)` list.** The LLM must not see variable names, constraints, or the answer.
- **New leakage-filter pass** after Phase 3: reject/repair outputs containing `h_\d+`-style tokens,
  `sum_exponents`, SMT keywords, or the answer verbatim in the problem body.
- **Round-trip gate:** for each retained variant, re-formalize the generated Vietnamese problem and
  check the answer with Z3; and/or re-solve with 1–2 independent LLM solvers under self-consistency.
- **Uniqueness gate (also a measurement instrument for N2/C3):** remove the `answer` assertion; if Z3
  still yields a unique solution, the item is "derived"; otherwise it is "hard-coded" and does not count
  as verified. For non-linear items, use `QF_NRA` / `QF_NIA` or fall back to SymPy.

### 6.2 The LLM augmentation arms (S2/S3)

**One fixed generative model.** S2 is "the strong-LLM arm" — a single, named model, or the condition is
not well-defined. Use **`Gemini 3.1 Flash Lite`** (Google's free tier: 15 RPM / 250K TPM / **500 RPD per
key**) for *both* S1 informalization and S2/S3 generation. With **4–5 free keys** rotated, ≈1,500–2,000
requests/day; paced over the ~3–4-week generation window (overlapping other phases) at N ≈ 8–10k per
arm, this covers the whole corpus. One model identity throughout (multi-key = same model, different
auth). Pin `model_version` + `temperature` + a fixed decoding config for reproducibility.

- **Why not self-host:** the $30/month Modal credit is reserved for training, and a self-hosted
  generator plus prompt iteration would not fit alongside it. Gemini Flash Lite is also *the model a
  budget-constrained person actually reaches for* — which is exactly the on-thesis "easy 2026 way".
- **Fragility mitigation:** (a) run generation **early**, not on the late critical path; (b) fallback
  ready — self-host `Qwen3-8B` on **Kaggle T4** (free, not Modal) if Gemini access degrades.
- Prompt: generate a variant of the seed at a specified difficulty + step-by-step solution + boxed
  answer. Encourage structural (not just numeric) variation; sample with diversity.
- **Generator-strength robustness check (answers "you used a weak LLM"):** an extra small-N arm
  **S2-strong** (~1–2k) generated with a stronger model — self-hosted `Qwen3-14B` on Kaggle free T4, or a
  stronger Gemini tier if free quota allows. Show the S1-vs-S2 ranking does not flip when the generator
  is stronger.
- **S2 check** — multiple *independent* solvers by design: an **independent** SymPy evaluation of the
  answer, plus LLM self-consistency across {Gemini Flash Lite, DeepSeek (free), a self-hosted small
  model} — majority vote. Solver-check volume is low, so free tiers suffice.
- S3: keep everything (no check).

**No paid services.** Free Gemini (multi-key) + free DeepSeek + free Kaggle compute + the $30/month
Modal credit (training only).

### 6.3 Fine-tuning protocol (fixes N3; identical across all arms)

- **Unsloth** LoRA, rank 16, `use_gradient_checkpointing="unsloth"`, **`packing=True`** (sequence
  packing — the single biggest speedup for short math examples).
- `max_seq_len` set from the p99 of the training-text length (do not truncate solutions before the
  answer marker).
- **Completion-only loss** (mask the instruction/problem tokens).
- No literal `<s>` in the text field (let the tokenizer add BOS once).
- 1 epoch, cosine schedule, effective batch 16–32, bf16 on L4 / fp16 on T4, seed logged.
- **Base models — use `-Base` (not `-Instruct`)**: the SFT arm *is* the instruction tuning, so a base
  model removes the instruction-data confound; 2026 augmentation papers (e.g. AgentMath, ICLR 2026) do
  the same.
  - **Primary: Qwen3-1.7B-Base** (April 2025, 36T-token pretrain, 119 languages incl. Vietnamese —
    current standard for small-model math papers).
  - **Secondary: Qwen3-0.6B-Base** (same family, smaller — tests whether the findings hold when the
    base is weak; the realistic low-resource case). Same-family size pair is clean for the paper.
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

## 7. Compute & tooling (entirely free)

*This section is purely instrumental — it exists to confirm the study in §§3–6 is runnable on free
resources. It is not a contribution, does not belong in the abstract, and no results section is
organized around it. The previous attempt failed on compute (Mistral-7B + Online DPO + a per-example
`model.generate` loop exceeded free quotas), so the plan removes every part of that: no 7B, no DPO/RL,
vLLM batched inference, sequence packing, a grid that fits one Modal month or ~2 weeks of Kaggle
quota. Beyond that, do not spend design effort here — spend it on the research questions.*

| Task | Where | Cost |
|---|---|---|
| C1 OCR + normalization | MathPix free tier / open OCR model + local | $0 |
| Solver verification (Z3 / SymPy) + all data-quality metrics | Kaggle CPU / local | $0 |
| Generation (S1 informalization + S2/S3) — **one fixed model** | **`Gemini 3.1 Flash Lite`** free tier, 4–5 keys rotated (~1.5–2k req/day), paced over ~3–4 weeks | $0 |
| Round-trip / self-consistency solver check — **multiple independent models by design** | Gemini Flash Lite + DeepSeek (free) + a small self-hosted solver on Kaggle T4; low volume | $0 |
| S2-strong robustness subset (~1–2k) + Gemini fallback | self-host `Qwen3-8B/14B` on **Kaggle free T4** (vLLM); frozen, inference only — not fine-tuned | $0 |
| **Training** (SFT LoRA, **≤1.7B only**) | **Kaggle free 2×T4** (Unsloth) by default; **Modal L4** ($0.80/h) to burst | $0 – ~$16 |
| **Eval inference** (≤1.7B adapters on our GPUs) | **Kaggle T4 + vLLM** | $0 |
| Larger-model leaderboard rows (>1.7B) | free hosted endpoints / published numbers | $0 |

**Concrete feasibility check.** A Qwen3-1.7B LoRA SFT with `packing=True` on ~20k short math examples on
one T4 ≈ 30–40 min (≈15–25 min on Modal L4). A 1.7B model in bf16 fits a 16 GB T4 with room to spare —
there is no memory pressure, unlike the previous 7B + DPO setup. A vLLM inference sweep of a 1.7B model
over ~2k problems ≈ 10–20 min. **Generation:** Gemini Flash Lite at ~1.5–2k requests/day (4–5 keys) ×
~3–4 weeks ≈ 30–55k calls — enough for N ≈ 8–10k per arm — at **zero GPU cost**, leaving the whole
Modal $30 for training. The Modal budget is never touched by generation.

**Modal account** `tran-hoang-dat-2312`, Starter plan: **$30 credit per month, resets each cycle, does
not roll over.** Keep the usage limit at $30 and **do not add a payment method** — this hard-caps spend
so the card can never be charged. L4 $0.80/h, A100-40GB $2.10/h. Use the default region and preemptible
execution to avoid the 1.5–3× multipliers.

**Optimization levers (apply all):** `packing=True` (up to 5× for short sequences) · Unsloth kernels
(≈2×) · **≤1.7B, never 7B** (the 7B + DPO + generate-loop combo is what broke the previous budget) · 1
epoch · LoRA r=16 · **vLLM** for inference with the adapter merged (10–20× vs. a `model.generate` loop) ·
`max_new_tokens` 512–640 · evaluate on a stratified 1k subset during the grid, full sets only for
finalists · Kaggle 2×T4 in parallel.
→ One ≤1.7B SFT run ≈ 15–40 min; one inference sweep ≈ 10–20 min.

**Grid size (~55 runs):** S0/S1/S2/S5 × 2 bases (Qwen3-1.7B-Base, Qwen3-0.6B-Base) × 3 seeds = 24 ·
S3/S4/S6 × 1 base × 2 seeds = 6 · data-efficiency (S0/S1/S2 × {2k,5k,10k} × 1 base × 2 seeds) = 18 ·
English control (S0/S1/S2/S3 × 1 base × 2 seeds) = 8. ≈ 15–22 GPU-hours ≈ **free on Kaggle (~1.5–2
weeks of quota)** or ~$12–18 on Modal L4. **Fits one month; do not spread across months.**

---

## 8. Timeline (~2.5–4 months calendar; GPU spend within one month)

Engineering is fast (AI-assisted coding). Data collection, API generation under rate limits, human
evaluation, analysis, and writing do not compress. Phases overlap.

| Phase | Duration | Work |
|---|---|---|
| **P0** | ~1 week | New repo structure; `math_verify` + eval harness; Unsloth-SFT + vLLM templates; **wire the Gemini Flash Lite multi-key generation client** (4–5 free keys); DeepSeek key for the solver check; anchor numbers: zero-shot Qwen3-1.7B-Base + DeepSeek-R1-Distill-Qwen-1.5B on Vi-GSM8K (+ free re-score of existing WizardMath predictions). |
| **P1** | 3–4 weeks *(parallel from P0)* | **C1: collect, OCR, verify, human-QA Vi-ExamMath.** |
| **P2** | 1.5–2 weeks | Fix the symbolic pipeline (§6.1); build the S1 dataset. |
| **P3** | 1–1.5 weeks | Build S2/S3/S4 datasets (LLM generation, paced against rate limits); run the full data-quality panel. |
| **P4** | 1.5–2 weeks | Training grid (~50 runs) + vLLM inference. |
| **P5** | ~1 week | Analysis (RQ-A … RQ-E). |
| **P6** | 2–3 weeks | Write; regenerate all figures from real numbers; `RESULTS_PROVENANCE.md`; advisor review; release artifacts; submit. |

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Method novelty is thin (symbolic synthesis is a crowded space in EN) | The paper is framed as a **comparison + resource + analysis**, not a new method. C1 is a standalone contribution. The 2026 question ("is symbolic still worth it") is timely and unaddressed. |
| C2's finding is boring ("LLM wins, as expected") | Still publishable as the first clean measurement; sharpen with cost/coverage/robustness axes where symbolic may still win. And C1 carries the submission regardless. |
| OCR of Vietnamese math is error-prone | The solver-verification step is the filter; report the OCR error rate as a finding. Human QA covers the benchmark. |
| Weak Vietnamese base → noisy downstream signal | Headline results are the **data-quality panel** (measured directly) and the **cost comparison**; downstream is one axis among several, reported with CIs. |
| $30/month Modal is not enough | The grid fits Kaggle's free weekly quota; Modal is an accelerator, not a requirement. Keep runs at 1.5B; cut the English control to 2 arms and non-primary seeds to 2 if needed. |
| Gemini free-tier limits / multi-key fragility | 4–5 keys ≈ 1.5–2k req/day; **start generation early** (Sprint 1–2), keep N ≈ 8–10k/arm; if Gemini access degrades, fall back to a self-hosted `Qwen3-8B` on Kaggle free T4 (same role, one model). |
| Exam-source licensing for the public release | Release only items from official public exams, with per-item provenance and a conservative non-commercial license; textbook-sourced items (if any) stay out of the release. |
| A nearby group (e.g. VNU-UET Vi-S1K authors) publishes something overlapping | The contamination-controlled *native-exam* benchmark and the symbolic-vs-LLM comparison are distinct from their translation-based work; monitor arXiv; differentiate explicitly in related work. |

---

## 10. Decisions (recommended — confirm or override)

1. **Base models:** Qwen3-1.7B-Base (primary) + Qwen3-0.6B-Base (secondary); optionally Sailor2-1B-Base
   and/or Qwen2.5-Math-1.5B. Use `-Base`, not `-Instruct`. No Mistral-7B, no WizardMath re-run, no DPO.
2. **Seed scope:** GSM8K + algebra/prealgebra MATH only; other subjects out of scope (justified by N2).
3. **"Strong LLM" for S2/S3:** `Gemini 3.1 Flash Lite` free tier, 4–5 keys, pinned config. Fallback:
   self-hosted `Qwen3-8B` on Kaggle free T4. Plus an S2-strong robustness subset (§6.2).
4. **Reference arms:** S5 = translate-train (`metaMathQA-vi`) is a first-class arm (existing best
   practice); DeepSeek-R1-Distill-Qwen-1.5B is a zero-shot reference on the leaderboard.
5. **Vi-ExamMath size:** target 600, minimum 400.
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

---

## 13. Immediate next steps (Phase P0)

1. **Modal:** install the client, run `get_started.py`, write one LoRA-SFT function (Unsloth) and one
   vLLM-inference function. Do **not** add a payment method.
2. `.env`: **4–5 `GEMINI_API_KEY`s** (free tier, for the generation client with round-robin rotation) +
   free **`DEEPSEEK_API_KEY`** (second solver). Benchmark generation quality + both solvers on 20
   Vietnamese problems; measure the real per-key RPD.
3. New repo skeleton separating: IR / symbolic-mutation / LLM-augmentation / informalization /
   verification / eval-harness. Stand up `math_verify`. Anchor numbers on Vi-GSM8K: zero-shot
   Qwen3-1.7B-Base (few-shot) + DeepSeek-R1-Distill-Qwen-1.5B; free re-score of the existing WizardMath
   predictions.
4. Prototype constrained constant substitution + the leakage filter on 100 already-formalized GSM8K-vi
   problems; measure the % integer-answer and % leakage vs. the current pipeline; read 30 outputs by
   hand.
5. Prototype the uniqueness gate on 500 base SMT programs; measure the genuinely-derived rate by
   subject.
6. Survey grade-10 entrance-exam sources (public provincial-department archives, recent years); OCR 20
   exams; measure the triple-extraction rate and LaTeX quality. Ask the advisor about official channels
   for obtaining exam sets.

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
- Dao et al., *VNHSGE* — arXiv:2305.12199 · *V-Math / NHSGMEs* — arXiv:2509.12251 · Bui et al., *Vi-S1K / Vi-Elementary-Bench* — arXiv:2604.17794 (Vietnamese neighbors)
- *VLSP 2025 Numerical Reasoning QA (ViNumQA)* — ACL Anthology 2025.vlsp-1.25 (Plan B venue)
- Yang et al., *Qwen2.5-Math* — 2024 — arXiv:2409.12122 · Dettmers et al., *QLoRA* — NeurIPS 2023 — arXiv:2305.14314
- *Qwen3 Technical Report* — 2025 — arXiv:2505.09388 (Qwen3-0.6B / 1.7B base models, 119 languages incl. Vietnamese) · *DeepSeek-R1* — Nature 2025 / arXiv:2501.12948 (R1-Distill-Qwen-1.5B)
- *Sailor2: Inclusive Multilingual LLMs for South-East Asia* — 2025 — arXiv:2502.12982 (Sailor2-1B, Vietnamese in pretraining)
- *VibeThinker-1.5B* — 2025 — arXiv:2511.06221 · *DeepScaleR-1.5B* / *FastCuRL-1.5B* (strong small-model math, 2025–2026)
- *AgentMath* — ICLR 2026 (uses Qwen3-1.7B-Base for SFT augmentation comparisons)
- Shin, *Can A Gamer Train A Mathematical Reasoning Model?* — 2025 — arXiv:2506.08935 (1.5B / single-GPU precedent)
- Unsloth documentation — sequence packing / kernel speedups (2026)
