# 01 — Technical design (evidence-based, 2026-11-07)

**Status:** proposal. Nothing here is executed until the user approves; open items are marked **⚑**.
**Principle (user, 2026-11-07):** the topic is **symbolic data for Vietnamese math**. Architecture follows what
was researched and measured, **not** the old pipeline. Every choice cites its evidence: research reports
`01–06`, tech reports `tech/T1–T6`, content reviews `content_review/R1–R6`, `[TK]` = re-verified by the lead.

---

## 1. Research problems (centred on symbolic data × Vietnamese)

| # | Problem | Evidence that it is real | Question |
|---|---|---|---|
| **P1** | **The certificate covers the number, not the problem.** The program→Vietnamese-text step is unverified. | R5: 97.5% text↔Z3 answer agreement, yet 10.5% teacher-acceptable; T2: of items passing the answer check, only 38.5% acceptable; R4: 45% of `status=pass` rows broken | How do we make symbolic Vietnamese data both verified **and** faithful/natural, and how big is the faithfulness gap? |
| **P2** | **Is verification worth more in Vietnamese than in English?** | Only head-to-head (2602.19187) is English, ±2 pts; LLM reasoning is less accurate in lower-resource languages (01 §2.3); measured systematic semantic errors in machine-made Vietnamese math data 2–19% (R1, R2, R6) | Does the symbolic/verification premium grow from English to Vietnamese (language × method interaction)? |
| **P3** | **Coverage.** Symbolic certificates exist only for part of the curriculum. | Usable certificate: GSM8K 54%, MATH 27% (Algebra 67%, C&P 0%) (R3, R4); VN grade-10 exams centre on equations/systems/quadratics/Vi-ét (04, W4 prototype) | What share of Vietnamese school/exam math is certifiable, and does training on certifiable types transfer to non-certifiable ones? |

**Contributions:** C1 Vi-ExamMath (instrument for P2/P3; solver-checked keys, type tags "certifiable or not",
temporal split) · C2 controlled comparison symbolic vs LLM-native, ± verification, × language (P2) ·
C3 faithfulness-gap attribution study + fixed pipeline (P1) · C4 does the advantage survive RLVR.
Translated data is a **baseline only**.

## 2. What is dropped from the old pipeline, and why

| Old component | Decision | Evidence |
|---|---|---|
| SMT-LIB as the representation | **Drop** | Renderings unnatural, worse than SymPy and NL (2602.19187); 81% of GSM8K programs are straight-line arithmetic, Z3 used as a calculator (R3); `QF_LRA` cannot express radicals/counting (R4) |
| Answer-expression mutation (`mutation_engine.py`) | **Drop** | Answer-tail is structural: a' = f(a, k) ⇒ the faithful rendering *is* "…then add k" [TK: `mutation_engine.py:18-71`]; 0% new problems, 84% tails (R5) |
| Informalize from the program + reflection loop showing the Z3 answer | **Drop** | Leak channel: 58% leaks; generator writes the fed-back answer into the problem (R5) |
| Phase-1 labels GOOD/FIXED/… and `z3_score` | **Drop** | Labels do not separate good from bad; FIXED = "sat", never compared to gold; `z3_score` = 1 on every row (R3, R4) |
| Batched-JSON translation | **Drop** | Escape bug (`\right`→CR, 175 rows [TK]) and precalculus misalignment came from batching (T4, R2) |
| Current `math_verify.py` | **Rebuild** | False positives `0.125≡125`, `(1,2)≡1.2` [TK]; partial-answer credit (04 H9) |
| QLoRA 4-bit + Mistral + DPO | **Drop** | Precision confound, weak base, DPO not needed (03, T3) |

## 3. Data generation architecture

### 3.1 Representation — typed quantity DAG (Python/SymPy)
- Nodes: `Quantity(value: Fraction, qtype, unit, label_vi, role, span)`; edges: typed `Op`.
  `qtype ∈ {COUNT, MONEY, MEASURE, RATE, PERCENT, RATIO, AGE, TIME, ABSTRACT}` (T1 §2.1).
- **Unit/type propagation** forces integrality and sign at *every* intermediate step for indivisible nouns —
  "0,5 cái kẹo" is impossible by construction.
- Exact arithmetic with `Fraction`/SymPy; **Z3 only** where it is needed: integer CSP sampling (divisibility
  chains) and uniqueness of solutions for equation-type problems. Not as a "verifier" of arithmetic chains.
- Labels are natural Vietnamese phrases, never identifiers (removes the `h_16296` leak channel).

### 3.2 Formalization (seed → IR)
- LLM returns JSON IR with character spans on the **Vietnamese** seed text (English original as an aid).
- Accept iff: executes exactly to gold; every given is span-anchored and its literal matches (Vietnamese
  numerals, "nửa", "một phần ba", `.`/`,` conventions); query is derived (depth ≥ 1, not a literal); all type
  constraints hold (T1 §2.1). Rejected seeds stay in the pool for the LLM arm → coverage measured (P3).

### 3.3 Mutation — premises and structure, never the answer
M1 value-resample (log-uniform [v/3, 3v], snapped to the type grid; Z3 Int for coupled integrality) ·
M2 inverse (FOBAR-style) · M3 insert a typed step (template library keyed by qtype/unit) · M4 relation change ·
M5 query shift (depth ≥ 2) · M6 compose two seeds (unit-compatible). Shared schedule
π = {.20, .20, .25, .15, .10, .10}, 3 slots/seed, identical for every arm (T1 §2.0).
**Answer-tail rule:** reject if q_new ≠ q₀ and A(q_new) \ A(q₀) \ {q₀} ⊆ new constants (T1 §2.2).
Identity rule: isomorphic DAG ∧ same givens ∧ same answer → reject.

### 3.4 Rendering — span-grounded minimal edit
M1 = deterministic substitution (no LLM). M2–M6 = LLM editor sees only the tagged original text and the edit
list (labels + given values), **never the new answer**; tag-multiset check; no other numerals allowed (T1 §2.2).

### 3.5 Verification (identical gates for every generated arm)
1. **Cycle consistency:** a model of a **different family** formalizes the new text blind → its executed answer
   must equal the program's answer (MathCAMPS: 97.7% faithful among kept) (T1 §1F).
2. **Validity:** 2 judges of different families must both pass: atomic, no conflict, complete, single question,
   realistic, naturalness ≥ 4/5 (T2 §2.4; MathQ-Verify decomposition).
3. **Answer:** blind solution writer (k ≤ 4, short Vietnamese CoT); keep the first trace matching gold.
4. Zero-tolerance leak regex; chrF ≤ 0.90 vs seed for M2–M6; MinHash dedup (≤ 3 per seed); 13-gram = 0 and
   cos < 0.92 vs every eval set (T1 §2.3).
Unverified variants (Y−, N−) skip steps 1–3 but keep 4, so "verification" is the only difference.

### 3.6 LLM-native Vietnamese arm (N)
Same seeds and operator schedule; CoT-Self-Instruct-style generation with a localization card; V1 =
self-consistency (≥3/5), V2 = back-formalization into the same IR (gives an N+V2 sub-arm) (T1 §2.4).

### 3.7 English mirror (for P2)
The identical pipeline on the English originals of the same seeds (GSM8K / MATH alg+prealg), same models,
prompts translated, same gates → arms Y±, N± in English. Primary test: language × verification interaction.

### 3.8 Matching
Primary analysis on paired slots (both arms succeeded); secondary on natural coverage; difficulty matched by a
reference model's pass rate; diversity reported (T1 §2.6).

### 3.9 Decontamination thresholds (calibrated, X5)
- **Generated training data (gate G4):** bge-m3 cosine ≥ 0.85 (Vi–Vi) / ≥ 0.80 (Vi–EN) OR number-templated
  13-gram hit → reject. Over-flagging is acceptable here. (0.92 missed 12–15% of re-translated/perturbed copies.)
- **Benchmark cleaning (Vi-ExamMath):** primary = review rule (cosine ≥ 0.90 or substantial 13-gram overlap;
  39 Core items flagged, mostly LuminarAI textbook copies); sensitivity = the G4 thresholds. Post-cutoff Core
  after review rule: 2026 = 110, 2025+ = 318, first_seen ≥ 2026-06 = 85. Flags: `data/contam/vi_exam_flags.jsonl`.
- Known external overlap to report: `vinhnq29` dataset holds a re-translated GSM8K test (1,280/1,319 by
  embedding); MetaMathQA-vi has 336 near-twins of Vi-MATH test → the T-off arm must be decontaminated.

## 4. Pilot gates before any scale-up
| Gate | Rule | Source |
|---|---|---|
| Faithfulness-attribution pilot (C3) | one-factor-at-a-time from the old config (operator, IR, renderer, answer exposure, model, prompt) + "old operator, everything else best" cell + literature baselines (GSM-Symbolic templates, MetaMath FOBAR/SV, 2602.19187 Sym/NL, MathCAMPS), 200 seeds | this design |
| Quality gate per arm (200 items) | teacher-would-write ≥ 70%; new (M2–M6) ≥ 60%; leak 0%; absurd ≤ 3%; judge-vs-human κ ≥ 0.6 | T1 §2.5 |
| **⚑ Generator choice** | deepseek-v4.1-flash (NIM, free, 40 RPM) vs Claude Haiku 5.5 vs Sonnet 5.5 on 100 seeds, by the quality gate and cost | 02, [TK] probe |
| Student gates G1–G8 | fp16 safety, packing boundaries, kernels on T4, vLLM on T4, throughput, quota accounting | T3 §6 |

## 5. Models

**Decision 2026-11-07 (user-approved): NIM is OFF the critical path.** Measured: one live key, 40 RPM total;
DeepSeek-v4.1-flash 300–600 s/call in the G1 pilot; ≈37 h per 10k accepted items per arm (T1). Generation and
judging run on Claude (Haiku/Sonnet; Batch at scale) + the Gemini flash-lite key pool. Nemotron may remain an
optional extra cross-family judge only. Final role assignment = `exec/G1_generator_pilot.md`.
Second family (S2, 2026-11-07): **LFM2.5-2.6B-Base** (4-shot Vi-GSM8K-300 60.3%, vLLM works on T4) —
conditional on an fp16 LoRA-SFT smoke on T4; fallback SmolLM3-3B. Gemma-4-E2B rejected (22.3%, no vLLM on T4).
Kaggle T4 fp16 SFT measured (plain HF, padded): 0.6B 2,352 / 1.7B 1,358 / 4B 426 tok/s; no fp16 overflow;
fp16≈bf16 parity at 1.7B (63.0 vs 62.7). Qwen3-4B → Modal L40S ($1.23/run). T4×2 bills 1 quota-h per hour.
Unsloth 2026.11.3 adds 1.74× on T4 / 2.4× on L4 with exact packing (O1) — adopt after a 300-step parity gate.
Eval → Modal L4 (vLLM 2,151 vs 341 tok/s on T4), generation capped at 1,024 tokens.
Student family decided: **Qwen3-Base** (pilot v1: 0.6B 38.3% / 1.7B 58.7% vs Qwen3.5 0.8B 29.3% / 2B 49.0%,
4-shot Vi-GSM8K-300 on T4).
| Role | Choice | Evidence |
|---|---|---|
| Generator | **⚑** pilot-decided (above) | NIM 40 RPM → ~37 h per 10k items; Haiku 5.5 Batch ~$4 per 10k, 4/4 on probe [TK] |
| Formalizer B / cycle check / judges | different family from the generator: nemotron-3-ultra + glm-5.3-flash (NIM) + Claude Haiku 5.5; Sonnet 5.5 arbitrates | T1 §2.3, T2 §2.3 |
| Benchmark QA, test-translation repair, audits | Claude Opus 5.5 (Batch) | only external model allowed to see test items |
| **⚑ Students** | Qwen3-Base 0.6/1.7/4B primary (only family proven end-to-end on T4); Qwen3.5 as gated reference rows | T3 §0.4 vs 02 §5.2 conflict |
| **⚑ Second family** | Gemma-4-E2B if fp16 gate G1 passes, else Modal bf16 | Gemma-3-1B rejected: GSM8K base 2.2 (02 §3.4) |

## 6. Training
SFT: completion-only NLL; plain-text template; fp16 LoRA r=32 all-linear, fp32 adapters; LR/epoch swept once on
S0 then frozen; 32 seq/step; padding-free; matched items (tokens as control); 3 seeds (5 in primary ~2B cells).
RL (C4): GRPO with Dr.GRPO loss, fp16, β=0, G=8, 32 prompts/step, 200 steps, binary reward, fixed
mid-difficulty pool, random-reward control ×3. (T3 §3–§4)

## 7. Evaluation
Typed-gold grader with Vietnamese number parsing (T2 §4). Eval: Vi-ExamMath Core ≥ 500 (native) + Vi-GSM8K-Platinum
600 + Vi-MATH alg/prealg 400 (translated) + English mirror sets. Decoding T=0.6, top-p 0.95, K=4. Binomial GLMM
with crossed item/run effects, Holm over confirmatory contrasts, cluster bootstrap, TOST. Power: ~4.6–4.9 pp
per model, ~3.4 pp pooled at this spec (T5 §2.4). IRT 2PL for item analysis and difficulty control.
Contamination: temporal DiD + corpus overlap + perturbation. Benchmark QA: 3 independent keys + Claude critique +
2 humans on flagged items + 150-item random audit (T5 §3).

## 8. Compute (needs re-sizing for the English mirror)
Kaggle T4×2 primary (30 h/week [TK]); Modal ($30 ≈ 37 L4-h at the real $0.80/h [TK]) for bf16 checks and
overflow; laptop for dev. T3's optimised estimate (≈ 560–680 T4-h, 9–11 weeks) predates the English mirror axis —
**re-size after the student gates measure real tokens/s.**

## 8b. Adopted defaults (2026-11-07, user said "continue" without objecting; reversible)
- Confirmatory family (Holm): **H1** = P2 language × verification; **H2** = Y± vs N± on Core; **H3** = verification
  effect on Core; **H4** = P3 (Y−N on certifiable vs non-certifiable Core items). P1 is measured by data-quality
  metrics + the attribution study, not by the GLMM. Fit: lme4 nAGQ=0 primary (Laplace changed ≤0.014 logit).
- Vi-ExamMath: do not fill the 2024 / HSG gap now (Core 1,585 ≫ 500; HSG only serves the Hard ceiling).

## 9. Open decisions (⚑)
1. Generator pilot (DeepSeek vs Haiku vs Sonnet). 2. Student family (Qwen3 vs Qwen3.5). 3. Second family
(Gemma-4-E2B gated). 4. Human raters (2–3 people). 5. Kaggle kernel `datuni/w2-pilot-v1` still running —
keep or delete. 6. Size of the English mirror (all arms vs Y±/N± only).
