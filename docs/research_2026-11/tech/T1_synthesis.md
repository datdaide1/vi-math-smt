# T1: Algorithms for synthesizing new math problems and solutions (state of the art, 2025–2026) and the two generation arms

**Agent:** T1 · **Date:** 2026-11-07 · **Brief:** `tech/BRIEF.md`

**Evidence rules.**
- Every paper in §1 had its arXiv abstract page opened this session. Rows marked **HTML** also had the full-text HTML read through a summarizing fetch tool. Numbers from HTML should be re-checked against the PDF before they go into a paper. Some details are marked **(abs.)**: the abstract does not state them.
- **[UNVERIFIED]** means the item was seen only in a search snippet or is from memory.
- **[ESTIMATE]** marks my own projection. Every yield or rate in §4 is an [ESTIMATE] until the pilot in §2.6 measures it.
- Local facts were obtained by running commands on the repo:
  - Seed counts: `data/translation/gsm8k/gsm8k_train_vietnamese.jsonl` has 7,473 lines. `data/translation/MATH/train/algebra_vi.jsonl` has 1,744 and `prealgebra_vi.jsonl` has 1,205.
  - Stubs exist and the design maps onto them: `vi_math_verified_aug/ir/typed_dag.py`, `symbolic_aug/{constant_sub,structural}.py`, `verify/{round_trip,uniqueness}.py`, `llm_aug/generate.py`.
  - NIM latencies come from the measured smoke test in `02_models.md` §2.

---

## 0. Bottom line

1. **The old failure has a formal name, and it can be forbidden mechanically.** The old operators only post-composed the answer, so the new query depended only on the old query plus new constants. That is an *answer tail*. The rule in §2.2 (the "ancestor-difference" test) rejects it by construction. Other leaks come from three places: free verbalisation from a program, the solver answer being fed back to the generator, and identifier names in the IR. All three are removed in arm A by **rendering through span-grounded minimal edits of the original Vietnamese text**. The renderer never sees the new answer, and the IR carries natural Vietnamese entity labels instead of variable names.
2. **The only head-to-head result from 2026 (arXiv 2602.19187) predicts a small difference between arms.** It compared symbolic-space and NL-space modification at the same seeds:
   - At 1.5B (RL), symbolic beats NL by about +1 to +2 points on average: 54.32 vs 53.16 PPO, and 54.71 vs 52.60 GRPO.
   - At 3B, the result is mixed: 62.50 vs 63.10 PPO, and 64.39 vs 62.57 GRPO.
   - SymPy has much higher embedding diversity than NL: 0.37 vs 0.07.
   - SMT-LIB renderings are "unnatural" and score below both SymPy and NL.

   ⇒ Use a **typed quantity DAG (Python/SymPy) IR, not SMT-LIB**. Expect arm differences of **≤ ±2 points**, so the comparison must be paired on seeds and operators and matched on difficulty (§2.4).
3. **Round-trip check: blind back-formalization (MathCAMPS cycle consistency), not blind-solve agreement.** In MathCAMPS, 30/245 generated problems were unfaithful. Cycle consistency removed 25 of them and only 7 faithful ones, and 210/215 (97.7%) of the kept problems were faithful (HTML). Blind-solve agreement with answer feedback is exactly what leaked answers in R5 (6.1% of items printed the gold value).
4. **Arm B: operator-conditioned, CoT-Self-Instruct-style native Vietnamese generation.** Each request gets a localization card. Arm B uses **the same seed and operator schedule as arm A**. Its optional verification labels are V1 self-consistency and V2 back-formalization into the same IR as arm A. The V2 label gives a within-arm measure of how much verification adds.
5. **Self-play (Absolute Zero, R-Zero, VHG) is not used as an arm.** It needs GPU-heavy RL, which is scarce at Modal's real rates. It also degrades at ≤4B:
   - R-Zero's pseudo-label accuracy falls from 79% to 63% (Qwen3-4B), and the 0.6B and 1.7B models collapse earlier (HTML).
   - AZR at 3B gains +7.7 on math, against +15.2 at 7B-Coder (HTML).
6. **Cost per 10k accepted items per arm** (§4):
   - About 85k NIM calls per arm, which is ≈ 37 hours on one 40-RPM key or ≈ 9 hours on four keys.
   - < $10 of Claude per arm (Haiku 5.5 Batch as second judge, Opus 5.5 Batch for audits).
   - About 3 T4-hours on Kaggle for difficulty labelling.
   - Expected acceptance rate is about 45% of candidates for arm A and about 50% for arm B [ESTIMATE].

---

## 1. Evidence table of methods

Abbreviations: **Sc** = guarantees a coherent NL problem (our failure mode)? **Y** = by construction, **P** = partly (checked post hoc), **N** = no.

### 1A. Program / computation-graph / symbolic-space synthesis

| Paper (arXiv) | Date | Mechanism | Reported numbers | Scale | Cost | Sc |
|---|---|---|---|---|---|---|
| **Yeo et al., Adaptive Problem Generation via Symbolic Representations** (2602.19187) HTML | Feb 2026 | GSM8K/MATH seeds are translated once to SymPy (SMT-LIB as an alternative). An LLM edits the *code*, the solver computes the answer, and a render prompt ("construct a coherent scenario" from the code) writes the NL. It is compared with NL-space modification. Prompt optimization (GPT-5-mini) adapts the modification strategy to the student. Guardrails: unique answer, executes, ≤50% longer than the seed. | 1.5B avg (GSM8K+MATH500): seed 51.14 → Opt-Sym 54.32 (PPO), 51.40 → 54.71 (GRPO). Opt-NL 53.16 / 52.60. 3B: Opt-NL 63.10 vs Opt-Sym 62.50 (PPO), 62.57 vs 64.39 (GRPO, Tab. 5). Diversity (MiniLM pairwise cosine distance, 10 variants × 100 seeds): Sym 0.37, NL 0.07. Filtering gives +1–1.5. SMT-LIB renders are "unnatural", "expose computational mechanics", and score below SymPy and NL. | Qwen2.5-1.5B/3B-Instruct, RLVR, 6k items (1,500 seeds × 4 prompts) | Not reported | **P**. Render is *from code, not from the seed text*; contexts and protagonists drift; validity is checked by GPT-5-mini |
| **RV-Syn** (2504.20426) HTML | Apr 2025 (EACL 2026) | Seeds are decomposed into Python functions, which form a library (28% dedup by AST). The graph composition samples 1–3 function nodes. Back-translation to NL+CoT. CoT answer must equal the executed answer. | 50k: LLaMA-3-8B avg 44.92 vs ScaleQuest 42.80, MetaMath 38.82. Qwen2.5-7B 64.04 vs 62.86. Phi-3-mini 55.64 vs 54.94. Yield: 16% of graphs fail to execute, 37% of pairs fail answer match. Problem error rate 0.9% (vs ScaleQuest 5.6%). | 3.8B–8B SFT | 72B generator | **P** (free back-translation) |
| **MathCAMPS** (2407.00900) HTML | Jul 2024 (COLM 2025) | Attribute grammar over 44 Common Core standards, generating symbolic problems by rejection sampling with 66 filters. Dependency-reachability pruning. GPT-4 verbalizes; **cycle consistency**: GPT-4 re-parses the NL blind, and the problem is kept iff the solved answers match. "Counterfactual" (change a constant) and "incremental" (add info) follow-ups are diffs on the symbolic form. | Human check of 245: 12.2% unfaithful. Cycle check removed 25/30 unfaithful and 7 faithful, leaving 97.7% faithful among the kept. About 9.6k problems for ~$330 (≈$0.034 per kept problem). | Eval | ~$0.03/item | **Y/P** (faithful to the symbolic form, but grammar text is formulaic) |
| **GSM-Infinite** (2502.05252) (abs.) | Feb 2025 | GSM8K abstracted as computational graphs. Generator with "infinite" difficulty. Noise nodes and edges added for long context. | Benchmark only | — | — | P |
| **GSM-Symbolic** (2410.05229) (abs.) | Oct 2024 (ICLR 2025) | Human-written symbolic templates from GSM8K; values resampled under conditions. | All models drop when only values change. One irrelevant clause gives up to −65%. | Eval | Manual templates | Y (hand templates) |
| **RE-IMAGINE** (2506.15455) (abs.) | Jun 2025 (ICML 2025) | Mutations in an intermediate symbolic representation at three levels of Pearl's ladder (association, intervention, counterfactual). | Models drop on variations (no numbers in abstract) | Eval | — | P |
| **Arrows of Math Data Synthesis** (2508.18824) (abs.) | Aug 2025 | Knowledge system plus tools produce executable programs, which are translated to NL problem-solution pairs. "Bilateral validation": program↔solution and program↔problem consistency. | 12.3M triples (no benchmark numbers in abstract) | — | — | P |
| **MathAgent** (2604.11188) (abs.) | Apr 2026 (ACL Findings) | Legislator adversarially evolves constraint-graph blueprints; Executor instantiates varied NL scenarios. | 1K samples beat LIMO/s1K on 8 benchmarks, 10 models (no numbers in abstract) | 10 models, 4 families | — | P |
| **Synthesis by Design** (2506.07664) (abs.) | Jun 2025 | Structure from solution code guides generation. Complexity = reasoning length. | 39K problems + 6.1K harder benchmark | — | — | P |
| **DAGGER** (2601.06853) [UNVERIFIED, snippet] | Jan 2026 | Solving as executable graph generation with distractor nodes | — | — | — | — |

### 1B. Inverse/backward, composition, perturbation

| Paper | Date | Mechanism | Reported numbers | Scale | Sc |
|---|---|---|---|---|---|
| **MetaMath FOBAR / SV** (2309.12284) HTML | Sep 2023 | FOBAR masks one number as x and appends "If we know the answer … is {a*}, what is x?". SV rewrites the Q+A as a statement and asks for x. | MetaMathQA 395K (SV 55K, FOBAR 55K). LLaMA-2-7B GSM8K: SFT 41.6, AnsAug+Reph 60.6, +SV+FOBAR **64.4** (+3.8). | 7B SFT | P (the template tail is unnatural) |
| **MathFusion** (2503.16212) (abs.) | Mar 2025 (ACL 2025) | Sequential, parallel and conditional fusion of two problems | +18.0 points over diverse benchmarks with 45K added | 7–8B SFT | P |
| **MATH-Perturb** (2502.06453) (abs.) | Feb 2025 | *Simple* perturbations keep the solution pattern; *hard* perturbations make the original steps inapplicable. | −16.5% (o1-mini) and −12.9% (gemini-2.0-flash-thinking) on MATH-P-Hard. Models apply memorized steps blindly. | Eval | Y (human) |
| **Distracting conditions** (2510.08615) (abs.) | Oct 2025 | Iterative prompts add distractors that "do not alter the original solutions". Storyline preserved. | No numbers in abstract | — | P |

### 1C. Concept / knowledge-point / curriculum-driven

| Paper | Date | Mechanism | Reported numbers | Scale | Sc |
|---|---|---|---|---|---|
| **MathScale** (2403.02884) (abs.) | Mar 2024 | Topics and knowledge points extracted from seeds form a concept graph; new questions are generated from it. | 2M QA. MathScale-7B is +42.9% micro over peers on MwpBench. | 7B | N |
| **KPDDS / KPMath** (2403.02333) (abs.) | Mar 2024 | Key points plus exemplar practices drive QA synthesis. | 800K+. Qwen1.5-72B: GSM8K 87.0, MATH 58.3. | 72B | N |
| **Self-evolving multi-role generation with fine-grained difficulty guidance** (2601.11792) (abs.) | Jan 2026 | Sampler, generator, evaluator, state machine and memory. Association-guided path sampling, difficulty model. CPT → SFT → GRPO. Dataset HSM3K-CN. | No numbers in abstract | — | P |
| **MathSmith** (2508.05592) (abs.) | Aug 2025 (AAAI 2026) | Random PlanetMath concept-explanation pairs, built from scratch. Nine difficulty strategies. RL on structural validity, complexity (trace length) and answer consistency. Weakness-focused variants. | "Consistent gains" on GSM8K/MATH500/AIME/Olympiad (no numbers in abstract) | — | P |
| **QueST** (2510.17715) (abs.) | Oct 2025 | Difficulty-aware graph sampling plus difficulty-aware rejection fine-tuning of the generator (code problems). | 100K problems → Qwen3-8B-base beats Qwen3-8B on LiveCodeBench | 8B | P |
| **PersonaHub** (2406.20094) (abs.) | Jun 2024 | About 1B personas condition generation for diversity. | (abstract gives no math numbers) | — | N |
| **Multilingual GSM-Symbolic** (2610.03367) (per `01_literature_novelty.md` C10) | Oct 2026 | 30k item-matched QA in 15 languages from symbolic templates | Eval | — | Y |

### 1D. Difficulty-controlled / evolutionary (Evol-Instruct successors)

| Paper | Date | Mechanism | Reported numbers | Scale | Sc |
|---|---|---|---|---|---|
| **WizardMath Evol-Instruct** (2308.09583) [snippet] | 2023 | Upward and downward evolution prompts | — | 7–70B | N |
| **CoT-Self-Instruct** (2507.23751) (abs.) | Jul 2025 | CoT plan on seed(s), then a new problem of similar complexity. Answer-Consistency filter (generator answer = majority vote) [snippet]. | Beats s1k and OpenMathReasoning on MATH500/AMC23/AIME24/GPQA-D (no numbers in abstract) | — | P |
| **SAND-Math** (2507.20527) (abs.) | Jul 2025 | Generate from scratch, then "Difficulty Hiking". Self-consistency as a well-posedness proxy [snippet]. | Difficulty 5.02 → 5.98. AIME25 46.38 → 49.23. 500 items give +17.85 over the next synthetic set. | — | P |
| **PromptCoT 2.0** (2509.19894) (abs.) | Sep 2025 | EM loop: rationale ↔ prompt | Qwen2.5-7B SFT on synthetic prompts only: AIME24 73.1, AIME25 65.6 | 7B / 30B | P |
| **Learning to Pose Problems** (2511.09907) (abs., v6 Oct 2026) | Nov 2025 | Generator plans its design reasoning; trained from multi-part human questions; solver-adaptive reward near the competence boundary. | +2.69 (1 round) / +3.39 (3 rounds) over seed-only, on 10 benchmarks | — | P |
| **SwS** (2506.08989) (abs.) | Jun 2025 | Failed RL problems → concepts → targeted synthesis | +10.0% (7B), +7.7% (32B) avg over 8 benchmarks | 7/32B RL | P |
| **ScaleQuest** (2410.18693) (abs.) | Oct 2024 (ACL 2025) | QFT + QPO on 7B solver models; questions from scratch | ~1M pairs | 7B generator | N |
| **FLAMES** (2508.16514) HTML | Aug 2025 (EMNLP 2025) | Controlled study of 10 agents (incl. FOBAR, Self-Verification, IQC, Suggester-Editor, Key Concepts) × 6 quality controls | Complexity-raising agents (IQC, Suggester-Editor) are best. **The solvability filter removes 30% of real MATH (50% of level 5) and hurts.** Majority+First > Majority: coverage > reliability at a fixed budget. Generator Qwen2.5-32B, solutions Qwen2.5-Math-7B. MATH 81.4 (Qwen2.5-Math-7B). | 7B | P |

### 1E. Self-play / proposer–solver loops

| Paper | Date | Mechanism | Reported numbers | Scale | Sc |
|---|---|---|---|---|---|
| **Absolute Zero (AZR)** (2505.03335) HTML | May 2025 | One model proposes and solves code tasks (deduction, abduction, induction). The executor verifies. Proposer reward = 1 − mean solve rate (0 if never solved). | Math avg: 7B-Coder 23.9 → 39.1, 7B 27.5 → 38.4, **3B-Coder 18.8 → 26.5**, 14B-Coder 20.2 → 43.0. Llama-8B only +3.2 overall. "Uh-oh" safety CoTs. | 3–14B RL | N (code tasks, not word problems) |
| **R-Zero** (2508.05004) HTML | Aug 2025 (v4 Feb 2026) | Challenger/Solver co-evolution. Challenger reward = 1 − 2·\|p̂ − ½\| minus a BLEU-cluster repetition penalty. Keep a question if 3–7 of 10 samples agree with the majority. 8k candidates per iteration. | Math avg: Qwen3-4B-Base 42.57 → 49.93, 8B 48.64 → 53.72, OctoThinker-3B 26.64 → 29.32. **Pseudo-label accuracy 79 → 69 → 63% (4B).** Gemini-labelled 1.7B: 69.4 → 45.2 by step 60. Smaller models collapse earlier. | 0.6–8B RL | N |
| **Self-Questioning LMs** (2508.03682) (abs.) | Aug 2025 | Proposer given only a topic; solver rewarded by majority vote | (no numbers in abstract) | — | N |
| **VHG: Verifier-Backed Hard Problem Generation** (2605.06660) HTML | May 2026 | Setter / solver / **verifier**. R = 1[valid] · (1 − Acc_solver). Hard verifier is SymPy; soft verifier is an LLM judge. | General math, Qwen3-4B-Base: 56.79 → VHG 69.01 vs vanilla GRPO 67.62, R-Zero 66.20. Setter valid rate 30.6% → 75.5% during training. | 4B RL | P (validity judged) |
| **Self-play position paper** (2603.02218) (abs.) | Feb 2026 (ICML 2026 position) | Loops plateau when learnable information does not grow; proposes asymmetric co-evolution etc. | — | — | — |
| **MathDuels** (2604.21916) [snippet] | Apr 2026 | Poser/solver benchmark with an independent verifier excluding ill-posed items | Eval | — | P |

### 1F. Well-posedness and diversity measurement

| Paper | Date | What it gives | Numbers |
|---|---|---|---|
| **MathQ-Verify** (2505.13903; v2 Mar 2026, KDD'26) [abs. opened in `01`; numbers from search snippet] | May 2025 | Five stages: format → formalize → atomic-condition errors → cross-condition conflicts → completeness. ValiMath has 2,147 labelled items. | Up to +25 F1 over direct verification. About 90% precision and 63% recall with lightweight model voting. |
| **FLAMES solvability analysis** (above) | 2025 | A solvability filter is unreliable on hard items | Removes 30% of real MATH |
| **TreeCut** [UNVERIFIED, snippet] | 2025 | Tree representation; remove a necessary condition to make the problem unanswerable | — |
| **DCScore** (2502.08512) (abs.) | Feb 2025 (ICML 2025) | Classification-based dataset diversity; satisfies diversity axioms | Better correlation with pseudo-truths, cheaper |
| **Embedding pairwise distance** (as used in 2602.19187) | 2026 | Mean pairwise cosine distance among variants of a seed (MiniLM) | Sym 0.37 vs NL 0.07 |
| **Vendi score** [UNVERIFIED] | 2022 | exp(entropy of the eigenvalues of the similarity matrix) | — |

### 1G. What the table says, condensed

- **Gains from synthetic problems on top of seeds are small at 1.5B–3B** (+2 to +3 avg in 2602.19187), and the symbolic-vs-NL difference is smaller still. Large gains (MetaMath +22.8, MathFusion +18) are older 7B SFT-on-weak-base results.
- **Nobody guarantees a coherent problem by construction.** Free rendering from a program (2602.19187, RV-Syn, Arrows) drifts away from the story. Hand templates (GSM-Symbolic) and grammars (MathCAMPS) are coherent but formulaic. The strongest post-hoc guarantee measured on humans is MathCAMPS cycle consistency (97.7% faithful among kept items).
- **Complexity-raising operators help more than paraphrase or value resampling** (FLAMES: IQC and Suggester-Editor best). "Hard" perturbations that change the solution steps are what models fail on (MATH-Perturb). ⇒ Arm A must favour structural edits (inverse, insert, relation change) over value resampling.
- **Verification of answers matters less than coverage** (FLAMES; 01 §2.1). Verification of *problems* (validity) matters for self-play stability (VHG). ⇒ Arm A spends its verification budget on problem faithfulness and well-posedness, not on answer certificates alone.

---

## 2. Recommended design for the two generation arms

### 2.0 Shared skeleton: one seed pool, one operator schedule, one solution writer

```
SEED_POOL = vi_gsm8k_train (7,473)  ∪  vi_math_train[algebra] (1,744)  ∪  vi_math_train[prealgebra] (1,205)
          = 10,422 seeds; seed_id = sha1(source || index)[:12]
DEV_SEEDS  = 5% of seeds by hash (≈ 520) → used only for the pilot and threshold calibration
                                          → never used to generate training items
OPERATORS  = {M1 value-resample, M2 inverse, M3 insert-step, M4 relation-change, M5 query-shift, M6 compose}
π (shared operator mix) = {M1: .20, M2: .20, M3: .25, M4: .15, M5: .10, M6: .10}
                          (M6 is GSM8K only; for MATH seeds its mass is redistributed proportionally)
SCHEDULE   = for each seed s, for slot j in 1..J (J = 3):
                 o[s,j] ~ π   (seeded RNG, 20261107)
             Both arms receive exactly the same (s, j, o[s,j]) triples.
SOLUTION_WRITER (shared by both arms; identical model, prompt and procedure; see §2.3)
COMMON_GATES    (shared by both arms; identical thresholds; see §2.3)
```

Why share the operator schedule? 2602.19187 compared symbolic and NL modification under the same *modification instruction* space (Base-NL vs Base-Sym), and that is what makes its comparison interpretable. Without a shared schedule, "arm" is confounded with "transformation type". FLAMES shows transformation type alone moves results by several points.

### 2.1 Arm A data structures (typed quantity DAG IR)

These replace SMT-LIB. They live in `ir/typed_dag.py`.

```python
from fractions import Fraction
from dataclasses import dataclass, field
from enum import Enum

class QType(Enum):          # semantic type → integrality, sign, rendering grid
    COUNT      = "count"      # indivisible nouns (kẹo, học sinh, quả, con): int ≥ 0
    MONEY      = "money"      # VND: multiple of 500 (localized); USD seeds: ≤ 2 decimals, ≥ 0
    MEASURE    = "measure"    # kg, lít, m, km, giờ, phút: rational, > 0, ≤ 2 decimals when rendered
    RATE       = "rate"       # derived unit (đồng/kg, km/giờ, quả/ngày): rational > 0
    PERCENT    = "percent"    # int in [1, 100] (grid 5 when resampled)
    RATIO      = "ratio"      # p/q with q ≤ 12 (một phần ba, ba phần tư)
    AGE        = "age"        # int in [0, 100]
    TIME_POINT = "time_point" # clock/date arithmetic (mod 60/24/7)
    ABSTRACT   = "abstract"   # MATH: SymPy assumptions (integer/positive/real) + domain

@dataclass
class Unit:
    lemma: str                # "kẹo", "nghìn đồng", "kg"
    classifier: str | None    # "cái", "quả", "con", "chiếc" (VN classifier; kept on edit)
    divisible: bool           # False for COUNT nouns; drives integrality propagation
    plausible: tuple[Fraction, Fraction]   # from the unit lexicon (e.g. giá 1 quyển vở: 5k–50k VND)

@dataclass
class Quantity:
    qid: int
    value: Fraction
    qtype: QType
    unit: Unit
    label_vi: str             # natural entity label, e.g. "số kẹo Lan có lúc đầu"; NEVER an identifier
    role: str                 # "given" | "derived" | "query" | "world_const" (7 ngày/tuần, 60 phút/giờ)
    span: tuple[int, int] | None   # char offsets of the literal in the seed text (givens must have one)
    sentence_id: int | None

@dataclass
class Op:
    oid: int
    kind: str                 # add, sub, mul, div, floordiv, mod, pct_of, ratio_part, solve_lin, pow, ...
    inputs: list[int]         # qids
    output: int               # qid
    sentence_id: int          # the clause that expresses this relation (alignment for edits)
    verb_vi: str | None       # "cho", "mua thêm", "chia đều", "gấp đôi" (for relation edits)

@dataclass
class Program:
    seed_id: str
    text: str                 # Vietnamese seed text
    sentences: list[tuple[int, int]]
    quantities: dict[int, Quantity]
    ops: list[Op]
    constraints: list[str]    # SymPy-evaluable: integrality/divisibility/bounds (auto + seed-specific)
    query: int
    answer: Fraction          # = execute(self)[query]
    kind: str                 # "story" (GSM8K, prealgebra word) | "symbolic" (algebra equations)
```

**Type propagation (this is what makes "0.5 candies" impossible).** Unit algebra infers each derived node's type: COUNT − COUNT → COUNT; MONEY ÷ COUNT → RATE(money/count); COUNT × RATE(count/day) → COUNT; and so on. Every node of type COUNT, AGE or PERCENT must be an integer. Every node with `divisible=False` must be an integer *at every intermediate step*, not only at the answer, and must also satisfy sign constraints such as "remaining ≥ 0". These are auto-added to `constraints` and checked on every resample and every edit.

**Formalizer acceptance (seed → Program).** One LLM call returns the JSON IR with spans. It may retry up to twice, with the validator's error message and the seed's gold answer. Giving the gold here is safe: the seed already exists, and nothing generated sees it.

The Program is accepted iff all of these hold:
- (i) `execute(P)[query] == gold`, exactly, as a Fraction.
- (ii) Every `given` has a span whose normalized literal equals its value. The normalizer handles Vietnamese numerals and number words such as "hai", "nửa" and "một phần ba", the "." thousands separator and the "," decimal mark.
- (iii) The query is derived, with depth ≥ 1, and is not a literal. This reuses `verify/uniqueness.classify == "derived"`, which addresses R3's hard-code finding.
- (iv) Every given is reachable to the query, apart from explicit distractors.
- (v) Every type constraint holds on the seed values.

Seeds that fail are excluded from arm A. They stay in the pool, and arm B still uses them, which is the coverage effect handled in §2.4.

### 2.2 Arm A: program/symbolic-verified generation (full pseudo-code)

```python
def arm_A(seed, op, rng) -> Item | Reject:
    P = FORMALIZED[seed.id]                 # cached; None if the formalizer rejected the seed
    if P is None: return Reject("no_ir")

    # ---- 1. mutate premises/structure in symbolic space (never the answer) ----
    P2, edits = MUTATE[op](P, rng)          # returns new Program + list of TextEdit
    if P2 is None: return Reject("op_not_applicable")

    # ---- 2. symbolic gates (deterministic) ----
    assert_types(P2)                        # every node satisfies QType/Unit constraints (§2.1)
    if not unique_solution(P2):             # SymPy solve has one solution in the domain;
        return Reject("non_unique")         # for Int problems, Z3 blocking-clause test
    if is_answer_tail(P, P2):               return Reject("tail")       # rule below
    if is_identity(P, P2):                  return Reject("identity")
    if not clean_answer(P2):                return Reject("ugly")        # thresholds in §2.5
    if not plausible(P2):                   return Reject("implausible") # unit lexicon ranges

    # ---- 3. render by editing the ORIGINAL text (span-grounded) ----
    if all(e.kind == "SUBST" for e in edits):
        text = apply_substitutions(P.text, edits)                 # deterministic, no LLM
    else:
        text_tagged = LLM_EDITOR(P.text_tagged, edits)            # sees edit list, NOT P2.answer
        if not tag_multiset_ok(text_tagged, P2):  return Reject("tag_mismatch")
        text = strip_tags(text_tagged)
    text = localize_surface(text, P2, card=LOCAL_CARD[seed.id])   # names/currency label only, via IR

    # ---- 4. leak / tail-phrase / novelty gates (zero tolerance) ----
    if leak_regex(text) or contains_value(text, P2.answer):      return Reject("leak")
    if tail_phrase_regex(text):                                   return Reject("tail_phrase")
    if chrF(text, P.text) > 0.90 and op != "M1":                  return Reject("near_dup")

    # ---- 5. round trip: blind back-formalization by a different model family ----
    P3 = FORMALIZER_B(text)                 # never sees P2; produces IR from scratch
    if P3 is None or execute(P3)[P3.query] != P2.answer:          return Reject("cycle")
    if structural_signature(P3) != structural_signature(P2):      flag("cycle_struct_drift")

    # ---- 6. common gates + shared solution writer (§2.3) ----
    return common_gates_and_solution(text, gold=P2.answer, meta=meta(P, P2, op))
```

**Mutation operators.** Every operator acts on premises or structure. None of them composes on top of the answer.

| Op | Symbolic action | Text edits produced | Applicability |
|---|---|---|---|
| **M1 value-resample** | For each given g (≥ 50% of givens, ≥ 1 on the query path): draw v' ~ log-uniform[v/3, 3v], snap to the type grid, then rejection-sample (≤ 200 tries) until all constraints hold. For coupled integrality (e.g. divisibility), solve with Z3 Int. | `SUBST(span, fmt_vi(v'))` | all |
| **M2 inverse** (generalized FOBAR) | Pick a given g on the query path with an `Op` that is invertible in the domain. New query = g. Old query q₀ becomes a *given* with its value. g's value is resampled first (M1 on g), so the new answer differs from the seed's g. | `UNKNOWN(span_g)`, `STATE_FACT(q0, value, label_vi, after=last story sentence)`, `NEW_QUESTION(label_vi(g))` | g must have a unique solution |
| **M3 insert-step** | Insert a new op on a *given or non-query intermediate* x strictly upstream of q₀: x' = f(x, c), with f drawn from a typed op-template library keyed by (QType, unit): money gets pct_of (giảm giá/thuế), +add (mua thêm) or −sub (trả lại); count gets −sub (cho đi), ×mul (mỗi hộp có), ÷div (chia đều, with divisibility); measure gets +/−. c is a new given (typed, plausible). | `INSERT_STEP(after=sentence(x), template, operands, value=c)` | at least one template matches |
| **M4 relation-change** | Replace an op on the query path with a type-compatible sibling: add↔sub, "gấp k lần"↔"gấp k' lần", pct_of p↔p', "nhiều hơn"↔"ít hơn". Then re-execute. | `REPLACE_RELATION(sentence_id, verb_old→verb_new, operand)` | sibling exists and constraints hold |
| **M5 query-shift** | New query = a different derived node of depth ≥ 2, or a new combination of two *story* nodes (difference or ratio of two derived quantities). The new query must not be a function of q₀ plus new constants only. | `NEW_QUESTION(label_vi(target))` | ≥ 2 derived nodes |
| **M6 compose** (MathFusion-sequential) | Seed s₂ is drawn from the same source with a given g₂ whose (QType, unit) matches q₀ of s₁. Substitute g₂ := q₀(s₁) as an *unstated* intermediate. Query = q₀(s₂). | `JOIN(s1.text minus question, s2.text with span_g2 → REFERENCE(label_vi(q0_s1)))` | GSM8K; unit-compatible partner exists |

**Operator pseudo-code.** M1 goes in `symbolic_aug/constant_sub.py`; M2–M6 go in `symbolic_aug/structural.py`.

```python
def M1_resample(P, rng, tries=200):
    G = [q for q in P.givens() if on_query_path(P, q)] + rng.sample(P.other_givens(), k=...)
    for _ in range(tries):
        vals = {q.qid: snap(loguniform(rng, q.value/3, q.value*3), GRID[q.qtype, q.unit]) for q in G}
        P2 = P.with_values(vals); env = execute(P2)          # Fraction arithmetic
        if env and all(check(c, env) for c in P2.constraints + type_constraints(P2)):
            return P2, [TextEdit("SUBST", P.q[qid].span, fmt_vi(v, P.q[qid])) for qid, v in vals.items()]
    if coupled_integrality(P):                               # e.g. "chia đều" divisibility chains
        return z3_int_sample(P, G, rng)                      # Int sorts, random objective, ≤ 2 s
    return None, None

def M2_inverse(P, rng):
    cands = [g for g in P.givens() if on_query_path(P, g) and invertible_on_path(P, g)]
    g = rng.choice(cands); P1, e1 = M1_resample_one(P, g, rng)          # new value for g first
    q0v = execute(P1)[P.query]
    P2 = P1.make_given(P.query, q0v).make_unknown(g.qid).set_query(g.qid)
    sol = sympy.solve(P2.equations(), sym(g.qid), domain=domain(g))      # must be unique
    if len(sol) != 1: return None, None
    return P2, [TextEdit("UNKNOWN", g.span), TextEdit("STATE_FACT", P.query, fmt_vi(q0v), P.q[P.query].label_vi),
                TextEdit("NEW_QUESTION", target_label=g.label_vi)] + e1_except(g, e1)

def M3_insert(P, rng):
    x = rng.choice([n for n in P.nodes() if n.role in ("given", "derived") and n.qid != P.query
                    and on_query_path(P, n)])
    tpl = rng.choice(OP_TEMPLATES[x.qtype, x.unit.lemma_class])          # e.g. ("pct_of","giảm giá")
    c = sample_typed_constant(tpl, x, rng)                               # new given, typed + plausible
    P2 = P.insert_op(after=x.qid, op=tpl.kind, const=c)                  # rewires x's consumers to x'
    return (P2, [TextEdit("INSERT_STEP", after_sentence=x.sentence_id, template=tpl.vi,
                          operands=[x.label_vi], value=fmt_vi(c))]) if feasible(P2) else (None, None)

def M5_query_shift(P, rng):
    T = [n for n in P.derived() if depth(P, n) >= 2 and n.qid != P.query] + pairwise_story_combos(P)
    t = rng.choice(T); P2 = P.set_query(t)
    return P2, [TextEdit("NEW_QUESTION", target_label=label_of(t))]

def is_answer_tail(P, P2):
    q0 = P.query
    if P2.query == q0: return False            # M1/M3/M4 keep the query; edits are upstream by construction
    if q0 not in P2.q or P2.q[q0].role == "given": return False
    new_consts = {q for q in P2.givens() if q not in P.q}
    return (ancestors(P2, P2.query) - ancestors(P2, q0) - {q0}) <= new_consts
```

**Formal answer-tail rule** (`is_answer_tail`). This rejects the R5 failure by construction.

```
Let A(n) = ancestor set of node n in P2, and C_new = givens that do not exist in P (new constants).
Reject iff q_new ≠ q0 and q0 ∈ P2 and q0 is not a given in P2 and   A(q_new) \ A(q0) \ {q0} ⊆ C_new .
(When q_new = q0, as in M1/M3/M4, the query is unchanged. M3's insertion must then lie strictly upstream
 of q0, which `M3_insert` enforces, so a tail cannot arise.)
```

In words: the problem is rejected if everything the new question adds on top of the old answer is a freshly invented constant. Old operators such as `answer·(answer+1)` and `(answer+k)·k` all fail this test. M2 passes because q₀ is a given in P2. M3 passes because the insertion is upstream of q₀. M5 passes because the new ancestors are story nodes.

**Identity rule** (`is_identity`). Reject if the canonical DAG of P2 is isomorphic to that of P **and** all given values are unchanged **and** the answer is unchanged. This catches R5's 18.4% `(e+k)−k` identities.

**Renderer: minimal span-grounded editor (LLM, used only for M2–M6).** The prompt outline, in Vietnamese in practice:

```
SYSTEM: Bạn là biên tập viên sách giáo khoa toán. Sửa TỐI THIỂU đoạn văn dưới đây theo đúng
        danh sách chỉnh sửa. Giữ nguyên nhân vật, bối cảnh, giọng văn. Mọi con số đã cho phải
        nằm trong thẻ <q n="ID">…</q>. Không thêm số nào khác. Không viết lời giải, gợi ý hay đáp số.
USER:   VĂN_BẢN (có thẻ): "Lan có <q n=1>12</q> cái kẹo. Lan cho em <q n=2>5</q> cái. …"
        CHỈNH_SỬA (JSON): [{"type":"UNKNOWN","q":1},
                           {"type":"STATE_FACT","q":7,"value":"7","label":"số kẹo Lan còn lại",
                            "after_sentence":2},
                           {"type":"NEW_QUESTION","target_label":"số kẹo Lan có lúc đầu"}]
OUTPUT: JSON {"text_tagged": "..."}
```

The renderer is never shown the new answer, the IR operator names, or any identifier. All it gets are `label_vi` phrases and the values of *givens*. This removes R5's three leak channels: variable names, "Đáp án Z3 là …" reflection, and SMT text. M1 needs no LLM at all. Vietnamese has no plural inflection and classifiers are unchanged, so a numeric substitution is always grammatical.

**Tag check** (`tag_multiset_ok`). Parse every `<q n=k>` and normalize its value. The multiset {(k, value)} must equal the givens of P2 exactly. The untagged text may contain no other numerals, apart from `world_const` items and ordinal or date words on a whitelist.

**Surface localization for arm A** (`localize_surface`). Person names are mapped to a Vietnamese name list. Currency is re-labelled through the IR: a "USD" unit becomes "nghìn đồng" with the value unchanged, so 12 USD reads "12 nghìn đồng". Values are not rescaled, so nothing is re-verified. This puts both arms on a Vietnamese surface, so that "localization" is not confounded with "arm". Only arm B changes the *scenario*. That is a property of the arm, and it is reported.

**Symbolic (MATH algebra) seeds.** The same IR with `kind="symbolic"` is used: SymPy symbols and equations, spans on the LaTeX numerals.
- M1 resamples coefficients under the answer-cleanliness rule, e.g. integer or small-denominator roots.
- M2 asks for a parameter given the result ("tìm a để …").
- M4 changes a relation, e.g. an inequality direction or the operation in a function composition.
- M5 asks for another derived quantity, e.g. the other root or f at another point.
- Rendering edits the LaTeX spans and the question sentence only.

### 2.3 Shared components: common gates and the solution writer

```python
def common_gates_and_solution(text, gold, meta) -> Item | Reject:
    # G1 language: fastText/langid "vi" prob ≥ 0.95; no CJK chars; LaTeX balanced
    # G2 length: 0.5×–1.6× seed length (2602.19187 used ≤1.5×; 1.6× allows M3/M6)
    # G3 leak regex (from R5): h_/d_ names, \banswer\b, SMT|ràng buộc|Z3, <BÀI, \boxed,
    #    "đáp (án|số) (là|=)", "gợi ý", "kết quả (mong muốn|cuối cùng phải)"
    # G4 decontamination: 13-gram overlap = 0 AND max cosine(bge-m3) < 0.92 vs
    #    {Vi-ExamMath, Vi-GSM8K test, Vi-MATH test, PolyMath-vi, DeepSEA-vi}
    # G5 dedup: MinHash (5-gram shingles) Jaccard < 0.8 vs all accepted items of the same arm;
    #    ≤ 3 accepted items per seed per arm (R5: 3,684 identical sibling pairs)
    # G6 well-posedness judge (MathQ-Verify-style decomposition, one call):
    #    J = JUDGE(text) → {atomic_ok, no_conflict, complete, single_question, realistic,
    #                       natural_1to5}
    #    accept iff atomic_ok ∧ no_conflict ∧ complete ∧ single_question ∧ realistic
    #               ∧ natural ≥ 4
    #    second judge (different family) runs on all items; disagreements go to a third
    #    (tie-breaker); a majority of 3 is needed
    # S  solution writer: blind-solve k ≤ 4 times (short Vietnamese CoT, \boxed{});
    #    keep the first sample whose answer == gold (math_verify, exact; 0.5% tolerance removed)
    #    if none matches → Reject("unsolved")   [this is also a difficulty signal, recorded]
    ...
```

**Solution-writer prompt.** "Giải bài toán sau bằng tiếng Việt, ngắn gọn từng bước, kết thúc bằng \boxed{đáp số}." The writer gets only the problem text, never the gold.

The writer is the same model with the same prompt and k for both arms, so the trace format and length are identical. Short CoT is used because ≤3B students learn worse from long CoT (`01` B1).

**Judge models.** NIM `nvidia/nemotron-3-ultra-550b-a55b` is the primary judge (the fastest strong model on NIM: median 9–15 s). It is not the same family as the generator, DeepSeek-v4.1-flash. The second judge is `claude-haiku-5-5` via the Batch API. The tie-breaker is Gemini free tier (`gemini-3.8-flash` or `gemini-3.5-flash-lite`, listed free on the official pricing page), restricted to training candidates. Reasons for the Gemini restrictions are in §3 D12.

### 2.4 Arm B: LLM-native Vietnamese generation

```python
def arm_B(seed, op, rng) -> Item | Reject:
    card = LOCAL_CARD[seed.id]          # sampled once per seed (same card used by arm A's surface pass)
    target = DIFF_TARGET[op](seed)      # "giữ nguyên độ khó" | "khó hơn một bậc" (M2/M3/M6)
    out = GENERATOR(seed.text_vi, op_instruction_vi[op], card, target, grade_hint(seed))
    # out = {"plan": str, "problem": str, "answer": str, "op_applied": str, "change_summary": str}
    text, a_gen = out["problem"], out["answer"]
    if out["op_applied"] != op:          return Reject("op_ignored")    # self-report + judge check
    if contains_value(text, a_gen):      return Reject("leak")
    item = common_gates_and_solution(text, gold=None, meta=...)        # gold unknown → below
    return item
```

**Arm B answer and label procedure** (it replaces "gold" in `common_gates_and_solution`):

1. Draw two blind solutions from the shared solution writer: s₁ and s₂.
2. **V1 (self-consistency)** passes if a_gen, s₁ and s₂ all agree. If they do not, draw s₃ and s₄. V1 passes if ≥ 3 of {a_gen, s₁..s₄} agree; that value becomes the gold, and the matching trace is kept. Otherwise reject. This is the CoT-Self-Instruct "Answer-Consistency" logic. It is the **main-arm filter**.
3. **V2 (program label, optional, not a filter in the main arm).** Run `FORMALIZER_B(text)` into the same typed IR as arm A, then apply the §2.1 acceptance rules (types hold, derived answer, every given spanned). V2 = 1 iff the executed answer equals the V1 gold. The label is stored. It defines the sub-arm **B-V2** (program-verified LLM data) for the "does the certificate matter at fixed generator" analysis (RQ-3 in `01`).

**Generator prompt outline.** The generator is `deepseek-ai/deepseek-v4.1-flash` on NIM. It is the same model that formalizes and edits for arm A, which holds the generator fixed. The prompt is written in Vietnamese:

```
SYSTEM: Bạn là giáo viên toán Việt Nam soạn bài tập cho học sinh {grade_hint}. Viết bằng tiếng Việt
        tự nhiên như sách giáo khoa Việt Nam (không dịch máy). Bài toán phải: tự chứa đủ dữ kiện, có
        đúng một câu hỏi, số liệu thực tế (số lượng đồ vật là số nguyên; giá tiền bằng đồng, hợp lý),
        không chứa lời giải, gợi ý hay đáp số.
USER:   BÀI GỐC: {seed_vi}
        PHÉP BIẾN ĐỔI: {op_instruction_vi}
          M1 "Đổi các số liệu, giữ nguyên cấu trúc lời giải."
          M2 "Đảo bài toán: cho biết kết quả của bài gốc, hỏi một dữ kiện ban đầu."
          M3 "Thêm đúng một bước tính hợp lý vào giữa câu chuyện (không thêm vào sau đáp số)."
          M4 "Thay đổi một quan hệ (thêm↔bớt, gấp k lần, phần trăm…)."
          M5 "Hỏi một đại lượng khác cần ít nhất hai bước tính."
          M6 "Ghép với bài sau thành một bài nhiều bước: {seed2_vi}"
        BỐI CẢNH ĐỊA PHƯƠNG: {card: bối cảnh, tên người, vật phẩm, khoảng giá, đơn vị}
        ĐỘ KHÓ: {target}
OUTPUT JSON: {"plan": "...(≤3 câu)", "problem": "...", "answer": "...", "op_applied": "M?",
              "change_summary": "..."}
```

The `plan` field is discarded. CoT-Self-Instruct plans before it writes, and keeping the plan internal stops it from leaking into the problem.

**Localization cards.** A bank of about 300 cards is built once. Each card holds: setting (chợ, Tết, trường học, ruộng lúa, xe buýt, siêu thị, …); 4 regional Vietnamese names; items with realistic VND price ranges; and units. The bank is drafted with about 300 generator calls and hand-checked in about 1 hour. Each seed gets one card, which both arms use: arm B for the scenario, arm A for names and the currency label.

### 2.5 Decision thresholds (both arms)

| Check | Threshold | Arm | Basis |
|---|---|---|---|
| Formalizer accepts seed | exact Fraction match to gold; every given spanned; query derived (depth ≥ 1); all constraints hold | A (and B-V2) | R3: 10.7% hard-coded answers, so "derived" is required |
| Resample range | log-uniform [v/3, 3v], snapped to the type grid (COUNT 1, VND 500, USD 0.01, PERCENT 5, MEASURE 0.5); ≤ 200 tries | A | GSM-Symbolic-style value domains |
| Clean answer | COUNT/AGE: int ≥ 0; MONEY: on grid; MEASURE: ≤ 2 decimals; RATIO/ABSTRACT: reduced p/q with q ≤ 100 (q ≤ 12 for story items); \|ans\| < 10⁷; ≠ seed answer (M1–M5) | A (B checked by judge `realistic`) | R5: 50.5% "ugly" golds |
| Answer tail | ancestor-difference rule (§2.2) | A | R5: 84% tacked |
| Identity | isomorphic DAG ∧ same givens ∧ same answer → reject | A | R5: 18.4% identity |
| Tag multiset | exact equality | A | — |
| Cycle consistency | executed answer of the blind back-formalization == P2.answer (exact) | A | MathCAMPS: 97.7% faithful among kept |
| Novelty (text) | chrF(variant, seed) ≤ 0.90 (M2–M6) | A, B | [ESTIMATE]; calibrate on the dev seeds |
| Novelty (structure) | GED(canonical DAG(variant), DAG(seed)) ≥ 1 for M2–M6 (variant DAG from back-formalization) | A, B-V2 | MATH-Perturb: "hard" > "simple" |
| Language | langid vi ≥ 0.95, 0 CJK | A, B | — |
| Leak | 0 regex hits; answer value string absent from the text (all VN numeric formats) | A, B | R5: 58% leak |
| Judge | 5 booleans true ∧ naturalness ≥ 4/5; majority of 3 judges on disagreement | A, B | MathQ-Verify decomposition |
| V1 self-consistency | ≥ 3 of 5 answers agree (incl. generator's), or 3/3 on the first pass | B | CoT-Self-Instruct |
| Dedup | MinHash Jaccard < 0.8; ≤ 3 per seed per arm | A, B | R5 siblings |
| Decontamination | 13-gram = 0 ∧ cos < 0.92 vs all eval sets | A, B | `01` C15 |
| **Pilot quality gate** (before scale-up; 200 items per arm, human + Opus audit) | "teacher would write" ≥ 70%; genuinely new (M2–M6) ≥ 60%; leak 0%; absurd ≤ 3%; Opus vs human κ ≥ 0.6 | A, B | R5 baseline: 10.5% / 0% / 58% / 15.5% |

Judge calibration: the threshold "natural ≥ 4" is tuned on 300 dev-seed items labelled by a human. It should give judge-accept precision ≥ 0.90 against the human "usable" label. If precision falls short at 4, raise the threshold to 5 [ESTIMATE].

### 2.6 Matching difficulty, diversity and coverage across arms

**Coverage (the confound FLAMES warns about).** Arm A can only use seeds it can formalize. Two analyses come out of one generation run:

- **Primary, paired:** keep only (s, j) slots where *both* arms produced an accepted item. Same seeds, same operators, N matched by construction. This isolates the mechanism.
- **Secondary, natural coverage:** each arm is subsampled to the same N from all of its accepted items. This measures the practical value, including arm A's coverage loss.

**Difficulty (stratified matching).**
1. Every accepted item gets a reference pass rate p̂ from a frozen self-run `Qwen3.5-2B`. This is reference solver row I in `02_models.md`. It runs on Kaggle T4×2 with vLLM: 4 samples at T = 0.7, plus greedy, max 512 new tokens.
2. Strata are (source ∈ {GSM8K, algebra, prealgebra}) × (MATH level or GSM8K step-count tercile) × (p̂ bucket ∈ {0, ¼, ½, ¾, 1}). The step count comes from the IR op count, or from the back-formalized IR for arm B.
3. In each stratum, sample min(n_A, n_B) items from each arm.
4. Check after matching: the standardized mean difference of p̂, op count and token length is < 0.1, and the KS test on p̂ gives p > 0.05.

**Diversity: report it, do not force-match it.** Diversity is a mechanism by which the arms differ: 2602.19187 found Sym diversity 0.37 vs NL 0.07. Forcing it equal would erase a real effect. The following are reported per arm and used as mediators in the analysis:

- (a) Within-seed variant diversity: mean pairwise cosine distance under `bge-m3`, a multilingual embedder (2602.19187 protocol; MiniLM is English-centric).
- (b) Corpus diversity: Vendi score on the same embeddings, and DCScore.
- (c) Distance to seed: chrF, and embedding cosine.
- (d) **Structural novelty:** the distribution of GED(variant DAG, seed DAG), plus the edit-class histogram (value-only / premise / structural / compose).
- (e) Scenario novelty: the share of items whose setting differs from the seed (judge field).

One sensitivity subset matches on (d) only, subsampling so the edit-class histograms are equal. It tests whether arm differences survive when structural novelty is held fixed.

---

## 3. Alternatives compared (one choice per decision)

| D1 Symbolic representation | Evidence | Coherence risk | Cost | |
|---|---|---|---|---|
| SMT-LIB (old pipeline) | 2602.19187: unnatural, below SymPy and NL. R3: 42% expose named story quantities; 10.7% hard-coded | High | Z3 local | ✗ |
| Free SymPy code edited by an LLM (2602.19187) | +1–2 at 1.5B; diversity 0.37 | Medium: render drifts from the story | 1 render call | ✗ |
| Python function graphs (RV-Syn) | +5% rel. over ScaleQuest at 50k | Medium | 72B back-translation | ✗ |
| **Typed quantity DAG with text spans + SymPy execution (chosen)** | Combines executable verification with alignment to the seed text. Typed integrality removes R5's ABSURD class (15.5%). | Low: edits are local to aligned spans | +1 formalizer call per seed (amortized) | **✓** |

| D2 Mutation locus | Evidence | | 
|---|---|---|
| Answer post-composition (old) | R5: 0% new, 84% tacked, 58% leak | ✗ forbidden by the tail rule |
| Value-only resampling | GSM-Symbolic/MATH-P-Simple: a "simple" perturbation, keeps the solution pattern | partial: capped at 20% (M1) |
| LLM-chosen free code edits | 2602.19187 Base-Sym; uncontrolled operator mix | ✗ (no shared schedule possible) |
| **Typed operator set M1–M6 on premises/structure (chosen)** | FLAMES: complexity-raising agents best; MetaMath: SV+FOBAR +3.8 GSM8K; MathFusion: composition +18; MATH-Perturb: hard perturbations are the informative ones | **✓** |

| D3 Rendering | Evidence | |
|---|---|---|
| Free verbalization from the program (old, 2602.19187, RV-Syn) | Leaks and drift (R5; 2602.19187 notes context change) | ✗ |
| Grammar/templates (MathCAMPS, GSM-Infinite) | Faithful but formulaic; needs hand grammars per standard | ✗ (only M1 is template-like, and it is span-substituted) |
| **Span-grounded minimal edit of the original text, LLM only for M2–M6, no answer shown (chosen)** | Keeps the seed's natural Vietnamese; the edit list bounds the change; tags make faithfulness checkable | **✓** |

| D4 Round-trip check | Evidence | |
|---|---|---|
| Blind solve + answer feedback/reflection (old) | R5: the reflection loop wrote golds into problems (6.1%) | ✗ |
| LLM judge only | Judges are hackable (`01` A11); VHG needed a hard verifier where possible | ✗ as the sole check |
| **Blind back-formalization by another model family + exact execution match (chosen)** | MathCAMPS: removes 83% of unfaithful items, keeps 97.7% faithful | **✓** (plus the judge as a common gate) |

| D5 Well-posedness | Evidence | |
|---|---|---|
| Solvability filter | FLAMES: removes 30% of real MATH, hurts | ✗ |
| Self-consistency only | SAND-Math/CoT-Self-Instruct proxy; can reward consistently wrong problems | used in arm B only |
| **Deterministic uniqueness/types (arm A) + MathQ-Verify-style decomposed judge (both) (chosen)** | MathQ-Verify ≈ 90% precision [snippet]; deterministic checks are exact | **✓** |

| D6 Arm B generation | Evidence | |
|---|---|---|
| Generate in English, then translate | Mixes translation into the "native" arm, which confounds RQ-1 in `01` | ✗ |
| Plain NL rewrite of the seed ("modify this problem") | 2602.19187 Base-NL; low diversity 0.07 | ✗ |
| Persona/concept-from-scratch (PersonaHub, MathScale, MathSmith) | No seed link, so the paired design breaks | ✗ (possible later extension) |
| Self-play generator (R-Zero/AZR/VHG) | Needs RL GPU-hours; pseudo-labels degrade 79→63%; small models collapse earlier | ✗ |
| **Operator-conditioned CoT-Self-Instruct in Vietnamese with a localization card (chosen)** | Same operator schedule as A ⇒ controlled; planning step (CoT-Self-Instruct, 2511.09907 "plan before pose") | **✓** |

| D7 Arm B verification | Evidence | |
|---|---|---|
| None | FLAMES: coverage > reliability | kept as the B-V0 ablation pool (items passing common gates only) |
| **V1 self-consistency as filter (chosen for the main arm)** | Standard (CoT-Self-Instruct); cheap | **✓** |
| V2 program label as filter | Shrinks coverage toward arm A's; confounds | label only → sub-arm B-V2 |

| D8 Solution writer | Evidence | |
|---|---|---|
| Template solution from the IR | Unnatural CoT, and arm-specific format ⇒ confound | ✗ |
| Answer-conditioned rationalization | Rationalizes wrong problems | fallback ✗ |
| **Blind solve k ≤ 4, keep the first match to gold; same model/prompt for both arms (chosen)** | Keeps format identical across arms; short CoT for ≤3B (`01` B1) | **✓** |

| D9 Difficulty control | Evidence | |
|---|---|---|
| Prompt "make it harder" (Evol) | Not solver-aware (2511.09907 critique) | ✗ |
| RL-trained generator (SwS, MathSmith, 2511.09907, VHG) | +2.7 to +10 but needs RL GPU-hours | ✗ |
| **Operator-based structural difficulty + post-hoc reference-solver p̂ + stratified matching (chosen)** | Free on the API side; about 3 T4-hours | **✓** |

| D10 Diversity measure | | |
|---|---|---|
| Distinct-n | Lexical only | ✗ |
| Embedding pairwise cosine (2602.19187) | Comparable to prior work | ✓ primary (bge-m3) |
| Vendi / DCScore | Corpus-level, axiomatic | ✓ secondary |
| **Structural GED on the IR + edit-class histogram (chosen as the key mediator)** | Directly measures "genuinely new reasoning", R5's headline failure | **✓** |

| D11 Coverage handling | | |
|---|---|---|
| Natural coverage only | Confounds mechanism with coverage (FLAMES) | secondary |
| **Paired (s, j) intersection as primary + natural coverage as secondary (chosen)** | Isolates mechanism and still reports practical value | **✓** |

| D12 Judge / API roles | Facts checked this session | Choice |
|---|---|---|
| NIM `nemotron-3-ultra` | Median 9–15 s (`02_models` smoke) | **primary judge**, back-formalizer B |
| NIM `deepseek-v4.1-flash` | Median 38 s easy, 96 s hard | **generator / editor / formalizer A / solution writer** (one fixed generator) |
| Claude `claude-haiku-5-5` Batch | $0.10 / $0.50 per MTok, −50% on Batch (claude-api skill, cached 2026-10-06) | **second judge on all items** |
| Claude `claude-opus-5-5` Batch | $4 / $20 per MTok, −50% on Batch | **audit judge** (pilot + final 400/arm); also safe for held-out test items |
| Gemini free tier | Official rate-limit page shows *no* free-tier numbers (limits are only visible in AI Studio per project). Free models listed include `gemini-3.8-flash` and `gemini-3.5-flash-lite`. | **tie-breaker judge on training candidates only**. Never send Vi-ExamMath or any eval item (contamination control). With third-party reports of ~1,500 RPD [UNVERIFIED], it cannot be a full-corpus judge. |

---

## 4. Cost per 10k accepted items, and expected yield

**Throughput math (NIM).** One key allows 40 RPM; budget 38 for safety. Sustaining 38 RPM at latency L needs about 38·L/60 requests in flight per key: ≈ 25 at L = 40 s, ≈ 57 at L = 90 s. Implementation: asyncio with a token bucket of 38 RPM per key and a semaphore of 64 per key. Timeouts are 300 s, with 2 retries (the Gemma-4 smoke test had one 410 s timeout). Wall-clock per key is (calls / 38) minutes.

**Arm A yield** [ESTIMATE; the pilot replaces every number]:

| Stage | Pass rate | Basis |
|---|---|---|
| Seed formalized (typed IR, spans, derived) | GSM8K 0.70, algebra/prealgebra 0.45 → **≈ 6,560 of 10,422 seeds** | R3: 58–65% faithful-deriving on SMT; spans add strictness, Python/SymPy removes Real-sort limits; R4 is worse for MATH |
| Operator applicable + constraints satisfiable | 0.85 | — |
| Symbolic gates (unique, tail, identity, clean, plausible) | 0.85 | — |
| Tag check (M2–M6; M1 ≈ 1.0) | 0.88 | — |
| Cycle consistency | 0.78 | MathCAMPS kept ~90% on K–8; edits plus Vietnamese lower it |
| Common gates (language, leak, dedup, decontamination, judge) | 0.80 | — |
| Solution writer finds a matching trace (k ≤ 4) | 0.92 | — |
| **Overall per candidate** | **≈ 0.33 (M2–M6), ≈ 0.55 (M1); mix ≈ 0.38** | — |

Candidates needed ≈ 10,000 / 0.38 ≈ **26k**. The capacity is ≈ 6,560 seeds × J = 3 slots ≈ 19.7k slots, which is **not enough**. Therefore J = 5 slots per seed, with the per-seed cap of 3 accepted items unchanged. Alternatively, accept that the paired (primary) set will be about 7k, matched N.

**NIM calls for arm A, per 10k accepted:**

| Step | Calls |
|---|---|
| Seed formalization (one-time, shared; 10,422 × 1.4 attempts) | 14.6k |
| Editor (≈ 80% of candidates are M2–M6) | 20.8k |
| Back-formalization (on tag-pass) | 18.0k |
| Primary judge (on cycle-pass) | 14.0k |
| Solution writer (≈ 11.2k × 1.5 attempts) | 16.8k |
| **Total NIM** | **≈ 84k → 37 h at 1 key; 9.2 h at 4 keys; 6.1 h at 6 keys** |
| Claude Haiku 5.5 Batch, second judge (14k calls × ~1.2k in / ~0.6k out incl. thinking) | ≈ 14k × ($0.06/M × 1.2k + $0.25/M × 0.6k) ≈ **$3** |
| Gemini tie-breaks (≈ 12% of judged) | ≈ 1.7k calls (spread over ≥ 2 days if free RPD ≈ 1.5k) |

**Arm B yield and calls, per 10k accepted:**

| Stage | Pass rate | Basis |
|---|---|---|
| Op followed + no leak | 0.93 | — |
| Common gates incl. judge | 0.78 | FLAMES/VeriGeo: many synthetic items are ill-posed |
| V1 self-consistency | 0.75 | R-Zero: 3–7/10 agreement band is typical on hard items; most seeds are easy |
| **Overall** | **≈ 0.54** → ≈ **18.5k candidates** | — |

| Step | Calls |
|---|---|
| Generator | 18.5k |
| Primary judge | 17.2k |
| Solution writer (2 on each judged-pass item, plus 2 more on ≈ 30% disagreements) | ≈ 13.4k × 2.6 ≈ 34.9k |
| V2 back-formalization of accepted items (× 1.3) | 13.0k |
| **Total NIM** | **≈ 84k → 37 h at 1 key; 9.2 h at 4 keys** |
| Claude Haiku 5.5 Batch, second judge | ≈ **$3.5** |

**Shared costs.**

| Item | Cost |
|---|---|
| Reference-solver p̂ on ≈ 2 × 12k items (accepted plus a matching buffer) × 5 samples × ~350 tokens ≈ 42M generated tokens | Kaggle T4×2 vLLM, Qwen3.5-2B: ≈ 3–6 h [ESTIMATE; Qwen3.5 on T4/vLLM support is [UNVERIFIED] → fallback Qwen3-1.7B]. **0 Modal hours.** |
| Embeddings (bge-m3, ~60k texts incl. seeds and eval sets) | Laptop 3050 Ti or CPU, < 1 h |
| Claude Opus 5.5 Batch audit: 200 (pilot) + 400 (final) per arm × (~2k in, ~1.5k out incl. thinking) | 1,200 × ($2/M × 2k + $10/M × 1.5k) ≈ **$23** |
| Localization card bank | 300 NIM calls + ~1 h of human review |
| **Total for both arms (10k each)** | ≈ **168k NIM calls** (≈ 18.5 h on 4 keys, ≈ 12.3 h on 6 keys), **≈ $30 Claude**, ≈ 5 Kaggle T4-hours, $0 Modal |

**One-day schedule.**
1. Pilot: 200 dev-seed slots per arm, ≈ 2.5k calls, about 1 hour on 4 keys. Run the Opus and human audit and the §2.5 gate.
2. Fix thresholds, then run both arms in parallel, 2–3 keys each, for about 12–18 hours.
3. Run the difficulty labelling on Kaggle as the batches land.

With fewer than 4 keys, generate 5k per arm first (the first data-efficiency budget point in `01` RQ-1) and finish 10k later.

**Expected final pools** [ESTIMATE]:
- Arm A ≈ 10k accepted from ≈ 26k candidates.
- Arm B ≈ 10k accepted from ≈ 18.5k candidates, of which ≈ 60% carry the V2 program label. That is the share back-formalizable into the typed IR, which is about arm A's seed-formalization rate.
- The paired primary set (both arms accepted on the same (s, j)) is ≈ 6–7k per arm before difficulty matching and ≈ 5–6k after.

**Expected quality** (to be checked at the pilot gate, against the R5 baseline of 10.5% natural / 0% new / 58% leak):
- Arm A ≥ 70% natural, ≥ 60% new for M2–M6, 0% leak and ≤ 3% absurd. These are achievable because M1 involves no LLM and M2–M6 are local edits checked by cycle consistency.
- Arm B ≥ 80% natural (no seed-text constraint), but ≈ 10–20% of V1-accepted items fail V2. This is the measured "verification gap" that RQ-3 needs.

**Expected downstream effect sizes** (for power planning in `01`):
- Generated vs seed-only: +2–3 points at 1.5B (2602.19187).
- Arm A vs arm B: |Δ| ≤ 2 points (2602.19187: +1.2/+2.1 at 1.5B, −0.6/+1.8 at 3B).
- ⇒ Only the paired-seed design, item-level mixed-effects models and ≥ 3 seeds can detect it. The bigger expected difference is in diversity (structural GED) and robustness slices, e.g. MATH-P-Hard-style or Vi-ExamMath. That makes them the mediators to report.

---

## 5. Open items and risks

1. **Formalization yield on Vietnamese MATH algebra/prealgebra** is the main unknown (R4: 17% named quantities under SMT). If the pilot shows < 35%, restrict arm A to GSM8K plus prealgebra word problems. Report the coverage loss; that is a finding, not a bug.
2. **Cycle-consistency false rejects.** MathCAMPS lost 7 of 215 faithful items, and Vietnamese numeral wording may lose more. Calibrate the numeral normalizer on the 7,473 GSM8K-vi seeds first: every seed literal must round-trip.
3. **Judge agreement.** If Haiku vs Nemotron κ < 0.5 on the calibration set, switch the second judge to `claude-sonnet-5-5` Batch ($2/$10 per MTok → ≈ $45 per arm; still within the $100 budget if the Opus audit drops to 200 per arm).
4. **Hosted-model churn** (`02_models` §0: the generator went EOL within a month). Freeze the corpus as soon as it is generated. Log the model ID, the date, the full request and the response `model` field.
5. **NIM rate limit with multiple keys** (per key or per account) was not measured [UNVERIFIED].
6. **Not verified this session:** exact numbers for MathScale, CoT-Self-Instruct, MathFusion per-benchmark, MathSmith, MathAgent (abstract-level only); TreeCut, Vendi and DAGGER (snippet only); and the MathQ-Verify precision/recall (search snippet of v2).
