# T2: Verification, labeling, quality filtering, data selection and answer grading (2026-11-07)

Scope: how generated Vietnamese items are checked (problem validity and answer label), how every arm is
deduplicated, decontaminated and subsampled the same way, and how Vietnamese answers are graded. Resource
constraint (coordinator update, 2026-11-07): GPU time is the scarce resource. Modal L4 costs $0.80/h, so $30
buys about 37 L4-hours. NIM (several keys, 40 RPM each), the Gemini free tier, $100 of Claude credit (~7k Opus
Batch calls) and a Kaggle T4 are available. **Every design below uses 0 GPU-hours. It spends API calls
instead.**

Evidence tags: `[arXiv id §/Table]` means I opened the page on 2026-11-07. `[RUN: file]` means a command I ran
(scripts and outputs are in the session scratchpad, see §7). `[UNVERIFIED]` means I did not open the source.

---

## 0. Decisions at a glance

| # | Decision | Choice | Main reason |
|---|---|---|---|
| D1 | What "checked" means for generated arms (N, Y) | **Problem-validity gate + answer-label consensus + independent re-formalization.** Answer checks alone are not enough | In the old pipeline, 195/200 augmented items passed the NL↔SMT answer check, but a teacher accepted only **75/195 = 38.5 % [31.9–45.5]** `[RUN: R5 labels]` |
| D2 | Validity checker | MathQ-Verify-style decomposed check with **2 different-family LLMs, both must say valid** ((2,2) vote) | (2,2): precision 89.6 / recall 62.7. A single model: 86.7 / 68.9 `[2505.13903 T3]` |
| D3 | Answer label | Reference answer (generator's or program's) must match **2/2 independent different-family solvers**. On a split, add 2 more solvers and accept at ≥3/4 | The Answer-Consistency filter scored 57.2 vs 53.0 unfiltered for a 4B student `[2507.23751 T1]`. Self-consistency labels alone drift: 79 % → 63 % `[2508.05004 T5]` |
| D4 | Formalization faithfulness | A third model writes an **independent SymPy program** from the Vietnamese text. Its result must equal the label, and its numeric literals must equal the text's literals | Round-trip equivalence predicts but does not guarantee faithfulness: NLI drift 28.7 % on failed vs 16.2 % on passed items `[2604.25031]` |
| D5 | PRM or LLM trace critic | **Not used** | Best open 7B PRM reaches ProcessBench F1 73.5, and the 1.5B reaches 36.4. BoN gain over maj@8 is +1.4 `[2501.07301 T6,T7]`. It needs GPU. GPT answer verification *lowered* OpenThoughts math 65.6 → 61.4 `[2506.04178 §4.5]` |
| D6 | Data selection across arms | **No quality or difficulty selection.** Apply the same dedup, decontamination and **seed-matched stratified random** sampling to every arm | Random is hard to beat at scale `[2503.01807][2410.09335]`. Quality selection would confound the checked/unchecked factor. Gradient methods cost GPU |
| D7 | Grader | **Typed structured gold, never extracted.** Locale-aware number parser, structure decided before number parsing, and math-verify 0.9.0 as the atom-equivalence backend | Prototype passes 60/60 targeted cases. The current scorer passes 39/60 and raw math-verify 38/60 `[RUN: test_out.txt]` |
| D8 | RLVR reward | **Rule-only typed grader, no model verifier in the loop.** Use checked items only | Model verifiers are hackable: general-verifier has a 22.1 % adversarial-prefix success rate `[2505.22203 T3]`. Noisy labels cost RLVR 8–10 points `[2603.16140]` |

---

## 1. Evidence table

| Method / finding | Source (date) | Measured numbers | Cost / scale | Use here |
|---|---|---|---|---|
| Rule-based verifiers (VERL, Qwen-Math, HF Math-Verify) | 2505.22203 (May 2025, v3 Aug 2026), Table 5 | Precision ≈ 1.00. Recall 0.78–0.95 on Math/DeepscaleR/ORZ/Skywork. HF Math-Verify recall 0.88–0.95. Recall ≈ 47 % on WebInstruct | CPU | The rule grader is a high-precision floor. Recall is the problem |
| Model-based verifiers | same, Table 1 | P/R: general-verifier 0.90/0.86. xVerify-3B-Ia 0.90/0.78. R1-Distill-Qwen-7B 0.79/0.59 | 1.5–7B GPU | Not as the primary grader |
| Hybrid (model only when rule says wrong) | same, §5 | Recall +~3 pts at >98 % precision. RL avg 55.0 → 57.3 with a 1.5B hybrid | +1 model call per rule-negative | Eval adjudication pattern only |
| Verifier hacking | same, Table 3 | Attack success rates: general-verifier 22.1 % adversarial prefix and 28.5 % "answer explanation". Qwen2.5-72B ≤1.3 %. xVerify-3B-Ia 0.0–1.1 %. A fine-tuned 1.5B verifier was hacked after ~450 RL iterations | – | No model verifier inside RL reward (D8) |
| TinyV | 2505.14625 (May 2025) | Prime verifier marks 38.5 % of incorrect-labelled pairs falsely (87K/226K). 42.1 % of prompts are affected. ~20 % of RL rollouts are false negatives. TinyV is 1.5B (Qwen2.5-1.5B-Instruct), called only on rule-negatives, adds 6 % overhead, and gives up to +10 % pass rate | GPU 1.5B | Motivates the recall fixes in our grader. Model not used (English, GPU) |
| xVerify | 2504.10481 (Apr 2025) | >95 % F1/accuracy on their VAR set. 3B-Ib > GPT-4o | 0.5–3B GPU | Cited alternative |
| CompassVerifier | 2508.03686 (Aug 2025), VerifierBench | Acc/F1: 3B 85.0/80.4, 7B 87.5/83.4, 32B 90.8/87.7. GPT-4.1 81.1/69.8. xVerify-8B 80.8/70.4. Baselines <40 F1 on sequence answers. Adds an "invalid" (truncated/repetitive) class | GPU | Cited alternative. Our typed grader handles sequences structurally |
| math-verify vs LLM judge on hard objects | 2603.18886 (Mar 2026) | On 168 human-labelled *disagreements*, o3 was right in 94.05 % and math-verify in 5.95 % | API | On disagreements, an LLM adjudicator beats rules. Use it only for adjudication |
| Math-Verify asymmetry | github.com/huggingface/Math-Verify | `verify(gold, pred)` is order-sensitive. Gold inequality vs pred interval → True, reverse → False. Extraction needs a LaTeX environment | CPU | Always pass gold first. Wrap bare answers in `$…$` |
| MathQ-Verify (problem validity) | 2505.13903 (May 2025) | 5 stages (leak/instruction, linguistic, atomic condition, cross-condition contradiction, completeness). (n,k) votes: (1,1) P 86.7 / R 68.9. (2,2) 89.6/62.7. (3,2) 86.2/76.0. (5,5) 92.4/50.8. In NuminaMath-derived synthetic data, 39.5 % of problems were invalid (contradiction 14.0 %, domain 8.0 %, instruction 6.7 %, nonsemantic 5.5 %, incomplete 5.4 %) | 1 call per model | Stage 1 of our pipeline (D2) |
| FaultyMath | 2410.18921 [search snippet only] | Of 635 GPT-4 "unsolvable" problems that survived self-verification, humans rejected 272 as actually solvable | – | LLM-only invalidity labels are noisy, so a human calibration set is needed |
| ReliableMath | 2507.03133 (Jul 2025) | "Small LLMs rarely show any progress" at flagging unsolvable problems, even with reliable prompts (qualitative) | – | Use large judges, not the student, for validity |
| MiP-Overthinking | 2504.06514 (Apr 2025) | Reasoning models produce much longer responses on missing-premise questions (qualitative) | – | Optional cheap signal: solver length outlier → re-check |
| Answer-Consistency filter (synthetic prompts) | 2507.23751 (Jul 2025), Table 1 | Qwen3-4B-Base with GRPO, avg of 4 benchmarks: CoT-Self-Instruct unfiltered 53.0 → +Answer-Consistency **57.2** (2,926 items). +RIP 56.2. s1k 44.6. OpenMathReasoning 47.5. Self-consistency filter threshold SC ≥ 0.5 | K solver samples per item | D3 |
| Majority-vote pseudo-label accuracy | 2508.05004 R-Zero (Aug 2025), Table 5 | 79.0 % → 69.0 % → 63.0 % over iterations. A 4B model fell from 71.0 % to 42.2 %. Kept questions have 3–7 of 10 agreeing | – | Single-model self-consistency is not a gold source |
| Majority vote vs reward accuracy | 2504.16084 TTRL (Apr 2025) | AIME24: label accuracy 37 %, yet reward accuracy 92 % ("lucky hit") | – | Noisy labels hurt less in RL than accuracy suggests, but see 2603.16140 |
| Panel of different-family judges | 2404.18796 (Apr 2024) | A panel beats a single GPT-4 judge in 3 settings × 6 datasets, is >7× cheaper, and has less intra-model bias | – | Different-family voters (D2, D3) |
| Round-trip autoformalization + SMT equivalence | 2604.25031 (Apr 2026, ICML 2026) | Equivalence rate 44.7 % → 85.3 % (Claude) with scoped repair. NLI drift: failed 28.7 % vs passed 16.2 % (ratio 1.78). Human check (n=25): 8 of 11 failures were actually faithful. Agreement is "necessary but not sufficient" | 6–26 LLM calls per repaired item | D4 (we use 1 call, no repair loop) |
| FormalAlign | 2410.10135 (Oct 2024) | Alignment-selection 99.21 vs GPT-4 88.91 (FormL4-Basic). 66.39 vs 64.34 (MiniF2F-valid) | Trained model, Lean | Not applicable (Lean, needs training) |
| BEq+ | 2406.07222 (EMNLP 2025) | Up to 45.1 % autoformalization accuracy, undergrad level. ProofNetVerif has 3,752 labelled examples | Lean | Not applicable |
| Old pipeline: SMT-"verified" ≠ valid problem | R3/R4/R5 + my recount of `aug.jsonl` | GSM8K SMT faithful 54.0 %. MATH: 45 % of `status=pass` rows broken. Augmented: NL↔SMT consistent 195/200, teacher accepts 75/195 (38.5 %), "would write" 21/195 (10.8 % [7.2–15.9]) | – | D1 |
| PRMs | 2501.07301 (Jan 2025), Tables 6, 7 | ProcessBench F1: Qwen2.5-Math-PRM-7B 73.5, -72B 78.3, Skywork-PRM-1.5B 36.4, -7B 42.1, Math-Shepherd-7B 31.5. o1-mini critic 87.9. BoN@8: PRM-7B 67.6 vs maj@8 66.2 | 7B GPU, English | D5: not used |
| OpenThoughts answer filtering | 2506.04178 (Jun 2025), §4.5, Table 7 | Math avg: no filtering 41.9, random 41.6, GPT verification **40.0** (math subset 61.4 vs 65.6). "No filtering strategy outperformed the baseline" | API | For SFT *traces*, correctness filtering may not help. Our gate is a factor, not an assumption |
| OpenThoughts question selection | same, §4.3, Table 5 | Math: LLM response-length filter 41.9 > AskLLM 36.3 > FastText 35.6. Best filter +4 % over random (math) | 1 LLM call per question | Alternative for D6 (rejected for fairness) |
| OpenMathInstruct-2 noise and diversity | 2410.01560 (Oct 2024), §2.2.3–2.2.4, §3.1 | "Little to no degradation with as much as 20 % incorrect solutions at ≥256K". Filtered vs unfiltered 43.0–43.8 vs 43.6. 1K → 6.5K unique questions gives +10.5 on MATH val. Decontam: embedding top-5 + LLM paraphrase check removed 50K/569K | – | Noise tolerance (§5). Decontam design |
| Wrong answers in long-CoT SFT | 2502.07374 (Feb 2025) | All-incorrect-answer samples: only −3.2 % accuracy vs all-correct. Shuffling or deleting steps hurts a lot (qualitative). Model size not stated in abstract [UNVERIFIED size] | – | §5 |
| Shape of Thought | 2512.22255 (Dec 2025) | 1.5–9B (Qwen/Llama/Gemma). Training on traces with wrong final answers still beats human CoT (no numbers in abstract) | – | §5 |
| Noisy labels in RLVR | 2603.16140 (Mar 2026) | Truly incorrect labels: −8 to −10 % on math benchmarks. Earlier "100 % noise is fine" results were contaminated with clean examples | – | D8 |
| Reward-flip noise | 2505.22653 (May 2025) | 40 % flipped rewards: Qwen2.5-7B still 5 % → 72 % (vs 75 % clean) | – | §5 (contested by 2603.16140) |
| LESS | 2402.04333 (Feb 2024) | 5 % of the data selected by gradient similarity to a target set often beats the full set | Gradient datastore (GPU) and a *target set* | Rejected: the target would be Vi-ExamMath, which leaks the test |
| LESSER | 2610.03702 (Oct 2026) | Output-layer gradients cut feature-extraction FLOPs 9.7× (SFT) and 3.0× (RL) | Still GPU | Rejected (same target problem) |
| S2L (SmallToLarge) | 2403.07384 (Mar 2024) | 11 % of MathInstruct matches full data. +4.7 % over other selectors. Proxy model up to 40× smaller | Training a proxy (GPU) | Rejected for the core design |
| Large-scale selection study | 2503.01807 (Mar 2025) | "Many recently proposed methods fall short of random". RDS+ (embedding mean-pool) is best and cheap | – | D6 |
| Random selection at scale | 2410.09335 (Oct 2024) | "Nearly all methods struggled to significantly outperform random" (~2M pool) | – | D6 |
| s1K selection (small scale) | 2501.19393, Table 2 | At 1K, AIME24: random 36.7, diverse-only 26.7, longest 33.3, s1K **50.0**, full 59K 53.3 | Claude grading + 2 solvers | Selection matters at 1K. Our arms are 2k–20k |
| Difficulty vs budget | 2605.12906 (May 2026, ICML 2026) | "No universally optimal difficulty". The optimum shifts harder as the data budget grows | – | Do not fix one difficulty band across scale points |
| LIMO | 2502.03387 | 63.3 % AIME24 with ~1 % of the data (abstract) | – | Context only |
| Small-model learnability gap | 2502.12143 (Feb 2025) | ≤3B models learn better from shorter, simpler chains (qualitative in abstract) | – | Keep short CoT. Do not select for "hardest/longest" |
| SemDeDup | 2303.09540 (Mar 2023) | Removes 50 % of web data with minimal loss | Embeddings + k-means | Within-arm semantic dedup |
| Dedup (Lee et al.) | 2107.06499 (ACL 2022) | Memorized emission 10× lower. >4 % of validation sets overlap train | CPU | Exact + MinHash dedup |
| Gemini free tier | aistudio.google.com/docs/pricing (fetched 2026-11-07) | Free tier for gemini-3.8-flash, 3.6-flash, 3.5-flash-lite, 3.1-flash-lite, among others. RPM/RPD are not published on the docs page (shown in AI Studio only) [UNVERIFIED limits] | $0 | 3rd-family verifier for **training-side items only**. Never send test items |
| Vietnamese number locale | CLDR via babel 2.11 `[RUN]` | `vi`: decimal `,`, group `.`. `1234567.125` formats as `1.234.567,125` | – | §4.3 |

---

## 2. Verification pipeline for generated items (arms N and Y)

### 2.1 Where it sits in the factorial

The checked/unchecked contrast is itself a factor (N-checked vs N-unchecked, Y-checked vs Y-unchecked), so
the pipeline is split into two tiers:

- **Tier H (hygiene)** applies to **every** arm, including unchecked, T, T-off and S0-derived data: schema/parse,
  Vietnamese language ID, leak regex, dedup, decontamination. It does not assess mathematical validity.
- **Tier V (verification)** is what "checked" means: validity gate, answer consensus and independent
  re-formalization. Only the *-checked arms get it. The *-unchecked arms are drawn from the **same raw pool**,
  after Tier H, with equal N.

For the factor to mean the same thing in both arms, the N and Y arms run the **identical** Tier V. The only
difference is the reference answer `a_ref`: the generator's answer for N, the SymPy program's answer for Y. Y
also has the program-side gates it needs to exist at all (domain types, Z3 uniqueness; reuse
`vi_math_verified_aug/verify` T1.1). Those are part of how Y is generated, not part of the checked factor.

### 2.2 Alternatives compared

**(a) Source of the answer label**

| Option | Evidence | Cost / 10k | Verdict |
|---|---|---|---|
| Generator's own answer | Old pipeline: answer consistent 97.5 %, valid problem 38.5 % `[RUN]` | 0 | Unchecked baseline only |
| Self-consistency of one model (maj@k) | Pseudo-label accuracy 79 → 63 % `[2508.05004]`. A 37 % label accuracy is possible `[2504.16084]` | k calls | Reject as gold |
| Program execution alone (SymPy/Z3) | Certifies a number, not the problem: 54 % (GSM8K) and 27 % (MATH) usable `[R3][R4]` | CPU | Necessary for Y, not sufficient |
| **`a_ref` must agree with 2 different-family solvers (3/4 on split)** | +4.2 avg for a 4B student `[2507.23751]`. Panels of different families beat a single judge `[2404.18796]` | ~2.5 calls | **Choose** |
| Strong reasoning model (Claude Opus) labels every item | Most accurate. ~7k calls total budget | 10k Opus calls > budget | Calibration and adjudication only |

**(b) Problem-validity checker**

| Option | Evidence | Verdict |
|---|---|---|
| None (answer checks only) | 38.5 % teacher acceptance among answer-consistent items `[RUN]` | Reject |
| One LLM, direct "is this valid?" | MathQ-Verify baselines are 2–9 F1 below decomposed checks for most models `[2505.13903 T2]` | Reject |
| **Decomposed check (conditions → atomic/contradiction/completeness) by 2 families, (2,2) vote** | P 89.6 / R 62.7 `[2505.13903 T3]` | **Choose.** Re-tune k on our calibration set |
| (3,2) vote with a 3rd family (Gemini) | P 86.2 / R 76.0 | Fallback if yield is too low |

**(c) Faithfulness of the formal representation**

| Option | Evidence | Verdict |
|---|---|---|
| Back-translate → re-formalize → SMT equivalence, with repair loop | 2604.25031: 6–26 calls per item. Agreement necessary, not sufficient | Too expensive, and repair adds a confound |
| Trained aligners (FormalAlign, BEq+) | Lean only, needs training | Not applicable |
| **One independent re-formalization (SymPy program from the Vietnamese text, by a non-generator family), executed locally. Result must equal `a_ref`, and the numeric-literal multiset of program inputs must equal the literals in the text** | Cheap version of round-trip equivalence. The literal check targets the measured "bolted-on tail" failure (84 %, `[R5]`) | **Choose.** Precision/recall to be measured (§2.6). Not yet measured |

**(d) Step-level checks (PRM / critic):** rejected, see D5.

### 2.3 Model roles (all free unless stated)

| Role | Model | Why |
|---|---|---|
| Generator (N) / rewriter (Y) | `deepseek-v4.1-flash` (NIM) | Project choice `[00_SOLUTION §5.1]`. **Never** used as a verifier of its own output (self-preference) |
| Validity judge V1 | `glm-5.3-flash` (NIM) | Different family. Fast |
| Validity judge V2 / solver S1 | `nemotron-3-ultra-550b-a55b` (NIM) | Strongest available non-generator family |
| Solver S2 / re-formalizer F | `glm-5.3-flash` (NIM), F at temperature 0.2 | Different family from S1 and the generator |
| Split breakers S3, S4 | `gemini-3.6-flash` (free tier) + `gemma-4-31b-it` (NIM) | Third and fourth families. Gemini sees training-side items only. Gemma is in the same family as student Gemma-4-E2B, so it only breaks ties and never decides alone |
| Difficulty tag (not a gate) | `nemotron-3.5-bolt-e4b` pass@2 | Weak-solver signal for stratification and reporting |
| Calibration rater, test-side adjudicator, decontam paraphrase check | Claude Opus (Batch) | Test items must not go to NIM/Gemini (contamination control). ~1.7k of ~7k calls (§2.5) |
| Exact checks | SymPy 1.14, Z3 5.1 (local), typed grader (§4) | Deterministic |

Gemini is an optional tie-breaker. Without it, S3/S4 = `gemma-4-31b-it` ×2 (temperature 0.7).

### 2.4 Pseudo-code and decision rules

```python
# Item produced by the generator, JSON-schema enforced:
# {id, seed_id, arm, problem_vi, answer:{type,value,vars?,target_var?,round_dp?}, solution_vi, program? (Y only)}

def tier_H(item):                                    # every arm
    reject_if(not schema_ok(item.answer))            # typed gold must parse with the §4 grader (gold side)
    reject_if(vi_ratio(item.problem_vi) < 0.85)      # share of Vietnamese syllable tokens; no English residue
    reject_if(leak(item))                            # reuse informalize/leakage_filter.py: answer string,
                                                     # snake_case/h_123 names, SMT tokens, "đáp án là" in problem
    reject_if(count_noun_type_violation(item))       # "bao nhiêu người/cái/học sinh" and non-int or negative answer
    # dedup / decontam run corpus-wide afterwards (§3.2)

def tier_V(item):                                    # *-checked arms only; identical for N and Y
    # V1: problem validity, MathQ-Verify style, one JSON call per judge
    #     output: {conditions:[...], target:"...", leak, language_error, domain_violation, contradiction,
    #              incomplete, multiple_answers, unrealistic_vn_context, tacked_tail, verdict: valid|invalid}
    if judge(glm, item).verdict != "valid":       return REJECT("V1-glm")          # (2,2): sequential, short-circuit
    if judge(nemotron, item).verdict != "valid":  return REJECT("V1-nemotron")

    # V2: answer-label consensus; solvers never see a_ref or the solution
    a_ref = item.answer                                     # generator answer (N) or SymPy result (Y)
    s = [solve(nemotron, item.problem_vi), solve(glm, item.problem_vi)]  # short CoT, typed boxed answer
    votes = sum(grade_typed(x, a_ref) for x in s)
    if votes < 2:
        if votes == 0 and grade_typed(s[0], s[1].answer):  # both solvers agree on a different answer
            return REJECT("V2-relabel-candidate")           # never auto-relabel; logged for analysis
        s += [solve(S3, item.problem_vi), solve(S4, item.problem_vi)]
        votes = sum(grade_typed(x, a_ref) for x in s)
        if votes < 3:                                       # >= 3/4, both families represented
            return REJECT("V2-consensus")

    # V3: independent re-formalization (round-trip, 1 call)
    prog = formalize(glm_lowtemp, item.problem_vi)          # SymPy program: named inputs, typed domains, answer
    r = run_sandboxed(prog, timeout=10)                      # subprocess, no network, CPU
    if r.error or not grade_typed(r.answer, a_ref):         return REJECT("V3-exec")
    if multiset(r.input_literals) != multiset(literals_vi(item.problem_vi)) - ALLOWED_CONSTANTS:
        return REJECT("V3-literals")                         # extra number in text = bolted-on tail;
                                                             # missing number = dropped condition
    # trace/label agreement (applies to the SFT target, not the problem)
    if not grade_typed(item.solution_vi, a_ref):
        item.solution_vi = regenerate_solution_once(item)   # same generator, same prompt
        if not grade_typed(item.solution_vi, a_ref):        return REJECT("trace-mismatch")
    item.tags.difficulty = bolt_e4b_pass_at_2(item)          # reported and stratified on, never gated on
    return ACCEPT
```

Thresholds and parameters (defaults, then tuned on the calibration dev half, §2.6):

| Parameter | Default | Tuning rule |
|---|---|---|
| V1 vote | (2,2) | If V1 recall < 0.60 on dev, switch to (3,2) with Gemini as 3rd judge |
| V2 consensus | 2/2, else ≥3/4 | Tighten to 4/4 if accept-precision CI lower bound < 0.90 |
| Solver temperature | 0.6, max 1,024 output tokens | – |
| `vi_ratio` | ≥ 0.85 | Set at the 1st percentile of Vi-GSM8K-train (clean translated) items |
| `ALLOWED_CONSTANTS` | {1, 2, 7 (tuần), 12 (tháng/chục), 24, 60, 100, 1000, 10 (chục)} + numbers written in words | Extend from V3-literal rejects that humans mark as false rejects |
| Program timeout | 10 s CPU | – |

Target operating point: **accept-precision ≥ 0.95** (lower 95 % CI bound ≥ 0.90) with yield ≥ 40 % of raw items.
5 % residual label noise sits well inside the SFT tolerance in §5. RLVR uses only items whose label and validity
were confirmed this way.

### 2.5 Cost per 10k raw generated items (N or Y arm)

Pass rates are planning assumptions taken from ValiMath (60.5 % valid) and CoT-Self-Instruct yields. Replace
them with the pilot's measured rates.

| Stage | Calls | Items in → out (assumed) |
|---|---|---|
| Tier H (local) | 0 | 10,000 → 9,000 |
| V1 glm | 9,000 | → 6,300 |
| V1 nemotron | 6,300 | → 5,700 |
| V2 two solvers | 11,400 | 75 % pass directly |
| V2 split breakers (25 %) | 2,850 | → 4,600 |
| V3 re-formalization | 4,600 | → 4,200 |
| Trace regeneration (~10 %) | 420 | → ~4,100 |
| Difficulty tag (bolt-e4b ×2) | 8,200 | – |
| **Total** | **≈ 43k API calls** (≈ 4.3 per raw item, ≈ 10.5 per accepted item) | **~41 % yield** |

- **Throughput:** at 40 RPM per NIM key, 3 keys give ≈ 120 RPM, so 43k calls take ≈ 6 h wall-clock per 10k raw.
  Without the difficulty tag it is ≈ 35k calls (≈ 4.9 h).
- **Money and GPU:** $0, 0 GPU-hours. Local CPU covers SymPy execution, regex and MinHash.
- **To deliver 20k checked items per arm:** ≈ 49k raw items, ≈ 210k calls, ≈ 29 h at 120 RPM. Run N and Y in
  parallel on separate keys. Freeze model versions and log every request (EOL risk `[00 §5.1]`).
- **Claude:** calibration (§2.6) 800 items × 1 call, decontam paraphrase checks ≤ 600, eval adjudication ≤ 300.
  That is **≤ 1.7k of ~7k Opus Batch calls**.

### 2.6 Measuring verifier precision and recall on a human-labelled subset

1. **Calibration pool, per generated arm (N, Y):** draw **400 raw items** (pre-Tier-H) with
   `random.Random(20261107)`, stratified by pipeline outcome so that ≥ 200 are ACCEPT. Rejects are spread across
   rejecting stages. Keep inclusion probabilities for inverse-probability weighting.
   200 accepted items give a ±3 pp half-width at p = 0.95 (n = 203 needed) `[RUN]`.
2. **Labels per item (blind to pipeline outcome):**
   - `valid ∈ {0,1}`: complete, consistent, unique answer, realistic in a Vietnamese context, no leak, no tacked tail.
   - `answer_correct ∈ {0,1}`.
   - `teacher_accept ∈ {0,1}`.
   - Error code from `content_review/PROTOCOL.md`.
   Two Vietnamese math-competent humans label independently. Claude Opus labels with the same rubric as a third
   rater. A human adjudicates disagreements.
   Report human–human Cohen's κ and κ(Claude, adjudicated label). If κ(Claude) ≥ 0.8, Claude may replace the
   second human in later rounds.
3. **Metrics** (IPW-weighted, Wilson or bootstrap 95 % CI), with GOOD = valid ∧ answer_correct:
   - Precision = P(GOOD | ACCEPT).
   - Recall = P(ACCEPT | GOOD).
   - Per-stage reject precision = P(¬GOOD | rejected at stage s), which shows which stage throws away good items.
   - Yield.
   - The same metrics for the answer-only baseline (V2 alone) and the validity-only baseline (V1 alone), so the
     paper can show what each stage buys.
4. **Tuning protocol:** split 50/50 into dev and test. Tune only the §2.4 thresholds on dev and report on test.
   Freeze before generating at scale.
5. **Gate:** if precision on test is < 0.90 at the lower CI bound, add the Gemini judge ((3,2) → (3,3)) or
   tighten V2 to 4/4, then re-measure on a fresh 200-item sample.

---

## 3. Data selection, identical for all arms

### 3.1 Alternatives

| Option | Evidence | Fairness risk | Cost | Verdict |
|---|---|---|---|---|
| **No quality selection; seed-matched stratified random to equal N** | Random ≈ SOTA selectors at scale `[2410.09335][2503.01807]`. OpenMathInstruct-2: filtered ≈ unfiltered (43.0–43.8 vs 43.6) | None | CPU | **Choose** |
| Difficulty band from weak-solver pass rate | Optimum depends on budget `[2605.12906]`. s1-style difficulty helps at 1K `[2501.19393]` | Generated and translated arms have different difficulty profiles, so a band changes each arm differently | ~2 API calls per item | Tag and report only. Use as a stratum for analysis |
| Embedding k-center / diversity | Inconsistent: diverse-only 1K gave AIME 26.7 vs random 36.7 `[2501.19393 T2]`. TAGCOS k-center inconsistent [search snippet] | Rewards whichever arm is more spread out, which is the outcome being measured | CPU | Reject for core. Optional ablation |
| LLM quality / response-length scorer | +4 % over random for math `[2506.04178 §4.3]` | Scorer preferences interact with arm style | 1 call per item | Reject for core |
| Gradient influence (LESS, LESSER, S2L) | 5 % beats full `[2402.04333]`. 9.7× cheaper features `[2610.03702]`. 11 % matches full `[2403.07384]` | Needs a target set. The natural target (Vi-ExamMath) is the test, so selecting toward it is leakage | GPU-hours | Reject |
| Correctness-based filtering inside "selection" | – | Duplicates the checked factor and breaks the unchecked arms | – | Forbidden |

**Argument for not selecting.** The research question compares data *sources*. Any learned selector selects
*for* a source style, so it becomes part of the treatment. At the 2k–20k scale the expected gain is a few points
`[2506.04178][2501.19393]`, which is the same order as the between-arm effects we want to detect. Large studies
find that random is rarely beaten `[2503.01807][2410.09335]`. Dedup, decontamination and composition matching
remove the confounds that do matter: duplicates, leakage, and topic mix.

### 3.2 Procedure (applied in the same order, with the same thresholds, to S0, T, T-off, N±, Y±, M)

1. **Normalise** text: Unicode NFC, lowercase, collapse whitespace, strip LaTeX delimiters. Produce a
   `numbers_masked` variant with every number replaced by `0`.
2. **Exact dedup:** SHA-1 of the normalised text.
3. **Near-dup (MinHash-LSH, within arm):** character 5-gram shingles, 128 permutations, Jaccard ≥ 0.8 → keep one.
   The 0.8 threshold is the common default [UNVERIFIED as a paper-specified constant]. Use `datasketch` on CPU.
4. **Template cap:** cluster on `numbers_masked` at Jaccard ≥ 0.9. Keep at most `c` items per cluster, where `c` is
   the per-seed cap below. This keeps legitimate Y-arm number mutations but stops 20 copies of one story.
5. **Semantic dedup (within arm):** `intfloat/multilingual-e5-base` embeddings [model choice UNVERIFIED for
   Vietnamese math], k-means with k = N/50, then within each cluster drop pairs with cosine ≥ 0.95 (SemDeDup
   `[2303.09540]`). On the laptop GPU or CPU, 20k items take minutes. 0 Modal hours.
6. **Decontaminate against every eval set** (Vi-GSM8K-Platinum, Vi-MATH test, Vi-ExamMath, PolyMath-vi,
   BenchMAX-vi):
   - Word 13-gram overlap: any hit → drop.
   - Embedding top-5 neighbours with cosine ≥ 0.85 → Claude paraphrase check in both orders
     (OpenMathInstruct-2 procedure `[2410.01560 §3.1]`). Claude, because eval items must not go to NIM/Gemini.
   - Implement in `eval/decontam.py` (currently a stub).
7. **Seed-matched stratified sampling to N** ∈ {2k, 5k, 10k, 20k}:
   - Strata = (source ∈ {GSM8K, MATH-alg, MATH-prealg}) × (MATH level or GSM8K step-count tercile).
   - Pick a nested seed subset `S_2k ⊂ S_5k ⊂ S_10k ⊂ S_20k` once, with stratum proportions equal to the seed pool.
   - For each arm, take ≤ `c = ceil(N/|S_N|)` items per seed, uniformly at random among that seed's surviving
     items.
   - If an arm cannot supply a seed's quota, draw a replacement seed from the same stratum. Log the shortfall
     rather than padding from other strata.
   - Every arm then covers the same seeds, topics and levels at each N.
8. **Report, do not equalise:** per arm, show the weak-solver difficulty distribution, length distribution,
   embedding spread and tokens/item. Item difficulty is a property of the arm.
   - Optional sensitivity analysis: refit the main model with difficulty-stratum weights.
   - Optional ablation (only if compute remains): on the best arm at N = 5k, compare "k-center within the middle
     difficulty tercile" against random, to show selection is not what drives the ranking.

---

## 4. Grader for Vietnamese answers

### 4.1 Alternatives

| Option | Measured on my 60 targeted cases `[RUN]` | Problems | Verdict |
|---|---|---|---|
| Current `eval/math_verify.py` | **39/60.** 6 false positives (`125≡0.125`, `(1,2)≡1.2`, `8≡(x=7;y=8)`, `3≡{2;3}`, `x=3≡{2;3}`) | Gold is extracted, and separators collapse without context | Re-architect |
| Raw HF math-verify 0.9.0 | **38/60.** 2 false positives (`2,5` vs set {2;5}, duplicated root `2;3;3` vs {2;3}) | Reads VN `1.000.000` as 1.0 and `2,5` as the set {2,5}. Drops `x=2 hoặc x=3` | Use as backend only |
| LLM judge as primary | 94 % right on disagreements `[2603.18886]`. Hackable `[2505.22203]` | Cost, non-determinism, test items cannot go to NIM/Gemini | Adjudicator only |
| CompassVerifier / xVerify | 80–91 % acc on VerifierBench | GPU. Vietnamese untested | No |
| **Typed gold + VN pre-normaliser + structure-first parsing + math-verify atoms + exact rationals** | **60/60** (prototype; these cases were written with the design, so this is not a generalisation estimate) | Must be validated on real outputs (§4.7) | **Choose** |

### 4.2 Typed gold schema (stored, never extracted)

```json
{"type": "int|rational|decimal|radical|expr|tuple|set|interval|ineq|choice",
 "value": "canonical LaTeX, '.' decimal, no digit grouping, ';' between container items",
 "vars": ["x","y"],          // tuple: names if the answer is a named system solution
 "target_var": "x",          // scalar: which variable the question asks for (if the solution names several)
 "round_dp": 2,              // only if the problem says "làm tròn đến hàng phần trăm"
 "percent": true,            // answer is P%; accept P or P% (not P/100 unless the problem asks for a ratio)
 "locale": "vi|en|mixed"}    // Vi-ExamMath = vi; translated sets = mixed
```

Examples:
- `{"type":"set","value":"2;3"}` for "x = 2 hoặc x = 3".
- `{"type":"tuple","value":"7;8","vars":["x","y"]}`.
- `{"type":"interval","value":"[1,3)"}`.
- `{"type":"radical","value":"\\frac{\\sqrt{3}}{2}"}`.
- `{"type":"decimal","value":"0.125"}`.

Migration: convert existing gold strings once with a script plus human review of every item whose type is not
int. Store `grader_version` with every score.

### 4.3 Locale-aware number parsing (prediction side only)

Rules for one numeric token. Separators are `.`, `,`, space and `\,`. `{,}` means a decimal comma.

1. Both `.` and `,` present → the **last** one is the decimal mark, the other is grouping (`1.234,5` and `1,234.5`
   both = 1234.5).
2. The same separator two or more times, with groups of 3 → grouping (`1.000.000`, `1,000,000`).
3. Space or `\,` between 3-digit groups → grouping (`1 000 000`).
4. A single separator followed by ≠ 3 digits, **or with a leading group of `0`** → decimal (`2,5`, `0.125`, `0,125`).
5. A single separator followed by exactly 3 digits (`1,500`, `3.500`) is **ambiguous**. It is resolved by the
   response's own convention: count unambiguous tokens in the whole response (`2,5` and `1.000.000` vote vi;
   `2.5` and `1,000,000` vote en). On a tie, fall back to `gold.locale` (vi: comma = decimal, dot = grouping, per
   CLDR `[RUN]`). Log `flag=ambiguous-locale`. If more than 1 % of graded items carry this flag, adjudicate them.
6. Integers compare exactly (big ints too). Decimals and rationals compare as exact `Fraction`s, so `0.333 ≠ 1/3`.
   Tolerance applies only with `round_dp`, as |pred − gold| ≤ ½·10^−dp.

### 4.4 Extraction rules (prediction only)

1. Remove `<think>…</think>`.
2. Take the last `\boxed{…}` (brace-balanced).
   - If the gold arity is > 1 and the last paragraph has several boxes, join them (`x=\boxed{2}` hoặc
     `x=\boxed{3}`).
   - If the gold is scalar and the last paragraph has ≥ 2 *different* boxes → **hedge → wrong**.
3. Otherwise use the last cue (`Đáp số`, `Đáp án (cuối cùng)`, `Kết luận`, `Vậy`, `The answer is`). Take the rest
   of that line, cut at a sentence-final dot that is not a decimal dot.
4. Otherwise use the last number or fraction `a/b`.
5. **Structure before numbers:**
   - A parenthesised `(a,b)` with no `;` is a pair, decided *before* locale parsing. So `(1,2)` ≠ 1.2.
   - Container separators: `;`, `hoặc`, `và`, `\vee`, and `,` followed by a variable name.
   - Sets accept `S = \{…\}`, `x_1 = …; x_2 = …`, `x ∈ \{…\}`.
6. **Arity must match** (`8` vs (7;8) → wrong; `2;3;3` vs {2;3} → wrong). Sets match as multisets. Tuples are
   ordered, unless both sides carry variable names, in which case they are matched by name.
7. **Scalar with several `=`:** if `target_var` is set, take that variable's RHS (`Vậy x=4 thì A=2` → x=4).
   Otherwise take the last RHS.
8. **Interval and inequality:** VN notation `[1;3)`, `(2;+∞)` and French-style `]1;3[` → `(1,3)`. Compare with
   math-verify `allow_set_relation_comp=True`, so `x ≥ 2` ≡ `[2,+∞)`. Openness must match.
9. Strip units and words (`đồng`, `nghìn đồng`, `km`, `cm`, `giờ`, `tuổi`, `học sinh`, `\text{…}`) after
   extraction.

### 4.5 Equivalence backend

- Numbers: exact `Fraction`.
- Everything else (radicals, π, expressions, intervals): `math_verify.verify(parse("$gold$"), parse("$pred$"),
  float_rounding=10)`, gold first.
- On Windows pass `parsing_timeout=None, timeout_seconds=None` and wrap the call in our own subprocess
  timeout. Run batch grading on Linux.
- Pin `math-verify==0.9.0` and `latex2sympy2_extended==1.11.0` `[04 §4.1]`.

### 4.6 Test cases that must pass

The prototype `grader_proto/vi_grader.py` passes all 60. The table also gives the old scorer's result.

| ID | Prediction | Gold (typed) | Expect | Old scorer returns |
|---|---|---|---|---|
| B1a/b | `125`, `\boxed{125}` | decimal 0.125 | ✗ | **✓ (FP)** |
| B1c | `3500` | decimal 3.5 | ✗ | ✗ |
| B1d | `Đáp số: 3.500 đồng` | int 3500 | ✓ | ✗ |
| B1e | `\boxed{0,125}` | decimal 0.125 | ✓ | ✓ |
| TUP1 | `1.2` | tuple (1;2) | ✗ | ✗ |
| TUP2 | `\boxed{(1,2)}` | decimal 1.2 | ✗ | **✓ (FP)** |
| TUP3/4 | `(1;2)`, `(1,2)` | tuple (1;2) | ✓ | ✓/✗ |
| TUP5 | `(2;1)` | tuple (1;2) | ✗ | ✗ |
| TUP6/7 | `x=7, y=8`; `y = 8; x = 7` | tuple (7;8) vars x,y | ✓ | ✓/✗ |
| B2a | `8` | tuple (7;8) | ✗ | **✓ (FP)** |
| B2b/c | `\boxed{3}`, `\boxed{x=3}` | set {2;3} | ✗ | **✓ (FP)** |
| B3a/b | `x = 2 hoặc x = 3` (either order) | set {2;3} | ✓ | ✓/✗ |
| B3c | `x_1 = 2; x_2 = 3` | set {2;3} | ✓ | ✓ |
| B3d | `S=\{2;3\}` | set {2;3} | ✓ | ✓ |
| B3e | `x=\boxed{2}` hoặc `x=\boxed{3}` | set {2;3} | ✓ | ✓ |
| B3f | `\boxed{2; 3; 3}` | set {2;3} | ✗ | ✗ (math-verify **✓ FP**) |
| B4a/b | `\boxed{1{,}5}`, `\boxed{2,5}` | decimal 1.5 / 2.5 | ✓ | ✗/✓ |
| B4c | `\boxed{2,5}` | set {2;5} | ✗ | ✗ (math-verify **✓ FP**) |
| B5a | `\frac{\sqrt{3}}{2}` | radical `\frac{\sqrt 3}{2}` | ✓ | ✓ |
| B5b | `\{-1; 2\}` | set {2;-1} | ✓ | ✗ |
| B5c/d | `\dfrac{1}{\sqrt2}`, `2\sqrt2` | `\frac{\sqrt2}{2}`, `\sqrt8` | ✓ | ✗/✓ |
| B6a/b | `Vậy x=4 thì A=2` | int 4 target x / int 2 target A | ✓ | ✗/✓ |
| B6c | `1,500` (vi default) | decimal 1.5 | ✓ | ✗ |
| B6d | `…1,500 dollars` | int 1500, locale en | ✓ | ✓ |
| B6e | `2,5 kg và 1,500 kg` | decimal 1.5 (doc votes vi) | ✓ | ✗ |
| B6f | `2.5 and 1,500` | int 1500 (doc votes en) | ✓ | ✓ |
| LOC1–3 | `1.000.000`, `885.000 đồng`, `1\,000\,000` | int | ✓ | ✓ (math-verify ✗) |
| LOC4/5 | `1.234,5`, `1,234.5` | decimal 1234.5 | ✓ | ✓ |
| NUM1–3 | `1/3` ✓, `0.333` ✗, `\frac26` ✓ | rational 1/3 | ✓/✗/✓ | ✓/✗/✓ |
| NUM4 | `0,5` | rational 1/2 | ✓ | ✓ |
| NUM5–7 | `-3` vs 3, `12` vs 120, 20-digit off-by-one | int | ✗ | ✗ |
| RND1/2 | `1{,}41` / `1{,}42` | radical √2, round_dp 2 | ✓/✗ | ✗/✗ |
| PCT1 | `20\%` | int 20, percent | ✓ | ✓ |
| INT1/2 | `[1;3)` / `[1;3]` | interval [1,3) | ✓/✗ | ✗/✗ here. With gold written `[1;3]`, the old scorer returns **True for `[1;3)` (FP via last-number)** `[probe_scorer.out]` |
| INT3 | `x \ge 2` | interval [2,∞) | ✓ | ✗ (with gold `[2;+\infty)` it returns True, but only because both sides' last number is 2) |
| INT4 | `(2;+\infty)` | interval [2,∞) | ✗ | ✗ |
| INT5 | `]1;3[` | interval (1,3) | ✓ | ✗ |
| HED1/2 | `\boxed{2} hoặc \boxed{3}`; `Đáp án: 2 hoặc 3` | int 2 | ✗ | ✗ |
| THK1 | `<think>…\boxed{5}</think> Vậy đáp số là 7.` | int 7 | ✓ | ✗ |
| MCQ1 | `Đáp án: C` | choice C | ✓ | ✓ |
| NONE | `Em không biết.` | int 3 | ✗ | ✗ |

Still to add (cases I did not write): `12 giờ 30 phút` vs 12.5 (mixed units) → decide policy, ✗ by default.
Degrees `30^\circ` vs `\frac{\pi}{6}` → ✗ unless gold says angle. `m > 2` vs `m \in (2;+\infty)` → ✓.
Matrices and vectors → out of scope for Vi-ExamMath Core.

### 4.7 Validating the grader itself

- Grow the suite to ≥ 150 cases, including ≥ 50 **real** outputs from Qwen3/3.5 students on Vi-ExamMath and
  Vi-GSM8K `[04 §4.2]`.
- Then score 300 real outputs, stratified by (answer type × grader verdict), with 2 humans. Report grader FP and
  FN rates with CIs in the paper.
- **Eval-time adjudication:**
  - Items where extraction fails or the flag is `ambiguous-locale` / `hedged` go to Claude Opus with gold +
    prediction (reference-guided, binary). Test items go to Claude only, never NIM/Gemini.
  - Report both "rule-only" and "rule + adjudication" accuracy. The primary metric is rule-only, so it is
    deterministic and identical across arms.
- **Port:** move the prototype into `vi_math_verified_aug/eval/math_verify.py`, keeping its public API (`is_correct`
  gains a `gold: dict` overload). Turn §4.6 into `tests/test_math_verify.py`.

---

## 5. How much label noise small-model training tolerates

| Setting | Number | Source |
|---|---|---|
| SFT, wrong-answer solutions mixed in | "Little to no degradation" up to **20 %** at ≥256K examples. At 40 % for incorrect pairing. Small-data behaviour is in their Fig. 5, which I did not read numerically | 2410.01560 §2.2.3 |
| SFT, long CoT, all answers wrong | **−3.2 %** accuracy vs all-correct. Structure perturbations hurt much more | 2502.07374 |
| SFT, 1.5–9B, wrong-final-answer traces from a stronger model | Still beat human-written CoT (no figures in abstract) | 2512.22255 |
| SFT trace verification | GPT verification filter: math 41.9 → 40.0 avg (no gain) | 2506.04178 |
| RLVR, 40 % flipped rewards (7B) | 72 % vs 75 % clean | 2505.22653 |
| RLVR, truly incorrect labels | **−8 to −10 %** on math benchmarks. Earlier robustness claims were contaminated | 2603.16140 |
| RLVR false negatives from the verifier | ~20 % of rollouts. TinyV recovers up to +10 % pass rate | 2505.14625 |

Implications:
1. For **SFT**, a 5 % residual noise target (§2.4) is safely inside the tolerated range. The checked-vs-unchecked
   SFT effect may be small; OpenThoughts and OpenMathInstruct-2 predict a near-null. So the factorial must be
   powered for small effects (3 seeds, mixed-effects model per `00 §2.3`), and a null is a reportable result.
   The old unchecked augmented data was invalid at a 61.5 % rate (problem validity, not answer correctness), far
   outside any measured tolerance. That suggests validity, not answer correctness, is the effect to expect.
2. For **RLVR (C4)**, label noise is costly. Use only Tier-V items and a deterministic rule reward (D8).

---

## 6. Risks and open items

- **Judges' Vietnamese ability is unmeasured.** The V1/V2 precision numbers come from English data. The
  calibration set (§2.6) is the only valid evidence; do not quote 2505.13903 numbers as ours.
- **V3 literal check false rejects:** numbers written as words (`hai`, `một nửa`), implicit constants and unit
  conversions. Estimate the rate on the calibration set and extend `ALLOWED_CONSTANTS`.
- **Gemini free-tier limits and licence for verification use:** unverified, so it is optional.
- **Gemma-4-31B as a tie-breaker** shares a family with the Gemma-4-E2B student. It is never the sole decider.
  Report the share of items it decided.
- **EOL risk** for NIM models: freeze the model IDs and request logs for each generation and verification window.

---

## 7. Evidence index

Commands run (outputs in the session scratchpad
`C:\Users\tranh\AppData\Local\Temp\claude\E--MATH-REASONING-THESIS-vi-math-smt\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad\`):

- `probe_scorer.py` / `probe_scorer.out`: current project scorer on 30 probes, 13 wrong (`125≡0.125`, `3500≡3.500`,
  `(1,2)≡1.2`, `8≡x=7;y=8`, `3≡2;3`, `[1;3)≡[1;3]`, `{,}` decimal, `0,5≢\frac12`, …).
- `grader_proto/vi_grader.py`, `grader_proto/test_cases.py`, `grader_proto/test_out.txt`: prototype, 60/60.
  `old_vs_cases.py`: old scorer 39/60. `mv_vs_cases.py`: raw math-verify 0.9.0 38/60.
  These use the math-verify venv `data/mvenv` from the 04 probe.
- Recount of `content_review/annotations/aug.jsonl` (n = 250): augmented NL↔SMT consistent 195/200, teacher
  accepts 75/195 = 38.5 % [31.9–45.5], would-write 21/195 = 10.8 % [7.2–15.9]. Original items: teacher accepts
  49/50.
- babel 2.11 CLDR: `vi` decimal `,`, group `.`.
- Sample-size computation: n = 203 for ±3 pp at p = 0.95.

URLs opened 2026-11-07:
- arxiv.org/abs or /html: 2505.14625, 2504.10481, 2508.03686, 2505.22203(v3), 2506.04178(v2), 2507.23751(v2),
  2508.05004, 2505.22653, 2502.07374, 2507.03133, 2504.06514, 2406.07222, 2410.10135, 2505.13903, 2604.25031,
  2503.01807, 2403.07384, 2410.09335, 2501.07301(v2), 2303.09540, 2502.03387, 2402.04333, 2410.01560(v2),
  2404.18796, 2501.19393(v3), 2502.12143, 2605.12906, 2610.03702, 2603.16140, 2512.22255, 2603.18886,
  2504.16084, 2502.17387, 2210.11610, 2107.06499.
- github.com/huggingface/Math-Verify.
- aistudio.google.com/docs/pricing and /docs/rate-limits.
- Search-snippet only (not opened): 2410.18921 (FaultyMath), TAGCOS 2407.15235, QaDS 2404.01067, 3DS 2410.10901.
