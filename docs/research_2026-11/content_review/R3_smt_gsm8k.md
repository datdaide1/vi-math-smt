# R3 — SMT content review: GSM8K formalizations (`smt_gsm8k`)

**File:** `data/formalize_output/gsm8k/gsm8k-smt-nvidia-FULL.jsonl` (7,473 rows, all `z3_score = 1`), produced by
`src/formalize/gsm8k/formalize_gsm8k_nvidia_langgraph.py` (gpt-oss-120b via NIM, QF_LRA / `Real` only, ReAct loop that
validates against the gold answer and retries up to 3x). Gold answers come from `data/generate/gsm8k/train.jsonl`, aligned by `index`. All 7,473 questions match exactly.
**Reviewer date:** 2026-11-07 · **Protocol:** `PROTOCOL.md` · **Annotations:** `annotations/smt_gsm8k.jsonl` (150 rows)
**Reproduce:** `tools/smt_gsm8k/` (`analyze.py` runs Z3 on the full corpus, `mutlevel.py` scores mutation support, `ann1.py` + `mutman.py` hold the hand annotations, `stats.py` computes all numbers below, `show.py` prints the sample for reading).

## 1. Method

- **Sample.** n = 150. Drawn uniformly with `sorted(random.Random(20261107).sample(range(7473), 150))`:
  `21 32 66 67 77 94 184 217 280 393 509 520 529 537 584 595 614 618 656 826 830 871 1059 1069 1097 1112 1227 1232 1251 1320 1371 1428 1531 1568 1669 1687 1746 1753 1782 1811 1871 1890 1901 1932 1943 1997 2038 2110 2137 2155 2316 2332 2362 2395 2482 2551 2558 2578 2597 2713 2749 2859 2863 2944 3058 3115 3281 3370 3460 3461 3490 3493 3530 3533 3601 3616 3652 3661 3708 3763 3779 3817 3964 4211 4228 4231 4245 4252 4327 4509 4564 4625 4638 4642 4681 4686 4763 4836 4884 4890 4928 4980 5019 5038 5043 5044 5113 5131 5143 5155 5318 5333 5344 5349 5367 5413 5478 5524 5617 5626 5636 5713 5716 5766 5768 5864 5890 5967 5994 6088 6108 6208 6232 6290 6408 6449 6527 6605 6616 6623 6631 6648 6674 6699 6765 6792 6864 6922 7223 7288`
- **Reading.** Every sampled program was read assert by assert next to its English question. For each one I checked whether every constraint corresponds to a stated fact, whether the answer is derived or written in as a literal, and whether quantities were folded by hand.
- **Execution.** Z3 5.1.0 via `.venv` with a 10 s timeout (maximum observed: 0.12 s). I compared the value to gold (tolerance 1e-4). The uniqueness check drops every `(assert (= answer <num>))` and asks whether `answer` is still forced. I also ran the repo's `verify/uniqueness.classify_verbose`.
- **Full corpus.** All 7,473 programs were executed, which took about 16 s.
- **New codes added** (allowed by the protocol):
  - `X-PRECOMP`: a story fact was combined by hand into a literal that appears nowhere in the problem (e.g. `total_pages 1200` = 3×400). `minor` when the folded step is trivial. `major` when the folded literal carries the main reasoning step.
  - `X-SORT`: the problem needs integer semantics (`mod`, a minimum integer, floor), which `Real`/QF_LRA cannot state.
- **Deriving** levels:
  - `full`: every step from the stated facts is done by Z3.
  - `partial`: some intermediate was computed by hand.
  - `none`: the answer, or the only non-trivial step, is a hand-written literal.

## 2. Headline: how many programs are faithful, deriving certificates?

| Definition (sample, n=150) | k | rate [Wilson 95% CI] |
|---|---:|---|
| Faithful (no wrong or invented constraint) | 134 | **89.3%** [83.4, 93.3] |
| Faithful + deriving, *any* form incl. hand-folded intermediates | 131 | 87.3% [81.1, 91.7] |
| Faithful + **fully** deriving (incl. closed-form one-liners) | 98 | 65.3% [57.4, 72.5] |
| **Faithful + fully deriving + non-trivial (named quantities)**: usable as a symbolic certificate | **81** | **54.0% [46.0, 61.8]** |
| … of which actually require *solving* (not a straight-line assignment chain) | 17 | **11.3%** [7.2, 17.4] |

**Full-corpus proxy.** The proxy is: classifier says `derived`, the value matches gold, the program is not a closed form, and it has no unexplained literal. It covers **4,879 / 7,473 = 65.3% [64.2, 66.4]**. Against my reading of the sample, the proxy flags 89 items vs my 81 (79 overlap), so it over-counts by about 10%. That puts the corrected corpus estimate at **≈ 58–60%** faithful-deriving-non-trivial. Only **1,029 = 13.8%** of the corpus is both proxy-good *and* needs actual solving.

## 3. Error rates per code (sample, n = 150)

| code | k | rate [Wilson 95% CI] | severity |
|---|---:|---|---|
| `X-PRECOMP` (hand-folded intermediate) | 35 | 23.3% [17.3, 30.7] | 29 minor / 6 major |
| `SMT-TRIVIAL` (single closed-form expression) | 29 | 19.3% [13.8, 26.4] | minor (2 also major via precomp) |
| `SMT-HARDCODE` (answer written as literal) | 16 | **10.7%** [6.7, 16.6] | critical |
| `SMT-NONUNIQUE` after dropping the answer literal | 15 | 10.0% [6.2, 15.8] | all 15 are the hard-codes (answer becomes unconstrained) |
| `X-SORT` (needs Int / mod / min-integer) | 2 | 1.3% [0.4, 4.7] | critical (both also hard-coded) |
| `SMT-MISMATCH` | 1 | 0.7% [0.1, 3.7] | major |
| `SMT-ERROR` | 0 | 0.0% [0.0, 2.5] | – |
| `SMT-UNFAITHFUL` (wrong relation/number, non-hardcode) | 0 | 0.0% [0.0, 2.5] | – |
| any code | 69 | 46.0% [38.2, 54.0] | |

- **Usable as-is** (no code, or minor only): **125 / 150 = 83.3% [76.6, 88.4]**
- **Fixable** (major): 9 = 6.0% [3.2, 11.0]
- **Discard the program** (critical, all hard-codes): 16 = **10.7%** [6.7, 16.6]. The problem itself is fine; only the SMT has to be redone.
- **Sorts.** 150/150 use `Real`. In the corpus, `Int` appears in 6 programs (0.1%).

Z3 reproduces gold on 149/150 sampled programs. This is **not evidence of faithfulness**: matching gold was the pipeline's acceptance criterion, so the selection guarantees it. Faithfulness had to be established by reading. In the sample, every unfaithful program is a hard-code. I found no "right answer, wrong model" program among the non-hardcoded ones.

## 4. Full-corpus sizing (all 7,473, Z3 executed)

| pattern | k | rate [Wilson 95% CI] |
|---|---:|---|
| `SMT-ERROR` (parse error / not SMT) | 22 | 0.3% [0.2, 0.4] |
| `SMT-MISMATCH` (sat, value ≠ gold) | 84 | 1.1% [0.9, 1.4] |
| ⇒ stored `z3_score = 1` **not reproducible** | 106 | 1.4% [1.2, 1.7] |
| `SMT-HARDCODE` (classifier) | 565 | **7.6%** [7.0, 8.2] |
|   … of which the whole program is `(assert (= answer N))` | 524 | 7.0% [6.5, 7.6] |
| non-unique after dropping answer literal | 564 | 7.5% [7.0, 8.2] |
|   … excluding hard-codes | 9 | 0.1% [0.1, 0.2] |
| `SMT-TRIVIAL` closed-form one-liner (non-hardcoded) | 1,363 | 18.2% [17.4, 19.1] |
| unexplained literal (`X-PRECOMP` proxy) | 1,188 | 15.9% [15.1, 16.7] |
| straight-line assignment program (each var defined once from earlier ones) | 6,065 | **81.2%** [80.3, 82.0] |
| needs real solving (implicit equation / inequality) | 1,386 | 18.5% [17.7, 19.4] |

**How well the heuristics match my reading:**
- Hardcode classifier: 16/16 agreement.
- Closed-form flag: 28/29 agreement.
- `X-PRECOMP` proxy: weak, catching 20 of my 35. Its corpus figure should be read as a lower bound of ≈ 15–25%.

**Hard-codes are not caused by gold leaking through retries.** The retry message does contain `expected {gt}` (`_validate_with_z3`), but the hard-code rate is 7.7% at 0 retries, 6.5% at 1, 9.3% at 2 and 6.8% at 3 or more. The model computes the answer in its "brief reasoning" and writes the literal.

**Mismatches (read 20 of 84).**
- 18 are SMT modelling errors. Examples: 1024 ignores the Monday/Tuesday hours; 131 drops "20 pages of chapter 11"; 220 ignores "back and forth"; 2549 counts horseshoes instead of horses; 5218 and 1868 hard-code a *wrong* literal.
- 1 is a GSM8K gold error: 356, where the solution text computes 64 but `#### 66`. The SMT's 64 is correct.
- 1 is a scale or interpretation disagreement: 618, where the SMT gives 0.2 and gold is 15 (%). The SMT's 20% reading is arguably more defensible than gold's.

**Pipeline bookkeeping is unreliable.**
- Every row says `z3_score=1, z3_error="NONE"`, including 2620, whose `smt_lib` is the string `ERROR: empty_response`, and 21 unparseable programs.
- Likely cause: the stored `smt_lib` is the last `tool_convert` output, while `status` is whatever the orchestrating LLM passed to `tool_validate`. The LLM copies the code into the tool call itself and may edit it on the way.
- 21 rows lack `retries`; 2 carry `validation_error`.
- **Never trust `z3_score` in this file; re-execute.**

## 5. Worst examples (quoted)

1. **2944** (HARDCODE). "Jane runs 3 kilometers in two hours… speed in meters per minute?" → the whole program is `(assert (= answer 25.0))`. The unit conversion, which is the entire problem, is skipped.
2. **217** (HARDCODE). The age-relation puzzle (Anne = 2×Emile, Emile = 6×Maude, Maude 8) → `(assert (= answer 96.0))`.
3. **529** (HARDCODE + X-SORT). "Minimum number of rides" → `(assert (> rides 25.0)) (assert (= rides 26.0))`. The integer-minimum step cannot be expressed in QF_LRA, so 26 is pinned by hand.
4. **4625** (HARDCODE + X-SORT). 47 carrots shared among 4 goats, leftover? → `(assert (= answer 3.0))`. This needs `mod`.
5. **1901** (PRECOMP, deriving none). `(= part_distance 50.0) … (= time2 3.0)`, `answer = 1+1+3`. The speed and remaining-time reasoning is done by hand, and `D` and `part_distance` are dead.
6. **4890** (TRIVIAL + PRECOMP). `(= answer (+ 1728 1200))`. Here 1200 = 2 tanks × 20 gal × 30 mpg, so Z3 only adds.
7. **537** (PRECOMP major). `(* (- num_total num_inexper) 2880.0)`. Here 2880 = $10 × 6/5 × 60 h × 4 weeks, all outside the solver.
8. **3533** (PRECOMP major). "Area of square A is 25 …" → `(= sideA 5.0) (= sideB 9.0)`. The square roots are taken by hand and the area facts are absent.
9. **6232** (PRECOMP). Solves `pg` from `pb = 2·pg` and then writes `(+ (* 40 50) (* g 25))`, re-typing 25 as a literal. The solved variable is dead.
10. **618** (MISMATCH). `answer = 0.4·0.25 + 0.2·0.5 = 0.2` vs gold `15`. It is flagged `z3_score=1` anyway.

## 6. Extra lens: can these programs support **story-level** mutation?

Levels (per program):
- **L3 story-mutable**: every story quantity is its own variable, or sits in an entity-defining relation (`donna = 2·maisie + 5`). Numbers can be changed, a step added or removed, or the question retargeted while the story stays coherent.
- **L2 partial**: some quantities are named, but others are inline in the answer expression or folded by hand (`quarters 9.0` = 4 + 5), so they can't be varied independently.
- **L1 answer-expression only**: a closed form or major precompute. The only thing to perturb is an arithmetic expression of opaque literals.
- **L0**: hard-coded literal, nothing to mutate.

| level (sample, hand-labelled) | k | rate [Wilson 95% CI] | full-corpus heuristic* |
|---|---:|---|---|
| **L3 story-mutable** | 63 | **42.0%** [34.4, 50.0] | 53.8% (≈ 50% after calibration) |
|   L3 **and** entity-named variables (≥ 50% of names ≥ 3 chars, e.g. `pages_per_day` not `x`) | 48 | 32.0% [25.1, 39.8] | – |
|   L3 **and** named **and** has a named intermediate (question target can be moved) | 39 | **26.0%** [19.6, 33.6] | 29.0% |
| L2 partial | 38 | 25.3% [19.0, 32.8] | 16.0% |
| L1 answer-expression only | 33 | 22.0% [16.1, 29.3] | 22.3% |
| L0 hard-coded | 16 | 10.7% [6.7, 16.6] | 7.6% |
| **L1 + L0: nothing but the answer to perturb** | 49 | **32.7%** [25.7, 40.5] | 29.9% |

\*`tools/smt_gsm8k/mutlevel.py`. It agrees with my labels on 133/150. Its errors: it over-calls L3 on 9 items (hand-folded literals and value collisions) and under-calls on 4 items whose numbers are written as words ("four", "twelve").

**Root cause of the bolted-on tails ("cho anh 1 cái và cộng thêm 0,5").** It is in the mutation engine *and* the data, in that order:

1. **The engine only ever touches `answer`.** `src/mutation_informalize/mutation_engine.py` has three operators:
   - `mutate_structure`: `answer' = answer + 2·h_k` with `h_k = uniform(0.5, 5.0)`.
   - `mutate_expression`: wraps `answer` in `(* (+ answer k) k)` and similar.
   - `mutate_difficulty`: `answer' = answer·(answer+1)`.

   None of them reads or changes the pinned story constants, even in the ~42–54% of programs where those constants exist as named, entity-tied variables. `rand_const()` draws 2-decimal floats, which explains the "0,5" tails. The fresh `h_*`/`d_*` names have no entity, so the informalizer can only render them as "add a number".
2. **The data caps what any engine could do.** About one third of programs (L1 + L0: 32.7% sample, 29.9% corpus) contain *only* an answer expression or a literal. A further quarter (L2) has quantities inlined or hand-folded. Even a story-aware mutator could do number-swap / add-step / retarget on only ~42–54% of seeds, and on only ~26–29% would it have entity names and an intermediate to retarget to.

## 7. Is "SMT-verified" meaningful at GSM8K level?

Mostly not, in this file's sense.
- **81% of programs are straight-line assignment chains.** Z3 acts as a calculator evaluating `let`-bindings, and another 18% are one-line closed forms. Only ~14–19% contain an implicit equation that Z3 actually has to solve (e.g. 393 `length = 2w, 2l + 2w = 30`; 6616 `D + I = 50, D + 2 = I − 2`).
- **The "verified" property was gold-matching.** The loop retried until Z3's value equalled the gold answer, so it certifies *consistency with the gold*, not that the program models the text. A hard-coded `(= answer 25.0)` passes identically.
- **The Real-only (QF_LRA) restriction pushes the integer-flavoured parts of GSM8K out of the solver.** `mod`, floors, the minimum integer and integrality of counts become hard-codes or hand precomputes.
- **Where the property does mean something:**
  1. As a **uniqueness certificate**. Non-hardcoded programs are essentially all unique (only 9 non-unique in the corpus).
  2. As a **structural scaffold for mutation**, when quantities are named (the L3 subset).

The value lies in the named-variable structure, not in the solver call.

## 8. Verdict: **FIX** (keep the majority; re-formalize about 1/3)

- **Keep as seeds (≈ 50–55%).** These are the L3, faithful, fully-deriving programs that match gold. Filter them with `classify == derived` ∧ re-executed match ∧ ¬closed-form ∧ ¬unexplained-literal. That leaves about 4.9k, or about 4.5k after the ~10% over-count correction.
- **Re-formalize (≈ 30%, free NIM budget).**
  - The 565 hard-codes and 106 error/mismatch rows (9.0% together).
  - The 1,363 closed-form one-liners (18.2%).
  - The `X-PRECOMP` major cases.

  Prompt for one pinned, entity-named variable per stated quantity and forbid answer literals. Run the uniqueness gate *without* gold in the generation loop: compare to gold only afterwards, and never echo `expected X` back into the prompt. Allow QF_LIA / `Int` (or QF_NIA) for the count, mod and min-integer problems. Cost is about 2.3k calls, roughly 5 h at 8 RPM.
- **Mutation engine (higher priority than the data).** Mutate the *pinned story constants* (integer-preserving, range-plausible), add or remove a named step, or retarget `answer` to an existing named intermediate. Stop appending arithmetic to `answer`, and drop `rand_const()` floats.
- **Gold.** Flag known GSM8K gold errors, e.g. 356 (`#### 66` vs the computed 64), rather than forcing the SMT to match them.
- **Never use the `z3_score` column.** It is 1 for all rows, including non-SMT strings. Always re-execute.
