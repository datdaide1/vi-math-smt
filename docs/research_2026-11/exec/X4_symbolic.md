# X4: symbolic core library (`vi_math_verified_aug/symbolic/`)

**Workstream:** X4. **Date:** 2026-11-07. **Brief:** `EXEC_BRIEF2.md`.
**Design implemented:** `tech/T1_synthesis.md` §2.1–§2.2 and §2.5, and `01_TECHNICAL_DESIGN.md` §3.1–§3.4.

The library is pure code. It makes no API calls and never imports `src/` or `verify/`. Following the
coordinator's correction, it was written from scratch. The stopped agent's partial files (`ir.py`,
`numerals.py`, `formalize.py`, `prompts/`) were moved out of the package to the session scratchpad and replaced.
The library has no LLM formalizer. The formalizer call belongs to the generation pipeline, which loads its JSON
with `from_expr_json` / `Program.from_dict` and accepts or rejects it with `validate_program`.

## 1. What was built

| Module | Contents |
|---|---|
| `units.py` | `QType` (9 types per T1 §2.1) and `Unit(lemma, classifier, divisible, plausible, per)`. The Vietnamese lexicon has 213 lemmas: people, animals, food, objects, VND/USD/cent money, mass/volume/length/area/time measures, age and percent. There are 62 classifiers (`cái, con, quả, chiếc, bông, cuốn, quyển, tờ, viên, …`). `resolve_unit("cái kẹo")` gives `kẹo` with classifier `cái`, indivisible. Rates are written `đô la/giờ` or `đô la mỗi giờ`. `grid_for` gives the resampling grids of §2.5: COUNT 1, VND 500, USD 0.01, PERCENT 5, MEASURE 0.5, and an integer-valued seed literal keeps a grid of at least 1. `requires_integer` / `requires_nonneg` drive the checks. |
| `numerals.py` | Normalizer: `parse_literal` returns the *set* of readings. `"1.000"` gives {1, 1000}, `"2,5"` gives 5/2, and `"1,200"` gives {6/5, 1200}. It handles `%`, `a/b`, LaTeX `\dfrac{1}{2}` / `\frac34`, and number words (`hai`, `mười lăm`, `hai mươi mốt`, `một trăm linh năm`, `ba nghìn`, `1,5 triệu`). It also handles `nửa`, `một nửa`, `rưỡi` (`hai rưỡi` = 2.5, `trăm rưỡi` = 150), `một phần ba`, `ba phần tư`, `gấp đôi`, `gấp 3 lần`, `tá`, `nửa tá` and `chục`. Formatter: `fmt_vi` uses Vietnamese conventions. `fmt_like(v, literal)` keeps the style of the replaced literal: digits stay digits with the same marks, money keeps fixed decimals (`1.50` becomes `2.25`), words stay words (`hai` becomes `ba`, `một nửa` becomes `một phần ba`), and `gấp đôi` becomes `gấp ba`. `int_to_words` round-trips 0–1199 plus large values. `numbers_in_text` and `contains_value` serve the leak and tag checks. |
| `ir.py` | `Quantity(qid, value, qtype, unit, label_vi, role, span, sentence_id, origin, distractor, fixed)`, `Op(oid, kind, inputs, output, sentence_id, verb_vi)`, `Equation(lhs, rhs)` and `Program(seed_id, text, sentences, quantities, ops, equations, constraints, query, answer, kind, meta)`. There are 19 op kinds: `add sub mul div floordiv ceildiv mod pct_of pct_inc pct_dec ratio_part pow avg min max neg abs floor ceil`. Graph helpers: `topo`, `ancestors` (an unknown's ancestors include the equations that pin it), `depth`, and `canonical`, which is ids-free and sorts commutative operands. **JSON:** `SCHEMA_VERSION = "vi-math-ir/1.0"`. Loaders accept any 1.x, reject 2.x and reject a missing schema. `ProgramBuilder` gives span-safe construction. `from_expr_json` loads the expression-style JSON used by the G1 pilot and the old formalizer prompt (`{"quantities":[{id, role, kind, unit, label, value, span, context, expr}], "equations":["l == r"], "query"}`). It compiles each expression into ops and infers types for anonymous nodes. |
| `executor.py` | `execute(P, values=None)` runs forward exact `Fraction` arithmetic. SymPy is used only when the program has unknowns, in three stages:
  1. Exact linear solve (`linear_eq_to_matrix` + rational LU).
  2. `sympy.solve`, but only on systems that are polynomial once denominators are cleared.
  3. For a univariate non-polynomial equation (unknown in an exponent, floor, ...), a sign-change scan with bisection, whose roots are rationalised and verified exactly. `sympy.solve` itself hangs in SymPy 1.14 on `2**x = 114`; a regression test covers this.

  All stages apply domain filtering. Results are memoised, and the unknown-free part is pre-checked with Fractions. The executor raises on division by zero, irrational values, and zero or several domain solutions. `type_violations` checks **integrality and sign at every node**. `infer_types` does the unit algebra: COUNT±COUNT gives COUNT, MONEY÷COUNT gives RATE, RATE(a/b)×b gives a, and COUNT+MONEY is a hard mismatch. `auto_constraints` and `check` cover type constraints plus seed constraint strings. `eval_constraint` evaluates constraints through a safe AST with no `eval`. |
| `validate.py` | `validate_program(P, gold)` checks acceptance rules (i)–(v): it executes to gold exactly, every given is span-anchored and its literal normalizes to its value, spans do not overlap, the query is derived with depth ≥ 1, every given reaches the query or is a distractor, and types, units and constraints hold. One addition: a `world_const` whose value is written in the text as an uncovered digit literal is rejected. This is the hard-coded-given failure from R3. |
| `rules.py` | `is_answer_tail`, `is_identity`, `clean_answer` / `answer_issues`, `plausible`, `unique_solution` (SymPy, then Z3 integer enumeration with a blocking clause), and `gate(P, P2, op)`, which runs the §2.2 step-2 gates in order. `is_semantic_tail` is an **optional** value-level check (§4.2). |
| `z3enc.py` | Z3 encoding of a Program. It is used only for M1 coupled-integrality sampling and for integer uniqueness (01 §3.1: "Z3 only where it is needed"). |
| `operators.py` | `m1_resample … m6_compose` each return `(Program, [TextEdit])` or raise `NotApplicable`. Every returned program has already passed `gate`. `mutate(name, P, rng)` wraps this as `MutationResult`. |
| `render.py` | `TextEdit` covers the eight edit kinds of T1 §2.2. `to_editor_dict()` exposes only labels and *given* values. `apply_substitutions` / `render_substitutions` are the deterministic M1 renderer; the second also re-anchors spans so the rendered program re-validates. The module also has `tagged_text`, `parse_tags`, `strip_tags`, `tag_multiset_ok` / `tag_issues` and `leaks_answer`. |
| `legacy_ops.py` | The old `mutation_engine.py` operators (all four `wrap` variants, `a+h*2`, `a*(a+1)`) re-expressed in the IR, plain and "renamed in place". They serve the C3 test and the "old operator, everything else best" pilot cell. |
| `fixtures.py` | Hand-written IRs for **23 Vi-GSM8K train seeds** (indices 0 1 2 3 5 7 10 13 20 25 30 40 50 60 70 80 100 150 200 250 300 400 500) and **5 Vi-MATH algebra train seeds** (52, 73, 69, 105, 145). The texts are byte-identical to the dataset files; a test checks this. All 28 pass `validate_program(P, gold)`. They cover rates and world constants (1, 50), unknowns and equations (10, 150), seed constraints (25, 400), distractors (5, 13, 25), word literals (`Mười`, `Hai`, `hai mươi`, `sáu`, `năm`, `ba`), `$0.10`/`$1.50`, and percent/pct_inc/pct_dec. |

### Using the library from G1-style code

```python
from vi_math_verified_aug.symbolic import from_expr_json, validate_program, mutate, render_substitutions, gate
P = from_expr_json(seed_id, text_vi, formalizer_json)        # same JSON shape as pilot_gen/ir_exec.py
errs = validate_program(P, gold)                             # [] = accepted; Violation(code, message) otherwise
r = mutate("M1", P, random.Random(seed))                     # r.program, r.edits, r.reason
text = render_substitutions(r.program, r.edits).text         # M1: no LLM
# M2-M6: send tagged_text(P) + [e.to_editor_dict() for e in r.edits] to the editor; then
# tag_multiset_ok(editor_output, r.program, seed=P) and strip_tags(editor_output)
```

`pilot_gen/` was not touched. G1's `Program`/`Q` maps to this library as follows:

| G1 (`Program` / `Q`) | This library |
|---|---|
| `id` | an int `qid` |
| `kind` | `qtype` |
| `expr` | ops |
| `locate_spans` | done in `from_expr_json` |
| `validate` | `validate_program` |
| `tag_text` | `tagged_text` |
| `parse_tags` / `strip_tags` | the same names |

## 2. Rules as implemented: where they refine the design text

1. **Answer tail.** The T1 formula is `A(q_new) \ A(q0) \ {q0} ⊆ C_new`. As written it has three gaps:
   - It rejects legitimate M5 targets that are intermediates of q0, because their difference set is empty.
   - It misses tails whose extra node is a new *derived* node, e.g. `d = a+1; a·d`.
   - It can be bypassed by re-defining the query qid in place.

   The implementation therefore does the following:
   - **(a)** It finds q0's *intact image* in P2 structurally: same op kind, with inputs that are images, so renaming cannot hide it.
   - **(b)** It requires that image to be an ancestor of q_new.
   - **(c)** It looks through new "plumbing" derived nodes.
   - **(d)** C_new = leaves absent from P whose origin is not `partner`. M6 partner material is story material, not an invented constant.
   - **(e)** It also rejects *intermediate tails*: a new query node equal to f(old intermediate x, ≥ 1 invented constant).

   When q0's computation is no longer intact (M2 makes it a given; M3/M4 change something upstream), there is no old answer to post-compose, so the rule cannot fire. This matches the doc's own argument for M2/M3/M4.
2. **Identity.** The doc rule is kept: isomorphic canonical DAG ∧ same given values ∧ same answer. It is extended with value identity: same answer, every original given kept with its value, and only invented leaves added. This catches `(a+k)−k` even though its DAG differs.
3. **Clean answer.** All §2.5 rules are implemented. One addition: **a zero answer is rejected for story items.** M3 produced "Betty needs 0 more" before this rule existed.
4. **PERCENT values** are stored as the percent number, so 20% is 20; `pct_of(x, p) = x·p/100`. A `ratio` written as `20%` is also accepted by the span check.
5. **Domain.** Solutions for unknowns must satisfy integrality and sign at every node *and* the seed constraints. So `x² = 9` has one solution for a COUNT unknown and two for an ABSTRACT one.

## 3. Operators

| Op | What it does | Edits |
|---|---|---|
| M1 | Resamples ≥ 50% of the resampleable givens, with ≥ 1 on the query path. Draws are log-uniform in [v/3, 3v] and snapped to the grid *inside* the interval. Ratios < 1 are drawn from p/q with q ≤ 12; multipliers stay integers ≥ 2; percents use a 5-grid. Givens linked by a seed constraint move together (100 + 75 = 175). The subset is re-drawn every 50 failed tries. The budget is 200 tries or 5 s; then the **Z3 Int CSP** runs: k·grid variables, integrality witnesses at every node, equations, seed constraints, answer ≠ old, and a randomized widening window. | `SUBST` only, rendered by `render_substitutions` |
| M2 | Picks a given g on the path and resamples it. The old q0 becomes a `stated_fact` given; its op moves to a copy node with the equation copy == q0. g becomes the unknown. The unique-solution check runs in `gate`. Works on seeds that already have unknowns, and on algebra seeds. | `UNKNOWN`, `STATE_FACT` (value of the *old* query only), `NEW_QUESTION` |
| M3 | Inserts x' = f(x, c) for x a given or derived node strictly upstream of q0; consumers of x and equations are rewired. Template library keyed by (qtype, unit class): count `cho đi / làm mất / mua thêm / được tặng thêm / tăng gấp / chia đều cho`; person `có thêm / rời đi`; money (USD, VND, nghìn đồng, cent) `giảm giá % / thuế % / phí / bớt`; measure `thêm / bớt / hao hụt %`; rate `tăng/giảm %`. Story programs only. | `INSERT_STEP(after_sentence, template, operands, value)` |
| M4 | Swaps a relation on the query path: binary add↔sub, pct_inc↔pct_dec, `gấp k lần` ↔ `bằng 1/k`, or factor k→k′. | `REPLACE_RELATION(sentence_id, verb_old, verb_new)` |
| M5 | Asks for another labelled derived node of depth ≥ 2 that is not downstream of q0 and, in story items, not ratio plumbing. Or asks for the difference / integer ratio of two *incomparable* derived nodes with the same type and unit. Givens the new question no longer needs become distractors. | `NEW_QUESTION` |
| M6 | Takes a partner seed with a given g2 whose (qtype, unit or money class) matches q0. g2 is replaced by q0 as an unstated intermediate, and the query becomes the partner query. Partner leaves get origin `partner`. GSM8K only. | `JOIN` (with drop-question info), `REFERENCE(span_g2 → label q0)` |

## 4. Measurements

Run locally on the laptop: one Python process, under the coordinator's RAM rule. Commands and outputs are below.

### 4.1 Tests

```
.venv/Scripts/python.exe -m pytest tests/test_symbolic_numerals.py tests/test_symbolic_ir.py \
    tests/test_symbolic_rules.py tests/test_symbolic_operators.py -q -p no:cacheprovider
```
Result: **210 tests collected, 210 passed**. By file:

| File | Tests |
|---|---|
| `test_symbolic_ir.py` | 79 |
| `test_symbolic_numerals.py` | 51 |
| `test_symbolic_operators.py` | 53 |
| `test_symbolic_rules.py` | 27 |

These are 69 test functions; the parametrized cases are counted separately. The slowest test is the M1 property test, at about 11 s. Earlier runs on this branch hit seven failures; all were fixed before the final run:
- M1 could emit a SUBST edit for an unchanged value.
- A grid snap could round just below v/3.
- Givens linked by a seed constraint were not resampled together.
- `sympy.solve` hung on a transcendental equation.
- Three of the failures were test-design errors.

Coverage of the required test classes:
- **Property: no operator ever produces an answer tail.** `test_property_no_operator_ever_produces_an_answer_tail[M1..M6]` runs every operator × all 28 fixtures × 3 seeds. On every output it also asserts no identity, no type violation, `gate` passes, and that no edit carries the answer.
- **"0,5 cái kẹo" is impossible.** Covered by `test_half_a_candy_is_impossible_{derived,given}`, `test_integrality_enforced_at_intermediate_nodes` (5/2·2 = 5 is rejected because 2,5 kẹo is an intermediate), `test_inferred_type_of_untyped_node_is_checked`, and the property test `test_property_operators_never_make_half_candies`.
- **C3 claim.** `test_legacy_operators_are_answer_tails_on_every_fixture` covers 6 legacy operators × {plain, renamed in place} × 28 fixtures × 3 constants: **1,008 / 1,008 rejected by `is_answer_tail`**, and `gate` rejects all of them. The independent value-level check `is_semantic_tail` also flags every forward GSM8K fixture × every legacy operator except the constant `(a+k)−k`, which `is_identity` catches on all 28 fixtures.
- **Round-trip serialization.** `test_json_round_trip[28 fixtures]`, plus schema-version and structural-error tests.
- **Z3.** Coupled integrality (`test_m1_z3_handles_coupled_integrality`, forced with tries=0); integer uniqueness (`test_underdetermined_integer_system_z3` gives non-unique, `…_with_unique_integer_point` gives unique).

### 4.2 Operator applicability on the fixtures (10 RNG seeds × 28 fixtures)

```
.venv/Scripts/python.exe -u scratchpad/stats.py      # mutate(op, P, Random(10007*s+1)) for s in 0..9
```
```
M1: gsm8k 229/230 algebra 46/50 | seeds_with>=1 28/28 | tail 0 | semantic_tail 0/275   | z3 11 | mean 0.078s max 3.98s
M2: gsm8k 229/230 algebra 41/50 | seeds_with>=1 28/28 | tail 0 | semantic_tail 0/270   | z3 0  | mean 0.036s max 0.88s
M3: gsm8k 230/230 algebra  0/50 | seeds_with>=1 23/28 | tail 0 | semantic_tail 68/230  |       | mean 0.004s
M4: gsm8k 190/230 algebra 30/50 | seeds_with>=1 22/28 | tail 0 | semantic_tail 0/220   |       | mean 0.001s
M5: gsm8k 170/230 algebra 10/50 | seeds_with>=1 18/28 | tail 0 | semantic_tail 0/180   |       | mean 0.001s
M6: gsm8k 100/230 algebra  0/50 | seeds_with>=1 10/28 | tail 0 | semantic_tail 100/100 |       | mean 0.001s
NotApplicable reasons: M1 try budget (5, all algebra-145/gsm8k-10); M2 no unique clean inverse (10);
M3/M6 story-only (50 each); M4 no compatible swap (60); M5 no depth-2 target or comparable pair (100);
M6 no unit-compatible partner among the 23 fixtures (130).
```

How to read this table:
- Every output passed `gate`, and the structural tail count is **0 for every operator**. Z3 rescued 11 M1 cases where rejection sampling failed: divisibility chains and the 100 + 75 = 175 constraint.
- **`is_semantic_tail` is a dependency test, not a quality verdict.** It is True when the new answer depends on the seed's leaves *only through the old answer*.
  - **M3 (68/230 = 30%).** An additive step inserted on an additive chain makes the new answer old ± c. The story is structurally different (the step is mid-story), but the arithmetic is an affine shift. Gating on this would drop about 30% of M3; that is left to the pilot (C3 attribution).
  - **M6 (100/100).** This holds by construction: sequential composition consumes the old answer through q0. The partner seed adds story material, which is why the structural rule accepts M6.
- M6 coverage is limited by the 23-fixture partner pool. With 7,473 seeds, unit-compatible partners are abundant.

## 5. Known limits

### 5.1 MATH algebra patterns

The IR is **rational-valued with one numeric query**.

**Can express:**
- evaluation of expressions at given values (69), including integer powers, floor/ceil/abs, and nested function application written inline (f(g(5)));
- linear equations and systems (52), rational equations with a unique admissible root (73);
- a parameter from a known root (105);
- exponential equations with a rational solution (145, via `sympy.solve`);
- word-problem algebra (GSM8K 10, 150);
- integer Diophantine uniqueness when every unknown is integer-typed (Z3).

**Cannot express:**
- (1) **irrational values**: `\sqrt{21}`, π, non-integer logs, so "rationalize the denominator" and radicals are out (an exact root is fine);
- (2) **set-valued or selected roots**: "the larger root", "sum of all solutions", "all x such that". A quadratic in an ABSTRACT unknown is non-unique and rejected. Such items are representable only if the formalizer writes a closed form, e.g. Vieta −b/a;
- (3) **non-numeric answers**: points/tuples such as the circle centre `(3,-1)`, intervals and inequalities, factorisations or simplified expressions, answers containing a free parameter;
- (4) **functions as objects**: piecewise definitions and continuity conditions (algebra-0) need hand-written equations per boundary, and there is no lambda node;
- (5) **Z3 limits**: the Z3 path does not encode floor/mod/min/max or a variable exponent, so M1 falls back to rejection sampling only. Example: gsm8k-10, where 2^(300/100) needs 847 ≡ 0 mod (2ⁿ−1). Nonlinear integer problems may return `unknown`, which is treated as non-unique.

**Operators on symbolic seeds:** M1, M2, M4 and M5 apply. M3 and M6 are story-only, as in the design. M4 has no inequality-direction swap, because inequalities are not answers.

### 5.2 Other limits

- **Tag check allows untagged numerals by value, not by position.** A stray `5` passes in gsm8k-0 because `tháng 5` is untagged in the seed (pinned by a test). Number words that are ambiguous on their own are not scanned: `một` (article), `năm` (year), `tư`.
- **Semantic tails.** The structural tail rule is the spec. Value-level analysis shows that an additive M3 step on a purely additive chain makes the new answer an affine shift of the old one: `is_semantic_tail` is True. The text is still structurally new, because the step sits mid-story. Whether to gate on this is a pilot decision; see the rate in §4.2.
- **Plausibility ranges** come from a coarse lexicon (per lemma, not per item: a ticket and a car are both "USD ≤ 1e8"). M6 can therefore produce odd but clean compositions, e.g. a print job priced off Julie's page count, with change = 0.10 USD. The editor judge and the cycle check have to catch these.
- **Unit inference** is a fallback for untyped nodes. Declared types win, and only hard qtype clashes are violations; a lemma mismatch such as táo + cam is not.
- **Uniqueness for transcendental equations is numeric, not proved.** Univariate non-polynomial equations are scanned on [−10⁴, 10⁴] for sign changes and exact zeros. Double roots that touch zero without crossing, and roots outside the range, can be missed. Several non-polynomial unknowns at once are unsupported (ExecError).
- **Performance.** Forward programs mutate in milliseconds. Programs with unknowns go through SymPy; the linear fast path, the polynomial-only `solve`, the pre-check and memoisation bring a call to ≤ 0.3 s. An unsolvable non-linear resample can still use the full 5 s M1 budget (algebra-145: 4 of 10 seeds end `NotApplicable`). Operator loops have wall-clock budgets: M1 5 s, M2 8 s.
- **Module name.** The executor module is `executor.py`, not `execute.py`, because the package re-exports the function `execute`, and a submodule of the same name is shadowed under `import … as`.

## 6. Files

- Library: `vi_math_verified_aug/symbolic/{__init__,units,numerals,ir,execute,validate,rules,z3enc,operators,render,legacy_ops,fixtures}.py`
- Tests: `tests/test_symbolic_{numerals,ir,rules,operators}.py`
- Backup of the replaced partial files (not in the repo): session scratchpad `symbolic_partial_backup/`.
