# X1: typed Vietnamese answer grader (2026-11-07)

**Built:** `vi_math_verified_aug/eval/math_verify.py` (GRADER_VERSION 3.0.2 (3.0.0 → 3.0.1 in §9, → 3.0.2 in §10)), written fresh from
`tech/T2_verification_selection.md` §4 and `04_data_eval.md` §4.2. The Sept-2026 file and a stopped agent's partial
edits were used only as a list of failure modes to test. None of their logic was kept.
**Tests:** `tests/test_math_verify.py` plus 333 labelled cases in `tests/grader_cases/`.
**Status:** all tests pass under both backends. 4 required false positives fixed. Self-match is 100 % on every gold.
On blind-labelled real outputs the grader made 0 FP and 0 FN. On the held-out style set it made 0 FP and 9/29 FN
on the first run (2/29 after fixes).

## 1. What the grader does

| Step | Rule (T2 §4) | Where |
|---|---|---|
| Gold | Typed dict `{answer_canonical, answer_type ∈ int, rational, decimal, radical, tuple, set, interval, expr, text, choice; locale vi/en/mixed; vars; target_var; round_dp; percent}`. T2 keys `type`/`value` and alias `ineq`→interval are accepted. **The gold is never extracted.** A legacy gold *string* is typed once, as a whole, by `infer_gold()`. Nothing is searched inside it. | `as_gold`, `infer_gold` |
| Extraction (prediction only) | Steps in order: drop `<think>…</think>`; last brace-balanced `\boxed{}`; `####`; last cue (Đáp số / Đáp án / Kết luận / Kết quả / Vậy / The answer is); a short pure-math response taken whole; for a container gold, the tail of the last line; the last number. Several *different* boxes in the final paragraph are joined for a container gold and count as a **hedge (wrong)** for a scalar gold. If the response ends in a repetition loop and has no box or cue, it has **no answer**. | `_extract`, `_degenerate` |
| Prose trimming | `tập nghiệm của phương trình là $S=\{1;\frac32\}$` becomes `S=\{1;\frac32\}`. Units and words are stripped (đồng, km, giờ, tuổi, `\text{…}`, `^\circ`). `(k∈ℤ)` is dropped and `k2\pi` becomes `2k\pi`. | `_trim_prose`, `_prenorm` |
| Structure before numbers | `(1,2)` is a pair. `x=2 hoặc x=3`, `x_1=2; x_2=3`, `S=\{2;3\}` and `x∈\{…\}` are sets. `\pm` expands to two items. `[1;3)`, `]1;3[`, `x\ge 2`, `m<-1 hoặc m>2`, `\cup` and `\mathbb R` are intervals. Arity must match. Sets compare as multisets. Tuples are ordered, or matched by name when both sides name their variables. `target_var` picks `x` in `Vậy x=4 thì A=2`. | `_container_items`, `_intervals`, `_target_value` |
| Locale numbers | `{,}` is a decimal comma. `\,`, a space, or a repeated separator between 3-digit groups is grouping. When both `.` and `,` appear, the last one is the decimal mark. A single separator followed by a number of digits other than 3, or preceded by a lone `0`, is a decimal mark. A single separator followed by exactly 3 digits is **ambiguous**: the response's own unambiguous tokens vote, then `gold.locale` decides. For `mixed` golds the gold type decides (int → grouping, otherwise decimal), and the verdict gets `flag=ambiguous-locale`. | `_votes`, `_context`, `_read_run` |
| Equivalence | Numbers compare as exact `Fraction`s. A decimal literal never equals an irrational gold. A rational gold needs an exactly rational prediction. Tolerance applies only with `round_dp`. Other atoms go to the backend (next row). | `_atom_eq` |
| Backend | `math-verify==0.9.0` (+`latex2sympy2_extended==1.11.0`) is used when it is importable. Its negatives get a second opinion from an exact SymPy zero-difference check, because MV 0.9.0 returns False for `x` vs `\frac x2+\frac x2` (measured). Otherwise a built-in SymPy path is used. `BACKEND` names the active one and `set_backend()` switches. On Windows MV runs with `parsing_timeout=None, timeout_seconds=None`. | `_sym_eq`, `_mv_eq`, `_sympy_eq` |

`grade(pred, gold)` returns `{correct, span, method, flag, conv, gold_type, backend, grader_version}`. Flags that
should go to adjudication: `no-answer`, `degenerate`, `hedged`, `arity p!=g`, `ambiguous-locale`, `prose-span`,
`unparsed-interval`, `text-gold`, `unclosed-think`.

**Which backend is used**

| Environment | Backend | Why |
|---|---|---|
| project `.venv` (3.11) | **sympy-fallback** | math-verify is not installed there |
| `E:\venv_mv` (new short-path venv: math-verify 0.9.0, latex2sympy2_extended 1.11.0, antlr4 4.13.2, sympy 1.14.0, pytest) | **math-verify-0.9.0** | installs cleanly on py3.11 / Windows |
| Kaggle snapshot (`infra/pilot/build_pilot_data.py` copies the file and the job does `from math_verify import …`) | **sympy-fallback, always** | A top-level module named `math_verify` cannot also import the HF package with that name. The guard `__name__ == "math_verify"` was verified: the snapshot imports and grades correctly with HF math-verify installed in the same env |

Every labelled verdict is identical under the two backends (test `test_backends_agree_on_labelled_cases`).

## 2. Compatibility path (thin wrappers, kept because existing callers need them)

| Caller (grep) | Uses | Status |
|---|---|---|
| `jobs/{build,kernels}/{w2,s2}-pilot-v1/run.py` (S2) | `is_correct(extract_answer(body), ground_truth_str)` | Works: the string gold is typed by `infer_gold`, never extracted. Recommendation: call `grade(full_output, gold)` instead. When a container gold is split across several boxes, those boxes can only be joined from the full response. |
| `vi_math_verified_aug/data_fix/mta_vi.py` (G1) | `is_correct(pred, gold_str)` | Works |
| `vi_math_verified_aug/vi_exam/build.py` (X2) | `normalize_answer`, `_to_float`, `_sympy_equal` | Shims kept and marked deprecated. `normalize_answer` defaults to `locale="vi"`. X2 should switch to `infer_gold`. |

Legacy kwargs `rel_tol` and `abs_tol` are accepted and ignored. Numbers are exact. For rounding, use `round_dp`.

## 3. Required false positives: fixed

`tests/test_math_verify.py::test_fp_*` (and cases B1a/b, TUP2, B2a, X01–X04, X15):

| Case | Old scorer | Now |
|---|---|---|
| `125` vs gold 0.125 | True (FP) | **False** |
| `(1,2)` vs 1.2 | True (FP) | **False** (pair decided before locale; `arity 2!=1`) |
| `8` vs `x=7; y=8` (typed tuple and legacy string) | True (FP) | **False** (`8` is a scalar against a 2-item gold, so arity fails) |
| `0.625` / `0,625` / `0{,}625` vs int 625, every locale | — | **False** (a leading `0` group is always decimal) |

## 4. Labelled cases (`tests/grader_cases/`, 333 total, ≥150 ✓, 102 real ✓)

| File | n | + / − | How labelled |
|---|---|---|---|
| `synthetic.jsonl` | 191 | 131 / 60 | T2 §4.6 (all 60 design cases), T2 "still to add", legacy failure modes, 7 regressions found by the probes in §6, and 4 follow-up cases (§9). Written alongside the grader, so this file is *not* a generalisation estimate. |
| `real_vi_gsm8k.jsonl` | 102 | 19 / 83 | **Real outputs** of Qwen3-0.6B/1.7B-Base and Qwen3.5-0.8B/2B-Base on Vi-GSM8K test (`jobs/outputs/w2-pilot-v1/results/*/preds.jsonl`). I labelled each one by reading it, **before running the new grader on it**, with the reasons stored in `scripts/make_real.py`. The sample is stratified toward hard cases: all 44 unboxed outputs (every one is a truncated repetition loop, labelled "no committed answer → wrong"), 40 with separators, decimals or fractions in the box (`90.000`, `14.000`, `2,25`, `0,725`, `19 1/3`, `23.33\%`…), and 18 random plain boxed integers. |
| `heldout_style.jsonl` | 40 | 29 / 11 | MATH / Vi-exam-style responses: sets, systems, intervals, `k2\pi` families, `12\,360\,000 đồng`, `12,36 triệu`, MCQ, units, `<think>`. I wrote and labelled them after the dev cases were frozen. First-run log: `heldout_firstrun.txt`. |

## 5. Error rates against the labels

Counts are FP/negatives and FN/positives. 95 % intervals are Wilson.

| Set | Grader run | FP | FN | Notes |
|---|---|---|---|---|
| Real Vi-GSM8K (blind) | first run = final | **0/83** (0 %, CI 0–4.4 %) | **0/19** (0 %, CI 0–16.8 %) | The old scorer had 1 FP here (a loop whose last number was the gold). |
| Held-out style | **first run** | **0/11** (CI 0–25.9 %) | **9/29 = 31 %** (CI 17–49 %) | All 9 were misses, never wrong accepts. Causes: a prose lead-in before `$S=…$` (H01, H02, H04); a cue-less list (H29); a cue-less `\dfrac` (H08); `triệu đồng` removed as a unit (H13); a trig family with `(k∈ℤ)` (H16); and H15/H19 below. |
| Held-out style | after fixes | 0/11 | 2/29 | **Not an estimate**: these cases were then used to fix the grader. |
| Synthetic (dev) | final | 0/60 | 0/131 | dev set |
| **All 333** | final | **0/154** | **2/179** | The 2 FN are `xfail(strict)` in the tests. |

Known errors (kept as `KNOWN_ERRORS` in the tests):
- **H15:** `Giá trị lớn nhất là 3, đạt được khi x = 1` has no cue, so the last-number fallback takes 1.
- **H19:** an English `1,234` against a `vi`-locale gold with no votes is read as 1.234. The verdict is flagged `ambiguous-locale`, so it goes to adjudication.

**Population check** (all 1,200 real outputs, `scripts/population.py`, gold typed `int`/`mixed`):
- The new grader disagrees with the stored old-scorer verdict on **1/1200**: Qwen3.5-0.8B #438, a loop ending "…cần 2 hộp…" with gold 2. Old: True. New: False, flagged `degenerate`. I judge the new verdict correct.
- Routes: 1156 boxed, 33 degenerate, 7 last-number, 3 cue, 1 none.
- 6 items carry `ambiguous-locale`. That is 0.5 %, under T2's 1 % adjudication trigger.

**Adjudicator:** not used. There are only 3 grader/label disagreements in total (H15, H19, R-…-438), and each one is unambiguous on reading, so no Claude call was made ($0, nothing in `logs/api/X1/`).

**Caveat:** the real outputs are all Vi-GSM8K with integer golds. **No real MATH or Vi-ExamMath student outputs exist
yet**, so the rich types (sets, intervals, radicals) are covered only by synthetic and held-out-style cases. T2 §4.7
asks for 300 real outputs stratified by type × verdict with 2 human raters. That should be done once S2 produces
Vi-ExamMath and Vi-MATH outputs.

## 6. Self-match (requirement 6) and false-positive probes

Command: `python tests/grader_cases/scripts/selfmatch.py {sympy|math-verify} CROSS.jsonl EXC.jsonl`, run from the
repo root. Pytest repeats the boxed-gold part (`test_self_match_*`).

- **MATH gold** = last `\boxed{}` of the reference solution, which is the dataset's own gold definition. English golds come from `data/generate/MATH/test/<subj>.json`, attached by position (lengths assert-equal for all 7 subjects).
- **Vi-MATH gold** comes from `data/test/math/<subj>_vi.jsonl`, typed with `locale=mixed`.
- **GSM8K gold** is `ground_truth`.

| Gold set | Boxed gold vs itself (sympy / MV) | Full reference solution vs its gold | Exceptions |
|---|---|---|---|
| MATH-en (5000) | **5000/5000** / **5000/5000** | 4989/5000 | 11, listed below |
| Vi-MATH (5000) | **5000/5000** / **5000/5000** | 4989/5000 | the same 11 items |
| Vi-GSM8K (1319) | **1319/1319** / **1319/1319** | **1319/1319** | none |

- **Self-match rate: 100 %, with no exceptions**, on 11,319 golds under both backends.
- With the string-identity shortcut switched off, so the backend must prove every gold equals itself, the result is still 100 % under MV and 99.92 % under SymPy. The 4 SymPy misses are the ratios `3:1`, `1:2`, `1:1` and the chain `f(2)<f(1)<f(4)`; the shortcut handles all four.
- **Type distribution of the 5000 MATH golds:** int 3296, rational 671, expr 380, radical 234, tuple 107, decimal 100, set 74, interval 66, text 43, choice 29.
- **11 full-solution exceptions.** These are reference solutions whose final paragraph boxes several *different* values, which the design rule grades as a hedge:
  - algebra#91 (`\frac{2\sqrt{53}+53}{53}` and `161`), algebra#414 (`-2`, `1`)
  - int_alg#12, #409, #516, #563, #576, #772
  - prealgebra#477 (`11`, `20`, `12`)
  - precalc#105, #406

  A student who writes this way is graded wrong and flagged `hedged`. This is T2's rule, kept on purpose.
- **Vi gold vs EN gold:** 4975/5000 agree. All 25 mismatches are translated `text` golds (`odd` vs `lẻ`, `Friday` vs `Thứ Sáu`, `ellipse` vs `elip`). There is no numeric translation drift. Vietnamese students should be graded against the **Vi** text gold, or these 25 should be adjudicated.
- **One typing difference EN vs VI:** precalc#196 `(0,1)` is an interval in English, cued by "ranges", and was a tuple in Vietnamese before the cue `chạy trên` was added. It is now consistent. Per T2 §4.2, every non-int gold still needs one human pass when golds are migrated.

**Cross-pair FP probe** (`scripts/crosspair.py`): each MATH gold is graded against the next item's gold, 4,993 pairs.
- 42 pairs graded True: 40 have identical strings, and 2 are equal by policy (`120^\circ`←`120`, `7\%`←`7`). **0 FP.**
- The probe found 1 real FP, now fixed and kept as test P01: `12` vs gold `2^{2005}`. The SymPy numeric fallback overflowed to `inf ≤ inf`. It now compares at 50 digits with a finite check.

## 7. Commands and evidence

```
.venv\Scripts\python -m pytest tests/test_math_verify.py -p no:cacheprovider
  -> 366 passed, 1 skipped (MV-agreement test: MV not in .venv), 2 xfailed in 9.8s
set PYTHONPATH=. & E:\venv_mv\Scripts\python -m pytest tests/test_math_verify.py -p no:cacheprovider
  -> 367 passed, 2 xfailed in 12.8s
.venv\Scripts\python -m pytest tests/  -> the only failure is tests/test_symbolic_numerals.py::test_fmt_vi_vietnamese_conventions (X4's file, not touched)
```

Case generators, with labels and reasons: `tests/grader_cases/scripts/make_{synth,real,heldout}.py`.

## 8. Open items

1. **Real outputs on rich types.** Collect them when S2 produces Vi-MATH and Vi-ExamMath generations. Then do T2 §4.7: 300 items, 2 raters, CIs in the paper.
2. **Cue-less prose (H15).** The remaining FN class. Route `method=last-number` items to adjudication when the gold is not an int.
3. **math-verify in the project `.venv`.** Not installed, so `.venv` runs the SymPy fallback. Both backends agree on all labelled cases. Pin `math-verify==0.9.0` if the lead wants MV in the main env. Run batch grading on Linux to get MV's own timeouts.
4. **S2 jobs.** Pass the full output to `grade()` and log `grader_version`, `flag` and `method` with every score.

## 9. Follow-up: X3's two self-match failures (grader 3.0.1)

**Reproduced.** I reran X3's own check from `eval_sets/build_base.py`:
`is_correct("\boxed{gold_raw}", r["gold"])` over `data/eval/_work/math_algprealg_base.jsonl` gives **2056/2058**.
The two failures are `math-test-algebra-0710` (`\$115,000`) and `math-test-prealgebra-0470` (`1.125`).

**Root cause.** It is the grader, not the stored golds. The base file's gold dicts carry no `locale` key
(`{"answer_canonical": "115000", "answer_type": "int"}`), and 3.0.0 defaulted a missing locale to `vi`. Under `vi`,
a single separator followed by exactly 3 digits with no other votes in the response is read the Vietnamese way:
`115,000` became 115.0 and `1.125` became 1125. The frozen `data/eval/*.jsonl` rows do carry `locale`
(`mixed` for VI, `en` for EN), and they already passed under 3.0.0. So the failure showed up only in the
locale-less base check.

**Fix (3.0.1).**
1. A gold with no `locale` now defaults to **`mixed`**. Unknown provenance gets the type-guided tie-break (int → grouping, otherwise decimal) plus `flag=ambiguous-locale`. An explicit `locale: "vi"` behaves as before. The only labelled case that relied on the old default, T2's B4c, now states `locale: "vi"`.
2. Under a tie, an unbracketed `2,5` that is split as a list is now flagged `ambiguous-locale` too.
3. Text golds ignore leading filler: `Ngày 20 tháng 6` = `20 tháng 6`, `the parabola` = `parabola`. Negations are never stripped. This answers X3's text-tolerance request.
4. New cases P08–P11. New test `test_self_match_frozen_eval_sets`, parametrised over **all 9 frozen sets**, VI and EN. For every record it checks three predictions against `record["gold"]`, the API the eval runner will use:
   - `\boxed{answer_canonical}`
   - `\boxed{gold_raw}`
   - `\boxed{gold_raw}` against the gold with its `locale` removed
   
   The `gold_raw` checks are skipped for the 5 VI rows whose text gold X3 localised (`east`→`đông`, `June 20`→`20 tháng 6`, `Monday`→`Thứ Hai`).

**New self-match rate.** `tests/grader_cases/scripts/selfmatch_eval.py`:

| Set | 3.0.0 (as X3 ran it, gold without locale) | 3.0.1 |
|---|---|---|
| base check (`_work/math_algprealg_base.jsonl`) | 2056/2058 | **2058/2058** |
| en_gsm8k_platinum / en_primary_1000 / en_math_algprealg / en_math500_algprealg / en_math500_full | — | 1209 / 1000 / 2058 / 206 / 500, all 100 % |
| vi_gsm8k_platinum / vi_primary_1000 / vi_math_algprealg / vi_math500_algprealg | — | 1209 / 1000 / 2058 / 206, all 100 % |
| **All frozen eval records** | — | **9446/9446 = 100 %**, under both SymPy and math-verify 0.9.0 |

**No re-freeze needed.** The current `infer_gold` reproduces every frozen gold's `(answer_type, answer_canonical)`
exactly: **0 differences** on all non-localised rows (scratch check `x3check.py`). The typing did not change, so
the data hashes stay valid.

**Regressions: none.**
- MATH/GSM8K self-match is still 11,319/11,319.
- The cross-pair probe still finds 0 FP on 4,993 pairs.
- The 1,200-output population check is unchanged (1 disagreement with the old scorer, the degenerate loop).
- Pytest: 366 passed, 1 skipped, 2 xfailed (`.venv`); 367 passed, 2 xfailed (`E:\venv_mv`).

**For X3.** I did not edit `data/eval`. These are stored-data observations, not grader errors:
- **Missing solutions.** 10 `vi_gsm8k_platinum` rows have `solution: null`: gsm8k-test-0288, 0403, 0454, 0649, 0749, 0823, 0952, 1035, 1071, 1309. Eight are `review_status: auto_fixed` and two (0649, 1071) are `flagged`. 4 of them are also in `vi_primary_1000`. The golds are fine; only the reference solution is missing.
- **algebra-0333.** The gold is `-3`, but the reference solution ends with "Note: `\boxed{3}` is also accepted as an answer." If both are meant to be accepted, the record needs an accept-also field. The grader will not accept 3 against gold `-3`.
- **Multi-box reference solutions.** algebra-0091 and prealgebra-0477 box several distinct values in the final paragraph. Self-match is unaffected, but these items' reference solutions grade as `hedged`.
- **Keep `locale` on every stored gold dict.** 3.0.1 now tolerates a missing key, but `vi` vs `mixed` changes the verdict on ambiguous `d.ddd` / `d,ddd` tokens.

## 10. Follow-up: `accept_also` (grader 3.0.2)

The X3 v1.1 eval sets add `gold.accept_also`: a list of alternative gold dicts with the same schema. Its first use
is `math-test-algebra-0333` (gold `-3`, accept_also `[3]`, VI and EN).

**Behaviour.**
- `grade()` tries the main gold first, then each alternative, and returns the first correct verdict.
- The result has a new field, `matched`: `"gold"`, `"accept_also[i]"`, or `None` when nothing matched.
- An alternative without a `locale` inherits the main gold's.
- A hedge stays wrong: `\boxed{-3} hoặc \boxed{3}` is hedged against every alternative.
- `is_correct()` inherits all of this.
- The single-gold logic moved unchanged into `_grade_one()`.

**Tests.**
- `test_accept_also` (unit).
- `test_accept_also_algebra_0333_frozen`: on both v1.1 rows, `\boxed{-3}` → `gold`, `\boxed{3}` → `accept_also[0]`, `\boxed{2}` → wrong.
- `test_self_match_frozen_eval_sets` now also requires every `accept_also` canonical to match itself.

**Results on v1.1.**
- Self-match over all 9 frozen sets: **9446/9446 = 100 %**.
- pytest: 368 passed, 1 skipped, 2 xfailed in `.venv`; 369 passed, 2 xfailed in `E:\venv_mv`.

**For X3.** No gold typing changed. The datacards record `3.0.1` as the typing grader; that stays valid, because
`infer_gold` is identical in 3.0.2.
