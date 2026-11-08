# X3 — Frozen evaluation sets, Vietnamese + English mirror (2026-11-07)

Specs: `tech/T5_evaluation.md` §2.1, `exec/E1_english_mirror.md` §5.6, `04_data_eval.md` §2.2–2.3, content reviews
R1/R2. Code (written from scratch; `data/clean/` and `data_fix/` from the stopped agent were not imported, copied or
reused): `vi_math_verified_aug/eval_sets/`. Outputs: `data/eval/`. No git commit was made.

## 1. What was built

| Set (VI file / EN file, same ids, same order) | Rows | Role |
|---|---|---|
| `vi_gsm8k_platinum` / `en_gsm8k_platinum` | 1,209 | Vi-GSM8K-Platinum, the full co-primary set; EN = secondary set (E1) |
| `vi_math_algprealg` / `en_math_algprealg` | 2,058 | MATH test, algebra (1,187) + prealgebra (871) |
| `vi_math500_algprealg` / `en_math500_algprealg` | 206 | the MATH-500 subset of the above (124 algebra + 82 prealgebra) |
| **`vi_primary_1000` / `en_primary_1000`** | **1,000** | **frozen primary sample: 600 GSM8K-Platinum + 400 MATH alg/prealg ⊇ all 206 MATH-500** |
| `en_math500_full` | 500 | English MATH-500, every subject; for comparison with the literature only (not parallel) |

- Every set has a `<name>.jsonl` and a `<name>.datacard.json`. `data/eval/SHA256SUMS` covers all 18 files.
  `sha256sum -c` gives 18/18 OK. Re-running `freeze` reproduces the same SHA256SUMS byte for byte.
- Each row has: `id`, `question`, `gold {answer_canonical, answer_type, locale}`, `gold_raw`, `solution`, `subject`,
  `level`, `in_math500`, `math500_unique_id`, `platinum_status`, `source_index`, plus the review fields
  (`review_status`, `review_triggers`, `deterministic_fixes` on the VI side; `vi_review_status` on the EN side).
- Gold `locale` is `mixed` in the VI files and `en` in the EN files.

Pipeline (each step runs from the repo root):

```
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.fetch          # pinned HF parquet/jsonl -> data/eval/_sources (+manifest)
.venv/Scripts/python.exe        -m vi_math_verified_aug.eval_sets.build_base     # alignment + golds -> data/eval/_work/*_base.jsonl, base_checks.json
.venv/Scripts/python.exe        -m vi_math_verified_aug.eval_sets.review_queue   # -> _work/review_queue.jsonl
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix run --stage propose|solve|judge|judge_orig
E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix decide
.venv/Scripts/python.exe        -m vi_math_verified_aug.eval_sets.freeze
```

Sources are pinned by HF revision; the raw-file and JSONL sha256 are in `data/eval/_sources/manifest.json`:

| Dataset | Revision |
|---|---|
| `madrylab/gsm8k-platinum` | `e762492` |
| `openai/gsm8k` | `740312a` |
| `EleutherAI/hendrycks_math` (algebra, prealgebra test) | `21a5633` |
| `HuggingFaceH4/MATH-500` | `6e4ed1a` |

The local VI and EN inputs are hashed in each datacard (`local_sources`).

Environment: `pyarrow 26.0.1` and `sympy 1.14.0` were added to `E:\venv_tools` (needed for parquet and for the
grader under the Claude client). Golds are typed with X1's `eval/math_verify.py`, read-only: version 3.0.0, sympy
fallback backend.

## 2. Alignment checks (`data/eval/_work/base_checks.json`)

### GSM8K

- The local EN test equals `openai/gsm8k` test: 1319/1319 questions.
- The VI `ground_truth` equals the EN gold in 1319/1319 rows, and so does the VI `####` line.
- Platinum → GSM8K index by exact question text:
  - 1209/1209 match, all unique, in monotone order; 110 items were removed.
  - Status counts: consensus 1100 / verified 99 / revised 10.
  - The 10 revised items keep their question text; only the gold changes:
    288 (3→4), 403 (81→135), 454 (150→240), 649 (162000→160000), 749 (3→6), 823 (14→18), 952 (360→1800),
    1035 (35→40), 1071 (251→257), 1309 (2280→2180).
  - Every gold is a typed `int`.

### MATH

The VI row k is checked against the HF EN row k:

| Check | algebra | prealgebra |
|---|---|---|
| Rows | 1187 | 871 |
| Local EN equals HF EN | 1187/1187 | 871/871 |
| Level match | 1187/1187 | 871/871 |
| Type match | 1187/1187 | 871/871 |
| VI last `\boxed` equals EN after normalisation | 1187/1187 | 871/871 |
| VI last `\boxed` graded correct against the EN gold | 1183/1187 | 870/871 |

The 5 grader misses are explained:
- 2 gold overrides (below);
- 2 strings the grader 3.0.0 reports as `ambiguous-locale`: `\$115,000` (algebra-0710) and `1.125`
  (prealgebra-0470). **For X1:** the typed gold does not self-match these, so `gold_self_match` = 2056/2058;
- 1 localised text answer (prealgebra-0024).

MATH-500 alignment:
- 206/206 MATH-500 alg/prealg problems match the hendrycks problems by exact whitespace-normalised text.
- The MATH-500 `answer` agrees with the hendrycks last `\boxed` in 206/206.

### Gold fixes (probe for English solutions with several distinct `\boxed`; all 8 hits read by hand)

- **algebra-0414.** "Find all solutions": the gold becomes the set `-2; 1` (the last `\boxed` held only `1`).
- **algebra-0333.** The gold becomes `-3`; the last `\boxed` was the note "3 is also accepted".
- **Localised text golds.** In 3 items the translation localised a text answer. Here the VI file keeps the VI
  wording and the EN file keeps the English:
  - algebra-0655: "June 20" → "20 tháng 6";
  - algebra-0798: "Monday" → "Thứ Hai";
  - prealgebra-0024: "east" → "đông".

### Deterministic VI repairs (MATH; code in `common.py`)

- A control character followed by a word is restored to the LaTeX command (CR+`ight` → `\right`), but only when
  that command occurs in the English text.
- A literal `\n` that is not a LaTeX command becomes a newline.
- 18 repairs in 15 rows:
  - prealgebra-0668: `\right` restored once in the problem and twice in the solution;
  - 14 literal `\n` in prealgebra solutions;
  - 1 in algebra solution 0970.
- After the repairs, the LaTeX command multiset of every VI problem equals the EN one (0/2058 mismatches).

## 3. Translation-defect review

### Review queue (`review_queue.py`)

- Triggers:
  - R1/R2 question-side annotations;
  - the 10 Platinum-revised items (forced);
  - numeral multiset EN≠VI (locale-agnostic: `,` and `.` between digits are ignored);
  - LaTeX multiset;
  - R1/R2 patterns: "N of the remaining", the calque "nhiều/ít hơn N … so với", "hơn N lần" without "N times",
    lone "thước";
  - 26 glossary rules, e.g. congruent→đồng dạng, common ratio, monic, kite, integer→nguyên tố,
    product→sản phẩm, nearest tenth/hundredth, imperial→metric swap;
  - English thousands comma left in the VI text;
  - unbalanced braces / `$`;
  - asy edits;
  - untranslated English.
- Patterns were refined after inspecting hits. False positives such as "kích thước" and `\prime` were removed.

### Opus procedure (`opus_fix.py`)

- Model `claude-opus-5-5`, effort `medium`, adaptive thinking, Message Batches API, structured JSON output. No
  fallback model was used.
- 1. **Proposer.** Sees EN + VI + triggers and returns `{verdict ok|fix, corrected_vi, reason}`.
- 2. **Check 1, parity (local).** The numeral multiset and LaTeX multiset of the candidate must equal the
  English. Extra VI numerals are allowed only where the English has a number, coin or month word.
- 3. **Check 2, solve.** A fresh call sees only the candidate VI text. `math_verify` must grade its answer
  correct against the English-derived gold.
- 4. **Check 3, judge.** An independent call sees EN + candidate VI, without the proposal, flags or reason, and
  must return `equivalent=true`.
- **Outcomes:**
  - `auto_fixed`: verdict fix and all three checks pass;
  - `cleared`: verdict ok and all three checks pass on the unchanged text;
  - otherwise `flagged`: the original text is kept.
- Each stage runs only on items that passed the previous one.

### Outcomes

| | GSM8K-Platinum | MATH alg/prealg |
|---|---|---|
| Queued | **96** (8 % of 1,209) | **52** (2.5 % of 2,058; 5 are MATH-500 items) |
| Proposer verdict fix / ok | 59 / 37 | 19 / 33 |
| **auto_fixed** | **56** | **18** (1 MATH-500: algebra-0683) |
| cleared (unchanged) | 35 | 28 (4 MATH-500) |
| **still flagged** | **5** | **6** (0 MATH-500) |

What the auto-fixes covered:
- **GSM8K (56):**
  - 31 "more than N … so với" calques;
  - 8 revised items re-translated;
  - 8 English thousands commas;
  - 6 of 7 R1 question-side annotations, including R1's critical test 130 ("10 of the remaining") and major
    test 382 ("40 more gold pesos").
- **MATH (18):**
  - 8 English thousands commas, including R2's major prealgebra-0765 "54,321";
  - "non-congruent → không đồng dạng" in algebra-1153;
  - 5 numeral or term errors, e.g. prealgebra-0740 "juniors/seniors" → "cấp 2/cấp 3";
  - the remaining 4: 1 broken `$` (prealgebra-0093), 2 "nhiều/ít hơn N … so với" calques, and
    "slope → độ dốc" (R2 algebra-0645).

Still flagged. The original text is kept, the item stays in the full sets with `review_status=flagged`, and it is
excluded from the primary sample in both languages (E1 exclusion rule):
- **Parity failed:**
  - gsm8k-0023 (Opus wrote "1 giờ chiều" for "1:00 PM");
  - gsm8k-0068, gsm8k-0201;
  - algebra-0057, algebra-0541, algebra-1011;
  - prealgebra-0349, prealgebra-0846.
  - Most of these are benign wording, e.g. "inclusive" rendered as "từ 1 đến 10".
- **Solve failed:**
  - gsm8k-0649 and gsm8k-1071 (Platinum-revised). The English itself is ambiguous: Opus picks the original
    reading.
  - algebra-0655. Opus answered "Ngày 20 tháng 6" against the text gold "20 tháng 6", a limit of text-gold
    matching.

**Judge sensitivity (negative control, `judge_orig`).**
- The same judge was run on the *original* texts of the 74 auto-fixed items. It rejected only 5:
  - wrong units and terms: gsm8k-0403, gsm8k-0261, gsm8k-0522;
  - algebra-1153 (congruent);
  - prealgebra-0695 (dimes).
- It accepted the originals of R1's critical gsm8k-0130 and major gsm8k-0382.
- So the judge is a lenient, necessary-not-sufficient check. Most of the detection comes from the proposer, while
  the solve and parity checks catch value changes. Hence `cleared` (unchanged after review) is weaker evidence than
  `auto_fixed`. The residual meaning-error rate of not-queued items is unmeasured here; R1's sample put it at
  1.3 % [0.4, 4.7] for GSM8K test before this pass.

## 4. Primary sample (`vi_primary_1000` = `en_primary_1000` ids)

- Seed 20261107. Eligible items are those not flagged: 1,204 GSM8K and 2,052 MATH.
- **GSM8K: 600**, stratified by solution length (lines), proportional allocation with largest remainder:

  | Solution lines | Population | Sampled |
  |---|---|---|
  | 2 | 306 | 152 |
  | 3 | 337 | 168 |
  | 4 | 273 | 136 |
  | 5 | 156 | 78 |
  | 6 | 78 | 39 |
  | 7+ | 54 | 27 |

  Platinum status in the sample: 541 consensus / 55 verified / 4 revised.
- **MATH: 400** = all 206 MATH-500 items + 194 drawn per subject × level stratum. The final per-stratum counts
  equal the proportional 400-item target exactly:

  | Level | algebra | prealgebra |
  |---|---|---|
  | 1 | 26 | 17 |
  | 2 | 39 | 35 |
  | 3 | 51 | 43 |
  | 4 | 55 | 37 |
  | 5 | 59 | 38 |

- Checks:
  - VI and EN ids are identical;
  - the sample contains all 206 MATH-500 items;
  - 0 flagged items;
  - VI and EN gold strings are equal in 999/1000 (prealgebra-0024 is a localised text gold);
  - gold types: 900 int, 39 rational, 19 expr, 15 decimal, 10 radical, 7 interval, 4 tuple, 3 text, 2 choice,
    1 set.
- Sampling details are in `vi_primary_1000.datacard.json` → `sampling`.

## 5. Cost and API use

| Stage | Calls | Input tok | Output tok | USD (batch rate $2/$10 per MTok) |
|---|---|---|---|---|
| propose (8 pilot + 140) | 148 | 182,495 | 41,327 | 0.778 |
| solve | 140 | 32,942 | 46,200 | 0.528 |
| judge | 137 | 94,473 | 3,736 | 0.226 |
| judge_orig (control) | 74 | 50,520 | 6,984 | 0.171 |
| **Total** | **499** | | | **$1.70 of the $4 budget** |

- All calls went to `claude-opus-5-5` through 6 batches, logged in `logs/api/X3/` (requests + results, no keys;
  grep for `sk-ant` finds nothing).
- No NIM, Gemini or Kaggle use. Test items went only to Claude.

## 5b. v1.1 (same day, after X1's review with grader 3.0.1)

1. **VI solutions of the 10 Platinum-revised items.**
   - New stage `opus_fix run --stage solution`: 10 calls, $0.025, batch `msgbatch_01GCyqJv…`. It translates the
     Platinum English solution.
   - A translation is accepted only if all three hold: the final `####` equals the gold, the `<<…>>` annotations
     are identical, and the numerals match.
   - 9/10 accepted.
   - gsm8k-test-1071 stays `solution: null`. Its translation restated arithmetic ("46 – 4 = …"), which broke
     numeral parity. The item is flagged anyway and is not in the primary sample.
   - New VI row field `solution_note` gives the provenance or the reason for a null solution.
2. **algebra-0333.**
   - Gold stays `-3`.
   - New field `gold.accept_also = [{"answer_canonical": "3", "answer_type": "int", "locale": …}]`, because the MATH
     solution says 3 is also accepted.
   - Grader 3.0.1 has no alternate-gold support (`grade()` reads one gold dict and ignores unknown keys). The
     contract, stated in every datacard's `gold_schema`: a prediction is correct if `grade(pred, gold)` **or**
     `grade(pred, alt)` for any `alt` in `accept_also`. **For X1:** honour `gold["accept_also"]` inside `grade()`,
     or the harness takes the OR.
   - Verified: `\boxed{3}` is rejected by the gold and accepted by the alternative.
   - The item is not in the primary sample.
3. **`locale` on every stored gold dict.**
   - Applies to `data/eval/_work/*_base.jsonl` (`mixed`), the VI sets (`mixed`), the EN sets (`en`) and every
     `accept_also` entry.
   - Check: 12,716 gold dicts, 0 without `locale`.

Set version is now **v1.1**: each datacard has a `changelog`, and SHA256SUMS was regenerated (18/18 OK).
Re-running `freeze` reproduces it byte for byte. The primary ids are unchanged, and grader 3.0.1 self-matches every
gold (`gold_self_match` 2058/2058). Total Claude spend is now **$1.73**.

## 6. Open points / for other workstreams

- **X1 (grader):**
  - Under 3.0.0, `\$115,000` and `1.125` did not self-match their own typed gold under locale `mixed`
    (`ambiguous-locale`). This is fixed in 3.0.1 (2058/2058 self-match).
  - Support `gold.accept_also` (§5b).
  - Text golds need tolerant matching ("Ngày 20 tháng 6" vs "20 tháng 6").
  - The grader moved from 2.0.0 to 3.0.0 during this run. The frozen golds were typed with 3.0.0. If the
    `infer_gold` typing changes again, re-run `build_base` → `freeze` and the hashes will change.
- **Residual risk:**
  - 5 GSM8K + 6 MATH items stay flagged.
  - Not-queued items were not re-read; a fresh 150-item human re-sample is the right way to bound the residual
    error (R1 §6 suggests < 1 %).
  - R1 also lists solution-side-only defects (tests 70, 154, 445, 629, 829). The question is unaffected; they
    matter only if VI solutions are used as few-shot exemplars.
- **T5/E1:** the 600/400 sizes and the inclusion of all 206 MATH-500 items are met. The 10 Platinum-revised
  items: 8 were re-translated and accepted; 2 are flagged and therefore excluded from primary.
