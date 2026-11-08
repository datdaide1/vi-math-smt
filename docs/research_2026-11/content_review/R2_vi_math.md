# R2 — Vi-MATH content review (train translation + Vi-MATH test)

Reviewer tag: `vi_math` · date 2026-11-07 · protocol: `PROTOCOL.md` (seed 20261107, Wilson 95% CIs)

Files reviewed

| split | Vietnamese file(s) | rows | English source | alignment used |
|---|---|---|---|---|
| train | `data/translation/MATH/train/*_vi.jsonl` (7 subjects) | 7,501 | `data/generate/MATH/train/*.json` (7,500) | `index` within subject; **precalculus re-aligned by content** (see §4.1) |
| test | `data/test/math/*_vi.jsonl` (7 subjects) | 5,000 | `data/generate/MATH/test/*.json` (5,000) | line position within subject (test rows have **no `index` and no `ground_truth` field**) |

Annotations: `annotations/vi_math_train.jsonl`, `annotations/vi_math_test.jsonl` (one row per sampled item; `index` = row position in the subject file, which equals the `index` field for train).

## 1. Sampling

Stratified by subject × level (proportional allocation, largest remainder), `random.Random(20261107)`, without replacement, strata iterated in sorted order. n = 150 train + 150 test.

- train allocation: algebra 36, intermediate_algebra 26, prealgebra 24, geometry 18, number_theory 18, counting_and_probability 14, precalculus 14
- test allocation: algebra 35, intermediate_algebra 27, prealgebra 27, number_theory 17, precalculus 16, counting_and_probability 14, geometry 14
- Exact index lists: the `subject`/`index` columns of the two annotation files, in draw order (re-draw: stratify `vi[s][i]["level"]` over the 7 files in the order algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus).

Every sampled item was read in full: Vietnamese problem + solution side-by-side with the English problem + solution, plus `ground_truth` (train) and every `\boxed{}` on both sides. `[asy]` blocks were compared character-for-character against English (identical blocks shown collapsed; any differing block shown in full).

## 2. Results on the read sample

### 2.1 Error rate per code (item-level, Wilson 95% CI)

| code | train k/150 | train % [CI] | test k/150 | test % [CI] |
|---|---|---|---|---|
| NL-MEANING | 3 | 2.0 [0.7, 5.7] | 3 | 2.0 [0.7, 5.7] |
| NL-LATEX | 5 | 3.3 [1.4, 7.6] | 2 | 1.3 [0.4, 4.7] |
| NL-FLUENCY (terminology) | 5 | 3.3 [1.4, 7.6] | 6 | 4.0 [1.8, 8.5] |
| NL-LEAK (English left over) | 0 | 0.0 [0.0, 2.5] | 2 | 1.3 [0.4, 4.7] |
| NL-OMIT / NL-UNANSW / NL-DUP | 0 | 0.0 [0.0, 2.5] | 0 | 0.0 [0.0, 2.5] |
| ANS-WRONG | 1 | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |
| ANS-FORMAT | 4 | 2.7 [1.0, 6.7] | 4 | 2.7 [1.0, 6.7] |
| SOL-INCONS | 1 | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |
| SOL-WRONG | 0 | 0.0 [0.0, 2.5] | 0 | 0.0 [0.0, 2.5] |
| X-MISALIGN (row ≠ English row of same index) | 2 | 1.3 [0.4, 4.7] | 0 | 0.0 [0.0, 2.5] |
| X-METADATA (wrong `level`) | 1 | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |
| X-DECSEP (English thousands comma in problem text) | 0 | 0.0 [0.0, 2.5] | 1 | 0.7 [0.1, 3.7] |
| **any code** | 20 | **13.3 [8.8, 19.7]** | 17 | **11.3 [7.2, 17.4]** |
| **major + critical** | 7 | **4.7 [2.3, 9.3]** | 5 | **3.3 [1.4, 7.6]** |
| critical | 1 | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |

Pooled (n = 300): any code 37/300 = 12.3% [9.1, 16.5]; major+critical 12/300 = 4.0% [2.3, 6.9].

### 2.2 Usable / fixable / discard

| | clean (no code) | minor only | usable as-is (clean + minor) | fixable (major/critical, English source exists) | discard |
|---|---|---|---|---|---|
| train | 130 (86.7%) | 13 | **143 = 95.3% [90.7, 97.7]** | 7 = 4.7% [2.3, 9.3] | 0 = 0% [0, 2.5] |
| test | 133 (88.7%) | 12 | **145 = 96.7% [92.4, 98.6]** | 5 = 3.3% [1.4, 7.6] | 0 = 0% [0, 2.5] |

Every defect found is fixable because the English original is available and correct (apart from the inherited English ground-truth bug in §4.4).

For eval use, what counts is the **problem statement + gold**. Problem-side major defects in the test sample: 2/150 = 1.3% [0.4, 4.7] (ia-760 literal `\n`, pa-765 `54,321`). No test item in the sample had a wrong or changed answer.

### 2.3 Qualitative picture

- The translation itself is **good**: numbers, conditions and asked-for quantities matched in 297/300 items. Every formula was copied correctly except where a *transport* bug hit (below). 300/300 boxed answers match English after normalisation. The Vietnamese is fluent and mostly uses standard terms (`số nguyên tố`, `ước chung lớn nhất`, `bội chung nhỏ nhất`, `nguyên tố cùng nhau`, `trục căn thức ở mẫu`, `công thức Vieta`, `đa tập`, `tâm đường tròn nội tiếp`).
- The "ba chấm" (sloppiness) shows up in the **pipeline**, not the language:
  1. JSON-escape corruption of LaTeX (`\right` → CR + `ight`).
  2. Literal `\n` in place of newlines.
  3. Dropped `}` after `\boxed{`.
  4. The LLM rewrote or moved `[asy]` code.
  5. The precalculus train file is out of alignment.
- There are also a few **systematic terminology errors**, the worst being `congruent` → `đồng dạng` (similar).

## 3. Worst examples (quoted)

1. **train algebra 387 (critical, NL-MEANING + SOL-INCONS).** The English asks "How many cubic **yards**"; the Vietnamese asks "bằng bao nhiêu **mét khối**" (cubic metres), while the solution and gold `4` still use `yard khối`. The gold is wrong for the Vietnamese question. A full-corpus scan found the same yard→mét swap in algebra 1000 ("square yards" → `mét vuông`) and a gallon→`lít` swap in counting_and_probability 444.
2. **train precalculus 527/528 (X-MISALIGN, file-level).** Both rows translate English row 527 ("Chu kỳ của $y = \cos \frac{x}{2}$ là bao nhiêu?" / "… là gì?"). From that point on, every row's `index` points one English row too early. See §4.1.
3. **train intermediate_algebra 798 (major, NL-LATEX, in the problem).** "Tìm \[\left|\left(1 + \sqrt{3}i⟨CR⟩ight)^4⟨CR⟩ight|\]". `\r` was decoded as a carriage return, so the problem no longer compiles. The same bug affects algebra 327, intermediate_algebra 665 and precalculus 740 in the sample, and 211 train rows overall.
4. **test intermediate_algebra 760 (major, NL-LATEX, in the problem).** "…bất đẳng thức`\n`\[\lambda xy + yz \le \frac{\sqrt{5}}{2}\]…". A literal two-character `\n` (an undefined TeX command) appears in the prompt. These rows cluster in batches: test ia 760–779 and train algebra 71–79.
5. **test prealgebra 765 (major, X-DECSEP).** "Thừa số lớn nhất của 54,321 mà nhỏ hơn 54,321 là gì?" In Vietnamese, `54,321` reads as the decimal 54.321. The gold `18,\!107` has the same problem. Compare test prealgebra 547, where `40,000` was correctly localised to `40.000`.
6. **test counting_and_probability 323 (major, NL-MEANING, in the solution).** English "5! divides 10!" became "5! chia hết cho 10!", which states that 10! divides 5!: the direction is reversed.
7. **test geometry 288 (major).** The solution translates congruent as "đồng dạng" (similar) and then concludes that the two triangles have equal area. A full-corpus scan of this pattern found **11 test problem statements** with `congruent` → `đồng dạng`. Four of them are counting questions ("How many **non-congruent** triangles…" → "không đồng dạng": test algebra 1153, counting_and_probability 294 and 336, geometry 301). Counting up to similarity rather than congruence can give a different answer, so these gold answers are not valid for the Vietnamese wording.
8. **train geometry 187 (major).** The solution has English "315° **counterclockwise** from (1,0)" → "theo **chiều kim đồng hồ**" (clockwise), which contradicts the boxed −√2/2. This is the stock solution template: 16 train solutions plus 1 problem have it, against 0 incorrect in test.
9. **train precalculus 26 (major, ANS-WRONG).** "Nhập hai nghiệm nhỏ nhất" asks for two answers. The solution boxes both 12 and 18, but `ground_truth = "18"`. This is inherited from the English `ground_truth` (it takes the last `\boxed` only); at least 9 train rows show the same truncation.
10. **train algebra 1339 (scan-confirmed, NL-UNANSW).** The `[asy]` figure was dropped from the Vietnamese problem, yet the problem still asks "Phần hình nào được tô bóng?" (what fraction of the figure is shaded). It is unanswerable as written.

## 4. Full-corpus pattern sizing (second pass; each pattern first seen while reading)

Rates are over all rows: train N = 7,501, test N = 5,000. The scripts are reproducible from the scratch logic described below: control-character and `\n` token diffs against English, brace/`$`/`\boxed` balance, `[asy]` diff, keyword probes.

| pattern | train rows | train % | test rows | test % | where |
|---|---|---|---|---|---|
| LaTeX escape corruption (`\r`→CR, `\t`→TAB, `\f`→FF inside `\right`, `\times`, `\text`, `\frac`…) | **211** (53 in problem, 11 in `ground_truth`) | 2.81 | 2 (1 problem) | 0.04 | 495 CR + 148 TAB + 1 FF occurrences in train |
| literal `\n` instead of newline | 7 | 0.09 | 22 (10 problem) | 0.44 | batch clusters (test ia 760–779, train algebra 71–79) |
| `\boxed{…` missing its `}` (`\boxed{\frac{5}{12}$`) | 8 sol + 2 problem | 0.13 | 0 | 0 | answer extraction fails |
| new unescaped `$` (`\$18.09` → `$18.09`) | 12 | 0.16 | 13 | 0.26 | breaks math mode |
| `[asy]` code edited by the translator | 21 (13 label-only/benign, 8 code edits incl. moved coordinates, 1 extra `)`) | 0.28 | 15 (3 label-only, 12 code edits, 1 extra `)`) | 0.30 | e.g. test precalculus 52 `A = (2,2)` → `A = (-5,3)` |
| `[asy]` dropped from or moved into the problem | 2 | 0.03 | 4 (2 dropped, 2 solution figures moved into the problem → possible answer leak) | 0.08 | train algebra 1339 unanswerable |
| `congruent` → `đồng dạng` (similar) | 20 rows (11 problems, 4 "non-congruent" counting questions) | 0.27 | 28 rows (**11 problems**, 4 counting) | 0.56 | ~20–37% of all "congruent" occurrences |
| `counterclockwise` → `chiều kim đồng hồ` | 16 (15 solutions, 1 problem) | 0.21 | 0 | 0 | train only |
| imperial→metric unit swap in the problem | 3 | 0.04 | 0 | 0 | algebra 387, 1000; counting_and_probability 444 |
| English thousands comma left in problem text (`54,321`) | 24 | 0.32 | 18 | 0.36 | about half of the English cases were localised to `.` and half were not |
| gold or last `\boxed` with a thousands comma (`1,\!296,\!540`) or a comma list (`0,1`) | 40 + 40 | 1.07 | 40 + 41 | 1.62 | ambiguous with the Vietnamese decimal comma; grader must handle |
| solution entirely untranslated (English) | 0 | 0 | 5 (algebra 101, 121; geometry 176; ia 307, 551) | 0.10 | problems all translated |
| problem untranslated | 0 | 0 | 0 | 0 | — |
| length-ratio outliers (VI/EN < 0.7 or > 2.5) | 36 | 0.48 | 18 | 0.36 | all explained by dropped/moved asy, misalignment or terse texts; **no truncation found** |
| last `\boxed` (VI) ≠ last `\boxed` (EN) after normalisation | 10 (7 broken braces, 3 misaligned) | 0.13 | **0** | 0 | 52 test diffs are only `\dfrac`/`\frac`, `\text{độ}` and similar |
| `ground_truth` ≠ English `ground_truth` | 84 raw: 11 control-char only, 69 equivalent (`\dfrac`→`\frac`, localised `\text{}`), 4 truly different (3 misaligned + 1 unit text) | 1.12 | n/a (no field) | — | |
| `level` ≠ level of the matching English row | 42 (all precalculus) | 0.56 | 0 | 0 | |
| any problem-side defect (union of the problem-text rows above) | 110 | 1.47 | **51** | **1.02** | |
| any LaTeX-transport defect (problem or solution) | 273 | 3.64 | 64 | 1.28 | |

### 4.1 The precalculus "stray" row (747 vs 746)

The surplus is not a single stray row. Content matching of all 747 Vietnamese rows against the English problems (token similarity plus `ground_truth`) shows:

- **Rows 0–477** are aligned. Rows **478/479 are swapped**.
- From **row 528** the file is a patchwork with **7 duplicated translations**. English rows 527, 599, 615, 639, 679, 719 and 743 are each translated twice, at Vietnamese rows (527,528), (599,600), (615,616), (639,640), (679,680), (719,720) and (743,744).
- **6 English rows are missing**: 583, 607, 631, 671, 711 and 735. The arithmetic is 746 + 7 − 6 = 747.
- **165 Vietnamese rows** carry an `index` that does not point to their English source.
- **42 rows** have a `level` copied from the neighbouring row, while the problem, solution and `ground_truth` in each row agree with one another.

The duplicates and gaps sit 24–40 rows apart, which looks like an off-by-one at batch boundaries in the translation job.

Two other spots are out of alignment:
- **counting_and_probability 355–357** in train hold a 3-cycle: row 355 carries English 357, 356 carries 355, and 357 carries 356.
- **No test row is out of alignment**: all 5,000 last `\boxed` answers match English position-for-position after normalisation.

### 4.2 Untranslated / truncated

No problem field is untranslated (0/12,501 have ≥ 4 English words and no Vietnamese diacritics). Five test solutions were left in English. No truncation was observed in reading (300/300 complete), and every length-ratio outlier traced to an asy or alignment cause.

### 4.3 Terminology (seen while reading, not sized individually)

Inconsistent or calqued terms:
- `common ratio` → "tỉ số chung" (should be "công bội"): 13 train and 13 test rows.
- `relatively prime` → "tương đối nguyên tố": 37/126 train rows. In other rows it is correctly "nguyên tố cùng nhau".
- `monic` → "đơn thức" (that word means *monomial*): 6 train and 5 test rows.
- `improper fraction` → "phân số không đúng quy cách" in one train row; elsewhere it is correctly "phân số không thực sự".
- `slope` → "độ dốc" instead of "hệ số góc".
- `kite` → "hình thoi" (rhombus).
- `palindrome` is kept as a loanword.

These are mostly minor, except `monic` and `congruent`, which name a different mathematical object.

### 4.4 Ground truth (train)

- After the control-character and normalisation fixes, train `ground_truth` equals the English `ground_truth` in 7,497/7,501 rows; the remaining 4 are the 3 misaligned rows and 1 unit-text case.
- **The English `ground_truth` itself is the last `\boxed` only.** For questions asking for several answers ("Enter all…", "two smallest…") it can hold just the last one; at least 9 rows were detected by a conservative regex (6 in train, 7 in test by the same probe on the English `\boxed`). This is a MATH-source issue that the Vietnamese files inherit.
- Seven train rows have `\boxed{…` without a closing brace, so any regex extractor returns nothing for them.
- Localisation put Vietnamese text inside some gold answers (`137.5\text{ độ}`, `\mbox{Thứ Bảy}`, `5\text{ chữ số}`). This is correct as Vietnamese, but a grader must strip `\text{}`.

## 5. Verdicts

### Vi-MATH train (7,501): **FIX — cheap; keep the content**

The language quality is high: 86.7% of items are fully clean and 95.3% [90.7, 97.7] are usable as-is. The "sloppy" impression is real but comes from pipeline hygiene, and almost all of it is mechanical:

1. **Escape restoration** (211 rows): map CR→`\r`, FF→`\f`, BS→`\b`, and TAB→`\t` **only when followed by a command stem** (`imes`, `ext`, `heta`, `an`, `o`, `frac`…). English has legitimate TABs in 39 train rows (asy indentation), so a blind replacement would be wrong. Also apply to `ground_truth`.
2. Replace literal `\n` with a newline (7 train rows; exclude legitimate `\neq`, `\nu`, `\not`…).
3. Close `\boxed{` braces (10 rows) and re-escape `$` amounts (12 rows).
4. Copy the English `[asy]` blocks back verbatim (23 rows).
5. **Rebuild precalculus** from the content map: drop the 7 duplicates, translate the 6 missing rows, reorder 478/479 and fix 42 `level` labels. Reorder counting_and_probability 355–357.
6. Targeted retranslation or review of semantic patterns (~60 rows): `congruent`→`đồng dạng` (20), `counterclockwise` (16), unit swaps (3), `monic` (6), plus multi-answer `ground_truth` (≥ 9; take the full answer list from the solution).

Cost: about 1 h scripting for items 1–5 and 2–3 h of manual review for item 6, with no paid API. The 6 missing precalculus rows can be translated with the free NIM generator.

### Vi-MATH test (5,000): **fit as an eval set after a small FIX (do not use raw)**

What makes it fit:
- Problems are complete and faithful: 0 untranslated, 0 truncated, 0 out of alignment. 96.7% [92.4, 98.6] of sampled items are usable as-is, and problem-side major defects are 1.3% [0.4, 4.7] in the sample and 51/5,000 = 1.0% in the census.

What must be fixed before scoring:
1. **There is no `ground_truth` field.** Attach gold from the English MATH test by position. All 5,000 rows are positionally aligned, and the Vietnamese and English last `\boxed` agree after normalisation. Prefer the English gold, plus the 7 multi-answer rows completed by hand, over Vietnamese `\boxed` text, which contains localised `\text{độ}` etc. Also add an `index` field.
2. **Repair the 51 problem-side defects.** Most important are the 11 `congruent`→`đồng dạng` problems; for the 4 "non-congruent" counting items the gold does not fit the Vietnamese wording. The rest are 10 literal-`\n` problems, 18 thousands-comma problems (localise `54,321` → `54.321` or `54 321`), 4 asy dropped/moved (2 moved figures may leak answers), 7 asy code edits, 3 `$` breaks and 1 CR.
3. Grader settings: `eval/math_verify.py` must treat `,\!` thousands separators and comma lists consistently with the Vietnamese decimal comma, and strip `\text{…}` units.

Cost: about 1 h scripting and about 2 h review of ~60 rows. After this the set is clean enough for headline numbers; residual minor terminology issues (~8% of items) do not change answers.

Caveat for the thesis: Vi-MATH test is a translation of the public MATH test set, which overlaps heavily with model pre-training and fine-tuning data. Report it as "translated-benchmark" accuracy, ideally alongside a MATH-500-subset number and the uncontaminated Vi-ExamMath benchmark (C1).

## 6. Notes

- `data/translation/MATH/train/geometry_vi.jsonl.bak` has the same size as `geometry_vi.jsonl`; it was not reviewed separately.
- No data file was modified. No paid API was used.
