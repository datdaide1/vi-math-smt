# R1 — Vi-GSM8K content review (train + test)

Reviewer tag `vi_gsm8k` · 2026-11-07 · protocol: `PROTOCOL.md` (seed 20261107, Wilson 95% CI)

## 1. Files reviewed

| File | Rows | English source (alignment) |
|---|---|---|
| `data/translation/gsm8k/gsm8k_train_vietnamese.jsonl` | 7,473 | `data/generate/gsm8k/train.jsonl`, by `index` (both 0..7472, verified) |
| `data/test/gsm8k/gsm8k_test_vietnamese.jsonl` (Vi-GSM8K test) | 1,319 | `data/generate/gsm8k/test.json`, by position (no `index` field; `ground_truth` matches EN in 1319/1319, and no row is shifted) |
| `data/test/gsm8k/gsm8k_test_vietnamese.json` | 1,319 | identical to the `.jsonl` row by row (1319/1319) |
| `data/finetune/mine/train_vietnamese_final.jsonl` | 14,870 | **What it is:** the merged SFT file, with 7,473 `type=Arithmetic` rows (GSM8K) and 7,397 MATH rows, plus an `smt` field. Its own `index` is 0..14869 and the rows are shuffled. All 7,473 GSM8K `problem`/`solution` strings are **byte-identical** to the translation file, so they were not re-reviewed. `ground_truth` there is an int. It also **inherits every corrupt row listed in §5** (for example, row 885 pairs the "Tracy … 99 books" problem and its x=9 solution with gt 210). |

## 2. Sample

```python
random.Random(20261107).sample(range(7473), 150)   # train, uniform, sorted below
random.Random(20261107).sample(range(1319), 150)   # test
```
- **train (n=150):** 21 32 66 67 77 94 184 217 280 393 509 520 529 537 584 595 614 618 656 826 830 871 1059 1069 1097 1112 1227 1232 1251 1320 1371 1428 1531 1568 1669 1687 1746 1753 1782 1811 1871 1890 1901 1932 1943 1997 2038 2110 2137 2155 2316 2332 2362 2395 2482 2551 2558 2578 2597 2713 2749 2859 2863 2944 3058 3115 3281 3370 3460 3461 3490 3493 3530 3533 3601 3616 3652 3661 3708 3763 3779 3817 3964 4211 4228 4231 4245 4252 4327 4509 4564 4625 4638 4642 4681 4686 4763 4836 4884 4890 4928 4980 5019 5038 5043 5044 5113 5131 5143 5155 5318 5333 5344 5349 5367 5413 5478 5524 5617 5626 5636 5713 5716 5766 5768 5864 5890 5967 5994 6088 6108 6208 6232 6290 6408 6449 6527 6605 6616 6623 6631 6648 6674 6699 6765 6792 6864 6922 7223 7288
- **test (n=150):** 5 8 16 19 23 26 27 40 46 54 70 71 97 98 127 128 130 132 134 146 148 153 154 164 171 197 206 207 217 243 262 264 267 274 278 294 306 308 312 324 330 342 357 382 389 392 417 421 436 438 445 452 467 472 475 478 483 484 485 499 509 527 530 534 535 538 560 579 583 590 598 609 620 629 630 637 639 644 647 649 678 687 696 714 715 723 736 764 776 778 820 829 842 855 858 865 872 873 882 883 900 904 913 915 917 927 940 944 954 991 999 1017 1052 1057 1060 1061 1063 1081 1084 1102 1103 1127 1141 1156 1159 1160 1170 1171 1174 1179 1190 1209 1221 1222 1228 1232 1245 1252 1254 1257 1259 1260 1261 1264 1268 1278 1282 1285 1288 1317

I read each of the 300 items in full, side by side with its English source: question and solution line by line, every number, unit, condition and question target, the `<<a*b=c>>` calculator annotations, `ground_truth` against EN `#### x`, fluency and leftover English. Annotations are in `annotations/vi_gsm8k_{train,test}.jsonl`. I added one extra field, `origin` (`translation` or `source`), which separates errors the translation introduced from GSM8K's own errors that were carried over.

## 3. Results (sample)

### Per code — Wilson 95% CI

| Code | train k/150 | train % [CI] | test k/150 | test % [CI] |
|---|---|---|---|---|
| NL-MEANING | 5 | 3.3 [1.4, 7.6] | 2 | 1.3 [0.4, 4.7] |
| NL-OMIT | 1 (minor) | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |
| NL-FLUENCY | 7 | 4.7 [2.3, 9.3] | 5 | 3.3 [1.4, 7.6] |
| NL-LEAK | 0 | 0.0 [0.0, 2.5] | 2 | 1.3 [0.4, 4.7] |
| X-NUMFMT (mixed VN/EN separators) | 2 | 1.3 [0.4, 4.7] | 3 | 2.0 [0.7, 5.7] |
| X-CULTURE (unlocalised US coins) | 1 | 0.7 [0.1, 3.7] | 0 | 0.0 [0.0, 2.5] |
| SOL-INCONS (all inherited from GSM8K EN) | 3 | 2.0 [0.7, 5.7] | 0 | 0.0 [0.0, 2.5] |
| SOL-WRONG (calculator annotation altered) | 0 | 0.0 [0.0, 2.5] | 1 | 0.7 [0.1, 3.7] |
| NL-LATEX, NL-UNANSW, NL-DUP, ANS-WRONG, ANS-FORMAT | 0 | 0.0 [0.0, 2.5] | 0 | 0.0 [0.0, 2.5] |

### Headline

| | train | test |
|---|---|---|
| **Meaning-changing translation error** (NL-MEANING/NL-OMIT, major or critical) | **5/150 = 3.3% [1.4, 7.6]** | **2/150 = 1.3% [0.4, 4.7]** |
| critical: VI text no longer supports the gold answer | 3 = 2.0% [0.7, 5.7] | 1 = 0.7% [0.1, 3.7] |
| major | 2 = 1.3% [0.4, 4.7] | 1 = 0.7% [0.1, 3.7] |
| minor only | 14 = 9.3% [5.6, 15.1] | 11 = 7.3% [4.1, 12.7] |
| clean | 131 = 87.3% [81.1, 91.7] | 137 = 91.3% [85.7, 94.9] |
| **Usable as-is** (clean or minor) | **96.7% [92.4, 98.6]** | **98.7% [95.3, 99.6]** |
| Fixable (major/critical; the EN source exists, so re-translation fixes it) | 3.3% | 1.3% |
| Discard | 0% | 0% |
| ground_truth ≠ EN gold | 0/150 | 0/150 |

Combined, the meaning-error rate over both samples is 7/300 = **2.3% [1.1, 4.7]**.

**Is it "ba chấm" (sloppy)?** Mostly no. Surface quality is high: the Vietnamese is fluent, units and numbers are carried over, `<<…>>` annotations are kept in every sampled item, and the translator even fixes several GSM8K typos (train 21 "90 snakes"→"90 chim", 2944 "120 hours"→"120 phút", 3817 "$.35"→0.75, test 1317 "cows have 2C legs"→"gà"). The real defect is a narrow, **systematic** class of MT errors on English comparatives and quantifier scope ("N of the remaining", "N more than k-times", "N more X than Y", "strictly"). On top of that, the train file has a small block of **misaligned or garbage rows** that only a full-corpus pass finds (§5).

## 4. Worst examples (quoted)

1. **train 2597**, critical: "6 more than twice as many" → "nhiều hơn **6 lần** so với gấp đôi", and "2 less than twice" → "ít hơn **2 lần** so với gấp đôi". The additive relations became multiplicative, so 27 cannot be derived.
2. **test 130**, critical: "then 10 **of the remaining** elves quit" → "sau đó **10 yêu tinh còn lại** bỏ việc" (= "the remaining 10 quit"). Under the VI text, nobody is left, but the gold is 30.
3. **train 520**, critical: "Four of the remaining are quartz" → "**Bốn lớp còn lại** là thạch anh" (= "the remaining four are quartz"). Same pattern as test 130.
4. **train 3661**, critical: "she got twelve more peaches than that" → "cô ấy có **nhiều hơn mười hai** quả đào" ("more than twelve peaches"). The +12 relation is lost.
5. **train 529**, major: "strictly cheaper per ride" → "rẻ hơn **đáng kể**" ("significantly cheaper"). The answer 26 = 25+1 depends on the strict inequality.
6. **train 6922**, major: "ten square yards" → "mười **thước vuông**". Colloquially "thước" means metre, so a VI reader reads 10 m².
7. **test 382**, major: "(Anna has) … 40 more gold pesos" → "và **nhiều hơn 40** đồng peso vàng". The comparison referent is gone, so it literally reads "more than 40".
8. **test 154**, minor SOL-WRONG: the annotation "4 x 2 = <<4*2=8>>" became "4 x 2 = **<<2*2=8>>**".
9. **test 445**, NL-LEAK: the VI solution contains "Con ngựa chạy 16 **miles**", "cần ăn 8 **bails of hay**".

## 5. Full-corpus sizing (second pass; every flag below was checked by hand)

| Check | train (7,473) | test (1,319) |
|---|---|---|
| `ground_truth` vs EN `#### x` and EN gt | 7473/7473 equal | 1319/1319 equal |
| Rows whose **VI Q/A is a different problem** from the gt/EN row | **8: idx 220–225** (a one-row shift; 220's question is a car+washing-machine hybrid), **2577** (a distorted copy of EN 2588 "doors", answer 70 vs gt 24), **3077** (duplicate of 3089 "Tracy 99 books", answer 9 vs gt 210). → `ANS-WRONG` critical | 0 |
| VI final `####` ≠ gt | 15 = the 8 above + 7 format-only (`#### 9.000`, `#### 285.000`, … VN thousands dot: 3311 3312 3327 4495 4798 5600 7332) | 0 (14 rows keep EN `#### 2,125` from the source; the gt field is clean) |
| An EN digit-number is missing or changed in the VI question (after normalising separators, number words and months; 44 / 4 flags checked by hand) | **12 genuine = 0.16% [0.09, 0.28]**: the 8 above plus 2345 (the sentence with the "previous" quantities, 30/30/10 curries, was dropped, so the problem is unanswerable), 3539 ("3½ times older" → "lớn hơn 3 tuổi rưỡi" = 3.5 years older), 5085 ("15 times the size" → "số cá mập nhiều hơn gấp đôi … 8 con", text from another problem), 7454 ("each of her 2 dachshunds": the 2 was dropped) | **0 [0, 0.29]** (all 4 flags benign) |
| "**N of the remaining**" (N an integer) | 3 occurrences, **3/3 mistranslated** as "N X còn lại" (520, 1801 "làm vỡ 15 quả trứng còn lại", 2911) | 1 occurrence, **1/1 mistranslated** (130) |
| Calque "nhiều/ít hơn N X so với Y" (reads as "more than N") | 202 = 2.7% | 38 = 2.9% |
| EN "N more/less than k-times" | 158 items; 35 use "…hơn k lần … N". Of 8 checked by hand: 2597 broken, 5727/6790 ambiguous, the rest standard VN phrasing | 26 items; 2 such renderings, both fine (470, 855) |
| Untranslated English sentence | 2 (3424, 6465: half of the question is in English) | 0 |
| Calculator annotations | identical in 7,446 rows. Of the 27 that differ: 6 are the misaligned rows; most others are extra annotations the VI added; 5 rows drop one | 2 differ (154 wrong, 693 extra) |
| Separator hygiene | EN thousands comma inside VI text: 114 (1.5%). VN dot-thousands: 241. VN decimal comma: 133. Dot-thousands and dot-decimal in the same item: 48 | 32 (2.4%) / 24 / 10 / 4 |
| Empty fields / truncation | 0 empty. Length ratio < 0.6: 3 (all explained: 224, 225, 5285) | 0 / 0 |
| Exact duplicates | 1 pair of VI questions (3077 = 3089). The EN source has 0 | 0 |
| Train–test VI question overlap | 0 | 0 |

The digit-level check catches only gross corruption. The relational errors in §4 keep every number intact, so the sample rate (3.3% train, 1.3% test) is the estimate to use. For train, 3.3% [1.4, 7.6] implies about 250 [105, 570] meaning-drifted rows.

## 6. Verdicts

### Train (`gsm8k_train_vietnamese.jsonl`, and the GSM8K part of `train_vietnamese_final.jsonl`): **FIX** (cheap and targeted)

- **Must fix:** 14 corrupt rows, each critical in an SFT set because it teaches a wrong answer. These are 220–225, 2577, 3077 (re-translate from EN and take gt from EN), and 2345, 3539, 5085, 7454, 1801, 2911. Also fix the 2 half-English rows (3424, 6465) and the 7 `#### 9.000` final lines. Rebuild `train_vietnamese_final.jsonl` from the corrected file, because it currently carries the same wrong gts.
- **Should fix:** the about 3% relational drift. Run an automatic solve-and-compare pass with a free model: solve each VI question and flag disagreements with the gold. Add pattern flags (§5 calques, "of the remaining", "k lần … N", "thước", "strictly"). Re-translate the flagged rows. Cost: one free-model pass over 7.5k items plus about 2–4 h of human review of flags.
- At about 3% label noise the data is still usable for SFT as-is. But the thesis claims "verified" augmentation, so the seeds should be verified as well.

### Test (`gsm8k_test_vietnamese.jsonl`, the planned co-primary eval): **FIT, after a one-off verification pass (FIX-light)**

- **Good:** gold labels are 100% faithful to GSM8K. There are no misaligned, duplicate, truncated or untranslated rows, and none of the 1,319 items has a lost or altered number. It is therefore far cleaner than the train file.
- **Risk:** 1.3% [0.4, 4.7] of items carry a translation that changes the problem (critical 0.7%). That is about 17 items [5, 62] where the VI text supports a different answer than the gold. This caps achievable accuracy at about 98.7%. It also adds label noise of the same order as the 1–3 pp effects the thesis expects between augmentation methods. Paired comparisons on the same items are mostly common-mode, but absolute VI-vs-EN gaps would be biased.
- **What fixing takes (about 0.5–1 day, free compute):**
  1. Automatically flag (a) items where a strong free model solving the VI question disagrees with the gold while solving the EN question correctly, and (b) the 38 calque, 26 additive-multiplicative and 1 "of-the-remaining" pattern hits. Expect about 60–150 flags.
  2. A bilingual human adjudicates and re-translates the flagged items at about 1–2 min each, so 2–4 h. Fix test 154's annotation. Optionally normalise the 14 `#### 2,125` final lines; this matters only if a grader parses `####`, since the `ground_truth` field is clean.
  3. Freeze the result as `Vi-GSM8K-test-v1.1` with a changelog. Report results on the verified version, and state the residual error bound, re-estimated with a fresh 150-item draw: it should fall below 1%.
- Until that pass is done, use the set but report it with the caveat "≈1.3% [0.4–4.7]% of items have translation-induced meaning drift". Graders must read the `ground_truth` field, not the `####` line.
