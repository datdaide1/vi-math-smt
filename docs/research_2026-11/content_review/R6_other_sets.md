# R6: the other Vietnamese sets (SVAMP-vi, ASDiv-vi, MetaMathQA-vi, Hungarian-vi, loigiaihay prototype)

Reviewer: Claude (R6), 2026-11-07. Protocol: `PROTOCOL.md`. Annotations: `annotations/{vi_svamp,vi_asdiv,metamath_vi,hungarian,vi_exam_proto}.jsonl`.
Scripts: `tools/r6_*.py`. No data file was modified and no paid API was called.

## Summary

| file | n read | any issue | major+critical | errors added by translation (major+crit) | verdict: eval set | verdict: training seed |
|---|---|---|---|---|---|---|
| `vi_svamp` (1,000) | 150 | 11.3% [7.2–17.4] | 2.0% [0.7–5.7] | 0.7% [0.1–3.7] | **KEEP** (after normalising Answer) | KEEP |
| `vi_asdiv` (2,305) | 150 | 7.3% [4.1–12.7] | 1.3% [0.4–4.7] | 0.7% [0.1–3.7] | **KEEP after a cheap fix pass** | KEEP |
| `metamath_vi` (395,000) | 150 | 50.7% [42.7–58.6] | **21.3% [15.5–28.6]** | **18.7% [13.2–25.7]** | n/a (training corpus) | **FIX or DISCARD**; do not use raw |
| `hungarian` (33) | 33 (all) | 33.3% [19.8–50.4] | 12.1% [4.8–27.3] | 3.0% [0.5–15.3] | **Not usable alone** (no gold answers) | no |
| `vi_exam_proto` (96 segments) | 96 (all) | 90.6% [83.1–95.0] | 85.4% [77.0–91.1] | n/a (native) | **Not yet**: 9/96 solver-ready | no (prototype) |

All intervals are Wilson 95%. "Usable as-is" means no code or only minor codes. "Fixable" means the worst code is major. "Discard" means the worst code is critical.

**Headline.** The suspicion that the content is sloppy is **wrong for SVAMP and ASDiv** and **right for MetaMathQA-vi**.
- In SVAMP and ASDiv, about 1 item in 150 has a translation error that changes the math. Most of the remaining codes are style issues (fluency, number format).
- In MetaMathQA-vi, about 1 row in 5 has a translation error that changes or breaks the problem. About 1 row in 11 is critical. The causes are systematic: "dozen" becomes "chục" (ten), "median" becomes "trung bình" (mean), LaTeX macros get translated (`\boxed` becomes `\đóng hộp`, `\angle` becomes `\góc`), Asymptote code gets machine-translated, and comparatives get reversed. These are sized across the full corpus below.

---

## 1. `vi_svamp`: data/translation_test_output/svamp/svamp-vi.jsonl

**Alignment.** The English source is `arkilpatel/SVAMP/SVAMP.json` from GitHub. It aligns by `ID` for 1000 of 1000 rows. Sampling used `random.Random(20261107).sample(range(1000),150)`, uniform. The index list is in the annotation file and in the appendix.

| code | rate (n=150) |
|---|---|
| NL-FLUENCY | 10 = 6.7% [3.7–11.8] (calque "Có bao nhiêu X … nhiều hơn …", "Baker" as a name) |
| NL-MEANING | 4 = 2.7% [1.0–6.7]: **3 of the 4 are silent fixes** of known SVAMP source bugs (641, 664, 702). Only 1 actually breaks the problem (595). |
| NL-UNANSW / ANS-WRONG (inherited from SVAMP) | 2 = 1.3% (781 critical, 924 major) |
| NL-LEAK | 1 = 0.7% |

- **Usable as-is: 98.0% [94.3–99.3].** Fixable: 1.3%. Discard: 0.7% (781, an inherited SVAMP error).
- **Worst examples**
  - **595:** "Jake has … 3 more apples than Steven" became "Jake có ít hơn Steven 13 quả đào **và nhiều hơn 3 quả táo**". This reads naturally as "more than 3 apples", so the gold answer 11 cannot be derived.
  - **781 (inherited):** "có 3 khách hàng. Sau khi một số người rời đi, vẫn còn 4". The gold is 5.
  - **641:** the question was changed from "chapters per day" to "mất bao nhiêu ngày để đọc xong mỗi chương". The VI version is now correct, but it is no longer parallel to the EN source.
- **Full-corpus sizing (n=1000)**
  - The leading number of `Answer` equals the EN gold in 1000 of 1000 rows.
  - The number multiset in Body+Question equals EN in 999 rows. The one exception is chal-774, where the EN source has a stray "3".
  - No leftover English stop-words.
  - The calque "Có bao nhiêu … nhiều hơn" appears in 42 rows.
  - The comparative anchor is dropped ("…nhưng ít hơn 13 lần gập bụng", which reads as "fewer than 13") in **8 rows**: 228, 440, 494, 595, 598, 744, 755, 769.
- **Answer format:** `Answer` is a Vietnamese string with a unit in 896 of 1000 rows (for example "51 đô la", "96 (quả đào đỏ)"). Grade on the leading number only.
- **Verdict: KEEP.**
  - As an eval set, take the numeric gold from EN `Answer`, or from the leading number.
  - Optionally fix the 8 anchor-dropped rows. This takes about 10 minutes by hand.
  - Know that 3 rows were silently "corrected" relative to SVAMP. This matters only if you report numbers as directly comparable to EN SVAMP.

## 2. `vi_asdiv`: data/translation_test_output/asdiv/asdiv-validation-vi.jsonl

**Alignment.** The English source is `chaochun/nlu-asdiv-dataset/dataset/ASDiv.xml` (2,305 problems). It aligns by position: `formula` is identical in 2303 of 2305 rows, and the 2 mismatches are whitespace only.

| code | rate (n=150) |
|---|---|
| X-NUMFMT | 6 = 4.0% [1.8–8.5] (English "$50.00"/"2.00 đô la" inside VI text) |
| NL-FLUENCY | 2 = 1.3% (ambiguous comparative "có ít hơn 8 đồng niken so với…") |
| NL-MEANING | 1 = 0.7% (625) |
| ANS-WRONG/NL-UNANSW (inherited) | 1 = 0.7% (435: asks for apples, gold counts sacks) |
| ANS-FORMAT (inherited) | 1 = 0.7% (1982: 51/9 given as "5.7") |
| NL-LEAK | 1 = 0.7% (1516: answer "25.12 (centimetres)") |

- **Usable as-is: 98.7% [95.3–99.6].** Fixable: 1.3%. Discard: 0%.
- **Worst examples**
  - **625:** "a second concert, which had 119 more people" became "có **nhiều hơn 119 người** tham dự" ("more than 119 people attended").
  - **1315 (full-corpus pass):** "If a number is subtracted from 32 …" became "**Nếu 32 được trừ từ một số**, kết quả là 23". The subtraction is reversed, and the answer is leaked into the problem.
  - **1434:** "takes 1 hour less" became "mất **ít hơn 1 giờ**" ("less than one hour").
  - **609/611/612:** EN "5,256" is kept as "5,256", which a Vietnamese reader parses as the decimal 5.256. 27 other rows localise correctly to "5.256".
- **Full-corpus sizing (n=2305)**
  - Number multiset differs from EN in 59 rows. All 59 were read: most are spelled-out numbers rendered as digits (benign). The real problems found among them are 1315 (reversed) and 609/611/612 (comma-thousands).
  - Comparative anchor dropped: 4 rows (625, 1002, 1395, 1434).
  - Comparative with an ambiguous "nhiều hơn N X so với Y" order: 16 rows.
  - English decimal point in text: 78 rows. "$" kept: 52 rows.
  - **2 rows have corrupted JSON keys** (`"    answer"` in row 1015, `"    solution_type"` in row 1947). This is an LLM output artefact and will break `row["answer"]`.
  - About 35 answers are non-numeric (Yes/No, names, "11:45 tối"). They were translated, so grading them needs string matching against VI strings.
- **Verdict: KEEP after a cheap fix pass (about 1 hour).**
  - Repair the 2 keys.
  - Normalise number formats (pick the VN convention consistently).
  - Hand-fix about 10 rows: 625, 1002, 1315, 1395, 1434, 609, 611, 612, 910.
  - Drop or keep-as-string the about 35 non-numeric golds.
  - Fit as eval set and as training seed.

## 3. `metamath_vi`: data/test/metaMATH/metaMathQA-395K_json_vi.jsonl (395,000 rows)

**Method.** The file was streamed, never loaded whole. A reservoir sample (Algorithm R) of 150 rows was drawn with `Random(20261107)`. Each row's `query_vi` and `response_vi` were read in full against `query_en`, with the EN response compared through a number/LaTeX alignment diff plus reading wherever the diff flagged something.

Sample type mix:

| type | n |
|---|---|
| GSM_Rephrased | 41 |
| MATH_AnsAug | 23 |
| GSM_AnsAug | 22 |
| MATH_Rephrased | 19 |
| GSM_SV | 17 |
| GSM_FOBAR | 13 |
| MATH_SV | 8 |
| MATH_FOBAR | 7 |

| code | rate (n=150) |
|---|---|
| NL-FLUENCY | 41 = 27.3% [20.8–35.0] |
| NL-LATEX | 18 = 12.0% [7.7–18.2] |
| NL-MEANING | 16 = 10.7% [6.7–16.6] |
| SOL-INCONS | 9 = 6.0% [3.2–11.0] |
| SOL-WRONG (mostly inherited MetaMath reasoning) | 5 = 3.3% |
| NL-LEAK | 3 = 2.0% |
| NL-OMIT | 2 = 1.3% |
| NL-UNANSW | 2 = 1.3% |
| ANS-WRONG | 1 = 0.7% |

- **Usable as-is: 78.7% [71.4–84.5].** Fixable: 12.7% [8.3–18.9]. **Discard: 8.7% [5.1–14.3].**
- **Errors added by translation (major or critical): 18.7% [13.2–25.7].** Translation NL-MEANING/OMIT alone: 11.3% [7.2–17.4]. Problems inherited from the EN source: 10.0%.
- The answer line (`Câu trả lời là: …`) matched the EN answer in every sampled row. **The damage is in the problem text and the reasoning, not in the final number.** For SFT this is the worst case: the model is trained to output the correct gold answer for a problem whose text, as rendered, no longer supports it.

**Worst examples** (row index in the 395k file):

| row | EN source | VI rendering | consequence |
|---|---|---|---|
| 26489 | "three cubic yards" | "ba **mét khối**" (cubic metres) | gold 81 is wrong for 3 m³ |
| 65187, 358748, 370620 | "four dozen", "3 dozen", "another dozen" | "bốn **chục**", "3 chục", "thêm chục quả" (chục = 10) | the responses still use 12 |
| 303780 | "median number of free throws" | "Số quả ném phạt thành công **trung bình**" (mean) | mean = 14.75, gold 15 is the median |
| 391354 | "Ben weighs 16 pounds **less** than Carl" | "Ben **nặng hơn** Carl 16 pound" (heavier) | |
| 364168 | "James has 3 more than 2 times" | "James có 3 chiếc, gấp 2 lần số Oreo mà Jordan có" | |
| 213083 | "takes 1/4 times more time" | "mất gấp 1/4 thời gian" | |
| 225263 | "Find $\sec 135^\circ$" | "Tìm **$\giây 135^\circ$**" | macro translated as "seconds" |
| 163063 | `\angle`, `\parallel`, asy `pair`/`dir`/`label` | `\góc`, `\song song`, asy "cặp A,B …; B = **thư mục**(-40); **nhãn**(…)" | |
| 314223 | asy `real ticklen=3`, `dot(…)`, `<`/`>` | "**cù thật**=3; không gian tích tắc thực=2 … dấu chấm((0,0)…)", `<`/`>` stripped | graph unrecoverable |
| 388563 | `\boxed{10}` | `\đóng hộp{10}` | breaks boxed-answer extraction |

**Full-corpus sizing** (streaming pass over 395,000 rows, `tools/r6_scan_metamath_*.py`). Precision was checked by reading flagged rows.

| pattern | rows | rate / precision |
|---|---|---|
| response newlines flattened to one line (EN has `\n`) | 385,797 / 393,353 | 98.1% (structural, every step run together) |
| "dozen" → "chục" (no "tá") | 1,582 / 3,598 dozen-rows | 44% of those; 10/10 true positives when read |
| "median" → not "trung vị" | 284 / 710 median-rows | 40% (7/12 true positives in a pre-fix read; final count case-insensitive) |
| `[asy]` code machine-translated (`nhãn(`, `dấu chấm(`, `thư mục(`) | 2,797 / 3,938 asy-rows | 71% |
| a LaTeX macro translated into Vietnamese (`\đóng hộp`, `\góc`, `\giây`, `\trái|`, `\phải|`, `\ tương đương`, `\ phân đoạn`, `\ sự cố` …) | 5,048 | 1.3% (snippets read: about 100% true) |
| `\boxed` present in EN response but lost in VI | 4,245 / 120,311 | 3.5% |
| macro split by an inserted space (`\ frac`, `\ sqrt`, `\ geq`) | 4,819 | 1.2% |
| `\begin{ pmatrix}`/translated env names | 1,382 | 0.35% |
| query `\times` → `\time` | 239 | |
| "2x" in EN response rendered as "gấp đôi" (double) in VI | 550 | 0.14% |
| question split by a misplaced "?" ("là bao nhiêu? tài khoản tiết kiệm…") | 7,831 | 2.0% |
| zero-width spaces in text | 7,595 | 1.9% |
| query numbers ≠ EN numbers (after VN-format normalisation) | 35,236 | 8.9%; 30 read: 8 major/critical [Wilson 14–44%], so **≈ 4.9k–15.5k rows** |
| answer line ≠ EN answer | 4,713 | 1.2%: 3,675 decimal comma only ("36,9"), 103 spacing, 935 other (thousands "32.000.000", "số 8", `\ phân đoạn{9}{4}`) |
| no VI answer line | 1,367 | 0.35% |
| empty VI field | 4 | |

**Extrapolation.** 18.7% [13.2–25.7] of 395k is **about 74k rows (52k–101k)** that need a fix or drop. By type in the sample, AnsAug/Rephrased-MATH rows carry most of the LaTeX damage and GSM rows carry most of the quantity/comparative damage.

**Verdict: FIX or DISCARD.** Do not use the file raw as a training seed. Options, cheapest first:
1. **Filter.** Drop rows that hit any regex above: translated macro, asy, dozen→chục, median→trung bình, query-number mismatch, `\boxed` lost, mid-question "?". This removes roughly 15–20% of rows and catches the critical patterns. Then restore newlines from `response_en`'s line structure. This is about 1 day of script work and needs no LLM. The residual major-error rate (comparative reversals, "each" dropped, wrong classifiers) stays at an estimated 5–10%, because those patterns are not regex-detectable.
2. **Re-translate** only `query` with a LaTeX-protected pipeline: mask `$…$`, `\[…\]`, and `[asy]…[/asy]` before translation, plus a glossary (dozen→tá, median→trung vị, remainder→số dư, cross product→tích có hướng, factor→ước). Keep EN `response` or regenerate it. This is the clean route, but it is a new translation run (free-compute only: NIM/local model).
3. For an augmentation-comparison thesis, the safest move is to **use the EN MetaMathQA plus a newly translated subset you control**, rather than this file.

## 4. `hungarian`: data/translation/hungarian_exam_vn.jsonl (33 rows, all read)

**Alignment.** The English source is HF `keirp/hungarian_national_hs_finals_exam` (33 rows, `Question` only). It aligns 1:1 by row.

The file contains **only `Question`. There are no answers or solutions**, and the HF source has none either: it was graded by humans with a rubric.

The translation is **good**: 0 NL-MEANING or NL-OMIT out of 33 (one-sided 97.5% CI upper bound 10.4%). The minor issues are:
- units "m\/s^2" (`\/` makes the slash vanish when rendered) and "Trial n:" left in English (rows 17, 18);
- English decimal points (17, 18, 19, 30);
- an English table header in row 28 ("| Year | Production volume of factory A (million Ft) |");
- "đường trung trực vuông góc" (redundant wording).

Inherited problems:
- 4 of 33 are not auto-gradable: draw a graph (14), prove (23, 26, 28).
- 3 depend on a missing figure (21, 22, 23).

**Verdict:** a fine small held-out *qualitative* probe. **Not usable as an auto-graded eval set** until someone writes gold answers. About 26 rows are short-answer and gradable with a key, which is roughly 1–2 h of work by someone who can solve them; the official Hungarian answer key could be used.

## 5. `vi_exam_proto`: _scratch/vi_exam/loigiaihay_exams.json (13 exams, 96 segments, all read)

**What it is.** This is native Vietnamese grade-10 entrance-exam data. A "segment" is a whole Câu or Bài, usually 2–4 sub-questions.

| kind | count |
|---|---|
| single closed answer | 28 |
| MCQ | 25 |
| multi-part, no proof | 21 |
| contains ≥1 proof | 22 |

**Live-page fidelity (11 items, raw HTML fetched).**
- Items checked: e0 Câu 1 and 2; e2 Câu I and V; e7 Bài IV; e10 Câu 5 and 7; e12 Câu 3, 13 and 14; e9 Câu 1.
- **The scraped problem text equals the live text, with difflib ratio 1.000 on 9 items and 0.999 on 1 item (two spaces).** LaTeX extraction (MathJax `\(..\)` source) is lossless.
- Gaps found in 3 Thanh Hóa items are on the source page itself:
  - Câu 3: "đồ thị hàm số  (a ≠ 0)", with y=ax² missing;
  - Câu 13: "theo đơn vị  (lấy π…)";
  - Câu 14: "vuông góc với  tại H".
- The 11th check, e9 Câu 1, is a segmentation check rather than a text diff: the live page has three different "Câu 1." blocks, and the scraped item is the last of them.

**Where it fails (rates over 96 segments):**

| code | rate | detail |
|---|---|---|
| X-NO-SOL | 24 = 25.0% [17.4–34.5] | Lào Cai's answer key is only an image (`dap-an-lao-cai.png`). Phú Thọ and Quảng Ninh publish no MCQ key. HN-2020 Bài IV has a **label-mismatch bug**: the solution header is "Câu IV" while the problem says "Bài IV", so the solution exists but was not joined. |
| X-SEG-COLLISION | entire exams affected | `segment()` keys a dict by "Câu N". In the 2026 three-part format (PHẦN I MCQ, PHẦN II true/false, PHẦN III short answer or essay) the numbering restarts, so later parts overwrite earlier ones. **Lào Cai: live page has 22 questions, 12 kept. Phú Thọ: 18 live, 12 kept. Quảng Ninh: at least 7 lost (inferred).** All PHẦN II true/false items are lost. |
| X-SCRAPE-JUNK | 17 = 17.7% | every last item carries "----- HẾT ----- Ad Tag - PC_Inread_ 1/3 bài2026_loigiaihay.com - ZMEDIA". Some carry the next section header ("PHẦN II. TỰ LUẬN (7,0 điểm)"). |
| X-FIGURE | 21 = 21.9% | image in the problem. One more (Lào Cai Câu 1) says "hình minh hoạ" but no image was captured. |
| final_answer_candidates quality | | on the 51 non-proof segments with a solution: ok 19, partial 14, wrong 10, none 8. The regex captures given data ("x = 9", "682 nghìn đồng"), domain conditions ("x > 0, x ≠ 1"), labels ("(d)", "AM", "\(P\)") and truncations ("9, ", "25 m" from "25 máy"). **Precision of the regex as an answer extractor is about 37%.** |

- **Solver-ready as-is** (single answer or MCQ with key, no figure, solution text present, extracted answer correct): **9/96 = 9.4% [5.0–16.9]**.
- With a correct answer extractor, 26/96 single or MCQ segments have a solution and no figure.
- Splitting multi-part segments into atomic sub-questions (pattern "1)", "a)") would add roughly 40–60 sub-questions, most numeric.

**Verdict: the method is sound, the prototype is not yet usable.** Native text and LaTeX are high-fidelity, so this is the right source for Vi-ExamMath. Fixes before scaling (about 1 day):
1. Segment by (PHẦN, Câu) and keep a list, not a dict.
2. Match solution headers loosely (Bài/Câu, Roman/Arabic, ignore "(TH)/(VD)").
3. Strip ad tags and section headers.
4. Split sub-questions "1) / a)" and tag proof sub-questions ("Chứng minh") as non-gradable.
5. Replace the regex with a "Vậy …" last-sentence plus LLM-free normaliser, or a short human pass. Hand-verify gold answers: the regex is about 37% precise.
6. Read MCQ keys from "Đáp án: X".
7. Record image-only answer keys as missing; do not OCR.

---

## Cross-cutting observations
- The quality gap tracks **source structure, not language**. SVAMP and ASDiv (short plain-text sentences, numbers only) survived translation almost intact. MetaMathQA (long LaTeX, asy, idioms such as dozen/median/2x) did not.
- **The final answers were never the problem in the translated sets.** The leading numeric answer matched EN in 1000 of 1000 SVAMP rows, in 150 of 150 sampled MetaMath rows, and in 98.8% of MetaMath rows corpus-wide (the remaining 1.2% are mostly decimal comma). The problems are in the text that the answer is supposed to follow from.
- Inherited source errors are real and should not be blamed on translation: SVAMP 2/150, ASDiv 2/150, and MetaMath 10% of rows (wrong reasoning in AnsAug or SV rows with correct final answers, broken FOBAR templates).

## Appendix: sampled indices
- vi_svamp (n=150, by line index; IDs in annotation file): 2,4,8,9,11,23,27,35,49,63,65,66,67,73,74,76,77,82,103,108,132,133,137,139,153,154,156,165,171,178,191,196,208,210,218,219,222,226,233,236,237,241,242,249,254,263,267,269,289,291,295,299,310,318,319,322,324,339,343,357,368,382,389,410,421,432,436,441,450,452,456,457,463,470,472,477,495,526,528,530,531,540,563,570,578,579,580,585,595,604,610,611,616,622,627,629,630,639,641,642,644,664,666,668,670,676,684,690,702,703,704,714,720,721,733,736,745,749,761,763,776,779,781,786,801,806,815,825,827,828,831,834,837,845,849,858,865,891,898,902,909,914,924,928,940,942,951,955,966,995
- vi_asdiv (n=150, line index): 10,16,33,38,47,52,54,61,80,92,95,108,140,142,195,196,254,256,260,264,268,271,292,297,307,309,328,343,394,413,415,435,450,486,524,529,534,548,556,588,613,616,625,648,660,685,686,689,714,765,778,784,834,843,863,873,876,891,897,905,935,945,950,956,966,969,971,995,998,1019,1048,1055,1060,1068,1070,1077,1121,1158,1166,1181,1195,1197,1217,1219,1241,1259,1260,1275,1279,1289,1294,1298,1326,1356,1374,1380,1392,1423,1429,1431,1446,1472,1516,1529,1552,1557,1577,1597,1602,1612,1640,1658,1685,1711,1716,1725,1730,1745,1746,1757,1765,1766,1800,1808,1826,1830,1835,1854,1881,1889,1908,1982,1998,2035,2100,2105,2114,2115,2121,2122,2126,2152,2160,2163,2168,2205,2206,2207,2254,2282
- metamath_vi (n=150, reservoir, line index): 3574,6807,15613,17258,21647,22462,26489,30606,33904,38611,39251,40447,41911,44807,49916,53617,55806,61342,65187,66733,67833,68156,79598,81282,84543,84704,87372,89044,91466,95641,98184,104876,105033,105136,105940,110085,112231,112286,112418,112526,112958,113141,118841,122857,126753,130146,130449,139094,140062,140581,143032,146043,147123,150789,151530,156256,160080,160105,160493,161293,161815,163063,165063,167328,174074,174209,175081,176616,186294,188062,188568,192657,198204,198841,201401,206490,209397,209729,213083,219091,222956,225263,225669,229935,230419,232272,236978,238978,239722,240165,240677,241910,247709,249486,254362,254984,261378,263006,267763,271487,272539,274754,278968,279094,282467,282735,282870,286510,287770,287802,292871,297789,303780,305740,314223,317514,318489,319261,320142,320694,320934,325292,329056,330846,333537,339027,340305,347303,349976,350123,350769,352315,352522,357147,358748,361300,362223,362707,364168,367058,368324,370620,373172,373842,379781,380399,386110,388563,391354,391404
- hungarian, vi_exam_proto: all rows.

Note for re-users: annotation `note` strings quote LaTeX with single backslashes. 8 MetaMath notes had their escapes repaired after writing: a shell heredoc had collapsed `\\` to `\`, and the repair mapped control characters back to `\t`, `\b` and similar.
