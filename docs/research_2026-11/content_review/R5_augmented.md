# R5: content review of the augmented training set (`aug`)

**File:** `data/finetune/augmented/train_augmented_final.jsonl` has 34,620 rows: 14,870 original and 19,750 augmented.
**Pipeline:** `src/mutation_informalize/` (phase 2 is `mutation_engine.py`, phase 3 is `phase3_informalize.py`, using gpt-oss-120b through NVIDIA NIM).
**Reviewer / date:** content-review agent, 2026-11-07. I read every sampled item in full next to its source problem, as PROTOCOL.md requires.
**Annotations:** `annotations/aug.jsonl` has 250 rows: 200 augmented and 50 original controls. Each row has codes, severity, a verdict, a novelty class, NL↔SMT consistency, the Z3 value and a quoted note.

---

## 0. Headline: what the augmented problems say

**The augmentation does not create new problems. Each augmented item is its source problem with an arithmetic tail bolted onto the source's answer, or an unchanged copy, or a gutted version.** The code guarantees this, and reading confirmed it:

* `mutation_engine.py` has exactly three operators. Not one of them touches a premise, a number in the statement, or a relation:
  * `mutate_structure`: answer → `answer + 2·h_NNNNN`, where h is a random 0.50–5.00.
  * `mutate_difficulty`: answer → `answer·(answer+1)`.
  * `mutate_expression`: answer → one of `(answer+k)·k`, `answer·k+k`, `answer·(k+k)` or `(answer+k)−k`. The last one is the identity.

  The informaliser is told to *"Sửa đổi BÀI_TOÁN_GỐC sao cho khớp với SMT_CONSTRAINTS mới"*. All it can do is restate the source problem and then verbalise the tail.

| Semantic measure (200 augmented items read side by side with their source) | n/200 | % [Wilson 95% CI] |
|---|---|---|
| **Genuinely new scenario or reasoning** | **0** | **0.0% [0.0–1.9]** |
| `NL-TACKED`: source story plus a bolted-on tail ("…rồi nhân với 2.19 và cộng thêm 2.19") | 168 | **84.0% [78.3–88.4]** |
| `NL-NEARDUP`: same problem, answer unchanged (identity mutation), paraphrase only | 26 | 13.0% [9.0–18.4] |
| `X-DEGRADED`: the source reasoning is removed. The problem is pre-solved, or replaced by a drill such as "Cho a=3. Tính a×¼×(2+2)" | 52 | 26.0% [20.4–32.5] |
| `NL-ABSURD`: real-world nonsense (fractional counts, unit clashes, impossible figures) | 31 | 15.5% [11.1–21.2] |
| `NL-LEAK`: h_NNNNN, "answer", SMT text, prompt tags, or the answer itself in the statement | 116 | 58.0% [51.1–64.6] |
| **A Vietnamese teacher would write it** (natural, coherent, no tail, no leak) | 21 | **10.5% [7.0–15.5]** |
| …and it is not just a copy or a gutted version of its source | **0** | **0.0% [0.0–1.9]** |
| Formally acceptable as a training problem (well-posed, correct, no artefacts; contrived is allowed) | 75 | 37.5% [31.1–44.4] |
| **Original control (50 MATH items): usable as-is** | 49/50 | **98.0% [89.5–99.6]** |

Quotable source → augmented pairs that show the bolted-on tail (full lists are in §5):

* **idx 17281**, source: "…có bao nhiêu học sinh nam không mang máy tính đến lớp?" → augmented: "…có bao nhiêu học sinh nam không mang máy tính đến lớp? **Sau khi lấy kết quả này nhân với 2.19 và cộng thêm 2.19**, giá trị thu được là bao nhiêu?" (gold 17.52 "students").
* **idx 27801**, source: "…số con trung bình trong các gia đình có con là bao nhiêu?" → augmented: "…trung bình bao nhiêu trẻ em mỗi gia đình có con, **sau đó cộng thêm 0,92 cho mỗi gia đình không có con** (tức là cộng 2 × 0,92)" (gold 4.34 children).
* **idx 17477**, source: "anh ấy đã vay bao nhiêu đô la?" ($132) → augmented: "…Jason được thêm một khoản thưởng cố định $3.70 … được nhân với 3.70 (ví dụ: do quy đổi sang một đơn vị tiền tệ khác). Hỏi **Jason đã vay bao nhiêu đô la ban đầu, tức là giá trị cuối cùng sau các bước trên?**" (gold $502.09).
* **idx 19151** (outside the sample): storm damage of 20 million USD → "tính (số đô la Mỹ) × (số đô la Mỹ + 1)" = 400,000,020,000,000.
* **idx 14214** (outside the sample): choose 2 co-presidents from 15 → "Theo một quy tắc đặc biệt, số cách chọn … = số thành viên nhân với (số thành viên trừ đi 2) rồi chia cho 2 … nhân kết quả với 1 và với tổng của 4.5 và 4.5" (gold 877.5 "ways").

**Full-corpus sizing (19,750 augmented items; regex and Z3 passes sized after reading):**

| Pattern | Augmented | Originals (7,397 MATH / 7,473 GSM8K) |
|---|---|---|
| Mutation is a post-composition of the source answer (by construction, from the SMT) | **100%** (structure 29.9%, difficulty 30.5%, expression 39.6%) | n/a |
| …of which the identity `(e+k)−k` (answer unchanged, a near-duplicate) | **3,633 = 18.4% [17.9–18.9]** | n/a |
| The random tail constant (k or h) appears verbatim in the problem text | 9,560/13,721 = **69.7% [68.9–70.4]** | n/a |
| Any tail phrase ("cộng thêm", "nhân với…", "k(k+1)"/"số liền sau"/"lớn hơn một đơn vị", "hai lần/2h", "(c+c)", "sau đó tính…") | **48.8% [48.1–49.5]** | ≤1.4% per phrase (k(k+1)-like 104/7,397; "cộng thêm" 14/7,397) |
| Doubled-decimal wrapper such as "(1.59+1.59)" | 6.1% | 0 / 0 |
| Hard leak (h_/d_ name, "answer" as a name, SMT/"ràng buộc", `<BÀI_…>` tag, answer announced, `\boxed` in problem) | **41.8% [41.1–42.5]** | 0 |
| — "answer" used as a variable name | 32.2% | 0 |
| — h_NNNNN or d_NNNNN | 13.0% | 0 |
| — leaked prompt tag `<BÀI_TỜÁN>` etc. | 4.8% | 0 |
| Gold answer printed in the problem (non-trivial values, absent from the source) | 6.1% | n/a |
| Source's answer (an integer ≥10) handed over in the augmented statement | 5.5% | n/a |
| Source is a counting question with an integer answer, but the augmented gold is **non-integer** | **902/1,810 = 49.8% [47.5–52.1]** | n/a |
| "Ugly" gold (terminating decimal with ≥2 places such as 1219/25, more than 14 characters, or a Z3 "?" marker) | 16.2% (sample reading: 50.5%) | 101/7,397 (1.4%) |
| Augmented text length ÷ source length (median) | **1.93×** | n/a |
| Token Jaccard to source (mean) | 0.356, against 0.160 for random same-subject pairs | n/a |

The lexical similarity is only moderate (Jaccard 0.36, character similarity 0.38). The generator pads with scaffolding, SMT names and verbalised tails, which roughly doubles the length. The *mathematical* similarity is total: every augmented SMT keeps all of the source's assertions and only rewrites `answer`. Lexical overlap therefore under-states the duplication.

**Breakdowns:**

*By the real strategy, inferred from the SMT.* The `mutation_strategy` field is useless: it holds `mutation_id` (a 0–4 counter), and every rate is identical across values 0–4. Sample of 200:

| Strategy | n | Teacher would write | TACKED | ABSURD | NEARDUP | LEAK | DEGRADED | Discard |
|---|---|---|---|---|---|---|---|---|
| difficulty `a(a+1)` | 59 | 1 | 58 | 9 | 0 | 35 | 9 | 23 |
| structure `a+2h` | 61 | **0** | 61 | 9 | 0 | **45** | 19 | 27 |
| expression (all forms) | 80 | 20 | 49 | 13 | 26 | 36 | 24 | 29 |
| …of which identity | 33 | (all 20 "write" items are here or degraded) | 0 | 0 | 26 | 9 | 7 | 9 |

*Full-corpus phrase and leak rates by strategy:* `(e+k)·k` puts its constant in the text 98% of the time and has an ugly gold 73% of the time. `structure` has a hard leak 57% of the time and names h_NNNNN directly. The identity form has a tail in only 5% and a leak in 22%, but it is a duplicate by definition.

*By subject.* "Would write" counts over the sample: algebra 1/42, counting and probability 5/20, geometry 3/21, intermediate algebra 6/36, number theory **0/23**, prealgebra 4/37, precalculus 2/21. NL-ABSURD concentrates in geometry (10/21) and prealgebra (10/37). Lengths, areas, angles and counts of people get `x(x+1)` or `·k+k` applied to them. Number theory has the highest discard rate (17/23): base-b answers such as "2677₉" get reread as decimals, and source answers get given away.

*GSM8K-type vs MATH-type.* **The augmented set contains no GSM8K-type items.** `config.SKIP_SUBJECTS = {"arithmetic"}` excluded all 7,473 GSM8K originals from mutation. Inside MATH, the story-worded subset (a heuristic on the source: ≤2 math delimiters, >120 characters; 3,886 items) behaves the same as the symbolic subset: 49% tail phrases in both. In the sample, story-type items are much more often absurd (14/36 against 17/164), because a tail on a story quantity produces "17.52 học sinh" or "4.34 trẻ em".

**Siblings make the duplication worse.** 5,670 source problems carry 19,750 children, and 2,256 sources have the maximum of 5. Of 30,571 sibling pairs, 3,684 have identical SMT apart from the random variable name, and 1,678 have ≥0.8 text similarity. Examples are idx 31344 and 654, both "(10·2677+9)(10·2677+10)", and idx 31915 and 27225, both "base=0, idx=628".

---

## 1. Sample and method

* Seed `random.Random(20261107)`, sampled without replacement.
* **Augmented:** n = 200, stratified proportionally over subject × the *real* strategy (21 strata, each ≥5). Allocation: algebra 13/16/13 (difficulty/expression/structure), counting and probability 6/8/6, geometry 6/8/7, intermediate algebra 10/15/11, number theory 7/9/7, prealgebra 11/15/11, precalculus 6/9/6.
* **Control:** n = 50 originals from the 7 MATH subjects only, allocated proportionally (arithmetic is never augmented).
* **Linking each item to its source.** Phase 4 shuffles and re-indexes, and it drops `source_index`. I re-joined each final row to its phase-3 row by (problem, smt_code), with a 100% match. From there I followed phase-3 `index` → phase-2 row → `source_index` → phase-1 or original problem (19,750/19,750 linked).
* **Reading and solving.** I read each item as a student would and compared it with its source. Z3 ran on every sampled program (10 s, through `vi_math_verified_aug.verify.z3_utils`). Then I ran Z3 again on all 19,750 programs, together with a uniqueness test that blocks the found answer.
* **Codes.** PROTOCOL codes, plus four additions:
  * `X-DEGRADED`: the source reasoning is removed or pre-solved.
  * `X-UGLY`: a random-decimal answer no human author would choose.
  * `X-MULTIASK`: several quantities are asked for, but there is one gold.
  * `NL-TACKED` replaces my first-pass label X-COSMETIC.
* **Verdicts.**
  * *usable*: natural and correct.
  * *usable-contrived*: correct and well-posed, but contrived (tail or near-duplicate).
  * *fixable*: a surface artefact such as a tag or a variable name that renaming or deleting would fix.
  * *discard*: meaningless, false, answer given away, or gutted.

Sampled indices (augmented, in order): 30124, 20245, 23662, 1634, 8082, 32730, 16572, 12659, 11319, 3777, 29650, 16510, 3304, 7376, 9858, 18540, 3915, 12864, 8825, 15315, 8126, 24992, 10835, 13183, 31288, 8430, 19393, 28504, 29672, 32823, 1436, 3974, 27896, 33929, 28166, 12271, 12461, 3258, 33912, 22992, 28192, 2635, 26005, 29408, 8502, 7495, 7175, 4185, 21223, 30383, 18649, 33904, 28351, 24442, 3765, 21378, 21472, 15753, 7591, 5117, 2359, 21475, 26525, 12280, 6781, 893, 14475, 19390, 14235, 30670, 15826, 30091, 24607, 20018, 30139, 1169, 16308, 20047, 15640, 504, 165, 13892, 10637, 24828, 28579, 7987, 23178, 5599, 15901, 28552, 14813, 24492, 4450, 12358, 10681, 21044, 89, 4597, 30854, 356, 28387, 14408, 29828, 16007, 12576, 28563, 2947, 19646, 28114, 22502, 13371, 4825, 506, 13774, 27734, 20568, 7704, 20083, 34147, 31344, 654, 16894, 17710, 33293, 33511, 2006, 17477, 27412, 20285, 10449, 30610, 28104, 24879, 34043, 25180, 2748, 680, 28184, 1186, 10399, 26934, 7348, 34302, 6104, 24228, 28958, 17359, 18481, 14281, 3848, 27103, 5157, 26320, 12012, 26954, 10281, 7406, 17915, 31809, 17281, 24910, 8230, 13335, 26506, 10612, 16518, 814, 28548, 1303, 25432, 10837, 20028, 15962, 28282, 6928, 27801, 25673, 25895, 34133, 21763, 11842, 20638, 16257, 13843, 19210, 7494, 31219, 29480, 2993, 30637, 15613, 34512, 15995, 31915, 19562, 24592, 27225, 14379, 6181, 15610.
Control: 5243, 16605, 26351, 1511, 32114, 26051, 16364, 1962, 1980, 12711, 25138, 11561, 275, 8101, 10201, 9317, 29924, 19306, 28966, 10233, 26420, 30935, 12098, 22613, 14365, 18499, 4315, 25535, 31382, 15486, 8806, 8615, 9029, 3734, 16751, 7581, 1711, 3588, 27268, 27200, 1259, 4290, 4853, 6035, 27065, 4586, 29006, 32544, 26232, 30484.

---

## 2. Error rates per code (Wilson 95% CI)

| Code | Augmented (n=200) | Original control (n=50) |
|---|---|---|
| NL-TACKED | 168 = 84.0% [78.3–88.4] | 0 = 0% [0–7.1] |
| NL-LEAK | 116 = 58.0% [51.1–64.6] | 0 |
| X-UGLY | 101 = 50.5% [43.6–57.4] | 0 |
| X-DEGRADED | 52 = 26.0% [20.4–32.5] | 0 |
| SMT-HARDCODE | 33 = 16.5% [12.0–22.3] | 6 = 12.0% [5.6–23.8] |
| NL-ABSURD | 31 = 15.5% [11.1–21.2] | 0 |
| NL-NEARDUP | 26 = 13.0% [9.0–18.4] | n/a |
| NL-MEANING | 18 = 9.0% [5.8–13.8] | 0 |
| X-MULTIASK | 15 = 7.5% [4.6–12.0] | 0 |
| SMT-UNFAITHFUL | 14 = 7.0% [4.2–11.4] | 0 |
| ANS-FORMAT | 11 = 5.5% [3.1–9.6] | 7 = 14.0% [7.0–26.2] (units, "\text{Thứ Bảy}", tuples) |
| NL-UNANSW | 7 = 3.5% [1.7–7.0] | 0 |
| ANS-WRONG | 4 = 2.0% [0.8–5.0] | 0 |
| NL-FLUENCY | 4 = 2.0% [0.8–5.0] | 2 = 4.0% [1.1–13.5] |
| NL-OMIT / NL-DUP (siblings) | 2 / 2 = 1.0% each | 0 |
| SMT-MISMATCH | 1 = 0.5% [0.1–2.8] | 5 = 10.0% (QF_LRA cannot hold vectors, √, text) |
| SMT-ERROR | 0 | 3 = 6.0% |
| NL-LATEX | 0 | 2 = 4.0% (the `\right`→"ight" escape bug; idx 26232 has it inside the problem) |
| **Severity critical** | 72 = 36.0% [29.7–42.9] | 0 = 0% [0–7.1] |

**Verdicts:**

| | usable | usable-contrived | fixable | discard |
|---|---|---|---|---|
| Augmented | **0 = 0.0% [0–1.9]** | 75 = 37.5% [31.1–44.4] | 46 = 23.0% [17.7–29.3] | **79 = 39.5% [33.0–46.4]** |
| Original | **49 = 98.0% [89.5–99.6]** | 0 | 1 = 2.0% | 0 |

The control originals are good Vietnamese translations of MATH problems. Their only defects sit on the SMT and answer-format side, not in the text the student reads.

---

## 3. Does symbolic verification make the NL problem correct? (C3)

**No. The verification guarantees that a number agrees, not that the problem is valid.**

* **NL↔SMT answer consistency is high.** It is 195/200 = **97.5% [94.3–98.9]**, and gold equals the Z3 value in **97.7% [97.5–97.9]** of the full corpus (19,290/19,750; 8 programs fail to parse). The answer is unique in 98.1% (363 programs are non-unique).
* **Yet that sits alongside poor problems:** 37.5% formally acceptable, 10.5% natural, 0% new. **Answer consistency and problem validity are separate properties.** The pipeline measures the first one and reports it as a "Consistency Rate" (the plots in `phase3_informalize.plot_evaluation_metrics`).
* **The verification loop itself creates leaks.** On failure, `REFLECTION_PROMPT` gets the feedback *"Đáp án chuẩn từ Z3 là: {answer}"*. The generator then writes the answer into the problem so the blind solver will match. Examples:
  * idx 12659: "Kết quả cuối cùng phải là một phân số (ví dụ 783/4)", where the gold is 783/4.
  * idx 7175: a "Gợi ý chi tiết" ending "= 185/1024".
  * idx 13892: "Kết quả mong muốn: 9727/25".
  * idx 26005: the full solution "= 14 × 15 = 210" inside the problem.

  Without any reflection it also happens spontaneously:
  * "(Đáp số là 10100.)", idx 28552.
  * "Bạn tính được x = 5256", idx 17359.
  * "(Theo các ràng buộc SMT, đáp án phải bằng N và N = 144 …)", idx 20285.

  In the full corpus, 6.1% print a non-trivial gold value in the statement.
* **The validator is lax.** `validate_generation` accepts a 0.5% relative error, splits the solver's answer on commas, and tries every fragment. Results:
  * idx 29408: gold **"0"**, but the true answer is 100000·100001 = 10,000,100,000. The fragment "000" of "10\,000\,100\,000" matched 0. Full corpus: **189** items have gold "0" while the SMT also admits another value, so the program is non-unique, and in 55 of them the stored solver answer is non-zero.
  * idx 15640: the text asks for surface area (1440); the gold is 1441.2. Accepted at 0.08% relative error.
  * idx 28192: "làm tròn đến hàng phần mười" (7.1) against a gold of 7.06.
  * idx 20638: the text says "kết quả chính là answer=20", while the true count is 4. The solution concedes "Mặc dù số lượng góc thực tế là 4, theo công thức đã cho 'answer' bằng 20".
* **Base SMTs that were wrong get faithfully verbalised.**
  * idx 21472: "a = 100 điểm, b = 120 điểm … số mặt phẳng tối đa là sum1 = 220". The base SMT faked C(12,3) = 220 as 100+120.
  * idx 28504: "(x·y − 31)" stands in for x²−y², because the base SMT faked 32 as xy − 31.
  * idx 2993: the answer list "2,4" was parsed as 24, which yields the unrelated drill "y=2x, z=3y, z−x=20, tìm z".

  Z3 "verifies" all of these.
* **Gold format against the project grader.**
  * 63.6% of augmented golds are Z3 plain `a/b` fractions (12,570).
  * 1,129 items (**5.7%**) have a stored solver answer that `eval/math_verify.is_correct` grades as wrong against the gold.
  * Side finding for whoever owns `eval/math_verify.py`: `_collapse_number_seps` reads any decimal with exactly 3 places as a thousands group. `normalize_answer("0.625")` returns "0625", so `is_correct("0.625","5/8")` is **False**. This affects 240 augmented golds and 164 of the 1,129 failures, and it will mis-score model outputs such as 2.525.

---

## 4. Secondary checks (solution and answer)

The stored solutions are the blind solver's outputs and were kept only when they matched Z3. In my reading they solve the problem *as written* and reach the gold. The exceptions are the ANS-WRONG items (29408, 28192, 15640, 20638), plus multi-ask items whose solutions report several values (X-MULTIASK, 7.5%). I found no systematic arithmetic errors in the solutions. The defect is upstream, in what the problem asks. Answers are "ugly" in 50.5% of the sample: 48.76, 601.7676, 1,467,023.9121, and gold strings such as `14142135623730951/10000000000000000` (√2 as a float) or `5.8412371462?` (with Z3's approximation marker).

---

## 5. Vivid source → augmented examples by failure mode (for C3)

### NL-TACKED: the source story plus an arithmetic tail
| idx | Source asks | Augmented asks |
|---|---|---|
| 17281 | boys without calculators (7) | "…Sau khi lấy kết quả này **nhân với 2.19 và cộng thêm 2.19**" (17.52) |
| 31809 | perimeter of one rectangle (90) | "cộng thêm 2.22 đơn vị vào chu vi … rồi nhân kết quả đó với 2.22" (204.7284) |
| 28548 | interior angle of a regular pentagon (108°) | "sau đó cộng thêm 4.16 độ và nhân kết quả với 4.16. Giá trị cuối cùng là bao nhiêu độ?" (466.5856°) |
| 2006 | 31₈ in decimal (25) | "hãy cộng 1 vào nó và nhân lại với giá trị thập phân ban đầu" (650) |
| 34302 | mean of 2/5 and 4/7 | "Tính tích của trung bình cộng … với một đơn vị hơn trung bình đó" (884/1225) |
| 3848 | largest prime factor of 1337 (191) | "tính giá trị của tích p×(p+1)" (36672) |
| 26954 | clock angle at 9:40 (50°) | "(góc …)×(1.31+1.31)" (131) |
| 30610 | smallest prime with digit sum 19 (199) | "Tìm giá trị của 3.38 lần số nguyên tố …" (672.62) |
| 13843 | det = −45 | "tích của định thức đó với một số lớn hơn nó một đơn vị" (1980) |
| 12864 | b in f⁻¹ (2) | "(b+2.43)×2.43" (10.7649) |

### NL-ABSURD: real-world nonsense
| idx | Quote |
|---|---|
| 27801 | "cộng thêm 0,92 cho mỗi gia đình không có con" → 4.34 children |
| 17477 | loan = "giá trị cuối cùng" after a $3.70 bonus and "nhân với 3.70 (quy đổi sang một đơn vị tiền tệ khác)" |
| 33904 | 9 people at a round table → "(factor×7!+3.6)×3.6" = 145164.96 arrangements |
| 24910 | number of chair arrangements → "(comb + 3.4) × 3.4" = 164.56 |
| 21472 | "a=100 điểm và b=120 điểm … số mặt phẳng tối đa … là sum1" (planes = 220 = number of points) |
| 13892 | a triangle with sides 100, 200, 84 (impossible) "cộng thêm hai inch" |
| 16308 | "BC = c = 0 … tam giác BCH trở thành một đoạn thẳng" |
| 20047 | "tổng của thể tích phần chóp nhỏ … và hai lần độ dài đoạn thẳng h_{85098}" (volume + length) |
| 1169 | "khối lập phương có cạnh s thỏa s²=3 … đoạn thẳng … có độ dài 2.68 … Chiều cao = (2s/3+2.68)×2.68" |
| 1303 | "một khu phụ có diện tích 3.43 acre và khu này được tính vào tổng diện tích … hai lần" |
| 14214* | "số cách chọn … = số thành viên nhân với (số thành viên trừ đi 2) rồi chia cho 2" → 877.5 ways |

\* outside the sample.

### NL-NEARDUP: unchanged problem, identity mutation `(e+k)−k`, 18.4% of the corpus
| idx | Quote |
|---|---|
| 30383 | "Chỉ sử dụng các chữ số 7, 8 và 9, có bao nhiêu số nguyên dương có bảy chữ số là số palindrome?" (verbatim; 81) |
| 2947 | ellipse-square problem copied with its [asy] code (8) |
| 89 | "Cho z = (−√3+i)/2. Tính giá trị của z⁶." (verbatim; −1) |
| 24607 | the AIME circle problem verbatim (254) |
| 30139 | the prism/cube problem verbatim (384) |
| 18649, 28351, 3765, 26506 | probability problems paraphrased, answer unchanged |
| 31344 / 654 | two "different" augmentations of the same source with identical SMT and answer 717141620 |

### X-DEGRADED: the source reasoning removed, a drill left
| idx | Source → augmented |
|---|---|
| 29828 | Σ1/(n(n+2)) → "Cho a = 3. Tính answer = a × ¼ × (2+2)" |
| 10281 | √(36×√16) → "Cho biến d = 1. Tính (d × 12) × (1.02 + 1.02)" |
| 27412 | "how many last digits can a square have?" → "sáu số thực d_i = 1.0 … (d₀+…+d₅)×(3.19+3.19)" |
| 7406 | (9×10⁸)÷(3×10³) → "Cho c = 9÷3. Tính (c×100000+4.89)×4.89" |
| 24992 | a hard radical equation → "Tìm giá trị lớn nhất của a thỏa mãn a² = 2" |
| 7704 | range of h(x) → "Giá trị x được cố định bằng −7" (the excluded point) "… result = 2x−6+2k" |
| 24879 | band-formation congruence → "N = 6 + 72 × t … Biết rằng t = 2" |
| 10399 | count n with n+S(n)+S(S(n))=2007 → "a=b=c=d=1, a²−2a−1=0" (contradictory) "… answer = 4 + 2·h_{88064}" |
| 19646 | minimal-degree polynomial → "total = 1000 số thực, có perfect = 30 số nguyên hoàn hảo …" |
| 29480 | area under arccos(sin x) → "hình vuông có cạnh bằng π (với π = 3.141592653589793) … (area+1.78)×1.78" |

### NL-LEAK (SMT artefacts)
| idx | Quote |
|---|---|
| 1436 | "answer = p + 2·h_{90478}, trong đó h_{90478} = 1.03" |
| 33929 | "(Đáp số cần đưa ra là giá trị của biến `answer` trong các ràng buộc SMT.)" |
| 3258 | "(Lưu ý: các điều kiện trên chính xác theo các ràng buộc SMT đã cho.)" |
| 9858 | "sum\_ab, prod\_ab, half\_prod, abs\_answer" asked as five quantities |
| 1634 | "Đặt một hằng số phụ dummy = 0 … tính answer = (2+dummy)×d" |
| 7376 / 29408 | "<BÀI_TỜÁN> … </BÀI_TỜÁN>" left in the problem |
| 4825 | "dummy = 1, scale = 108, h_{42713} = 3 … (Ở đây answer = S²)" (and false: S² = 108) |
| 5157 | "a) sum\_ab … b) prod\_ab … c) ratio\_sp … d) d_{45445} = a/b+1 … e) answer" |
| 31915 | "base = 0, idx = 628 … sum1 = base + idx … answer = sum1×(2.29+2.29)" |
| 28282 | "(1/2)³ là **half_cubed**, 1/7 là **seventh** … 2 nhân **h_99614** …" |

### NL-LEAK (answer or source answer given away)
| idx | Quote |
|---|---|
| 12659 | "(ví dụ 783/4)", where the gold is 783/4 |
| 28552 | "Tính \boxed{100(x+y+89)}. (Đáp số là 10100.)" |
| 17359 | "Bạn tính được x = 5256." |
| 24492 | "(Kết quả yêu cầu là 156.)" (and the cubic has only one real root) |
| 5599 | "(Kết quả mong muốn là answer = 56 × 57 = 3192.)" |
| 28958 | "(Đáp án đúng là 286/225.)" |
| 15995 | "(Đáp án là \|s\|·D = 80.)" |
| 17710 | "Số có bốn chữ số lớn nhất mà tổng các chữ số bằng 16 là 9700. Tính tích của số này với số liền sau nó" |
| 10612 | "mức lương ở vị trí này là $23 000. Tính 0.54×(mức lương trung vị)+0.54 đô la" |
| 7494 | "Nếu biết góc trung tâm ∠ACB bằng 120°, hãy tính (∠ACB+3.75)×3.75" |

### False statements and wrong gold (verified, yet wrong)
| idx | Quote |
|---|---|
| 29408 | gold "0" for N(N+1) = 10,000,100,000 (validator matched "000") |
| 20638 | "kết quả chính là answer = 20" (true count 4) |
| 15640 | gold 1441.2 against the asked surface area of 1440 |
| 12280 | "quãng đường mà P đã di chuyển bằng p_in(p_in+1) … = 156" (true answer 12) |
| 24492 | "các nghiệm thực, phân biệt p, q, r" of a strictly increasing cubic |
| 13371 | "Cho r = 1 … phần dư … khi chia cho r−1" (division by zero) |
| 30091 | "biểu diễn câu trả lời dưới dạng π" for 0.97(27π+0.97); gold is a 19-digit fraction |

---

## 6. Verdict: DISCARD the 19,750 augmented rows (keep the 14,870 originals)

**Justification:**

1. **The defect is in the operator, not the wording.** Answer post-composition cannot produce a new problem. 100% of items are tail, identity, or a gutted version of the source. 0/200 are new [0–1.9%], and 10.5% are even natural.
2. **Fixing is regeneration.** Removing tags and renaming `h_NNNNN` would move about 46 of 200 items from "fixable" to "contrived". They would still be `NL-TACKED` duplicates of their source with random-decimal answers, which teaches the model to append "×2.62+2.62" to word problems. 39.5% [33.0–46.4] are discard-grade anyway: answer leaked, false statements, absurd, gutted.
3. **Training on it is harmful for the thesis.** It inflates the "symbolic" arm with 19,750 near-copies of 5,670 sources (≤5 children each, 3,684 sibling pairs with identical SMT). That distorts any symbolic-vs-LLM comparison (C1) and contaminates evaluation through source-answer leakage.
4. **Value for C3.** Keep the file *as an exhibit*. "Z3-verified" (97.7% gold = Z3, 98.1% unique) and "valid problem" (37.5% acceptable, 0% new) are separable. The verify-and-reflect loop actively creates answer leaks.

**If a symbolic-augmentation arm is still needed,** rebuild it. This is an estimate, and it costs no money: free NIM plus a local Z3.

* Mutate *premises*: perturb literal inputs in the source assertions (only where `uniqueness.classify == derived`), re-solve, and require an integer or clean answer for count-type questions.
* Never pass the Z3 answer to the generator. Drop the "Đáp án chuẩn từ Z3 là" feedback.
* Add a semantic gate before acceptance. It must have zero tolerance for these leak regexes (h_/d_, answer, SMT, `<BÀI`, `\boxed`, a gold value in the text) and for tail phrases. It must also include an integer check for count nouns, an LLM-judge or teacher check for "is this a natural new problem", and a near-duplicate check against the source (identity answer → reject).
* Use exact answer matching. Drop the 0.5% tolerance and the comma-fragment matching.
* Separately, fix the 3-decimal thousands bug in `eval/math_verify._collapse_number_seps` before scoring anything.
