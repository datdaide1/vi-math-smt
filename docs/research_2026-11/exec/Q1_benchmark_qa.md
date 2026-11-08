# Q1: Vi-ExamMath answer-key QA with LLM judges (2026-11-07)

Workstream Q1. It applies T5 §3.1–3.3 (QA protocol) to X2's 1,585 Core items. The user replaced the human raters
with LLM judges (Claude Opus 5.5). X2's and X5's files were only read. The queue was verified unchanged:
`qa_queue.jsonl` hashes to X2's freeze sha256 `0f58ee1c…e36aeaaf` once line endings are normalised to LF. X2's
`write_text` wrote CRLF on Windows, so the raw-bytes hash differs.

## 0. Update 2026-11-08: core_v1.1 (lead approved +$5.5, cap $15.5)

**Bottom line (v1.1, current).**
- **Frozen file:** `data/vi_exam_qa/core_v1.1.jsonl` has **1,545 items**: **1,367 VERIFIED** (was 1,029) and **178
  ADJUDICATED_LOW_CONFIDENCE** (was 526). **40 items are INVALID** and were dropped to `invalid_v1.1.jsonl`.
- **SHA256:** `ceac95defa5ed8f2b93c31835660551089743d583335c3002d4ad9d5b8c0046c`.
- **Traceability:** `core_v1.jsonl` is unchanged and still hashes to `48bfd1c9…c534dc`. The v1 decisions, stats and
  audit are kept as `decisions_v1.jsonl`, `q1_stats_v1.json` and `audit_results_v1.jsonl`.
- **Fresh audit #2:**
  - design: 150 items drawn uniformly from all Core items, new seed 20261108, INVALID items skipped; 18 items
    overlap with audit #1;
  - result: **2 label errors**, so the **estimated residual label-error rate is 1.3 %, 95 % CI [0.16 %, 4.7 %]**
    (two-sided Clopper–Pearson), and the **one-sided 95 % upper bound is 4.1 %**;
  - conservative reading: if every grader mismatch counted as an error (10/150), the upper bound is 11.0 %;
  - VERIFIED stratum: 2/132 errors, one-sided upper bound 4.7 %. LOW_CONF stratum: 0/18, upper bound 15.3 %.
  - Both errors were corrected. They are one item and one segmentation defect (below), not wrong arithmetic.
- **Post-cutoff Core after QA and review-rule contamination removal:**

  | slice | after QA + contamination | of which VERIFIED |
  |---|---|---|
  | 2026 | **109** | **98** |
  | 2025+ | **313** | **272** |
  | first_seen ≥ 2026-06 | **84** | **75** |

- **Certifiable split (H4), contamination-clean:**

  | set | certifiable (yes) | non-certifiable | maybe | no |
  |---|---|---|---|---|
  | VERIFIED | **970** | **360** | 283 | 77 |
  | all Core | **1,021** | **485** | 392 | 93 |

- **Claude cost:** **$15.36 of the $15.5 cap** (ledger `spent_usd 15.361113`). Of that, $5.43 is this update:
  - adjudication: $4.24;
  - audit #2: $1.10;
  - judge: $0.10.

  No batch is pending. Every result row is appended per item, and Batch IDs are in `data/vi_exam_qa/batches.jsonl`.

**Recommendation for T5 evaluation: VERIFIED is the primary analysis set, and all of Core (VERIFIED + LOW_CONF) is
the sensitivity analysis.** In both, drop items with `contamination.flag` from post-cutoff claims. The VERIFIED share
differs by certifiable tag (95 % of yes, 73 % of maybe, 82 % of no among valid items), so for H4 also report the
contrast on all valid Core items.

**What was done.**
1. **Adjudication of the two-Claude-only items.**
   - All 382 items in this group were adjudicated: Opus, effort medium, Batches API, 11 batches, 382/382 ok.
   - Order: post-cutoff first (2026 or first_seen ≥ 2026-06: 30; then 2025+: 46; then the rest by queue rank).
   - The reference solution was clipped to 2,000 characters (was 3,000) to fit the budget; the prompt is otherwise
     unchanged.
   - Adjudicator confidence: high 358, medium 24.
   - Outcomes:
     - 328 → VERIFIED (two Claude solves + adjudicator high);
     - 12 → VERIFIED (independent keys incl. non-Claude);
     - 33 → LOW_CONF (grader cannot confirm, or confidence medium);
     - 9 → INVALID (4 text defects, 3 ill-posed, 1 proof, 1 other).
2. **Audit #2.**
   - 54 items were re-solved at effort high. The other 96 were re-solved at effort medium to stay under the cap:
     effort high cost $0.0094 per item on this sample, which would have breached it.
   - Errors by effort: high 1/54, medium 1/96.
   - All re-solves used a fresh draw (`sample_idx=1`), so no cache was reused from audit #1.
   - The 10 grader mismatches went to the Opus audit judge: 8 both_acceptable, 2 key_wrong.
3. **The 2 errors are one error class: the item text contains sibling sub-questions.**
   - Example: `lgh-a186376.s0q1.1.b` shows "a) Thực hiện phép tính … b) Rút gọn …". The key covered only part b,
     although a model reading the text would answer both parts.
   - Both items came through the X2-consensus route.
   - `lgh-a186376.s0q1.1.b` now has the tuple gold (17; √x) and is LOW_CONF (`audit_revised_key`).
   - `lgh-a133370.s0q2` was dropped (INVALID, `audit_multi_part_item`): the judge's "gold" was a prose list of parts.
   - Scan of `core_v1.1.jsonl` for problem texts holding ≥ 2 line-initial `a)`/`b)` labels: **6 items, all LOW_CONF,
     0 VERIFIED**. Labels written inline or as "2.1" would not be caught.
   - This is a segmentation defect for X2 to fix, not a QA-key issue.

**v1.1 status by X2 key status.**

| X2 key status | VERIFIED | LOW_CONF | INVALID |
|---|---|---|---|
| consensus_3 | 384 | 1 | 0 |
| consensus_2 | 533 | 35 | 1 |
| conflict | 228 | 84 | 13 |
| unverified | 222 | 58 | 26 |

**Routes:**
- X2 consensus confirmed by the grader: 894.
- Two Claude solves + adjudicator high: 370.
- Keys agree after solve #2: 88.
- Independent keys incl. non-Claude: 15.
- LOW_CONF: insufficient agreeing keys 175, audit-revised 3.
- INVALID: 40 in total — text_defect 15, ill_posed 10, proof_or_open 7, not_gradable 5, other 3.

**Still open (v1.1).**
- The 175 LOW_CONF items with insufficient key agreement were adjudicated, but the grader could not confirm two
  agreeing keys. Most are format gaps; the adjudicator judged the keys correct.
- The 88 "keys agree, not adjudicated" items and the 894 X2-consensus items were never validity-checked by an LLM.
  Audit #2 found its 2 errors in the X2-consensus route (2/82 audited in that route).
- No humans were involved.

**For KNOWN_ISSUES** (grader 3.0.2; X1 owns it; nothing was edited). These 3 gaps decide whether a correct model
answer gets credit:
1. **Named points.** `M(0;0), M(2;2)` is graded wrong against the set gold `(0; 0); (2; 2)`. Only bare coordinates
   parse.
2. **Compound conditions.** `0 \le x < 4, x \ne 1` and `m \ne 1` fail with `unparsed-interval` against the interval
   golds `[0,1)\cup(1,4)` and `(-\infty,1)\cup(1,\infty)`.
3. **Prose last-value.** A prose span holding several labelled values is read as its last value.
   - "Đội I: 48 ngày; Đội II: 24 ngày" is graded **correct** for the gold `24` (false positive).
   - `\max A = 16 khi x = y = -1` is graded correct for `-1`.
   - The same colon-labelled answer is graded **wrong** against the tuple gold `48; 24` (false negative).

   Q1 works around this for key agreement only (the strict check in §2). Model evaluation is still exposed.

---

## v1 report (2026-11-07; superseded by §0, kept for traceability)

**Bottom line (v1).**
- **Frozen file:** `data/vi_exam_qa/core_v1.jsonl` has **1,555 items**: **1,029 VERIFIED** and **526
  ADJUDICATED_LOW_CONFIDENCE**. **30 items are INVALID** and were dropped to `invalid_v1.jsonl`.
- **SHA256:** `48bfd1c9287d88d5b7f22556e0914b033048410e7d8232aa420e3e1688c534dc`.
- **Audit:** 150 random items, **3 label errors** → estimated residual label-error rate 2.0 %, **95 % upper bound
  5.1 %** (Clopper–Pearson). The 0-error target (≤ 2 %) is **not met**. All 3 errors were corrected, and the error
  class behind 2 of them was fixed pipeline-wide (§5).
- **Post-cutoff Core after QA and removal of review-rule contamination:**
  - 2026: **109** (72 VERIFIED);
  - 2025+: **314** (209 VERIFIED);
  - first_seen ≥ 2026-06: **84** (52 VERIFIED).
- **Certifiable split (H4), clean Core:** yes 1,029 vs non-certifiable 487 (maybe 392 + no 95).
- **Claude cost:** **$9.93 of the $10 cap.** The ledger overstates it slightly (cache reads are priced at 0.10×;
  Opus 5.5 bills 0.05×).

**Main deviation: the budget did not cover adjudicating every item.** The protocol asks for a second solve plus an
adjudication on all 631 conflict / <2-key items. The measured prices at effort high made that impossible inside $10:
- second solve: $0.0090 per item → 664 items = $5.99;
- adjudicator at effort medium: $0.0127 per item → all items would have cost about $8.4.

All 664 items selected at the time got the effort-high second solve. **218 items were adjudicated**, chosen by need (§2): every item where
no two keys agree, plus the items that had no usable gold. The rest were settled by key agreement under the grader.
Without an adjudicator call, an item whose only agreement is between the two Claude keys is
ADJUDICATED_LOW_CONFIDENCE, never VERIFIED.

---

## 1. What was run

Code is in `vi_math_verified_aug/vi_exam_qa/` (new):
- `data.py` — paths and loaders;
- `keys.py` — re-types the keys and checks agreement with the grader;
- `llm_stage.py` — Claude stages through `common/llm.py` and the Batches API;
- `finalize.py` — statuses, audit, contamination, freeze.

Prompts are in the registry: `prompts/q1_solve.v1.md`, `q1_adjudicate.v1.md`, `q1_audit_solve.v1.md`,
`q1_audit_judge.v1.md`.

Tests: `.venv/Scripts/python.exe -m pytest tests/test_vi_exam_qa.py -q` → **5 passed** (no API).

| Step | Command (repo root; LLM steps with `PYTHONPATH=. E:\venv_tools\Scripts\python.exe`, the rest with `.venv`) | Result |
|---|---|---|
| keys | `-m vi_math_verified_aug.vi_exam_qa.keys prepare` | `keys_table.jsonl`, `selection.json`. Re-checks X2's agreements with grader 3.0.2 and picks the adjudication set and the audit sample (seed 20261107). |
| solve #2 | `-m …llm_stage run solve2 --max-tokens 4000` (pilot `--limit 20`; 1 truncated item re-run at 12k) | 664/664 ok, 9 batches, **$5.99** |
| triage | `-m …keys triage` | T1 no agreement 187 · T2 only the two Claude keys agree 398 · T3 a non-Claude key agrees 104 |
| adjudicate | `-m …llm_stage run adjudicate --effort medium --ids-file …` (pilot 10, T1 138, rounds 2–3) | 218/218 ok, 16 batches, **$2.78** |
| audit | `-m …llm_stage run audit --effort high` (150 + 1 retry + 5 reserves) | 155/156 ok, 6 batches, **$1.04** |
| audit judge | `-m …finalize audit-prep`, then `-m …llm_stage run audit_judge --effort medium` | 12/12 ok, 2 batches, **$0.12** |
| freeze | `-m …finalize freeze` | `core_v1.jsonl`, `invalid_v1.jsonl`, `decisions.jsonl`, `audit_results.jsonl`, `q1_stats.json`, datacard, sha256 |

**Guards.**
- Every request is `claude-opus-5-5` with `split="benchmark"`, so `common/llm` refuses Gemini and NIM for it. No
  Gemini or NIM calls were made.
- All calls are logged to `logs/api/Q1/claude.jsonl`. Batch submissions and collections are in
  `data/vi_exam_qa/batches.jsonl` (62 events, 0 pending). Driver logs are in `logs/Q1/*.log`.
- Spend ledger: `.cache/vimath/spend.sqlite`, scope Q1. `llm_stage spend` → `spent_usd 9.928386`.
- The hard cap is enforced per process. Each chunk's worst case (input estimate + max_tokens) must fit under
  min($10, spent + stage budget).
- The shared system prompt is cached: 110 of 111 requests in a batch read it from the cache.
- Every process stayed under 300 MB. The heaviest step, `freeze`, takes about 35 s.

## 2. Protocol as executed

1. **Independent solve (664 items).**
   - The set is X2's 631 conflict / <2-key items plus 33 X2 "consensus" items the grader did not confirm at that
     time (rule below).
   - The later labelled-values guard (§5) moved 25 more consensus items into the adjudication set (689 in total).
     They have no solve #2.
   - Opus 5.5, effort **high**, a Vietnamese "teacher re-solving for a reference key" prompt in a different phrasing
     from X2's.
   - The answer goes in `\boxed{}` using the grader's conventions.
   - X2's solve was effort low, Vietnamese, with an "ĐÁP ÁN:" line.
2. **Key agreement is decided only by grader 3.0.2, made strict for keys.** Key strings are first cleaned of MathType
   artefacts (`\left( {x;y} \right)`, `{\rm{ }}`). A match is then rejected when:
   - the matched span is prose or hedged;
   - the gold is a single value and the span holds several labelled values;
   - the span holds an equality-case clause ("khi …").

   The grader was written for model outputs. It reads "Đội I: 48 ngày; Đội II: 24 ngày" as `24`, so a partial page
   key would otherwise "agree" (§6).
3. **Adjudication.**
   - The Opus adjudicator (effort medium) sees:
     - the problem;
     - the rule-based tags;
     - all keys with their provenance (SymPy definition and values; the page key with extractor method and
       confidence; both Claude finals with the tail of their reasoning);
     - the third-party solution (≤ 3,000 chars).
   - It must re-derive the answer and may not take a majority vote.
   - It returns the typed gold (grader schema), validity (with reason), `well_posed`, `figure_needed`,
     `multiple_answers`, `accept_also`, confidence, a per-key verdict, a certifiable re-tag and a justification.
   - Order of adjudication: every T1 item (187), then the items with no usable gold. 31 T2 items were also reached
     (10 in the pilot, the rest because they lost their gold under the stricter rule).
4. **Status rule** (`finalize.decide_all`):
   - **VERIFIED** if either:
     - ≥ 2 keys match the final gold under the grader, at least one of them non-Claude (SymPy or page), and the
       adjudicator's confidence is not low; or
     - Claude #1 and Claude #2 both match and the adjudicator's confidence is **high**.
   - X2 consensus items re-confirmed by the strict grader (896) are VERIFIED without an adjudicator call: they
     already have ≥ 2 agreeing keys, at least one non-Claude.
   - Items not adjudicated get their gold from the first typed key that another key confirms. They are VERIFIED only
     under the first condition above.
   - **INVALID** = the adjudicator says the item is not valid, or the audit judge says `item_ambiguous`.
5. **Audit.**
   - 150 items drawn uniformly at random from **all** Core items (seed 20261107; INVALID items replaced from 30
     seeded reserves).
   - Each was re-solved blind: Opus, effort high, a third prompt (English instructions).
   - Answers were compared with the frozen key by the grader. 12 mismatches went to an Opus audit judge
     (key_correct / key_wrong / both_acceptable / item_ambiguous).
   - The sample, the mismatches and the verdicts are frozen (`audit_sample.frozen.json`,
     `audit_mismatches.frozen.jsonl`), so later fixes cannot hide an error the audit found.
6. **Contamination.**
   - Source: X5's `data/contam/vi_exam_flags.jsonl`.
   - `contamination.flag` = X5 §5.3 review rule. Flagged items are kept in the file and marked.
   - `g4_any` (G4 thresholds) is kept for sensitivity analyses.

## 3. Counts per status

| | VERIFIED | LOW_CONF | INVALID | total |
|---|---|---|---|---|
| **All Core** | **1,029** | **526** | **30** | 1,585 |
| X2 consensus_3 | 385 | 0 | 0 | 385 |
| X2 consensus_2 | 513 | 56 | 0 | 569 |
| X2 conflict | 57 | 259 | 9 | 325 |
| X2 unverified (<2 keys) | 74 | 211 | 21 | 306 |
| certifiable = yes | 826 | 228 | 9 | 1,063 |
| certifiable = maybe | 164 | 240 | 7 | 411 |
| certifiable = no | 39 | 58 | 14 | 111 |

**Routes:**
- X2 consensus confirmed by the strict grader: 896.
- Keys agree after solve #2 (incl. non-Claude), not adjudicated: 88.
- Adjudicated:
  - two Claude solves + adjudicator high: 42;
  - independent keys incl. non-Claude: 3;
  - insufficient agreeing keys → LOW_CONF: 142.
- Two Claude solves agree, not adjudicated → LOW_CONF: 382.
- Audit-revised: 2.
- INVALID: text_defect 11, ill_posed 7 (incl. 1 from the audit), proof_or_open 6, not_gradable 4, other 2.

**Adjudicator.**
- Confidence: high 182, medium 35, low 1.
- Its certifiable re-tag matches X2's tag on 192 of 218 items. 12 of the mismatches are yes → no: text-only geometry
  that X2's regex tagged certifiable.

**Why LOW_CONF is large.** 382 items (T2) have two agreeing Claude solves, but no adjudicator call was affordable.
Among the 144 adjudicated LOW_CONF items with confidence high or medium, 118 would become VERIFIED if the
adjudicator's own per-key verdicts counted as agreement. They do not count: agreement is grader-only. Those items
mostly fail on grader parsing gaps (§6). LOW_CONF is therefore a conservative label, not a measured error: the audit
found 1 error in 54 LOW_CONF items, against 2 in 96 VERIFIED items.

## 4. Agreement matrices

**Pairwise key agreement** over all Core items where both keys exist (strict grader). The claude2 and audit columns
come from selected subsets: claude2 was run only on the hard items, so its pairs are not population rates.

| pair | agree | disagree | rate |
|---|---|---|---|
| sympy ~ page | 309 | 29 | 91.4 % |
| sympy ~ claude1 | 338 | 16 | 95.5 % |
| page ~ claude1 | 934 | 565 | 62.3 % |
| sympy ~ claude2 (hard subset) | 7 | 7 | 50.0 % |
| page ~ claude2 (hard subset) | 85 | 511 | 14.3 % |
| claude1 ~ claude2 (hard subset) | 490 | 142 | 77.5 % |
| sympy ~ audit | 38 | 1 | 97.4 % |
| page ~ audit | 96 | 53 | 64.4 % |
| claude1 ~ audit | 134 | 19 | 87.6 % |
| claude2 ~ audit | 58 | 7 | 89.2 % |

Under the stricter rule, X2's lax comparator overstated agreement. For example, X2 reported provisional vs SymPy
96.6 %; it is 91.4 % here. A partial key ("1" for the system answer (2; 1), or one root of two) had counted as a
match.

**Each key vs the final gold** (non-INVALID items with a gold):

| key | all Core: agree / disagree / absent | adjudicated items: agree / disagree |
|---|---|---|
| SymPy program | 413 / 25 / 1,117 | 1 / 2 |
| page solution (regex) | 905 / 579 / 71 | 23 / 159 |
| Claude #1 (X2, effort low) | 1,397 / 158 / 0 | 51 / 138 |
| Claude #2 (Q1, effort high) | 600 / 34 / 921 | 135 / 29 |

**Grader vs adjudicator per-key verdicts** (adjudicated, non-INVALID):

| grader verdict | adjudicator verdict | n |
|---|---|---|
| agree | correct | 210 |
| disagree | correct | 182 |
| disagree | partial | 110 |
| disagree | incorrect | 36 |

The grader and the adjudicator never conflict the other way: no case where the grader agrees and the adjudicator
says incorrect. The strict grader under-confirms; it does not over-confirm.

## 5. Audit and residual error bound

| | n | match | both_acceptable | key_wrong | item_ambiguous | label errors | 95 % upper bound |
|---|---|---|---|---|---|---|---|
| **All** | **150** | 138 | 9 | 2 | 1 | **3 (2.0 %)** | **5.1 %** |
| VERIFIED at audit time | 96 | 92 | 2 | 2 | 0 | 2 | 6.4 % |
| LOW_CONF at audit time | 54 | 46 | 7 | 0 | 1 | 1 | 8.5 % |

- **Conservative bound:** if every grader mismatch counted as an error (12/150), the upper bound is 12.6 %.
- **both_acceptable (9):** grader format failures, not label errors. Examples: an audit answer bundling extra parts
  ("15; A = 19/3"), labelled points `A(-2; 1)`, "Không thể ra khơi vì …" against the text gold "Không".
- **The 3 errors:**
  - `lgh-a186409.s0q2.c` (route: keys agree, not adjudicated). The question asks both speeds; the key was "25"
    only. It came from a partial page key "confirmed" by Claude #1's prose answer "Bình: 20 km/h, An: 25 km/h".
  - `lgh-a165624.s0q5.3` (X2 consensus route). OE and the area are asked; the key was the area only.
  - `lgh-a188291.s1q4.c` (two Claude keys, not adjudicated). The text never defines D; the item was made INVALID.
- **Corrections.** The two keys are corrected (`audit_revised_key`, LOW_CONF). The error class, a multi-quantity
  question keyed by one quantity, was then fixed for every item by the labelled-values guard (§2 step 2). That fix
  took 9 items off the "keys agree, not adjudicated" VERIFIED route (97 → 88) and 4 items out of the confirmed
  X2-consensus set (900 → 896).
- **Scope of the bound.** It describes the pipeline *before* that fix. A re-audit (about $1) would measure the
  pipeline after it.

## 6. Contamination and post-cutoff Core

- 39 Core v1 items carry `contamination.flag` (X5 review rule). They are kept.
- `g4_any` is the sensitivity column below.

| slice | X2 Core | after QA (not INVALID) | **after QA + contamination** | of which VERIFIED | G4-clean (sensitivity) |
|---|---|---|---|---|---|
| exam year 2026 | 112 | 111 | **109** | 72 | 87 |
| exam year ≥ 2025 | 333 | 329 | **314** | 209 | 241 |
| first_seen ≥ 2026-06 | 86 | 85 | **84** | 52 | 70 |

## 7. Certifiable vs non-certifiable (H4)

The tag is X2's rule-based tag (`certifiable`); the adjudicator's re-tag is in `qa.adjudicator_certifiable`.

| scope | yes | maybe | no | **certifiable (yes) vs non-certifiable (maybe + no)** |
|---|---|---|---|---|
| Core v1 | 1,054 | 404 | 97 | 1,054 vs 501 |
| Core v1, contamination-clean | 1,029 | 392 | 95 | **1,029 vs 487** |
| VERIFIED and clean | 802 | 159 | 38 | 802 vs 197 |
| exam ≥ 2025, clean | 179 | 118 | 17 | 179 vs 135 |
| exam 2026, clean | 58 | 44 | 7 | 58 vs 51 |

**Caveat for H4.** Non-certifiable items are much more often LOW_CONF: 59 % of maybe/no items vs 22 % of yes items.
Run H4 on all valid Core items and repeat it on VERIFIED only as a robustness check, so that key quality is not
confounded with the contrast.

## 8. Cost (Claude Opus 5.5, Message Batches, 50 % price)

| stage | requests ok | input tok | output tok | USD |
|---|---|---|---|---|
| solve #2 (effort high) | 664 (+1 truncated, re-run) | 247,557 | 540,384 | 5.99 |
| adjudicator (effort medium) | 218 | 439,542 | 180,846 | 2.78 |
| audit re-solve (effort high) | 155 (+1 truncated, re-run) | 67,474 | 90,812 | 1.04 |
| audit judge (effort medium) | 12 | 17,593 | 8,570 | 0.12 |
| **total** | | | | **9.93** (cap 10) |

The input column excludes cached tokens.

## 9. Findings for other workstreams

**X1 grader** (gaps hit while comparing keys; no grader file was edited).

Model answers the grader marks wrong:
- named points: `M(0;0), M(2;2)` against the set gold `(0; 0); (2; 2)`;
- compound conditions: `0 \le x < 4, x \ne 1` and `m \ne 1` (unparsed-interval);
- colon labels for a tuple: `Đội I: 48 ngày; Đội II: 24 ngày`;
- word labels or prose in the box: `A = 3 \text{ khi } x = 4`;
- a named tuple gold (vars `[hoa hồng, hoa cúc]`) against a prediction naming `x, y`. Q1 therefore keeps `vars`
  only for variables the problem itself uses; otherwise tuples are positional and the names go to `quantities`.

Model answers the grader marks right although they hold more than the gold:
- a prose span with several labelled values is read as its last value ("Đội I: 48 ngày; Đội II: 24 ngày" passes
  for a gold of 24);
- `\max A = 16 khi x = y = -1` passes for a gold of -1.

Gold typing:
- `infer_gold("(x;y)=(2;1)")` returns a *set*; Q1 rewrites it to a tuple with vars;
- `\text{label} = v` items become `= v`.

7 Core golds fail the round-trip check. All are LOW_CONF tuples or decimals with angles, points or text; they are
listed by `qa.gold_roundtrip=false`.

**X2 benchmark.**
- The page key is right on only 905 of 1,484 Core items (61 %).
- X2's "consensus" counted partial matches: 58 of 954 consensus items fail the strict grader. Of the audit's 2
  wrong keys, 1 was an X2 consensus item.
- 30 items are invalid:
  - 11 text defects (formula or condition missing from the page, e.g. a level-1 electricity quantity, "phương trình
    (1)" absent);
  - 7 ill-posed;
  - 6 proof/open;
  - 4 not gradable;
  - 2 other.

**H4 / analysis.** Use `status` and `contamination.flag` as filters. The certifiable tag disagrees with the
adjudicator on 12 % of adjudicated items.

## 10. Limits and open items

- **No humans.** Every status rests on Claude judgements plus the grader. The audit judge and the adjudicator are the
  same model family as two of the keys, so a shared Claude error can survive. Only 413 items have a SymPy key that
  is independent of Claude.
- **Coverage.** 471 items of the 689-item adjudication set were not adjudicated (budget): 367 T2 and 104 T3.
  Adjudicating the T2 items (about $4.7 at the measured $0.0127 per item) would turn most of the 382 two-Claude
  LOW_CONF items into VERIFIED or INVALID. The 30-reserve audit design is in place for a re-audit.
- **Possibly hidden invalid items.** Validity was LLM-checked only for the 218 adjudicated items. Other Core items
  (consensus and key-agreement routes) may still hide text defects; the audit found one of them.
- **Audit bound vs target.** 5.1 % does not meet T5's ≤ 2 % target (that needs 0 errors in 150). Report it as
  "estimated label-error rate ≤ 5.1 % (95 % upper bound)".
- **Release.** Third-party solutions stay out of `core_v1.jsonl`; adjudicator justifications are internal
  (`decisions.jsonl`, `adjudications.jsonl`, which can paraphrase them; not for release). The canary GUID is
  carried on every row.

## 11. Files

- `data/vi_exam_qa/`:
  - frozen release files: `core_v1.jsonl`, `core_v1.sha256`, `core_v1.datacard.md`, `invalid_v1.jsonl`;
  - QA and audit records: `decisions.jsonl`, `keys_table.jsonl`, `triage.jsonl`, `selection.json`,
    `audit_sample.frozen.json`, `audit_mismatches.frozen.jsonl`, `audit_results.jsonl`;
  - raw LLM outputs (internal): `solve2.jsonl`, `adjudications.jsonl`, `audit_solves.jsonl`,
    `audit_judgements.jsonl`;
  - run records: `batches.jsonl`, `q1_stats.json`, `*.txt` id lists.
- Code: `vi_math_verified_aug/vi_exam_qa/{__init__,data,keys,llm_stage,finalize}.py`.
- Prompts: `prompts/q1_*.v1.md`.
- Test: `tests/test_vi_exam_qa.py`.
- Logs: `logs/Q1/*.log`, `logs/api/Q1/claude.jsonl`.
- v1.1 (2026-11-08):
  - release files: `core_v1.1.jsonl`, `core_v1.1.sha256`, `core_v1.1.datacard.md`, `invalid_v1.1.jsonl`;
  - QA records: `decisions_v1.1.jsonl`, `q1_stats_v1.1.json`;
  - audit #2: `audit2_sample.frozen.json`, `audit2_mismatches.frozen.jsonl`, `audit2_solves.jsonl`,
    `audit2_judgements.jsonl`, `audit2_results.jsonl`;
  - id list: `adj_round4_lowconf.txt`;
  - logs: `logs/Q1/{adjudicate_r4,audit2*,freeze_v11}.log`;
  - commands: `finalize audit2-sample | audit2-prep | freeze --version 1.1`;
    `llm_stage run adjudicate --ref-chars 2000 …`; `llm_stage run audit2 | audit2_judge`.
- Nothing was committed.
