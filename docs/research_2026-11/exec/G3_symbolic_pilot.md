# G3: end-to-end pilot of the symbolic arm (arm Y) (2026-11-07)

**Brief:** `EXEC_BRIEF2.md` rules, `01_TECHNICAL_DESIGN.md` §3–§4, `tech/T1_synthesis.md` §2.2–§2.5, `exec/X4_symbolic.md`,
`exec/G1_generator_pilot.md`, `content_review/PROTOCOL.md`. Coordinator addendum: X5's bge-m3 threshold cos 0.85 (Vi–Vi)
for the near-duplicate and decontamination gates (`exec/X5_decontam.md` §5).

**Code:** `vi_math_verified_aug/pilot_symbolic/`. It contains `seeds`, `common` (the shared `common/llm.py` client,
ws G3, $5.9 hard cap), `editor` (the editor-facing view of X4 edit lists and the tag check), `gates`, `pipeline`
(stages 1–5), `embed_modal` + `cosine` (bge-m3), `judge` and `report`.
**Prompts (new, versioned):**
- `prompts/formalizer.v2.md`: the G1 `formalize_v1`, verbatim.
- `prompts/back_formalizer.v1.md`: the same prompt, blind (no English, no gold).
- `prompts/editor.v2.md`: run A.
- `prompts/editor.v3.md`: run B, the final run.

Judging reuses `validity_judge.v2` (the G1 judge of record), `validity_judge.v3` (the G2 cheap-judge rubric) and
`solution_writer.v2`.

**Data:** `data/pilot/symbolic/`. Shared files: `seeds.jsonl`, `slots.jsonl`, `formalize.jsonl`. `run_b/` holds every
stage, the judges, `cosine.jsonl`, `emb.json`, `metrics.json` and `report_tables.md`, which has every table below.
`run_a_x4_as_delivered/` holds X4 before my fixes: formalize, mutate and render, plus 9 partial cycle rows.
**API ledger:** `logs/api/G3/{claude,gemini}.jsonl`.
**Spend:**
- Claude **$4.21 of $6**: Opus judge $1.92, Opus seed baseline $0.84, Opus blind-solve $1.17, Haiku editor $0.23 (runs A+B), Haiku formalizer $0.05.
- Modal **$0.004**: one bge-m3 call on L4/A10G; billing before/after is in `run_b/modal_billing_*.txt`.
- Gemini: free tier, dev split only.
- No NIM, no Kaggle. No test or benchmark item was sent anywhere.

**Tests:**
- `tests/test_symbolic_g3_fixes.py` covers the X4 bug fixes (6 cases). All 6 fail on the as-delivered X4 code (checked on a copy) and pass now.
- `tests/test_pilot_symbolic.py` covers the editor view, tag check and gates.
- The whole X4 suite still passes: `pytest tests/test_symbolic_*.py tests/test_pilot_symbolic.py` gives **245 passed**.

## 0. Answer

**Arm Y as specified FAILS the T1 §2.5 quality gate.** The judge is Opus 5.5 (`validity_judge.v2`) on the 66 items
that survive every gate:

| Gate (T1 §2.5) | Threshold | Arm Y survivors | Verdict |
|---|---|---|---|
| teacher-would-write | ≥ 70% | **35% [24–47]** (23/66) | FAIL |
| genuinely new (M2–M6) | ≥ 60% | **55% [41–69]** (26/47) | FAIL (CI straddles) |
| leak | 0% | **0% [0–6]** (0/66; regex 0, answer-in-text 0, judge 0) | PASS |
| absurd | ≤ 3% | **12% [6–22]** (8/66) | FAIL |
| (answer faithfulness) Opus blind solve = program answer | — | **100% [94–100]** (66/66) | — |

**What this means for C3 (the faithfulness gap):**
- The symbolic pipeline closes the **answer**-faithfulness gap. The answer leak is gone (R5: 58% → 0%), the answer tail is mostly gone (R5: 84% → 14%, and 9 of those 9 are M6 or M4), and the program answer is the answer a strong blind solver gets on 100% of survivors.
- It does **not** close the **text**-faithfulness gap. Only 35% of survivors are worksheet-quality. Fluency issues affect 76% (natural 3.2/5); absurd or unanswerable items still pass the cycle check (12% and 11%).
- **The cycle check is the component that buys answer faithfulness.** Opus agrees with the program answer on 98% [93–99] of cycle-passed renders and 56% [39–72] of cycle-failed ones.
- **The rest is a ceiling inherited from the seeds.** The 60 machine-translated seed texts are only **17% [9–28]** teacher-acceptable under the same judge (fluency issues 88%). A minimal-edit renderer cannot rise far above its seed.

**Against the old pipeline and G1:**
- vs R5: teacher-acceptable 10.5% → 35%; tail 84% → 14%; leak 58% → 0%.
- vs G1's simplified-edit-list editor, same judge and rubric, all tag-passing M2–M6 renders: teacher 21% [9–43] (4/19) → **25% [18–33] (28/114)**; absurd 32% → 25%; valid 32% → 20%.
- **The real X4 edit lists did not improve editor output quality**; the gates then lift M2–M6 teacher to 32% (15/47).
- vs the arm-N native generator on other dev seeds (G2, Haiku `native_generator.v3`): teacher 92% [80–97], absurd 0/40. Arm Y is far behind on text quality. Arm N, however, has no answer certificate.

## 1. Setup

- **Seeds:** 60 new Vi-GSM8K train dev seeds.
  - Pool: the T1 dev pool (`sha1("vi_gsm8k_train||i")[:12]` ≡ 0 mod 20, 369 seeds), minus the 16 R1 corrupt/half-English rows, the 100 G1 seeds and the 40 G2 seeds, leaving 229.
  - Draw: `random.Random(20261108).sample(…, 60)`; there is no overlap with G1 or G2.
  - Schedule (T1 §2.0): J = 3 slots per seed, o ~ π with `Random(20261108)`. That gives 180 slots: M1 27, M2 43, M3 44, M4 35, M5 11, M6 20.
- **Pipeline (T1 §2.2):**
  1. Haiku 5.5 (effort low) formalizes, with ≤ 3 attempts and X4 validator feedback. The program is accepted only if `from_expr_json` loads it and `validate_program(P, gold)` returns no violations.
  2. X4 `mutate(op, P, Random(f"{slot}:20261108"))`. M6 partners are the other 58 accepted dev programs. X4's internal `gate` is wrapped (read-only) so that every candidate verdict is counted, and every returned program is re-checked independently.
  3. Rendering:
     - M1: `render_substitutions` (deterministic), then re-validated on the new text.
     - M2–M6: the Haiku editor (`editor.v3`, effort low) gets the tagged seed, a numbered sentence list and `[e.to_editor_dict() for e in edits]`. JOIN char offsets are dropped, and the M6 reference literal is wrapped in `<ref>`. The editor never sees the answer.
     - `symbolic.tag_issues` must be empty. There are ≤ 2 attempts, and the retry gets only the tag issues.
     - An optional `warning` field lets the editor say an edit is unnatural. It is recorded and not used as a gate.
  4. Cycle check: gemini-3.1-flash-lite (`back_formalizer.v1`, temperature 0, JSON) formalizes the stripped text blind.
     - The executed answer must equal the program answer exactly (Fraction).
     - There are ≤ 2 attempts, retried only on parse, IR or exec errors, and never with the answer.
     - API failures are not verdicts: they are re-run.
  5. Common gates:
     - leak regex (the G1/R5 `LEAK_RE`);
     - answer value written anywhere in the text (`contains_value`; T1 zero tolerance);
     - chrF ≤ 0.90 vs the seed for M2–M6;
     - **bge-m3 cosine < 0.85** (X5 τ, Vi–Vi) vs 13,552 Vietnamese eval items: Vi-GSM8K test, Vi-MATH test, Vi-ExamMath raw and items. These come from X5's target embeddings, built with the same encoder settings.
     - bge-m3 cosine < 0.85 vs any item made from a disjoint seed set;
     - 5-gram Jaccard < 0.8, with ≤ 3 per seed.
  6. Judging, run on **all 139 tag-passing renders** (survivors and rejects), so that every gate's discrimination is measured:
     - Opus 5.5 judge (`validity_judge.v2`, effort medium);
     - Opus 5.5 blind solve (`solution_writer.v2`);
     - the second judge, gemini-3.1-flash-lite with `validity_judge.v3`. This is the non-Claude judge G2 keeps, as one leg of its cascade. G2's report appeared after this run and found no better non-Claude judge.
     - Opus also judged the 60 seed texts as a baseline.
- **Labels:** taken from G1/G2 (`lab_v2`).
  - teacher = `teacher_would_write`;
  - new = `new_vs_seed`;
  - absurd = NL-ABSURD, tail = NL-TACKED, near-dup = NL-NEARDUP, leak = NL-LEAK;
  - valid = 5 booleans ∧ natural ≥ 4.
- **CIs:** Wilson 95%.

## 2. Stage-by-stage yield per operator (run B; cumulative, count and % of the scheduled slots)

| op | slots | IR | mutated | tag ok | cycle ok | no leak/answer | chrF ≤ .90 | cos < .85 | dedup | survivor |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 27 | 26 (96%) | 25 (93%) | 25 (93%) | 23 (85%) | 21 (78%) | 21 (78%) | 21 (78%) | 19 (70%) | **19 (70%)** |
| M2 | 43 | 43 (100%) | 38 (88%) | 37 (86%) | 32 (74%) | 22 (51%) | 22 (51%) | 22 (51%) | 22 (51%) | **22 (51%)** |
| M3 | 44 | 44 (100%) | 42 (95%) | 42 (95%) | 26 (59%) | 26 (59%) | 9 (20%) | 9 (20%) | 9 (20%) | **9 (20%)** |
| M4 | 35 | 33 (94%) | 16 (46%) | 16 (46%) | 13 (37%) | 12 (34%) | 6 (17%) | 6 (17%) | 5 (14%) | **5 (14%)** |
| M5 | 11 | 11 (100%) | 7 (64%) | 7 (64%) | 6 (55%) | 5 (45%) | 5 (45%) | 5 (45%) | 5 (45%) | **5 (45%)** |
| M6 | 20 | 20 (100%) | 12 (60%) | 12 (60%) | 7 (35%) | 6 (30%) | 6 (30%) | 6 (30%) | 6 (30%) | **6 (30%)** |
| all | 180 | 177 (98%) | 140 (78%) | 139 (77%) | 107 (59%) | 92 (51%) | 69 (38%) | 69 (38%) | 66 (37%) | **66 (37%)** |

**Per stage:**
- **Formalize:** 59/60 = 98% [91–100] accepted, pass@1 92% [82–96]. That is close to G1's 100%, even with X4's stricter validator.
  - Violations over the 66 attempts: `type` 5, `unit` 4, `span` 3, `reach` 3.
  - The one rejected seed (idx 4175) has an intermediate of 1/5 snacks, which the validator correctly refuses.
- **Mutate:** 140/177 applied. NotApplicable reasons:
  - M4 "no type-compatible relation swap" 17 (after my describability fix; 11 before);
  - M6 "no unit-compatible partner" 8 (only 58 partners);
  - M2 "no unique clean inverse" 5; M5 "no depth-2 target" 4.
  - X4 `gate` verdicts over every candidate tried:
    - M2: 208 implausible vs 38 pass;
    - M3: 109 type violations (half-candies caught), 76 implausible, 6 structural tails;
    - M1: 21 implausible, 19 ugly.
  - **Uniqueness and identity never fired.** The re-check of all 140 returned programs: gate passes, no structural tail, no identity.
- **Render:**
  - Tag pass: 139/140 within 2 attempts. Pass@1: M2 89%, M3 98%, M4 88%, M5 100%, M6 100%.
  - M1 re-validates 25/25.
  - The editor raised a `warning` on 34/115 M2–M6 renders.
- **Cycle check (gemini-3.1-flash-lite):** 107/139 = 77% [69–83], with 137/139 executed.
  - By operator: M1 92%, M2 86%, M5 86%, M4 81%, **M3 62%**, **M6 58%**.
  - Back-formalized DAGs match the program's canonical DAG for only 25/137, so structure-level comparison is not usable as a gate. A different but equivalent formalization is the norm.
- **Answer-in-text:** 18 items, 15 among cycle-passed items.
  - All 18 are **coincidences**: the answer equals a different given ("4 thỏi son" and answer 4), or "thứ **Ba**" (Tuesday) is read as 3.
  - One is a real trivialization: M4 sets the ratio to 4:4, so 44 gallons gives 44 miles.
  - Opus finds no leak in any of them, and they are as good as the rest (teacher 33%). As implemented, the zero-tolerance rule costs **10 of the 32 cycle-passing M2 items** for no benefit (§6).
- **chrF ≤ 0.90:** this removes **M3 26 → 9** and M4 12 → 6. Median chrF to the seed is 0.92 for M3 and 0.91 for M4. *A minimal-edit renderer and a 0.90 chrF novelty gate conflict by design.* The near-dup items it removes are judged near-dup by Opus 85% of the time and teacher-acceptable 21% of the time, so it is not removing good items.
- **bge-m3 gates (X5 τ = 0.85):**
  - **0/139** items reach 0.85 vs any Vietnamese eval item. The max is 0.819, Vi-GSM8K test #692, all three slots of one seed.
  - 0/139 reach 0.85 vs an item from a disjoint seed (max 0.814).
  - **Cosine to the item's own seed is ≥ 0.85 for 133/139** (median 0.968; M6 0.851).
  - Used as a near-dup-vs-seed gate, τ = 0.85 would reject every M1–M5 item. X5 calibrated it for "same problem / re-translation / number-perturbed copy" detection. Span-grounded edits *are* number-perturbed copies of their seed by construction. So I apply 0.85 for decontamination and cross-seed dedup, and keep T1's chrF for the seed-novelty gate (§6).
- **Dedup:** 3 rejected, all sibling slots of one seed (two are identical M4 outputs, J = 1.0).

## 3. Quality gate (Opus 5.5 judge), survivors and the other sets

| set | n | teacher (≥70%) | new M2–M6 (≥60%) | leak (0%) | absurd (≤3%) | tail | near-dup | valid | unanswerable | fluency | natural | Opus solve = program |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **survivors, all ops** | 66 | **35% [24–47]** | **55% [41–69]** | **0% [0–6]** | **12% [6–22]** | 14% [7–24] | 58% | 27% | 11% | 76% | 3.23 | **100% [94–100]** |
| survivors M1 | 19 | 42% [23–64] | – | 0% | 5% | 0% | 100% | 32% | 5% | 68% | 3.32 | 100% |
| survivors M2 | 22 | **55% [35–73]** | **91% [72–97]** | 0% | 9% | 0% | 9% | 36% | 9% | 73% | 3.32 | 100% |
| survivors M3 | 9 | 33% [12–65] | 33% [12–65] | 0% | 22% | 11% | 67% | 33% | 22% | 67% | 3.33 | 100% |
| survivors M4 | 5 | 0% [0–43] | 0% | 0% | 20% | 40% | 100% | 0% | 0% | 100% | 3.00 | 100% |
| survivors M5 | 5 | 0% [0–43] | 0% | 0% | 0% | 0% | 100% | 20% | 0% | 80% | 3.20 | 100% |
| survivors M6 | 6 | 0% [0–39] | 50% | 0% | 33% | **100%** | 17% | 0% | 33% | 100% | 2.67 | 100% |
| all tag-passing renders | 139 | 26% [19–34] | 41% [33–50] | 1% | 22% | 14% | 64% | 22% | 21% | 78% | 3.13 | 88% [82–93] |
| cycle check passed | 107 | 31% [23–40] | 46% | 0% | 14% | 10% | 62% | 24% | 11% | 78% | 3.20 | **98% [93–99]** |
| cycle check FAILED | 32 | 9% [3–24] | 27% | 3% | **47%** | 25% | 72% | 12% | **53%** | 78% | 2.91 | **56% [39–72]** |
| chrF > .90 (gate fired) | 39 | 21% | 15% | 3% | 26% | 8% | 85% | 18% | 31% | 77% | 3.10 | 77% |
| M2–M6 tag-passing (cf. G1 D2) | 114 | 25% [18–33] | 41% | 1% | 25% | 17% | 56% | 20% | 24% | 79% | 3.10 | 86% |
| **the 60 seed texts (no transformation)** | 60 | **17% [9–28]** | – | 0% | 8% | – | – | 15% | 8% | **88%** | 3.12 | – |

**Reading the table:**

1. **The gates work in the right direction, and the cycle check does most of the work.**
   - Cycle-failed renders are 5× more often absurd/unanswerable (47%/53% vs 14%/11%).
   - They disagree with the strong solver 44% of the time (vs 2%).
   - The cycle check removes 32 items, of which only 3 were teacher-acceptable.
   - This is the T1 claim, measured: the certificate is only worth something together with the round-trip.
2. **Survivors are still mostly not worksheet-quality, and the dominant defect is language, not math.**
   - 76% have fluency issues. Opus's notes, again and again: "đô la", "feet", "dặm", "ounce", foreign names, "cô ấy" repetition, English decimals ("247.5").
   - These are inherited from the machine-translated seeds; the seeds themselves score 17% teacher-acceptable.
   - Paired by seed, an item is teacher-acceptable 48% (10/21) of the time when its seed is, and 22% (26/118) when it is not.
   - T1's `localize_surface` step (đô la → nghìn đồng, Vietnamese names) was **not** in this pilot's brief, and the data shows it is needed.
3. **Operator ranking:**
   - **M2 (inverse) is the one operator that works:** teacher 55%, new 91%, 0 tails.
   - M1 is clean but by definition not new.
   - M3 survives poorly (20% of slots) and is mostly judged near-dup (one step inserted into the same story).
   - **M4, M5 and M6 produce 0 teacher-acceptable survivors** (0/16 together).
   - **M6 is judged a tail 6/6.** Sequential composition with an artificial bridge ("Agatha có số tiền bằng đúng tổng số tiền Charlotte đã chi…") reads as the R5 failure, even though X4's structural rule accepts it (partner material is not an "invented constant"). X4 predicted this: semantic tail 11/12 by construction.
4. **The editor warning is a free, useful filter.** M2–M6 renders with a warning are 9% teacher-acceptable (3/34); those without are 31% (25/80).
5. **Answer correctness on survivors is 100% (66/66).**
   - Some survivors are still ambiguous (7 unanswerable), for example "đắt hơn 30% … phải trả thêm thuế 15%".
   - The program and both readers chose the same reading, so neither the cycle check nor the blind solve detects *ambiguity*. Only the judge does.

**Second judge (gemini-3.1-flash-lite, `validity_judge.v3`) vs Opus, all 139:**

| label | teacher | new | valid | absurd | tail | near-dup | leak | unanswerable |
|---|---|---|---|---|---|---|---|---|
| κ | 0.14 | 0.32 | 0.06 | 0.27 | −0.03 | 0.34 | 0.00 | 0.15 |
| positive rate (Opus / Gemini) | 36 / 114 | 47 / 101 | 30 / 125 | 30 / 10 | 19 / 2 | 89 / 37 | 1 / 0 | 29 / 3 |

- Gemini rates **92%** of survivors teacher-acceptable (Opus 35%) and solves 80% to the program answer.
- This is the same leniency G1 (κ 0.35) and G2 (κ 0.19, "accepts 86%") found. It is unusable as a second judge and adds nothing to the gate verdict.
- The gate verdict rests on Opus. G2 measured Opus re-test κ at 0.93, and Opus showed no self-preference.
- G2's recommended screen (Sonnet or Haiku with v3) was not run here, to keep within the budget.

## 4. X4's open question: M3 items whose new answer is old answer ± c (or k·old)

`is_semantic_tail` holds for 9/42 M3 renders. I read all 9 with Opus's verdict (table and texts in `run_b/report_tables.md`;
rows in `run_b/{mutate,judge_opus}.jsonl`):

| M3 subset | n | teacher | new | tail (judge) | near-dup (judge) | cycle ok | survivor |
|---|---|---|---|---|---|---|---|
| semantic tail | 9 | 1 | 2 | 1 | 7 | 8 | 4 |
| not semantic tail | 33 | 5 | 4 | 4 | 29 | 18 | 5 |

- **Additive cases (old ± c), n = 2.** Neither is judged a tail.
  - "Sau đó có thêm 52 người đến dự" (134 + 52 = 186).
  - "mỗi quả trứng còn phải trả thêm 8 xu phí đóng gói" (10 − 8 = 2).
  - Both are mid-story steps that change an intermediate, and the cycle check passes. Both are judged near-dup and not teacher-acceptable: the first because the step is unmotivated ("who are these 52?"), the second because it has the seed's structure.
- **Multiplicative cases (k·old or old/k), n = 7.**
  - The one judged tail is "cô chia đều khoảng thời gian đạp xe thành 5 giai đoạn… Hỏi trong mỗi giai đoạn…". The insertion re-asks the old total per part, so it is a tail in substance.
  - One is the best M3 item: successive discounts 15% then 40%.
- **Decision:**
  - The ± c items are **not tails**. They are acceptable *structurally*, but low-novelty.
  - `is_semantic_tail` should **not** be a gate. It does not predict the judged tail (1/9 vs 4/33) or teacher-acceptability (1/9 vs 5/33).
  - The real M3 defect is semantic fit: templates applied regardless of story role. Examples: "Bob mua thêm 5 con hàu" raising his shucking *rate*; "chia đều cho 5" on Emiliano's apples, giving 3.75 oranges.
  - It should be fixed at the template level (§6, fix 3).

## 5. Bugs found in X4 and fixed (minimal fix plus test, `tests/test_symbolic_g3_fixes.py`)

All of these come from real Haiku-formalized programs: `from_expr_json` output never carries `verb_vi` and can contain anonymous
nodes, while the hand-written fixtures have neither. Run A (as delivered) vs run B (fixed), same seeds, slots and RNG:

| # | Bug (file) | Effect in run A | Fix | Run B |
|---|---|---|---|---|
| 1 | M4 `_flip_verb` invents `verb_old` "thêm"/"bớt" when the op has no verb (`operators.py`) | 19/22 M4 edits named a word not in the sentence. **16/22 M4 renders came back as the unchanged seed** ("không có từ 'bớt'… giữ nguyên bài toán gốc"). | When there is no verb, describe the operation ("phép trừ (bớt, ít hơn, …)" → "phép cộng (…)"); `editor.v3` explains this | 0/16 no-op renders |
| 2 | op `sentence_id` is None when all inputs are derived (`ir.py` `from_expr_json`) | M3 INSERT_STEP without `after_sentence` (6), M4 without `sentence_id` (10) | Inherit the latest sentence of the inputs' producers (topological pass) | 1 and 0 |
| 3 | M3 inserts on anonymous (unlabelled) nodes (`operators.py`) | 4 edits with `operands: [""]` and labels like "sau khi thêm" | Skip unlabelled nodes | 0 |
| 4 | M4 swaps a **world-constant** factor and operands without labels (`operators.py`) | "/100" in a percent became "×100": the phone plan cost 450.150 đô la and passed the gate; 8 edits with empty operands | Require labelled, non-world-const operands; mul↔div only on a *given* factor | 0; M4 applicability 22 → 16 |
| 5 | M4 factor change names the derived output's qid instead of the factor's tag (`operators.py`) | The editor had to guess which tag to change | `qid` = the factor given | — |

The X4 suite still passes: 210/210 original tests plus the new ones.

## 6. Concrete fixes needed before arm Y can pass the gate (ordered by expected effect)

1. **Localize and naturalize the surface (T1 `localize_surface` plus a fluency pass).** Fluency issues are 76% of survivors and 88% of seeds.
   - Add one Haiku pass after rendering. It keeps every `<q>` tag (the same tag check), converts currency labels and units (đô la → nghìn đồng, dặm → km, values unchanged), swaps in Vietnamese names, and fixes the Vietnamese style. Cost is about $0.001 per item.
   - Re-run the cycle check *after* this pass.
   - Alternatively, use the 01 §3.7 native-Vietnamese seeds (Vi-ExamMath-train-like) where they are available.
   - This is the largest lever: seed quality alone separates 48% from 22% teacher-acceptable.
2. **Drop M6, or make the bridge a story element.**
   - It is judged a tail 6/6, with 0 teacher-acceptable items.
   - Either redistribute its π mass to M2, or restrict partners to the same characters and setting and gate the bridge phrase with the judge.
3. **M3: add a story-role check to the template library.**
   - Needed: rates are not changed by "mua thêm"; "chia đều" applies only to a count being shared; the inserted quantity must be consumed after its sentence.
   - This should be added to X4 `TEMPLATES` as (qtype, unit class, **role**: stock / rate / price / total) and should also unblock the chrF gate.
4. **M4/M5: restrict to swaps and targets with a story reading.**
   - Required: add↔sub only where both readings are natural (not "tổng giá lẻ và giá khay"), and no M4 on ratio literals (4:7 → 4:4).
   - M5 is judged near-dup 5/5, so new questions over the same story need a "different situation" component, or M5 should be dropped from the mix.
5. **The formalizer must encode redundant givens as constraints, not as distractors.**
   - In "đặt 18 món hàng: 6 + 4 + 8", 18 was marked a distractor. M1 then produced 6 + 7 + 3 = 16 ≠ 18 and M2 produced a contradictory total, and both passed. M1 was cycle-checked as consistent because the back-formalizer also ignored 18.
   - Fix: a validator rule that a distractor equal to a derivable sum or product is a `constraint`. Also add a clean-value rule for **STATE_FACT** values: M2 stated "400/7 giờ".
6. **Gate calibration changes:**
   - (a) **Answer-in-text:** only count *untagged* occurrences, and not day names. The current rule drops 10/37 M2 items on coincidences.
   - (b) Keep chrF ≤ 0.90 only after fix 3. Measure it on post-fix data before keeping or loosening it.
   - (c) Use bge-m3 0.85 for decontamination and cross-seed dedup only. As a seed-novelty gate it rejects 100% of span-grounded edits by design.
   - (d) Use the editor `warning` as a reject signal (3/34 acceptable).
7. **Judge:** Opus on survivors (or G2's Haiku/Sonnet-v3 cascade with Opus on disagreements). gemini-3.1-flash-lite cannot serve as the second judge (κ 0.14 on teacher).

After fixes 1–5 the gate must be re-run on ≥ 200 items (T1 §2.5). These 66 survivors are not enough to certify absurd ≤ 3%.

## 7. Cost and time

**Measured per call (direct, list prices):**

| Call | Model | $ per call | p50 latency |
|---|---|---|---|
| formalize | Haiku | $0.00081 (1.1 attempts per seed) | 5.0 s |
| editor | Haiku | $0.00093 (1.08 attempts per render) | 5.8 s |
| judge | Opus | $0.0139 | 6.3 s |
| blind solve | Opus | $0.0084 | 3.4 s |
| cycle | gemini-3.1-flash-lite | $0 | 4.0 s |

The Gemini cycle run covered 139 items with 271 attempts (1.95 per item) in 8.2 min on 6 free-tier keys, about **17 items/min**.

**Projection for 10k surviving items** (yield 0.367 per slot; the pipeline as is):

| Item | Value |
|---|---|
| Slots / seeds needed | 27.3k slots / 9.1k seeds at J = 3. That exceeds the 7.47k GSM8K pool, so J ≥ 4 or MATH seeds are needed |
| Formalize | 9.1k × $0.00081 × 1.1 ≈ **$8** ($4 Batch), < 1 h at 24 in flight |
| Editor | 0.64 renders per slot × 1.08 attempts ≈ 18.9k calls ≈ **$18** ($9 Batch), ≈ 2 h |
| Cycle (Gemini) | ≈ 21k items ≈ **21 h** on 6 free keys at 17 items/min. This is the wall-clock bottleneck, and free-quota limits apply |
| Opus judge on survivors | 10k × $0.0139 = **$139** direct ($70 Batch). Opus blind-solve is not needed in production (100% agreement with the program after the cycle check) |
| **Total** | **≈ $165 direct / $83 Batch, ≈ 24 h wall-clock** |

**For 10k teacher-acceptable items:** the yield is 0.128 per slot, so 78k slots and 26k seeds are needed (> 2.5× the whole seed pool). With
the judge as a filter on 28.7k survivors, the judge alone costs about $400. **Scaling is pointless before fixes 1–5.**

## 7a. Five best and five worst survivors (Opus verdicts; idx = Vi-GSM8K train index)

**Best:**
1. idx 1141, M2 (natural 4, no issue): "Một chủ cửa hàng thú cưng có 12 con chim, vài con chó con chưa kiểm đếm, 5 con mèo và 15 con nhện… Lúc này, cửa hàng còn 20 con vật. Ban đầu cửa hàng có bao nhiêu con chó con?" (4). Opus: well-posed inversion, whole numbers.
2. idx 5625, M3: "Hannah đang mua táo với giá 5 đô la mỗi kilôgam. Sau đó, cửa hàng giảm giá 15%… được giảm thêm 40% trên giá… đã giảm" (25,5). Opus: a real mid-story step that adds new reasoning (successive percentages).
3. idx 999, M2: "…quỹ lớp của cô Grayson còn lại $90. Chi phí chuyến đi cho mỗi học sinh là bao nhiêu?" (3). Opus: correct reverse reasoning; only "$" and the foreign name are minor flaws.
4. idx 3938, M2: "Frank… cắt khay bánh thành các cột đều nhau và 3 hàng đều. Mỗi người có thể ăn 4 cái… 6 người… thành bao nhiêu cột?" (8).
5. idx 7094, M3: "Trong thời gian đó, em trai cô mượn xe đúng 440 ngày, nên cô chỉ đạp xe vào những ngày còn lại" (1965). Opus: the step sits mid-story, not after the answer.

**Worst** (all passed every gate):
1. idx 3476, M6: "Hannah có ba người anh em trai… đều 8 tuổi. Tuổi của Hannah gấp đôi tổng số tuổi… Molly… đạp xe… cho đến khi cô ấy bằng tuổi của Hannah" (Hannah would be 48, with an artificial join; NL-ABSURD, NL-TACKED).
2. idx 2020, M3: "Bob mua thêm 5 con hàu nữa, và số hàu anh ấy có thể tách trong cùng khoảng thời gian ấy cũng tăng thêm đúng bằng số đó" (buying oysters raises his shucking rate; NL-ABSURD).
3. idx 1692, M2: "Tổng thời gian cô ấy đi từ New Orleans đến San Francisco là 400/7 giờ" (an unclean stated fact; ambiguous layover; NL-UNANSW).
4. idx 4517, M6: "Nếu một gói 50 thẻ ghi chú có giá bằng đúng tổng chi phí của giỏ trái cây nói trên…" (a 28-đô-la pack of note cards via an arbitrary link).
5. idx 6556, M4: "số que kem làm được từ nước ép sẽ được cộng thêm vào số que kem trong gói" (popsicles added to the sticks that make them; a nonsense relation flip).

## 8. Caveats

- n = 60 seeds and 66 survivors. The CIs are wide (±11–12 pp on the headline rates), but the teacher-gate failure is far outside them (upper bound 47% vs 70%).
- Opus alone is the reference; there is no human rating (G1 and G2 had none either). The seed baseline used a "no transformation" instruction, M0, so its new and op fields are meaningless and were not used.
- Run A was not judged, to save budget. Its comparison is mechanical (§5), and its cycle check is partial (9 rows).
- The `localize_surface` step of T1 §2.2 was not implemented (it was not in the brief). §3 shows it matters most.
- Nothing has been committed. No test, benchmark or eval item went to Gemini or Modal: only train dev seeds and their derivatives were embedded, and X5's eval embeddings were reused locally.

## 9. Reproduce

```
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic.seeds
set G3_RUN=run_b
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic.pipeline formalize|mutate|render|cycle
E:\venv_tools\Scripts\modal run -m vi_math_verified_aug.pilot_symbolic.embed_modal
.venv\Scripts\python -m vi_math_verified_aug.pilot_symbolic.cosine
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic.pipeline gates
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic.judge opus|solve|seeds|second
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic.report
```

Every LLM call is cached, so a re-run is free. `z3-solver==5.1.0.0` was installed into `E:\venv_tools`, so the X4 library
and the anthropic SDK run in one process (no torch locally).
