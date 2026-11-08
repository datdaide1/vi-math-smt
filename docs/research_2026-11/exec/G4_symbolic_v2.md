# G4: symbolic arm (arm Y), second iteration — renderer and operator fixes (2026-11-07)

**Brief:** coordinator task G4. Rules from `EXEC_BRIEF2.md`. Inputs: `exec/G3_symbolic_pilot.md`, `exec/G2_judge_prompt.md`, `tech/T1_synthesis.md` §2.2–2.5 and `exec/X4_symbolic.md`.

**Code:** `vi_math_verified_aug/pilot_symbolic_v2/`:
- `common`: ws G4 client with a $5.9 hard cap.
- `localize`: IR-level USD→đồng rescale, cards and the R2 IR view.
- `roles`: operator fixes.
- `pipeline`: stages mutate, render, cycle and gates.
- `judge`: Message Batches.
- `embed_modal` + `cosine`: bge-m3.
- `report`.

The package reuses G3's `pilot_symbolic` (editor view, tag check, gates, `load_ir`), X4's `symbolic` library and `common/llm.py`. G3's and X4's files were not edited.

**Prompts (new):**
- `prompts/story_check.v1.md`: Haiku story-role check.
- `prompts/fluency.v1.md`: the R1 pass.
- `prompts/ir_render.v1.md`: the R2 renderer.
- `prompts/fluency.v2.md`: R1b. It was rendered only (§8).

The renderer of variant E, `editor.v3`, and the judges `validity_judge.v2/v3` and `solution_writer.v2` are unchanged.

**Data:** `data/pilot/symbolic_v2/`:
- `mutate.jsonl`;
- `render_{E,R1,R2}.jsonl`, `cycle_{E,R1,R2}.jsonl`, `gates.jsonl`, `cosine.jsonl`;
- `judge_opus.jsonl`, `solve_opus.jsonl`, `judge_sonnet.jsonl`, `native_N.jsonl`;
- `report_tables.md` (every table below), `metrics.json`.

**Tests:** `tests/test_pilot_symbolic_v2.py` (8 tests) plus G3's `tests/test_pilot_symbolic.py`: **11 passed**.

**Spend:**
- **Claude $4.10 of $6** (logged in `logs/api/G4/claude.jsonl`; judges at the Batch price):
  - Opus judge $1.80;
  - Opus solve $0.99;
  - Sonnet-v3 $1.00;
  - Haiku $0.31 (story checks $0.05, editor $0.01, fluency $0.05, IR render $0.09, native v3 $0.09).
- **Modal ≈ $0.01** (one bge-m3 call; metered $5.03 → $5.04, `modal_billing_{before,after}.txt`).
- **Gemini:** free tier, dev items only. No test or benchmark item was sent anywhere. No NIM.

## 0. Answer

The comparison is **paired** with G3: the same 60 seeds, the same 180 slots, the same cached Haiku IRs and the same RNG. M6 is dropped, so **160 slots** are compared. The judge of record is Opus 5.5 with `validity_judge.v2`. Every rate below is on **survivors** (cycle check + all gates), with Wilson 95% CIs.

| variant | survivors / 160 slots | teacher (≥70%) | new M2–M5 (≥60%) | leak (0%) | absurd (≤3%) | accept* | accepted per slot | cycle pass | Opus solve = program |
|---|---|---|---|---|---|---|---|---|---|
| G3 as published (incl. M6) | 66 / 180 | 35% [24–47] | 55% [41–69] | 0% | 12% [6–22] | 24% | 0.09 | 77% | 100% (66/66) |
| **E** = G3 renderer + operator fixes | 58 | 47% [34–59] | **76% [61–87]** | 0% | **7% [3–16]** | 33% | 0.12 | 88% [79–93] | 98% (57/58) |
| **R1** = E + Haiku fluency/localization pass | 77 | **61% [50–71]** | 67% [53–78] | 0% | 21% [13–31] | **60%** | **0.29** | 86% [77–91] | **100% (77/77)** |
| **R2** = free render from the IR | **85** | 51% [40–61] | 70% [57–80] | 0% | 24% [16–34] | 47% | 0.25 | **98% [92–99]** | 99% (84/85) |
| N = native v3, same slots (no certificate) | 91 parsed | **80% [71–87]** | **82% [71–89]** | 1% | 5% [2–12] | 78% | 0.44† | – | 97% (generator answer) |

\*accept = teacher ∧ valid ∧ ¬absurd ∧ ¬leak ∧ ¬tail. †N: accepted per *mutated* slot (91), with no verification gate.

1. **No arm-Y variant passes the T1 §2.5 gate.**
   - Every variant passes leak (0%) and new (≥ 60%).
   - None reaches teacher ≥ 70% or absurd ≤ 3%.
   - The native arm also fails the absurd threshold on these seeds (5% [2–12]).
2. **Fluency was the teacher-acceptability bottleneck, and the fluency pass fixes it.**
   - On the same slots, R1 vs E (tag-passing, exact McNemar):
     - teacher 37% → 53% (22 vs 7 discordant, p = .008);
     - valid 29% → 56% (p = .0001);
     - Opus "fluency issue" 71% → 32%;
     - natural 3.3 → 3.9.
   - R1 also stops the chrF ≤ 0.90 gate from deleting M3: E keeps 4/18 M3 items, R1 keeps 14/18.
   - Accepted items per slot rise from 0.12 (E) and 0.09 (G3) to 0.29.
3. **The new defect is absurdity, and most of it comes from my localization rule, not from the fluency pass itself.**
   - Absurd rose from 11% (E) to 27% (R1) on the same tag-passing items (15 vs 1 discordant, p = .0005).
   - On slots whose program was money-rescaled (USD ×1000 → đồng, 38 slots), R1 is absurd on **15/38**, against 4/38 for E.
   - On slots without money (52), R1 is absurd on 9/52, against 6/52 for E.
   - Opus's notes name the cause: "400 đồng per egg", "field trip 3.000 đồng per student", and "can holds 12 gam" (ounces relabelled as grams with the number unchanged).
   - The fix is **a realistic money scale and no relabelling of feet or ounces**. It is implemented as variant R1b (`fluency.v2`, ×10,000).
   - R1b is rendered (tag pass 88/91), but its cycle check was **not** completed: the Gemini flash-lite free tier hit its daily quota (500 RPD/project) mid-run (§8). It is therefore not judged.
4. **Faithfulness is not traded away. R2 does not lose it; it gains.**
   - R2 cycle pass is 98% [92–99], against 88% (E) and 86% (R1).
   - Opus's blind solve agrees with the program on 99% of R2 survivors and 100% of R1 survivors.
   - So rendering from the IR without the seed text does **not** cost answer faithfulness. It removes the seed's ambiguities, which the round trip then confirms.
   - The C3/P1 trade-off measured here is therefore text quality (teacher, absurd), not faithfulness.
5. **Operator fixes help, at a large cost in yield.**
   - The Haiku story-role check rejects most M3 and M4 candidates: story_meaning = false on 107/125 M3 and 11/15 M4 candidates.
   - M3 survives on 18/44 slots, against 42 in G3. M4 survives on 4/35, against 16.
   - Items that pass are better. E vs G3 on survivors: new 55% → 76%, absurd 12% → 7%, tail 14% → 0%.
   - Answer-tail / identity remain 0 by construction, and the judge-flagged tail rate is 0–2%.
6. **Recommendation: R1 (edit + fluency pass), with the R1b localization fix**, as the arm-Y renderer.
   - It has the highest teacher rate and the highest accepted-per-slot yield (0.29 vs R2 0.25, E 0.12), and 100% answer agreement.
   - R1 vs R2 is **not** significantly different on any label (teacher 62% vs 55% on 73 both-surviving slots, p = .38). R1 is chosen on point estimates.
   - R2 is the fallback whenever the seed text is unavailable, or when an English render from the same IR is needed. R2 is also one Haiku call instead of two.
   - **Projection (R1, as measured): about $63 Batch / $125 direct per 10k accepted items**, needing about 35k slots (≈ 11.6k seeds at J = 3). The derivation is in §7.

## 1. What changed vs G3 (applied to E, R1 and R2 alike)

- **M6 dropped:** 20 slots. G3 judged its survivors tail on 6/6; X4's semantic-tail check fires on 100/100 by construction. Its π mass is not redistributed in this paired run (§7 uses π renormalised over M1–M5).
- **M3/M4/M5 story-role checks** (`roles.py`). Candidate order per slot: G3's own program first, so unchanged slots stay byte-identical to G3; then up to 12 fresh X4 draws (`Random(f"{key}:20261108:g4r{r}")`).
  - (a) **Deterministic filter.** M3 templates are filtered by the role of the node they act on: count/person templates only on a count *stock* (not a rate, per-item or difference node); giảm giá/thuế/phí never on income or savings; "phí" never on a per-item price. M4 never swaps on ratio literals or on unlabelled outputs, and add↔sub only between operands of the same type and unit. An M5 target must be a story quantity, not a ratio, percent, time point or unit conversion. The filter rejected 8 M3, 8 M4 and 1 M5 candidates.
  - (b) **Haiku story check** (`story_check.v1`, effort low; it sees the seed and a plain description of the change, never the new answer). The candidate is accepted iff type_ok ∧ story_meaning, with at most 4 checks per slot.
  - A semantic-tail M3 (new answer = old ± c or k·old) is rejected whenever its step has no story meaning: 32 of the 34 semantic-tail candidates were rejected.

| op | slots | G3 mutated | G4 mutated | same program as G3 | new program | rejected by checks | Haiku checks |
|---|---|---|---|---|---|---|---|
| M1 | 27 | 25 | 25 | 25 | 0 | 0 | 0 |
| M2 | 43 | 38 | 38 | 38 | 0 | 0 | 0 |
| M3 | 44 | 42 | **18** | 9 | 9 | 24 | 125 |
| M4 | 35 | 16 | **4** | 3 | 1 | 12 | 15 |
| M5 | 11 | 7 | 6 | 6 | 0 | 1 | 6 |
| M6 | 20 | 12 | 0 (dropped) | – | – | – | – |

The Haiku verdicts look right on reading:
- "gives away a book *after* shelving, and the question asks the starting count" is rejected;
- "sum instead of difference for 'cao hơn bao nhiêu'" is rejected;
- "donated turtles reduce Owen's count" is rejected;
- "15% tax on the tray price" is accepted;
- "2 more turtles than Owen" is accepted.

Its strictness is the main yield cost: **M4 as implemented by X4 (add↔sub swaps) almost never yields a sensible story.**

**Variants on the same mutated programs:**
- **E:** M1 deterministic; M2–M5 Haiku `editor.v3`, exactly G3. 81 of the 91 rows are G3's own render, cycle, judge and solve rows, reused. The 10 re-mutated slots are new.
- **R1:** E's tagged output, then Haiku `fluency.v1`.
  - Inputs: a localization card (G1 `cards_v1`, one per seed), Vietnamese conventions and natural phrasing.
  - Every `<q>` tag and value is preserved. Money tags are rewritten to the localized đồng values.
  - Never sees the answer. The **tag check and cycle check are re-run** on the new text.
- **R2:** Haiku `ir_render.v1` writes the problem from the localized program: labelled givens with values, world constants, relations as formulas over `[ids]`, unknown/equation, and the question label, plus the card.
  - It never sees the seed text, a derived value or the answer.
  - Then the tag check and the cycle check.
- **Localization** (`localize.py`):
  - Applies to R1/R2 programs with USD or cents: every money node is rescaled ×1000 (cent ×10) to đồng and the program is re-executed.
  - The rescale is accepted only if every node scales by exactly its expected factor; otherwise the currency is kept.
  - 38/91 programs were rescaled, and none failed the homogeneity check.
  - R1/R2 tag values, cycle targets and solve targets use the localized program.
- **Gates** (G3's, with G3 §6 fixes (a)/(c)):
  - answer-in-text counts only the **untagged** text, excluding day names;
  - chrF ≤ 0.90 vs the seed applies to M2–M5;
  - bge-m3 cos < 0.85 is used for decontamination (13,552 Vi eval items) and cross-seed duplicates only: 0 hits in any variant, max cos to eval 0.819 / 0.807 / 0.791;
  - 5-gram Jaccard dedup.
  - The cycle check is gemini-3.1-flash-lite `back_formalizer.v1`, blind, with exact Fraction equality.

## 2. Stage yield (count, % of the 160 non-M6 slots)

| variant | mutated | tag ok | cycle ok | no leak/answer | chrF ≤ .90 | cos < .85 | survivor |
|---|---|---|---|---|---|---|---|
| G3 (its non-M6 slots) | 128 (80%) | 127 (79%) | 100 (62%) | 86 (54%) | 63 (39%) | 63 | 60 (38%) |
| E | 91 (57%) | 90 (56%) | 79 (49%) | 79 | 61 (38%) | 61 | 58 (36%) |
| R1 | 91 | 90 (56%) | 77 (48%) | 77 | **77 (48%)** | 77 | 77 (48%) |
| R2 | 91 | 87 (54%) | **85 (53%)** | 85 | 85 | 85 | **85 (53%)** |

- The fixes cut mutation coverage from 80% to 57% of slots, almost all of it in M3/M4.
- R1 and R2 recover the losses at the chrF gate, because their texts are no longer near-copies of the seed. Median bge-m3 cos to the seed is 0.98 for E, 0.83 for R1 and 0.79 for R2.
- **Tag check:** E 90/91; R1 90/91 (pass@1 83); R2 87/91 (pass@1 71). Most failures are untagged numerals ("hai", "đôi") written outside tags.
- **Renderer warnings:**
  - R2 raised a warning on 38/87 items (teacher 45% with a warning vs 53% without).
  - The warning stays a useful E filter: 17% vs 40%.

## 3. Quality (Opus 5.5) — survivors, all tag-passing items, by operator

All tag-passing items (before cycle check and gates):

| set | n | teacher | new M2–M5 | absurd | valid | unansw | fluency issue | natural | accept* |
|---|---|---|---|---|---|---|---|---|---|
| G3 (no M6) | 127 | 28% [21–37] | 38% | 20% | 24% | 19% | 76% | 3.20 | 20% |
| E | 90 | 37% [27–47] | 58% | 11% [6–19] | 29% | 11% | 71% | 3.30 | 26% |
| R1 | 90 | 53% [43–63] | 62% | 27% [19–37] | 56% | 13% | 32% | 3.88 | 52% |
| R2 | 87 | 49% [39–60] | 69% | 25% [17–35] | 52% | 13% | 39% | 3.74 | 46% |

Survivors by operator (teacher / absurd / new / n):

| variant | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| E | 40% / 5% / – / 20 | 52% / 10% / 93% / 29 | 75% / 0% / 50% / 4 | – | 20% / 0% / 0% / 5 |
| R1 | **74%** / 13% / – / 23 | 58% / 23% / **100%** / 31 | 57% / 36% / 29% / 14 | 75% / 25% / 0% / 4 | 20% / 0% / 20% / 5 |
| R2 | **75%** / 12% / – / 24 | 48% / 30% / 94% / 33 | 29% / 29% / 59% / 17 | 25% / 50% / 0% / 4 | 33% / 0% / 17% / 6 |
| N | 80% / 0% / – / 25 | 89% / 8% / 95% / 38 | 61% / 11% / 61% / 18 | 100% / 0% / 75% / 4 | 67% / 0% / 67% / 6 |

- **M2 remains the operator that makes genuinely new items** (new 93–100%).
- M5 is near-dup in every arm-Y variant (new 0–20%).
- **M3 absurdity under R1 (36%)** is the largest single cell. It combines a story-check acceptance that is still too permissive (e.g. "cho đi 1 quả bơ" changes a basket's *cost*) with the localization prices.
- Seed-inherited formalization defects also reach R1/R2 survivors. Example: idx c8357de93a24, "đặt 18 món hàng" = 6 + 7 + 3 = 16, where the redundant total was formalized as a distractor (G3 §6 fix 5, not in this brief).

**Paired comparisons** (exact McNemar; both renders exist for the slot):

| pair | set | n | label | A-only | B-only | rates A → B | p |
|---|---|---|---|---|---|---|---|
| E vs R1 | tag-passing | 90 | teacher | 7 | 22 | 37% → 53% | **.008** |
| E vs R1 | tag-passing | 90 | absurd | 1 | 15 | 11% → 27% | **.0005** |
| E vs R1 | tag-passing | 90 | valid | 7 | 31 | 29% → 56% | **.0001** |
| E vs R2 | tag-passing | 86 | teacher | 15 | 25 | 38% → 50% | .15 |
| E vs R2 | tag-passing | 86 | absurd | 3 | 16 | 9% → 24% | **.004** |
| R1 vs R2 | both survive | 73 | teacher | 13 | 8 | 62% vs 55% | .38 |
| R1 vs R2 | both survive | 73 | absurd | 2 | 4 | 21% vs 23% | .69 |
| R1 vs N | both survive | 77 | teacher | 8 | 22 | 61% vs 79% | **.016** |
| R1 vs N | both survive | 77 | absurd | 15 | 2 | 21% vs 4% | **.002** |
| R2 vs N | both survive | 84 | teacher | 6 | 31 | 50% vs 80% | **<.0001** |

**How close arm Y gets to arm N** (native v3, same seeds and slots, same card, same judge):
- Arm N leads by about 19 pp on teacher (R1 61% vs N 80%; G2 had 92% on other seeds) and about 16 pp on absurd.
- Arm Y's distinctive property is the certificate: 100% answer agreement after the cycle check. Arm N's generator answer is wrong on 3/91 items.

**Second judge (Sonnet 5.5, `validity_judge.v3`)** vs Opus on all 358 judged items:
- κ teacher 0.64, valid 0.63, new 0.63, absurd 0.53 (per variant: R1 0.73 / 0.75 / 0.59 / 0.58; E 0.49 / 0.41 / 0.59 / 0.44).
- Sonnet ranks the variants the same way: teacher on survivors E 41%, R1 53%, R2 50%, N 75%.
- Sonnet is *stricter* on absurd: E 17%, R1 35%, R2 36%, N 15%. That is the same direction G2 saw.
- The gate verdict (FAIL for every arm-Y variant) holds under both judges.

## 4. Answer faithfulness (reported separately)

| variant | cycle pass / tag-passing | Opus agrees: survivors | Opus agrees: cycle passed | Opus agrees: cycle failed |
|---|---|---|---|---|
| G3 (no M6) | 79% [71–85] | 100% (60/60) | 98% (98/100) | 48% (13/27) |
| E | 88% [79–93] | 98% (57/58) | 99% (78/79) | 91% (10/11) |
| R1 | 86% [77–91] | **100% (77/77)** | 100% (77/77) | 92% (12/13) |
| R2 | **98% [92–99]** | 99% (84/85) | 99% (84/85) | 0% (0/2) |
| N | – | – | – | generator answer = Opus on 97% (88/91) |

- **R2 does not lose faithfulness.** Its cycle pass by operator is M1 25/25, M2 33/34, M3 17/18, M4 4/4 and M5 6/6. Writing from the explicit relations seems to produce *less ambiguous* text than editing a translated seed.
- The two R2 cycle failures are both absurd items that Opus also disagrees with.
- The fluency pass (R1) keeps the program answer: 77/77. It loses 2 cycle passes relative to E (77 vs 79); 13 items fail in both.

## 5. Best and worst survivors (Opus verdicts; full texts in `report_tables.md` §H)

**E**
- Best:
  - `bd4abb643a98:1` M2: "…Lúc này, cửa hàng còn 20 con vật. Ban đầu cửa hàng có bao nhiêu con chó con?" Opus: well-posed inversion, x = 4.
  - `68d313f51190:1` M2: "…Tổng thời gian Adam ở trường trong 3 ngày này là 21 giờ. Vậy vào thứ Ba, Adam có bao nhiêu tiết học?"
  - `15db8654b240:1` M2: the spa discount inversion, flawed only by "247.5 đô la" and stiff wording.
- Worst:
  - `d46a5281e278:1` M2: "Tổng thời gian … là 400/7 giờ" (ugly stated fact, ambiguous layover; absurd + unanswerable).
  - `22968fe49dd0:3` M1: "dạy 12 tiết mỗi ngày".
  - `1bf27fde2604:2` M2: a 103-xu saving per egg.

**R1**
- Best:
  - `79ee7b636558:3` M2: "Nam nuôi một số con rùa, còn Lan có ít hơn Nam 5 con rùa… Lúc này, Nam có tất cả 25 con rùa. Hỏi lúc đầu Nam đã nuôi bao nhiêu con rùa?" (natural 5).
  - `316a32678874:2` M4: "nhiều hơn lần đầu 70 điểm" (natural 5, but near-dup).
  - `68d313f51190:3` M5 (natural 5, near-dup).
- Worst:
  - `d46a5281e278:1` (the inherited 400/7).
  - `c8357de93a24:3` M1 and `c8357de93a24:2` M2: the 18-item contradiction plus "dép 3.200 đồng", "son 1.250 đồng" from the ×1000 rescale.

**R2**
- Best:
  - `7930ae84d804:3` M2: "Máy A đóng nắp được 12 chai trong 1 phút… tổng số chai cả ba máy đóng nắp được là 185 chai. Hỏi cả ba máy cùng đóng nắp trong bao nhiêu phút?" (natural 5; a new division step).
  - `bd4abb643a98:1` M2: the pet-shop inversion, rewritten fluently.
  - `2e98b473b0c4:2` M1: a theatre ticket with an online fee.
- Worst:
  - `c8357de93a24:2`/`:3` (inherited contradiction, unrealistic prices).
  - `203d8a2add9c:1` M3: "phần nước ngọt nặng 12 gam" (ounce → gam relabel) and "thêm 2 lon rỗng để cầu chắc chắn hơn".

**N** (reference)
- Best: `c739e9383ea8:3` M2 (wedding thank-you cards, 90% → 80% → 250 invitations); `5e98d9f5dcfc:1` M3 ("2 hộp sữa đã hết hạn … trả lại").
- Worst: `d7ebf53a5d18:2` M3 (ambiguous whether the extra trip counts) and `df5a7d0912c8:2` (20 bus tickets including airport tickets).

## 6. Recommendation for arm Y: renderer R1 (edit, then fluency/localization pass)

Evidence, all paired on the same 160 slots:
- **(i) Highest teacher rate:** 61% [50–71] on survivors, and the only arm-Y variant whose CI reaches 70%.
- **(ii) Highest accepted yield:** 0.29 accepted per slot, against R2 0.25, E 0.12 and G3 0.09.
- **(iii) Perfect answer agreement** on survivors (77/77), with no tail and no leak.
- **(iv) Absurdity is attributable and fixable:** the excess over E is concentrated in money-rescaled slots (15/38 vs 9/52), and the Opus notes name the ×1000 price scale and the ounce/feet relabel.
- R1 vs R2 is a statistical tie (p ≥ .38 on every label). R2's advantages are one call, 98% cycle pass and no dependence on the seed text. That makes R2 the right renderer for the English mirror, which renders from the same IR.

**Required before scaling:**
1. **Use R1b localization:** a realistic money scale (×10,000 đồng per USD, cents ×100) and no feet/ounce relabel. R1b is built (`fluency.v2`, `localize(usd=10000)`) and rendered 88/91 (`render_R1b.jsonl`), but not yet cycle-checked or judged (§8).
2. **Add a cheap absurd/validity gate**, e.g. the G2 Haiku-v3 ∧ flash-lite cascade with Opus on disagreements. Sonnet-v3 has κ 0.58 with Opus on absurd for R1.
3. **Fix the formalizer for redundant totals** (G3 §6 fix 5): the 18 = 6 + 7 + 3 contradiction reaches survivors in every variant.
4. **Re-weight π** away from M4/M5: M4 survives on 4/35 slots, and M5 is near-dup.

## 7. Projected yield and cost per 10k accepted items (R1, as measured)

- **Yield:**
  - accepted per scheduled non-M6 slot = 46/160 = 0.29;
  - so **≈ 34.8k slots ≈ 11.6k seeds at J = 3**, which exceeds the 7.47k GSM8K train pool (use J ≥ 5 or add MATH seeds);
  - survivors per slot 0.48, so 16.7k survivors to screen.
- **Generation cost**, measured over the 160 slots ($ direct):
  - formalize 60 seeds × $0.00089 = $0.053;
  - story checks $0.054;
  - editor ≈ $0.065 (G3 rate);
  - fluency $0.055;
  - total ≈ **$0.0014 per slot**, i.e. $49 direct / **$25 Batch** for 34.8k slots.
- **Cycle check:** about 19.5k items (tag-passing).
  - On the gemini-3.1-flash-lite free tier at 500 RPD per project × 6 projects ≈ 2.3k items/day, that is **≈ 8–9 days**.
  - With Haiku as back-formalizer in Batch it is ≈ $10 and hours. This is a different-family requirement (T1) that would then be violated, so record it.
- **Screening survivors** with the G2 cascade (Haiku-v3 + flash-lite, Opus on ≈ 20% disagreements, 5% audit): ≈ $0.0034 direct per item, so 16.7k × $0.0034 ≈ $57 direct / **$28 Batch**.
- **Total ≈ $63 Batch / $125 direct per 10k accepted items.** The wall-clock bottleneck is the cycle check on free Gemini quota.
- For reference: G3's pipeline needed ≈ 78k slots for 10k teacher-acceptable items; arm N (G2) costs ≈ $31 Batch per 10k accepted, without a certificate.

## 8. Caveats

- **n:** 58–85 survivors per variant. CIs are ±10–12 pp. The absurd ≤ 3% gate cannot be certified on these n (T1 §2.5 asks for 200 items plus a human audit).
- **In-sample tuning:** the story-check prompt and the role filters were written from G3's failure analysis of these same 60 seeds. The examples in the prompt are paraphrased, not copied. M3/M4 gains may therefore be optimistic. The next run must use fresh seeds.
- **R1b is incomplete:**
  - All 6 Gemini projects hit the flash-lite free-tier daily cap (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 500; 429s in `logs/api/G4/gemini.jsonl`) after 4 R1b cycle rows.
  - R1b is not judged, and its effect on absurd is a hypothesis backed by the localized/non-localized split in §0.
  - **This quota is also the scale bottleneck for any Gemini-based cycle check (§7).**
- The judges are LLMs (Opus as reference, Sonnet as second judge), with no human labels. G3's E rows were judged with direct calls; the new rows used Batch with the same model, prompt and effort.
- N (native v3) was run on the 91 mutated slots, so its seeds and operators match arm Y. It has no verification step beyond Opus's blind solve.
- Nothing was committed.

## 9. Reproduce

```
set PYTHONPATH=.
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline mutate      # fixed operators + Haiku story checks
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline render      # E / R1 / R2
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline cycle      # gemini-3.1-flash-lite, blind
E:\venv_tools\Scripts\modal run -m vi_math_verified_aug.pilot_symbolic_v2.embed_modal
.venv\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.cosine
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline gates
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.judge native        # arm-N reference
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.judge all           # Batch: Opus v2, Opus solve, Sonnet v3
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.report
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline render_r1b  # R1b (rendered only)
```

Every LLM call is cached (`.cache/vimath/llm_cache.sqlite`), so a re-run is free.
