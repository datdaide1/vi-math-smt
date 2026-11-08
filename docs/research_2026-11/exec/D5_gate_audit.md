# D5 — Joint quality-gate audit of the FINAL training sets (2026-11-08)

## Method

**Scope:** all 8 arm × language cells of the final `select_v1` sets, `data/train/final/{Y,N}/{vi,en}/train_{plus,minus}.jsonl`, n = 1,808 each.
- VI: 100 items per cell.
- EN: 50 items per cell. The EN audit was approved later with +$1.4; at the measured $0.0068 per item, 100 per cell would not fit.
- EN uses `validity_judge_en.v1`, a line-by-line English twin of v2.

**Judge:** one judge for both arms: Opus 5.5 with `validity_judge.v2` (the G1–G4 judge of record), effort medium, via the Message Batches API.
- `max_tokens` was 1,500. Judge outputs run about 500–700 tokens, so no verdict was truncated: 390/390 parsed.
- An item that appears in both the plus and minus samples is judged once.
- Batches are recorded in `data/audit/d5/pending_batches.jsonl`, so a re-run never pays twice.

**Sampling:**
- In each cell, the 100 seeds with the lowest sha1("D5aud|vi|<variant>|<seed>") are taken.
- Y+ and N+ therefore share the same 100 seeds, and so do Y− and N−. This allows paired McNemar tests within each sign.
- Samples are frozen in `data/audit/d5/sample_*`.

**Code and data:**
- `vi_math_verified_aug/analysis/c3_d5_audit.py` (sampling, metrics, manipulation check) and `c3_judge.py` (crash-safe Batches judge).
- Tests: `tests/test_c3_d5_audit.py`, 2 passed.
- Judgments: `data/audit/d5/judge_opus.jsonl`. Tables: `tables.md`, `metrics.json`.

**Spend** (all Batch):
- VI: $2.665 for 390 items. 7 N− items were left unjudged when an earlier cap was reached, so the VI N− cell has n = 93.
- EN: $1.159 for 198 items.
- D5 total: **$4.40 of $4.8**, which includes C3's R1b judging at $0.576. The ledger is `logs/api/D5/claude.jsonl`.

## 1. Gate metrics (Wilson 95%; T1 §2.5 thresholds)

| Cell | n | teacher (≥ 70%) | new M2–M6 (≥ 60%) | leak (0%) | absurd (≤ 3%) | Gate | valid | accept* | unanswerable | fluency | near-dup | critical | natural |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Y+ vi** | 100 | 66% [56–75] | 89% [79–95] | 1% [0–5] | **12% [7–20]** | FAIL 1/4 | 68% | 66% | 5% | 36% | 11% | 0% | 3.95 |
| **Y− vi** | 100 | 50% [40–60] | 86% [75–92] | 1% [0–5] | **21% [14–30]** | FAIL 1/4 | 53% | 48% | 11% | 53% | 13% | 1% | 3.71 |
| **N+ vi** | 100 | **89% [81–94]** | 76% [65–84] | 1% [0–5] | 5% [2–11] | FAIL 2/4 | 88% | 87% | 3% | 15% | 24% | 0% | 4.31 |
| **N− vi** | 93 | 75% [66–83] | 70% [58–80] | 1% [0–6] | 9% [4–16] | FAIL 2/4 | 77% | 75% | 11% | 17% | 30% | 0% | 4.17 |
| **Y+ en** | 50 | 74% [60–84] | 83% [67–92] | 0% [0–7] | 6% [2–16] | FAIL 3/4 | 74% | 70% | 4% | 26% | 17% | 0% | 4.02 |
| **Y− en** | 50 | 52% [39–65] | 79% [63–90] | 6% [2–16] | 10% [4–21] | FAIL 1/4 | 54% | 50% | 14% | 44% | 21% | 2% | 3.78 |
| **N+ en** | 50 | **96% [87–99]** | 65% [50–78] | 0% [0–7] | 2% [0–10] | **PASS 4/4** | 96% | 96% | 0% | 8% | 35% | 0% | 4.42 |
| **N− en** | 50 | 82% [69–90] | 69% [54–81] | 4% [1–13] | 2% [0–10] | FAIL 3/4 | 94% | 82% | 2% | 16% | 28% | 0% | 4.42 |

\*accept = teacher ∧ valid ∧ ¬absurd ∧ ¬leak ∧ ¬tail (G4 definition).

- **EN N+ is the only cell that passes all four T1 §2.5 thresholds** (on the point estimate; n = 50, so absurd 1/50 has a CI up to 10%).
- EN scores higher than VI on teacher-acceptability in every cell: Y+ +8, Y− +2, N+ +7, N− +7 pp.
- EN has fewer fluency issues: Y+ 36 → 26%, Y− 53 → 44%, N+ 15 → 8%, N− 17 → 16%.
- This fits C3's finding that translationese is inherited from the seed: EN seeds are the originals.
- These are different items and samples (n = 50 vs 100), so these are descriptive comparisons, not tests.
- In VI, no cell passes. The two "+" cells come closest:
  - N+ passes teacher and new, but its absurd rate is 5% against a 3% threshold;
  - Y+ passes new only.
- Leak is 1/100 in every cell, so the 0% bar is missed by one item each time.
- These are the training sets as they are; the gate is reported, not enforced, because training had already started.

D1's separate pool-level audit is consistent and is **not pooled** with this one, because it sampled the pool, not the final file. It covered 80 random N+ pool items (`exec/D1_N_scale.md` §5): absurd 2.5% [0.7–8.7], teacher 78.8%.

## 2. Y vs N, paired on the same seeds (exact McNemar; RD = Y − N)

| Sign | not valid | not teacher | absurd | unanswerable | fluency |
|---|---|---|---|---|---|
| + (100 seeds) | **+20.0 [+8.8, +30.7]**, p = .0008 | **+23.0 [+12.2, +33.3]**, p = .0001 | +7.0 [−0.9, +15.3], p = .12 | +2.0 [−4.3, +8.6] | **+21.0 [+9.7, +31.7]**, p = .0005 |
| − (93 seeds) | **+25.8 [+12.0, +38.3]**, p = .0005 | **+25.8 [+12.2, +38.1]**, p = .0004 | **+12.9 [+3.0, +22.9]**, p = .017 | +1.1 [−8.4, +10.6] | **+34.4 [+20.9, +46.2]**, p < .0001 |

This is consistent with C3: on the same seeds, the arms differ mainly in fluency and teacher acceptability, and also in absurdity among the minus items.

The C3 attribution is in `exec/C3_attribution.md`:
- the remaining Y−N gap comes from operator value choices and near-dups;
- the translationese is seed text carried through the renderer.

## 3. E1 manipulation check (R8): does verification lower the error rates? VI vs EN

RD = error(−) − error(+) in pp, with Newcombe CIs (independent samples):

| Arm | Lang | not valid | not teacher | absurd | unanswerable |
|---|---|---|---|---|---|
| Y | VI | **+15.0 [+1.5, +27.8]** | **+16.0 [+2.3, +28.9]** | +9.0 [−1.4, +19.3] | +6.0 [−1.8, +14.1] |
| Y | EN | **+20.0 [+1.2, +37.0]** | **+22.0 [+3.1, +38.9]** | +4.0 [−7.7, +16.0] | +10.0 [−1.8, +22.5] |
| N | VI | +10.6 [−0.1, +21.3] | **+13.7 [+2.9, +24.5]** | +3.6 [−3.9, +11.6] | **+7.8 [+0.5, +15.9]** |
| N | EN | +2.0 [−8.2, +12.6] | **+14.0 [+1.5, +27.1]** | 0.0 [−8.7, +8.7] | +2.0 [−5.3, +10.5] |

**VI − EN difference of the manipulation** (normal approximation):

| Arm | not valid | not teacher |
|---|---|---|
| Y | −5.0 [−27.8, +17.8] | −6.0 [−28.8, +16.8] |
| N | +8.6 [−5.0, +22.2] | −0.3 [−16.3, +15.8] |

**First-attempt verification rates** (`D5_selection.md`): Y 54.5% VI / 63.4% EN; N 80.1% VI / 87.1% EN.

**Reading for H1 (P2):**
- **The manipulation works in both arms and both languages on "teacher-acceptable":** verification lowers the not-teacher rate by 14–22 pp in every arm × language cell.
- On the strict "valid" criterion it is clear for Y in both languages, but for N only in VI. EN N− is already 94% valid, so there is little left for verification to remove.
- **The size of the manipulation does not differ detectably between VI and EN** (every DiD CI covers 0 and is ±15–23 pp wide at these n).
- So a null H1 cannot be blamed on the manipulation being absent in one language. But the check is too coarse to rule out a language difference in manipulation strength of ≲ 20 pp.
- The more precise signal is the verification rate gap: verification discards 45% (VI) vs 37% (EN) of Y items, and 20% vs 13% of N items. So VI is the language where verification removes more.

## 4. Reproduce

```
.venv\Scripts\python -m vi_math_verified_aug.analysis.c3_d5_audit sample vi            # en: sample en --n 50
E:\venv_tools\Scripts\python -m vi_math_verified_aug.analysis.c3_d5_audit judge vi --go     # Batches, crash-safe
.venv\Scripts\python -m vi_math_verified_aug.analysis.c3_d5_audit report
```
