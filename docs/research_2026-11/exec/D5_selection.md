# D5 — Final training-set selection (`select_v1`) (2026-11-08)

## 0. Answer

**Final n = 1,808 per arm × language**, in all 8 files: {Y+, Y−, N+, N−} × {VI, EN}. Published 07:03 in `data/train/final/`.
- The launcher reads `manifest_{vi,en}.json` → `n_effective = 1808`.
- The 1,500 threshold is met.

**Strata:** every file has the same per-stratum counts. All seeds are GSM8K; steps = seed-IR op count.

| Stratum | Steps | Count |
|---|---|---|
| t0 | ≤ 3 | 1,061 |
| t1 | 4 | 356 |
| t2 | ≥ 5 | 391 |

**Files:**

| Set | Train file sha256 (Y+, Y−, N+, N−) |
|---|---|
| VI | 4bb48193…, 0b51bebf…, 9c3e61f6…, fc35ac44… |
| EN | c0def46f…, fd5481ba…, 242419d2…, f3e5d832… |

Each `{Y,N}/<lang>/SHA256SUMS` holds the full hashes.

**Read first: two facts the analysis must carry.**

1. **R8 manipulation check.** The ± contrast is much stronger in Y than in N, and stronger in VI than in EN.

   | Measure | VI | EN |
   |---|---|---|
   | First-attempt verification rate, Y | **54.5%** (2,782 / 5,106) | **63.4%** (2,486 / 3,921) |
   | First-attempt verification rate, N | **80.1%** (7,102 / 8,870) | **87.1%** (2,081 / 2,389) |
   | Selected minus items that pass verification, Y− | **56.5%** | **61.1%** |
   | Selected minus items that pass verification, N− | **84.4%** | **87.6%** |

   - In N, verification removes only ~15% of items. In Y it removes ~45%.
   - H3 (verification) and H1 (language × verification) must be read with this asymmetry in mind.
   - Source: manifest `first_attempt_verification_rate` and `summary.minus_share_plus`.
2. **Question length is NOT balanced across arms.**
   - SMD Y vs N on question words is **0.46–0.65** (VI: Y+/N+ 0.50, Y−/N− 0.52; EN: 0.46, 0.53).
   - KS D is 0.18–0.27, p < 1e-27.
   - This is an arm property: G5's R2b render vs D1's native generator. It is not a selection artefact.
   - Solution words, Y vs N: SMD 0.05–0.22. Verified items have shorter traces: Y+ vs Y− −0.27 (VI) / −0.33 (EN).
   - **Correction (07:20):** these lengths are arm-level properties. Every run of an arm trains on the same 1,808 items, so in the tranche GLMM they are **aliased with arm** and cannot be estimated as covariates.
   - Instead, R1 uses generated-output tokens (item × run) as the covariate.
   - Training length is a mediator that only a **length-matched training subset** can identify. `select_v1` can draw one (equal question-length histograms Y vs N, within strata), but it needs retraining. This is a successor task.
   - Report H2 with this caveat.

**What is balanced:**
- Seed step count: |SMD| ≤ 0.04 for every pair of arms.
- Source mix: SMD 0 (GSM8K only).
- Strata: identical counts in all 8 files.
- Seeds: Y+ and N+ use the same 1,808 seeds (S+); Y− and N− use the same 1,808 seeds (S−); VI and EN use the same seeds.

**Selection uses no outcome information from student models or eval sets.** It reads only:
- the generators' own verification flags;
- the operator;
- seed metadata: source, IR op count, S3 solvecheck screen.

**Code:**

| File | Contents |
|---|---|
| `vi_math_verified_aug/analysis/select_v1.py` | selector, checks, secondary sets, projection, funnel, coverage |
| `select_v1_diversity.py` | Vendi, pairwise cosine distance, chrF and cosine to the seed |
| `select_v1_embed_kaggle.py` | bge-m3 on a private Kaggle T4; Modal is at its limit |
| `select_v1_embed_modal.py` | not used |

**Tests:** `tests/test_select_v1.py`, 12 passed. They cover:
- seed matching;
- identical quotas;
- the minus draw never reading `verified` or `wave`;
- dev, FAIL and UNINF exclusion;
- the two-part selection;
- op-matched histograms equal;
- that a placeholder or a changed pool raises, with nothing published (staged atomic write).

## 1. Design and deviations from T1 §2.6 (each approved by the lead)

| T1 §2.6 | Here | Why |
|---|---|---|
| Primary = slot-paired, the same (seed, slot) in both arms | **Seed-matched + operator class reported.** M6 excluded | Y (G5, π M1 .25 / M2 .40 / M3 .25 / M4 .05 / M5 .05, no M6) and N (D1 `slots_v1`, π with M6) use different slot schedules, so no common slot exists |
| One seed set for all arms | **Two seed sets, each shared by Y and N:** S+ (verified) feeds Y+/N+; S− feeds Y−/N−. Same strata quotas, same n, ≤ 1 item per seed, the same seeds in VI and EN | Requiring Y+ at a seed would make the J=1 Y− item equal to the verified one |
| "−" = unverified | **"−" = the non-adaptive first item only** (Y wave-1 slot 1, N j=1): one per seed, fixed before any verification | G5's waves 2/3 ran only on seeds without a Y+, so any draw over all waves would depend on verification |
| "+" | Any wave, `verified == True` | — |
| Strata with p̂ | source × step tercile; **p̂ pending** (`add_difficulty()` hook) | No reference-solver run yet |
| Equal n | Strict per-stratum quotas: S+ takes min(S− quota, available) per stratum, and S− is trimmed to the same counts | One n, identical strata in every arm |

Seed universe and refills:
- **Seed universe:** pool_v1 (sha256 `1b5654110f0ae891…`, 9,649) minus dev_v1 (`0109576a…`, 512) minus solvecheck FAIL (101) and UNINF (77) = 9,471 allowed. PENDING2 seeds are allowed and flagged: 19 in S+, 43 in S−.
- **S− refills:** 148 seeds were replaced from the 300-seed reserve, within the same stratum, so that the first item is eligible in all 4 arm × language cells. Causes: 97 N-EN ineligible (D1 count); the remaining 51 are Y-VI (decontam) or Y-EN ineligible (not broken down per cause). None was dropped.
- **Retries:** "+"-only retry rows (D1 EN-N `attempt_no=2`; G5 EN2 re-renders) are admitted to S+ and never to S−.
  - EN S+ items from retries: Y+ 158, N+ 177.
  - **S0 slice** (S+ seeds with a first attempt in all 4 cells): **1,493 seeds**, listed in `selection_vi_v1.json` → `s0_plus_seeds`.
  - EN N+ items not on the VI slot: 200. These are D1 retries; the EN N item is an independent generation, not a translation.
- **Bug found and fixed before publishing:** the first run dropped "+"-only rows from the plus candidates because of `minus_eligible = False`, which gave n = 1,493.

### Verification selection effect (P3; lead request)

| Set | t0 / t1 / t2 | Mean seed steps |
|---|---|---|
| S+ (verified in Y and N, both languages) | 0.587 / 0.197 / 0.216 | 3.51 |
| Unconstrained S− (part 1, 3,000 seeds, proportional to the eligible population) | 0.537 / 0.206 / 0.257 | 3.72 |

- Total-variation distance between the two = **0.050**.
- Verification rate by stratum (S+ candidates / S− candidates): t0 0.60, t1 0.52, t2 0.46.
- So verification keeps easier seeds. The matched quotas remove this shift, which means **H3 is identified as "correctness at matched difficulty"**.
- Note: the 07:03 manifest field `unconstrained_random_S-` held S+ by mistake. It was patched at 07:05 (`note_d5_patch`); the train files are unchanged.

## 2. Coverage (P3) and lost seeds

**The matched sets are GSM8K-only.** G5's Y formalizer was run on GSM8K seeds only. For allowed pool_v1 seeds:

| Seeds | Allowed | Formalized by Y | Unreachable by Y |
|---|---|---|---|
| GSM8K | 6,922 | 6,876 | 0.66% |
| MATH | 2,549 | 0 | **100%** |

The natural-coverage set keeps N's MATH items.

**Lost-seed funnel** (`select_v1 funnel`; sequential, so each step names the seeds it loses):

| Step | Seeds | Lost |
|---|---|---|
| Allowed seed universe | 9,471 | — |
| GSM8K only (Y formalizer scope) | 6,922 | 2,549 |
| Y formalization accepted | 6,876 | 46 |
| VI Y wave-1 item exists (operator applicable, tag / leak / boxed trace / decontam) | 4,973 | 1,903 |
| **VI Y+** (Qwen cycle, chrF/dedup, trace = program, Haiku-v3 screen) | 2,734 | **2,239** |
| VI N item exists (minus-eligible, op ≠ M6) | 2,533 | 201 |
| VI N+ (agree + Haiku-v3) | 2,268 | 265 |
| EN Y item exists | 2,251 | 17 |
| **EN Y+** (incl. EN2 retries; 74 EN2 rows unchecked because of the Modal cap) | 1,887 | **364** |
| EN N item exists | 1,871 | 16 |
| EN N+ (incl. D1 retries) | **1,808** | 63 |

**Costed path to a larger n, for the successor.** Costs are per item from the logs (Haiku gen/render ≈ $0.0009, solve $0.0002, Haiku-v3 screen $0.0003; Batch halves them). Expected gains use the measured rates: VI Y+ 54.5% first attempt, VI N+ given Y+ ≈ 0.83, EN Y+ first attempt 63% (≈ 0.8 with one retry), EN N+ 0.90.

| Action | Seeds | Expected Δn | Cost |
|---|---|---|---|
| Finish the cycle on G5's 74 unchecked EN2 rows | 74 | ≈ +60 | ≈ $0.1 Modal |
| EN N 2nd retry | 63 | ≈ +50 | ≈ $0.07 Claude |
| VI N retry, next slot, on Y+ seeds that failed VI N+ | 265 | ≈ +150 after EN | ≈ $0.3 Claude |
| **VI Y wave 2** on the 2,239 VI-Y+-failed seeds | 2,239 | ≈ 1,230 VI Y+ × 0.83 × 0.8 × 0.9 ≈ **+735** | ≈ $1.6 Batch (VI) + Qwen cycle on Modal + ≈ $2 Batch (EN) |
| Y formalizer on MATH algebra/prealgebra | 2,549 | unknown | formalizer only, ≈ $0.0008/seed; MATH certificate rate 27% (R4) |

Wave 2 was stopped on 11-08 because of the Modal budget. Wave 2 plus the cheap retries gives n ≈ 2,800.

## 3. Op / edit-class mix (reported, not matched)

| Arm | M1 | M2 | M3 | M4 | M5 | value / premise / structural |
|---|---|---|---|---|---|---|
| VI Y+ | 643 | 916 | 215 | 2 | 32 | 643 / 918 / 247 |
| VI Y− | 621 | 903 | 219 | 0 | 65 | 621 / 903 / 284 |
| VI N+ | 451 | 375 | 483 | 314 | 185 | 451 / 689 / 668 |
| VI N− | 416 | 401 | 502 | 303 | 186 | 416 / 704 / 688 |

EN is the same for Y. EN N+ is 454 / 392 / 463 / 319 / 180.

**Op-matched sensitivity set:** `data/train/final_opmatched/`, n = **1,387 (VI) / 1,404 (EN)**.
- Inside each sign it is the largest seed subset whose Y and N edit-class histograms are equal (max circulation over (Y class, N class) cells).
- VI: + = 451 / 689 / 247; − = 399 / 704 / 284.
- Limitation: it was selected per language. VI and EN differ by 17 seeds; the successor should intersect them before running.

## 4. Secondary sets

| Set | n | Rule | Use |
|---|---|---|---|
| `final_shared/` | 1,679 | One S = S+ for all four arms; minus = the first items of the S+ seeds | Sensitivity only. Its minus items are 100% / 97% verified (VI Y−/N−) and 92% / 92% (EN), so it does **not** identify H3 |
| `final_natural/` | VI 2,734 / EN 2,325 | Each arm independently from all its accepted items, coverage-maximising; minus = first items; N keeps MATH and M6 | Queued for the successor (T1 secondary) |
| `final_opmatched/` | 1,387 / 1,404 | §3 | Queued |

## 5. Diversity per arm (P3)

**Method:**
- Encoder: bge-m3 (CLS, max_length 256, fp16) on the private Kaggle kernel `datuni/d5-select-embed` (T4, about 3 min, free quota). The input is the private dataset `datuni/d5-select-texts`: 15,450 VI and 11,374 EN training-split texts.
- Metrics come from `select_v1_diversity`, with results in `data/audit/d5/select_diversity_{vi,en}.json`.
- Every set has the same n per arm, so Vendi values compare across arms.

**Matched sets** (n = 1,808):

| Lang | Arm | Unique seeds | Vendi | Mean pairwise cos distance | Cos to own seed | chrF to own seed |
|---|---|---|---|---|---|---|
| VI | Y+ | 1,808 | **40.0** | 0.563 | **0.756** | 0.426 |
| VI | Y− | 1,808 | 41.8 | 0.569 | 0.753 | 0.417 |
| VI | N+ | 1,808 | **35.0** | 0.550 | **0.664** | 0.404 |
| VI | N− | 1,808 | 36.3 | 0.556 | 0.659 | 0.400 |
| EN | Y+ | 1,808 | **48.5** | 0.595 | **0.828** | 0.468 |
| EN | Y− | 1,808 | 50.4 | 0.600 | 0.827 | 0.465 |
| EN | N+ | 1,808 | **41.6** | 0.580 | **0.682** | 0.453 |
| EN | N− | 1,808 | 42.5 | 0.583 | 0.678 | 0.450 |

**Reading:**
- Y is *more* diverse as a corpus than N: Vendi +5 (VI) and +7 (EN); pairwise distance +0.013 to +0.015.
- But each Y item is *closer to its own seed*: cosine +0.09 (VI) and +0.15 (EN).
- The span-grounded symbolic edit keeps the seed's scenario, so Y inherits the 1,808 seeds' diversity. N's native generator, with its localization cards, moves away from the seed but converges on similar Vietnamese settings.
- This matches 2602.19187's direction (Sym 0.37 vs NL 0.07 pairwise distance) only at the corpus level, and adds the seed-distance trade-off.
- Verification costs a little diversity in both arms (Vendi + < −: −1.7 Y and −1.3 N in VI).
- The op-matched, shared and natural sets show the same ordering. Natural VI: Y+ 41.5 vs N+ 39.7.
- Use Vendi and seed distance as mediators, as T1 §2.6 prescribes; do not force-match them.

## 6. Part-1 record (S− at 03:17, for provenance)

- Part 1 drew S− from the eligible population:
  - Y wave-1 eligible: 5,108;
  - N j=1 eligible (op ≠ M6): 8,400;
  - both: 4,388.
- It used proportional quotas 1,611 / 619 / 770 (n = 3,000) with a 300-seed reserve, and read no verification information.
- File: `selection_vi_v1.json` (part 1 → part 2, history in `data/train/final/_history/`).
- The S+ quotas then bound below S− in every stratum, so S− was trimmed to the S+ counts.

## 7. Reproduce

```
.venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1 part1 vi            # S- (03:17)
.venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1 part2 vi,en --force # final + shared + natural + opmatched
.venv\Scripts\python -m vi_math_verified_aug.analysis.select_v1 funnel x | coverage x | project x
```

The selection is deterministic: sha1 orders, no RNG state. It reads only the pools hashed in each manifest's `inputs`.
