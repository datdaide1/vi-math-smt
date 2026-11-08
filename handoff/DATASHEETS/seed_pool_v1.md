# Datasheet — seed pool `pool_v1` (+ `dev_v1`) and the S0 anchor sets

Builder: S3 (`vi_math_verified_aug/seeds/`, `python -m vi_math_verified_aug.seeds.build <stage>`), report
[`exec/S3_seed_pool.md`](../../docs/research_2026-11/exec/S3_seed_pool.md). S0 sets: T1r (`train/build_s0.py`,
`train/build_dev.py`), report [`exec/T1r_first_runs.md`](../../docs/research_2026-11/exec/T1r_first_runs.md) §1.
Status: **FROZEN v1** (2026-11-08 02:06:34 +07; `git_head b6723b7` in the datacard).

## Files and hashes (`data/seeds/SHA256SUMS`, verified 6/6 OK on 2026-11-08 08:40)

| File | Rows | sha256 |
|---|---|---|
| `pool_v1.jsonl` | **9,649** | `1b5654110f0ae8917a34cea0db8811db174e193422a5633d54ba822cf506deca` |
| `dev_v1.jsonl` | 512 | `0109576a3c94f0ab057f3d943a1dbacf8f489ad9bc6e66d2f88669b822eef049` |
| `dropped_v1.jsonl` | 261 | `ff89aa63e2d5ec1c5866e124d7840dc834986ce5a3aeed4b54c69e8ba06684e8` |
| `changelog_v1.jsonl` | 313 changes, each with a reason | `fc5f4859b796ad8ebca4d92f8f8320a4f2ba277a3c3f49318da8bca3b57222c6` |
| `pool_v1.datacard.json` | — | `5e18d4575876ee928ba52ca79c992b6283231aec9a2add174e03618fed8845f5` |
| `CHANGELOG.md` | — | `c2d5870c2f0e93a950bf092f2f72e1e6c07423a16e8527a8cda601354a58378c` |

Superseded hash: `cb918a93…` (a 02:06:0x write before per-seed checker models were filled in; texts identical).

## 1. Motivation
Every arm (S0, Y±, N±, VI and EN) draws from the same seeds (T1 §2.0), so arm differences are not seed differences.
Seeds must be clean (translation defects repaired), typed (gold for the grader), parallel (aligned English original
for E1) and decontaminated against every test set. `dev_v1` is the DEV bucket held out from generation, used as the
training dev set.

## 2. Composition
- **Pool:** Vi-GSM8K train 7,100 · Vi-MATH train algebra 1,468 · prealgebra 1,081 = **9,649**; every row carries the
  English original (`en.problem`, `en.solution`, `en.gold_raw`).
- **Schema** `vi-math-seed-pool/1`: `seed_id`, `problem`, `solution` (VI), `gold` (typed dict, 100 % typed with grader
  3.0.2), `en.*`, `provenance` (`original` | `det_repair` | `mta_vi`), `flags`, `screens` (solve-check verdict,
  coverage and **checker model per seed**, gold-agreement checks, G4-only decontam flags, max same-/cross-language cosine).
- `seed_id = sha1(f"{source}||{index}")[:12]`; DEV = `int(seed_id,16) % 20 == 0` → `dev_v1` 512 (369 GSM8K + 143
  MATH); all 200 pilot seeds (G1/G2/G3) are in DEV, none in the pool.
- **Drops (261):** 259 by X5's review rule (255 MATH, 4 GSM8K; **0 against Vi-ExamMath**), 2 because the gold cannot be
  represented (two-intersection answers).
- **Solve-check verdicts** (pool + dev, 10,161): PASS 9,678 · FAIL 108 (all GSM8K; 107 re-translated and accepted) ·
  UNINF 84 · **PENDING2 291** (second sample + English control batch `msgbatch_01M9aNn5D9Ny6Ss5sDt9u3VB` submitted
  02:03; v1.1 not built — KNOWN_ISSUES). D5's selection excludes FAIL (101) and UNINF (77) and flags PENDING2.
- VI and EN gold agree 10,161/10,161 after 11 gold resolutions.

## 3. Collection process
- Upstream English: GSM8K train (MIT), MATH train algebra + prealgebra (MIT), aligned by `index` (checked, not
  assumed; R1's misalignments caught by `detect`).
- Vietnamese text: the project's legacy translation files (`data/translation/...`, sha256 in the datacard;
  translator per `06_packaging.md` §4.3 = Gemini 2.5 Flash, **UNVERIFIED**). **Kept faithful, not rewritten for
  fluency** (brief). Translationese audit (Opus, 60 seeds): teacher-acceptable **45.0 % [33.1, 57.5]** (GSM8K-only
  34.1 %).
- Repairs: **49 deterministic** (escape damage 37, literal `\n` 6, `#### 9.000` 7, `[asy]` 6 before drops) and
  **253 MTA-vi re-translations kept** (166 rows sent: R1/R2 must-fix rows + gate triggers; translator Haiku 5.5,
  escalation Sonnet 5.5, MQM span QE by Gemini; accepted by Haiku 246, Sonnet escalation 8, tie-break 8).
- Solve-check ladder: gemini-3.5-flash-lite → 3.1-flash-lite → Haiku → Sonnet, English control for FAIL vs UNINF.
  **Checker per seed:** Haiku Batch 8,040, gemini-3.5-flash-lite 1,333, gemini-3.1-flash-lite 783, mixed 5.
  **Gemini quota deviation:** the free tier's per-day limit (500 requests/day/project/model) was hit; `LanePool`
  waited forever on a 16 h `RetryInfo`, fixed to raise `LaneStall` (S3 §3.2; common/ratelimit tests); the remainder
  went to Haiku via Batch. Only train seeds went to Gemini.
- Decontamination: X5 gate + review rule vs 13,655 eval targets (9 eval sets) + Vi-ExamMath 4,209; bge-m3 on Modal
  (`s3-seed-embed`, L4, $0.026).
- **Spend:** Claude **$2.53** of the $4 cap (stage-2 batch ≤ $0.10 pending); Gemini free tier (~3.6k calls); no NIM.
  Ledgers `logs/api/S3/{claude,gemini}.jsonl`.

## 4. Uses
Seed problems for arms Y and N (generation input; D5 used only GSM8K seeds because the Y formalizer was run on
GSM8K only), the S0 anchor's text, and `dev_v1` as the dev set of every training run. Not an evaluation set.

## 4b. S0 anchor sets (built from the same translation, `data/t1r/`)
| File | Rows | Role |
|---|---|---|
| `s0_train.jsonl` / `s0_train_en.jsonl` | 6,788 / 6,788 (same ids) | S0 pool: Vi-GSM8K train (EN: GSM8K originals) minus R1's 14 corrupt + 2 half-English rows, minus dev300 and minus the DEV bucket; `<<…>>` annotations stripped, answer line rebuilt from the integer gold. Each S0 run trains on a seeded subsample of n = 1,808 (matched to the arms) |
| `dev_seed_vi.jsonl` / `dev_seed_en.jsonl` | 512 / 512 | `dev_v1` in the training template (dev loss + quick-eval) |
| `dev300*.jsonl`, `fewshot4_{vi,en}.jsonl`, `eval_qhashes.json` | 300, 4, 11,245 hashes | pilot dev set, 4-shot exemplars for base rows, overlap guard |
Hashes: `data/t1r/MANIFEST.json`, `dev_seed.MANIFEST.json`. **Note:** S0's Vietnamese text is the legacy translation
(not S3's repaired pool) — S0 and the arms therefore differ in seed-text cleanliness as well as in data type.

## 5. Distribution and licenses
GSM8K / MATH derived → keep the MIT notices; our translations, repairs and annotations: proposed CC BY 4.0 (06 §4.5),
disclosing machine translation (legacy Gemini; Claude MTA-vi on 253 rows; Claude provenance of those rows in
`README.md`). Internal: bundle `data`.

## 6. Maintenance
`pool_v1` is frozen; the PENDING2 resolution ships as `pool_v1.1` (`solvebatch collect2` → `mta` → `build freeze
--version v1.1`, S3 §7) with a changelog; seed ids are stable. The training sets of 11-08 used v1.
