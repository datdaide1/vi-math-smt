# S3 — The clean, frozen seed pool (pool_v1 / dev_v1), 2026-11-07/08

S3 owns this work. Nothing in it is committed; the lead commits.
- **Code:** `vi_math_verified_aug/seeds/` (new).
- **Prompts:** `prompts/seed_{translate,postedit,solve,solve_en,mqm}.v1.md`.
- **Tests:** `tests/test_seeds.py` (8 tests) and 5 new tests in `tests/test_common_ratelimit.py`.
- **Outputs:** `data/seeds/`.
- **Sources:** T1 §2.0 (pool definition), T4 §3–4 (MTA-vi algorithm and glossary), R1 and R2 (known defects), X5 §5 (G4 gate and review rule).

## 0. Headline

| | |
|---|---|
| Frozen files | `data/seeds/pool_v1.jsonl` (**9,649**), `dev_v1.jsonl` (**512**), `dropped_v1.jsonl` (261), `changelog_v1.jsonl` (313 changes, each with a reason), `CHANGELOG.md`, `pool_v1.datacard.json`, `SHA256SUMS` |
| sha256 of pool_v1.jsonl | `1b5654110f0ae8917a34cea0db8811db174e193422a5633d54ba822cf506deca` (re-frozen at 02:07 to fill in per-seed checker models; the 02:06 hash sent earlier is superseded) |
| Pool composition | Vi-GSM8K 7,100; Vi-MATH algebra 1,468; Vi-MATH prealgebra 1,081. Every row carries the English original (`en.*`) for E1 |
| Repairs | 49 deterministic repairs. 253 MTA-vi re-translations kept in the frozen files (262 accepted, 9 of them later dropped by decontamination). 11 gold resolutions |
| Drops | 259 by X5's review rule (255 MATH, 4 GSM8K; **0 against Vi-ExamMath**). 2 because the gold cannot be represented |
| Typed gold | 100 % (grader 3.0.2). The VI and EN gold fields agree 10,161/10,161 after the fixes |
| Translationese (Opus, 60 seeds) | Teacher-acceptable **45.0 % [33.1, 57.5]**. GSM8K-only 34.1 % [21.9, 48.9], against G3's 17 % [9, 28]. Not rewritten (brief) |
| Solve-check coverage | 10,161/10,161 seeds checked at least once. **291 are `PENDING2`**: the Vietnamese solve missed once, and the second sample plus English control is queued in a Haiku batch. This will ship as an update and does not block |
| Cost | Claude $2.53 of the $4 cap (stage-2 batch pending, ≤ $0.10). Modal $0.026 (L4, 107 s). Gemini: free tier. No NIM |

## 1. Pool definition (`align`)

- **Sources.** Vi-GSM8K train (7,473) plus Vi-MATH train algebra (1,744) and prealgebra (1,205), 10,422 rows in total. Each row is aligned by `index` to `data/generate/{gsm8k,MATH}/train`.
  - The alignment was checked, not assumed. The `ground_truth` fields agree on every row except prealgebra 72 and 955, which are unit text only.
  - The content misalignments of R1 (GSM8K 220–225, 2577, 3077) are caught by `detect` (see §2).
- **Seed ids.** `seed_id = sha1(f"{source}||{index}")[:12]`, where `source` is one of `vi_gsm8k_train`, `vi_math_train_algebra`, `vi_math_train_prealgebra`.
- **DEV split.** DEV is `int(seed_id,16) % 20 == 0`: 522 rows before drops and 512 after.
  - This is exactly the G1/G2/G3 pilot rule. All 200 pilot seeds (G1 100, G2 40, G3 60) fall in `dev_v1` and none in `pool_v1`.
- **Record schema** (`schema: vi-math-seed-pool/1`):
  - `problem`, `solution` (Vietnamese) and `gold` (typed dict);
  - `en.{problem,solution,gold_raw}`;
  - `provenance` (`original` | `det_repair` | `mta_vi`) and `flags`;
  - `screens`, which holds the solve-check verdict, coverage and checker models, the gold-agreement checks, the G4-only decontamination flags, and the maximum same-language and cross-language cosines.

## 2. Repairs (`detect`, `mta`). Every change is logged with its reason in `changelog_v1.jsonl`

### 2.1 Deterministic repairs (T4 §3.8 step 1)

53 rows had deterministic repairs before drops; 49 changes remain in the frozen files.

| Repair | Rows | What it does |
|---|---|---|
| Escape damage | 37 | `\right`→CR+`ight`, `\times`→TAB and similar are restored with the W3 `repair_control_chars`, applied to problem, solution and gold. A TAB is repaired only before a command stem |
| Literal `\n` | 6 | Replaced with a real newline |
| `#### 9.000` | 7 | The VN thousands dot in the final line is replaced with the EN gold |
| `[asy]` | 6 | Blocks the translator edited are replaced by a verbatim copy of the English block |

Precalculus is not in the pool, so it was not re-aligned (as the brief anticipated).

### 2.2 Gates on the existing translations

The gates are deterministic, adapted from MTA-vi G0/G1 (`seeds/checks.py`):
- **G0:** control characters, brace balance, `$` parity, prose swallowed by a broken `\$`, new or translated LaTeX macros, unclosed `\boxed`, English residue, length ratio.
- **G1:** prose number multiset, final answer against the EN gold, GSM8K calculator annotations, the triggered glossary forbid-list, and CR-1/2/3/5.

The glossary is `seeds/glossary_vi_math_s3.json`, which is W3's v0.1 plus the changes below. I fixed false positives after reading the samples:
- **CR-3:** W3's forbid regex also matched the *correct* template "nhiều hơn k lần … là N". It is now computed from the additive N.
- **G3 median:** no longer fires when the text also has mean or average.
- **G54:** "kích thước" (size) is no longer read as "thước".
- **Units:** a unit forbid is skipped when the English itself uses the metric unit.
- **Numbers:** coordinates and coin values are no longer counted as changed numbers.

**Result:** 166 rows were sent to re-translation. 23 of them are **must-fix** rows:
- R1's 14 corrupt rows and 2 half-English rows;
- the R1/R2 major or critical annotations 520, 529, 2597, 3661, 6922 and algebra 387.

### 2.3 MTA-vi re-translation (Claude, no NIM)

The pipeline (`seeds/mta.py`) runs these steps:
1. Mask math, `[asy]`, `<<calc>>` and `\boxed`.
2. Add the triggered glossary entries and construction rules to the prompt.
3. Translate with Haiku.
4. Run G0 (placeholder bijection plus structure) and G1.
5. **G2 blind solve.** The ladder is gemini-3.5-flash-lite → 3.1-flash-lite → 3.5@0.6 → Haiku → Sonnet. An English control decides between FAIL and UNINFORMATIVE.
6. **G3 MQM span judge** (Gemini, a different model family from the translator).
7. Up to 2 Haiku post-edit rounds with the external feedback, then escalation to Sonnet (translate plus 1 post-edit).
8. A Sonnet J2 tie-break when only the span judge objects.

**Acceptance requires G0, G1 and G2 = PASS, and no major or critical span.**

| Trigger | Rows | Accepted | Not accepted |
|---|---|---|---|
| Must-fix (R1/R2) | 23 | **23** (2597 via J2 tie-break; the English "half fewer" is itself ambiguous) | 0 |
| Gate failures (§2.2) | 143 | 134 | 9: original kept and flagged. Reasons: the G1 number check misreads "Thứ 2…6" and a proper name "Miles"; the EN source's own `$\$$30` formatting; or no solver reproduces the gold even in English (6760) |
| Solve-check FAIL (§3.2) | 108 | 107 | 1, kept and flagged |

- **Who accepted:** Haiku 246, Sonnet escalation 8, J2 tie-break 8.
- **Examples:**
  - 2597 "6 more than twice" → "nhiều hơn 2 lần … là 6";
  - 520 "Four of the remaining" → "Bốn trong số các lớp còn lại";
  - 6922 / algebra 1000 "square yards" → "yard vuông";
  - algebra 387 "cubic yards" → "yard khối";
  - algebra 1458 "common ratio" → "công bội";
  - 189 "13 times older" (the VI had "lớn hơn 13 tuổi") fixed;
  - 565 "20 more than twice Michelle" fixed.
- **Not re-translated:** 9 rows whose accepted re-translation had come from CR-3 false positives in run 1. Their originals were kept; the run-1 log is in `_work/mta_run1_full.jsonl`.

### 2.4 Infrastructure incident and shared-code fix (lead-requested)

The first solve-check run hung for about 25 minutes.

**Root cause.** Gemini free-tier 429s were *per-day* quota errors (`generate_content_free_tier_requests, limit: 500`), and they carried `RetryInfo "16h40m3s"`. `Lane.rate_limited` honoured that with a 60,003 s cool-down, and `LanePool.acquire` then waited forever without raising.

**Fix (backward-compatible).**
- `common/ratelimit.py`:
  - new `LaneStall` exception;
  - `LanePool(max_block_s=300)` raises when no lane can start within 300 s, or after 300 s of total waiting. `None` restores the old behaviour;
  - new opt-in `Watchdog(timeout_s)`, which raises after 5 minutes without a completion.
- `common/llm.py`: the budget reservation is released if `acquire` raises.
- **Tests:** 5 new tests, covering lane recovery after a cool-down, pool recovery, a raise on a 16 h cool-down, the unbounded mode, and the watchdog. They pass together with all existing ratelimit and llm tests (47 tests).

**Restart.**
- 12 Gemini lanes (6 keys × {3.5, 3.1}-flash-lite), with fallback to the other model and then to Haiku.
- The checker model is recorded per item.
- The remainder went to Haiku through the Message Batches API.
- All intermediate results are append-only (`_work/*.jsonl`) and the response cache is kept. Freeze writes are atomic (`.tmp` + rename) and the stage is idempotent.

## 3. Gold and screens

### 3.1 Typed gold (`gold`)

`infer_gold` (grader 3.0.2) is applied to the EN ground truth with `locale="mixed"`, which is the X3 convention. Type counts in pool + dev:

| Type | int | rational | expr | decimal | radical | interval | tuple | text | set | choice |
|---|---|---|---|---|---|---|---|---|---|---|
| Count | 9,488 | 319 | 147 | 77 | 42 | 35 | 35 | 10 | 5 | 3 |

**Disagreements found** (VI `ground_truth` field, VI solution final, EN solution, all graded against the gold). I read all of them and fixed them in `gold.GOLD_FIXES`, logged as `gold_replace` / `gold_confirm`:

| Case | Rows | Fix |
|---|---|---|
| Day/month/ordering text golds ("Friday" vs "Thứ Sáu") | 7 | Added an `accept_also` with the Vietnamese text |
| `\mbox{four}` | prealgebra 1078 | Typed as int 4 |
| `\approx 8.24 \text{ mph}` | algebra 1556 | Typed as decimal 8.24 instead of expr |
| `6, 8\text{, and }10` | algebra 409 | Typed as {6, 8, 10}. `infer_gold` had read it as {6, 810}, which is an X1 grader issue worth reporting |
| First `\boxed{2}` is an intermediate step | algebra 1425 | Gold 4 confirmed |
| Two intersection points; ground_truth kept only the last box, and the grader cannot type a set of tuples | algebra 334, 504 | **Dropped (D3)** |

After the fixes, the VI field and the EN gold agree on 10,161/10,161 rows, and the VI solution's final answer agrees on 10,160/10,161 (1425 confirmed).

### 3.2 Solve-check (screen, T4 §3.8 step 2)

Every Vietnamese seed is solved blind. Seeds missed twice are re-checked with an English control: if the English is solved, the verdict is **FAIL** (the translation changed the math); if not, it is UNINF.

| | PASS | FAIL | UNINF | PENDING2 |
|---|---|---|---|---|
| All (10,161 in pool + dev) | 9,678 | 108 | 84 | 291 |
| GSM8K | 7,079 | 108 | 84 | 198 |
| MATH algebra | 1,525 | — | — | 30 |
| MATH prealgebra | 1,074 | — | — | 63 |

- **Checker per seed** (`screens.solvecheck_models`): Haiku-batch 8,040; gemini-3.5-flash-lite 1,333; gemini-3.1-flash-lite 783; mixed 5.
- **FAIL rate on GSM8K: 108/3,493 Gemini-checked = 3.1 %**, which matches R1's 3.3 % sample estimate. 107 of the 108 were re-translated and accepted through MTA-vi, whose own G2 must pass.
- **PENDING2:** stage 2 (`msgbatch_01M9aNn5D9Ny6Ss5sDt9u3VB`, 700 requests) was submitted at 02:03. When it is collected, `solvebatch collect2` → `mta` → `freeze` produces v1.1. Expect about 3 % of these 291 to become FAIL and be re-translated.

### 3.3 Decontamination (`decontam`, X5 Gate)

**Targets** (13,655 items):
- every `data/eval/*.jsonl`: VI 4,473 and EN 4,973;
- `data/vi_exam/items.jsonl`: 4,209.

**Method.**
- Each seed is checked with both its VI and its EN text.
- Lexical part: X5's `variant_tokens` / `NgramIndex(13)` over exact, nodiac and template variants, computed locally (≤ 300 MB).
- Embeddings: bge-m3, the X5-calibrated encoder, run on a Modal L4 (`seeds/embed_modal.py`), keeping the top 3 per (seed side, target language).

**Review rule (§5.3) → DROP: 259 seeds.**
- By rule:
  - template or 13-gram coverage on a ≥ 20-token item only: 83;
  - cos ≥ 0.90 only: 53;
  - 13-gram coverage ≥ 0.5 + long + cos: 47;
  - 13-gram coverage ≥ 0.5 + long: 36;
  - long + cos: 36;
  - 13-gram coverage ≥ 0.5 only: 3;
  - 13-gram coverage ≥ 0.5 + cos: 1.
- By target set: en_math_algprealg 224, vi_math_algprealg 168, en/vi_primary_1000 34/24, MATH-500 alg/prealg 18/11, MATH-500 full 22, GSM8K-platinum 2/1. **Vi-ExamMath: 0.**
- By source: algebra 189, prealgebra 68, GSM8K 4. 10 of the 259 are DEV seeds.
- **Reading.** These are MATH train/test template twins, e.g. algebra 56 "Emily mentally figures $30^2$…" versus test 0638 "$40^2$…", and prealgebra 538 heptagon versus hexagon diagonals. This is the MetaMath-style leak that X5 found.
  - Two drops are short-stem false positives: prealgebra 83 and 1171 contain the 10-token test item "Có bao nhiêu số nguyên dương có ba chữ số?". I kept them dropped because the rule is the rule.

**G4-only flags (recorded in `flags` / `screens`, not dropped): 502 seeds.** Each seed can match more than one rule:

| G4 rule | Seeds |
|---|---|
| Any shared 13-gram (mostly boilerplate such as "Express your answer as a common fraction", e.g. a 1-gram share of a 21-token item) | 197 |
| cos ≥ 0.85 same-language / ≥ 0.80 cross-language | 174 |
| Template coverage ≥ 0.5 | 98 |
| Combinations | 33 |

The arms' own G4 gate on generated items still applies.

**Caveat.** Embeddings were computed at 21:58, before the 107 solve-check re-translations. The lexical rules were re-run on the final texts at 02:04. For those seeds the cosine uses the earlier Vietnamese wording (same content) and their English side.

### 3.4 Translationese (Opus, `translationese`)

Judge, prompt and settings are the same as G3's seed baseline: Claude Opus 5.5, `validity_judge.v2`, effort medium, operator M0.
- **Sample:** `random.Random(20261107).sample` of 60 seeds from the candidate pool (non-DEV, not dropped). It contains 44 GSM8K, 9 algebra and 7 prealgebra seeds.

| Set | n | Teacher-acceptable | Natural ≥ 4 | NL-FLUENCY | Valid (5 booleans + natural ≥ 4) | Natural mean |
|---|---|---|---|---|---|---|
| All | 60 | **45.0 % [33.1, 57.5]** | 41.7 % [30.1, 54.3] | 65.0 % [52.4, 75.8] | 40.0 % [28.6, 52.6] | 3.43 |
| GSM8K | 44 | 34.1 % [21.9, 48.9] | 31.8 % [20.0, 46.6] | 72.7 % [58.2, 83.7] | 29.5 % | 3.27 |
| MATH algebra | 9 | 55.6 % [26.7, 81.1] | 44.4 % | 66.7 % | 44.4 % | 3.44 |
| MATH prealgebra | 7 | 100 % [64.6, 100] | 100 % | 14.3 % | 100 % | 4.43 |

**Reading.**
- GSM8K seed texts are mostly understandable but stiff ("translated"): 73 % carry NL-FLUENCY. Opus's severity labels are mostly minor (38/60); 2 are major.
- The GSM8K rate (34 % [22, 49]) is higher than G3's 17 % [9, 28] on 60 other GSM8K dev seeds, but the CIs overlap. The judge is stochastic, and the samples differ (pool versus dev).
- **The ceiling remains.** About two thirds of seed texts would not pass a Vietnamese teacher, as written, so fluency must be handled in the arms. The seeds were left as the faithful translation, as the brief requires.

## 4. Freeze (`freeze`)

- `pool_v1.jsonl` / `dev_v1.jsonl` contain one record per seed (schema in §1).
- `dropped_v1.jsonl` gives the rule and evidence for each dropped seed (test ids, coverages, cosines).
- `changelog_v1.jsonl` holds every change with its reason; re-translations carry the old and new problem text. `CHANGELOG.md` is the human-readable summary.
- `pool_v1.datacard.json` records:
  - input sha256s, counts, the decontamination and translationese summaries;
  - the glossary sha and prompt shas;
  - the grader version and spend.
- `SHA256SUMS` covers all of the above.
- **Re-running freeze** is deterministic from `_work/` and atomic.

| File | sha256 |
|---|---|
| pool_v1.jsonl | `1b5654110f0ae8917a34cea0db8811db174e193422a5633d54ba822cf506deca` |
| dev_v1.jsonl | `0109576a3c94f0ab057f3d943a1dbacf8f489ad9bc6e66d2f88669b822eef049` |

**Residual flags in the frozen pool.**
- 502 seeds have G4-only decontamination flags.
- 9 seeds failed a gate, but MTA did not accept a re-translation, so the original was kept.
- 1 seed is a solve-check FAIL that was not fixed.
- 291 seeds are PENDING2.

## 5. Cost and evidence

| Item | Cost |
|---|---|
| Claude, total | $2.53 of the $4 cap (`logs/api/S3/claude.jsonl`) |
| MTA-vi (Haiku translate and post-edit) | $0.12 |
| MTA-vi (Sonnet escalation) | $0.38 |
| G2 solves and the J2 judge | about $0.10 |
| Live Haiku solve-check fallback | about $0.45 |
| Opus translationese | about $0.9 |
| Haiku batch stage 1 (8,117 requests) | about $0.55 |
| Haiku batch stage 2 | pending, ≤ $0.10 |
| Modal | $0.026 (L4, 107 s; `data/seeds/_work/logs/modal_billing_{before,after}.txt`) |
| Gemini free tier (`logs/api/S3/gemini.jsonl`) | $0, about 3.6k calls |

- **Commands:** `python -m vi_math_verified_aug.seeds.build {align,detect,mta,solvecheck,gold,embed-inputs,decontam,translationese,freeze}`, plus `python -m vi_math_verified_aug.seeds.solvebatch {submit1,collect1,submit1b,collect1b,submit2,collect2}` and `modal run -m vi_math_verified_aug.seeds.embed_modal`.
- **Environments:**
  - LLM stages run in `E:\venv_tools` (it has anthropic and httpx2);
  - local numpy stages run in the project `.venv`;
  - no torch was used locally.

## 6. Open items

1. **v1.1.** Collect stage 2 (`solvebatch collect2`), run `mta` on the new FAILs, then run `build freeze --version v1.1` (291 PENDING2 seeds).
   - This writes **new** files: `pool_v1.1.jsonl` and `dev_v1.1.jsonl`, appended to `SHA256SUMS`.
   - v1 is write-once. `freeze` refuses to rewrite any existing version (`FROZEN = ("v1",)`).
   - **Hash history.** The `cb918a93…` hash I first reported belonged to a v1 write at 02:06:0x. I re-ran the freeze at 02:06:34 to fill in `solvecheck_models` for the 783 seeds from Gemini runs 1–2, which had been logged without a model. The texts, golds and drops are unchanged; the only difference is the `screens.solvecheck_models` field. **The frozen v1 is `1b5654…`**, which matches the datacard and SHA256SUMS (`sha256sum -c` passes).
2. **Grader.** The `infer_gold` defects ({6, 810}; sets of tuples) should go to X1.
3. **Glossary.** The SGK [UNVERIFIED] entries of the glossary still need a teacher check (T4 §4).
