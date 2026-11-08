# G1 — Generator-comparison pilot (2026-11-07)

**Scope (approved by the user):** this pilot only. Its job is to pick the native generator, formalizer, editor, solution writer and judge on evidence, with the user's brief "light for speed, strong for quality".
**Code:** `vi_math_verified_aug/pilot_gen/` (pipeline.py, clients.py, ir_exec.py, mutate.py, analyze.py, report_tables.py, latbench.py, judge_render.py; prompts versioned in `pilot_gen/prompts/*_v1|v2.txt`).
**Data:** `data/pilot/generator/` (one JSONL per stage and generator, `items_scored.jsonl`, `metrics.json`, `report_tables.md`).
**API logs:** `logs/api/G1/{nim,claude,gemini}.jsonl`, one record per call with ts, provider, model, params, request, raw response, latency, usage and cost. No keys are logged. The Gemini probe is in `gemini_probe.jsonl`.
**Claude spend:** **$7.55** of the $10 cap (from logged usage at list prices, direct calls). No Vi-GSM8K/Vi-MATH test item or benchmark item was sent to NIM or Gemini; the client refuses any item id that is not a train seed or a G1-generated item.

## 0. Answer per role (details in §4)

| Role | Pick | Runner-up | Evidence (Opus 5.5 judge, same paired 30 seeds for every generator unless stated) |
|---|---|---|---|
| **Native generator (arm N)** | **Claude Haiku 5.5, effort=low** | gemini-3.1-flash-lite (free; comparison only) | Yield (Opus-valid ∧ Opus blind-solve agrees ∧ no leak): **Haiku 67% [49–81]**; Sonnet 40% [25–58]; 3.1-lite 40% [25–58]; 3.5-lite 37% [22–54]. Paired exact McNemar for Haiku vs Sonnet/3.5-lite/3.1-lite: p = .057/.035/.077. Answer-correct: Haiku 97%, Sonnet 69%, 3.1-lite 60%, 3.5-lite 53%. Cost: $0.00096 per call. |
| **Formalizer (arm Y step 1)** | **Claude Haiku 5.5** | Claude Sonnet 5.5 (same pass rate, 14× the cost) | pass@1 / pass ≤3 attempts on 100 seeds: Haiku 99% / **100%**, Sonnet 98% / 100%, 3.1-lite 88% / 94%, 3.5-lite 83% / 94%. Cost: $0.0008 per seed. |
| **Minimal-edit editor (arm Y renderer)** | **Claude Haiku 5.5** (Sonnet if budget allows; quality gap within noise) | Claude Sonnet 5.5 | Tag pass on 64 edit lists: Haiku 94%, Sonnet 95%, 3.1-lite 95%, 3.5-lite 75%. Teacher-acceptable rendered texts (Opus, n=19): Sonnet 26%, Haiku 21%, 3.1-lite 16%. **Every editor scores low; the bottleneck is the edit design, not the editor (§4.3).** |
| **Solution writer** | **Claude Haiku 5.5** | Claude Sonnet 5.5 | Seeds 97% vs 98%; common accepted set 95% vs 100%. Haiku p50 is 1.4 s and its cost per call is 1/14 of Sonnet's. |
| **Judge** | **Claude Opus 5.5 (effort medium)** as judge of record; gemini-3.5-flash-lite **not usable** as a judge | — (no other judge was validated) | Agreement between Opus and Gemini-lite: κ = 0.31 (valid), 0.35 (teacher-acceptable), 0.12 (absurd). Gemini-lite marks 87% of items valid vs Opus 62%, and flags absurdity on 6% vs 21%. **Nemotron is not recommended**, even as an extra judge: it was never run as a judge, and it is slow (p50 ≈ 48 s, ~4k output tokens per native call). |

**Headline:** under the user's brief, Haiku 5.5 is both the light model and the best quality option. Sonnet 5.5 at effort=low was *worse* than Haiku: it writes more complex problems and gets their answers wrong (69% agree with Opus; 4 of 29 items were critical-severity contradictions). The "strong for quality" slot therefore goes to **Opus as the judge and blind-solver**, not to Sonnet as the generator. For 10k accepted arm-N items, Haiku generation costs about **$14 direct ($7 Batch) and takes ~1.5 h** at 24 in flight. Verification is the dominant cost: Opus judge plus blind-solve on every raw item is about $345 direct or $172 Batch, so the verification design must change before scale-up (§5).

## 1. What was run (and what changed mid-pilot)

- **Seeds:** 100 Vi-GSM8K train dev seeds (seed_id hash ≡ 0 mod 20; `random.Random(20261107)`). The 14 R1 corrupt rows and the 2 half-English rows are excluded. Each seed gets one operator from π (M1 17, M2 26, M3 22, M4 18, M5 6, M6 11), one localization card from a hand-written 12-card bank (`pilot_gen/cards_v1.json`), and a partner seed for M6. Every generator received identical (seed, operator, card) triples (`data/pilot/generator/seeds.jsonl`).
- **Tasks per generator:**
  - (a) Native generation, T1 §2.4 prompt (`native_v1`).
  - (b) Formalization to the typed quantity DAG with spans (`formalize_v1`), executed by `ir_exec.py` (Fraction arithmetic; linear unknowns solved exactly). Pass requires all three: exact match to gold, every given span-anchored (its literal parses to its value), and a derived query. Up to 3 attempts, with validator feedback.
  - (c) Minimal-edit rendering (`render_v1`) of deterministic M2–M5 edit lists built by `mutate.py` from one reference IR per seed (the first passing IR in the order Sonnet → Haiku → 3.5-lite → 3.1-lite). The renderer sees only the tagged seed and the edit list, never the answer. Pass = the tag multiset is exact and no numerals are added beyond those already untagged in the seed.
  - (d) Blind short Vietnamese CoT with \boxed{} on the seeds, and on a common set of 40 accepted (a)-items (10 per generator).
- **Judging:** two judges.
  - Opus 5.5 (effort medium; rubric in a cached system prompt, `judge_v2`; 1,047 cached tokens read per call).
  - gemini-3.5-flash-lite (`judge_v2`).
  - Rubric = PROTOCOL codes (NL-FLUENCY/LEAK/UNANSW/TACKED/ABSURD/NEARDUP/LATEX, X-OP-IGNORED) + T1 §2.3 booleans + natural 1–5 + teacher-would-write + new-vs-seed + op-followed.
  - Policy as specified: Claude items → Gemini primary; Gemini items → Opus primary; NIM items → both.
  - Answer-correct = Opus blind-solves the item; its answer must agree with the generator's.
- **Changes from the brief, all instructed by the coordinator/user during the run:**
  1. **NIM removed from the main pipeline.** DeepSeek runs at 220–620 s per call and got 504s at the ~300 s gateway limit before I switched to streaming. Finished NIM rows are kept as *partial* reference rows: nemotron n=11, glm n=8–11, deepseek n=1–8.
  2. **Gemini candidates swapped to flash-lite.** gemini-3.5-flash → gemini-3.5-flash-lite, plus gemini-3.1-flash-lite as a second candidate; the judge changed from gemini-3.8-flash to gemini-3.5-flash-lite.
  3. **Haiku and Sonnet run at effort=low in every generator-side role.** A 5-seed smoke run at medium is kept in `_v1_smoke/` and is not used here.
  4. **Pipelined, resumable orchestrator.** NIM: 38 RPM bucket, 35 in flight. Claude: 24 in flight. Gemini: one lane per (key, model), 12 RPM, 3 in flight.
- **Budget-driven sampling:** Opus judged and blind-solved a fixed **30-seed subsample** (`opus_subsample.json`) for all four candidates. Every yield and answer-correct number is on these 30 paired seeds. Gemini-lite judged all items until throttling stopped it (judged n: Haiku 95, Sonnet 93, 3.5-lite 81, 3.1-lite 98).
- **Bugs found and fixed during the run:**
  - A thread race in the cost tracker briefly reported "$30 spent" while the logs showed $0.91. The run was stopped before any extra spend, and a lock now guards the cache load.
  - Transient Windows `PermissionError` on log appends; writes are now retried.
  - Latency first logged included queueing (Gemini "p50 150–200 s"). Pure HTTP latency is now logged, and a clean latency bench was added (§3).

## 2. Results

**Read Table B, not Table A, to compare generators.** Table A follows the specified judge policy, which gives Claude items the lenient Gemini-lite judge and Gemini items the strict Opus judge. That inflates Claude rows and deflates Gemini rows. Table B uses one judge (Opus) for everyone.

Opus-consistent yield on the 30 paired seeds (Opus-valid ∧ Opus blind-solve agrees ∧ no regex leak). "Strict" also requires teacher-acceptable and new (M2–M6):

| Generator | yield | strict yield |
|---|---|---|
| Claude Haiku 5.5 | **67% [49–81] (20/30)** | **57% [39–73] (17/30)** |
| Claude Sonnet 5.5 | 40% [25–58] (12/30) | 27% [14–44] (8/30) |
| gemini-3.5-flash-lite | 37% [22–54] (11/30) | 20% [10–37] (6/30) |
| gemini-3.1-flash-lite | 40% [25–58] (12/30) | 30% [17–48] (9/30) |

Paired exact McNemar test, Haiku vs other (discordant pairs Haiku-only / other-only): Sonnet 11/3 (p=.057), 3.5-lite 12/3 (p=.035), 3.1-lite 12/4 (p=.077). With n=30 only the Haiku vs 3.5-lite gap is significant at .05. The direction is the same on every quality column and in the all-100 Gemini-judge table.

## 2a. Tables (generated by `report_tables.py` from `metrics.json`)

Section G of `report_tables.md` (a projection that uses the policy-judge yield) is superseded by §3 below, which uses the Opus-consistent yields.

### A. Native generation (arm N), Opus subsample (n = 30 seeds, paired), primary judge per policy

| Generator | primary judge | yield (valid ∧ Opus-agrees ∧ no leak) | strict yield (+teacher, new, no leak) | teacher-acceptable | new (M2–M6) | leak | absurd | tail | near-dup | answer-correct (Opus) |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | Gemini-lite | 83% [66–93] (25/30) | 83% [66–93] (25/30) | 87% [70–95] (26/30) | 96% [80–99] (23/24) | 0% [0–11] (0/30) | 3% [1–17] (1/30) | 3% [1–17] (1/30) | 27% [14–44] (8/30) | 97% [83–99] (29/30) |
| Claude Sonnet 5.5 | Gemini-lite | 53% [36–70] (16/30) | 53% [36–70] (16/30) | 79% [62–90] (23/29) | 100% [86–100] (23/23) | 0% [0–12] (0/29) | 14% [5–31] (4/29) | 7% [2–22] (2/29) | 17% [8–35] (5/29) | 69% [51–83] (20/29) |
| gemini-3.5-flash-lite | Opus | 37% [22–54] (11/30) | 20% [10–37] (6/30) | 57% [39–73] (17/30) | 38% [21–57] (9/24) | 3% [1–17] (1/30) | 20% [10–37] (6/30) | 7% [2–21] (2/30) | 67% [49–81] (20/30) | 53% [36–70] (16/30) |
| gemini-3.1-flash-lite | Opus | 40% [25–58] (12/30) | 30% [17–48] (9/30) | 57% [39–73] (17/30) | 58% [39–76] (14/24) | 3% [1–17] (1/30) | 13% [5–30] (4/30) | 7% [2–21] (2/30) | 53% [36–70] (16/30) | 60% [42–75] (18/30) |

### B. Same subsample, one judge for everyone (Opus 5.5, effort medium) — rank cross-check

| Generator | valid | teacher-acceptable | new (M2–M6) | leak | absurd | tail | near-dup | natural (mean) |
|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 67% [49–81] (20/30) | 63% [46–78] (19/30) | 75% [55–88] (18/24) | 0% [0–11] (0/30) | 23% [12–41] (7/30) | 7% [2–21] (2/30) | 40% [25–58] (12/30) | 3.93 |
| Claude Sonnet 5.5 | 52% [34–69] (15/29) | 48% [31–66] (14/29) | 61% [41–78] (14/23) | 7% [2–22] (2/29) | 31% [17–49] (9/29) | 0% [0–12] (0/29) | 52% [34–69] (15/29) | 3.66 |
| gemini-3.5-flash-lite | 53% [36–70] (16/30) | 57% [39–73] (17/30) | 38% [21–57] (9/24) | 3% [1–17] (1/30) | 20% [10–37] (6/30) | 7% [2–21] (2/30) | 67% [49–81] (20/30) | 3.57 |
| gemini-3.1-flash-lite | 63% [46–78] (19/30) | 57% [39–73] (17/30) | 58% [39–76] (14/24) | 3% [1–17] (1/30) | 13% [5–30] (4/30) | 7% [2–21] (2/30) | 53% [36–70] (16/30) | 3.83 |

### C. All 100 seeds, Gemini-lite judge for everyone (same-family for the two Gemini generators)

| Generator | parsed | valid | teacher-acceptable | new (M2–M6) | leak (regex ∨ judge) | absurd | tail | near-dup | answer value in text | chrF to seed (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 96% [90–98] (96/100) | 94% [87–97] (89/95) | 94% [87–97] (89/95) | 97% [91–99] (77/79) | 0% [0–4] (0/95) | 1% [0–6] (1/95) | 3% [1–9] (3/95) | 24% [17–34] (23/95) | 5% [2–12] (5/96) | 0.28 |
| Claude Sonnet 5.5 | 94% [88–97] (94/100) | 77% [68–85] (72/93) | 76% [67–84] (71/93) | 93% [86–97] (71/76) | 0% [0–4] (0/93) | 13% [8–21] (12/93) | 9% [4–16] (8/93) | 24% [16–33] (22/93) | 5% [2–12] (5/94) | 0.26 |
| gemini-3.5-flash-lite | 100% [96–100] (100/100) | 77% [66–84] (62/81) | 73% [62–81] (59/81) | 86% [76–93] (56/65) | 0% [0–5] (0/81) | 9% [4–17] (7/81) | 9% [4–17] (7/81) | 36% [26–47] (29/81) | 6% [3–12] (6/100) | 0.34 |
| gemini-3.1-flash-lite | 100% [96–100] (100/100) | 88% [80–93] (86/98) | 87% [79–92] (85/98) | 93% [85–97] (75/81) | 1% [0–6] (1/98) | 3% [1–9] (3/98) | 9% [5–17] (9/98) | 21% [14–31] (21/98) | 3% [1–8] (3/100) | 0.23 |
| deepseek-v4.1-flash (partial) | – | – | – | – | – | – | – | – | – | – |
| nemotron-3-ultra (partial) | 73% [43–90] (8/11) | – | – | – | – | – | – | – | 0% [0–32] (0/8) | 0.29 |
| glm-5.3-flash (partial) | 88% [53–98] (7/8) | – | – | – | – | – | – | – | 0% [0–35] (0/7) | 0.22 |

### D. Formalizer (b), editor (c), solution writer (d)

| Model | formalize pass@1 | formalize pass (≤3 attempts) | passes with a secondary flag | render pass (tags exact, no extra numerals) | solve seeds (gold) | solve common accepted set |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 99% [95–100] (99/100) | 100% [96–100] (100/100) | 2% [1–7] (2/100) | 94% [85–98] (60/64) | 97% [92–99] (97/100) | 95% [83–99] (38/40) |
| Claude Sonnet 5.5 | 98% [93–99] (98/100) | 100% [96–100] (100/100) | 0% [0–4] (0/100) | 95% [87–98] (61/64) | 98% [93–99] (98/100) | 100% [91–100] (40/40) |
| gemini-3.5-flash-lite | 83% [74–89] (82/99) | 94% [87–97] (93/99) | 3% [1–9] (3/93) | 75% [63–84] (48/64) | 91% [84–95] (91/100) | 97% [86–100] (35/36) |
| gemini-3.1-flash-lite | 88% [80–93] (88/100) | 94% [88–97] (94/100) | 4% [2–10] (4/94) | 95% [87–98] (61/64) | 92% [85–96] (92/100) | 98% [87–100] (39/40) |
| deepseek-v4.1-flash (partial) | 100% [21–100] (1/1) | 100% [21–100] (1/1) | 0% [0–79] (0/1) | – | 100% [68–100] (8/8) | – |
| nemotron-3-ultra (partial) | 100% [74–100] (11/11) | 100% [74–100] (11/11) | 0% [0–26] (0/11) | – | 91% [62–98] (10/11) | – |
| glm-5.3-flash (partial) | 100% [72–100] (10/10) | 100% [72–100] (10/10) | 0% [0–28] (0/10) | – | 82% [52–95] (9/11) | – |

Formalize failure reasons (final attempt): Claude Haiku 5.5: {}; Claude Sonnet 5.5: {}; gemini-3.5-flash-lite: {'wrong answer': 5, 'carson_saved': 1}; gemini-3.1-flash-lite: {'people_count': 1, 'r1': 1, 'wrong answer': 4}

Render failure reasons: Claude Haiku 5.5: {'tag ids': 2, 'extra numerals': 2, 'no text_tagged': 1}; Claude Sonnet 5.5: {'extra numerals': 2, 'tag ids': 1}; gemini-3.5-flash-lite: {'no text_tagged': 2, 'tag ids': 14, 'extra numerals': 1}; gemini-3.1-flash-lite: {'extra numerals': 2, 'tag ids': 2}

### D2. Editor output quality: Opus 5.5 judge on tag-passing rendered texts (Opus subsample seeds)

| Editor | n | valid | teacher-acceptable | new vs seed | op followed | absurd | fluency issue |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 19 | 32% [15–54] (6/19) | 21% [9–43] (4/19) | 32% [15–54] (6/19) | 95% [75–99] (18/19) | 32% [15–54] (6/19) | 68% [46–85] (13/19) |
| Claude Sonnet 5.5 | 19 | 32% [15–54] (6/19) | 26% [12–49] (5/19) | 32% [15–54] (6/19) | 84% [62–94] (16/19) | 26% [12–49] (5/19) | 53% [32–73] (10/19) |
| gemini-3.5-flash-lite | 13 | 8% [1–33] (1/13) | 15% [4–42] (2/13) | 8% [1–33] (1/13) | 69% [42–87] (9/13) | 31% [13–58] (4/13) | 77% [50–92] (10/13) |
| gemini-3.1-flash-lite | 19 | 16% [6–38] (3/19) | 16% [6–38] (3/19) | 32% [15–54] (6/19) | 74% [51–88] (14/19) | 21% [9–43] (4/19) | 84% [62–94] (16/19) |

### E0. Clean API latency (no queueing; 12 calls per cell, 3 in flight)

| Model | native p50 s | native p90 s | native out tok | solve p50 s | solve p90 s | solve out tok |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 8.7 | 10.8 | 1771 | 1.4 | 2.5 | 281 |
| Claude Sonnet 5.5 | 4.0 | 6.8 | 567 | 2.0 | 2.5 | 222 |
| gemini-3.1-flash-lite | 2.4 | 3.5 | 214 | 1.9 | 2.5 | 180 |

### E. Latency, tokens, cost per call (this run, uncached successful calls)

| Model | task | calls | errors (by HTTP status) | wall p50 s (incl. queue) | wall p90 s | HTTP p50 s (n) | HTTP p90 s | in tok | out tok (incl. thinking) | $ / call |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | formalize | 101 |  | 36.3 | 57.3 | – (0) | – | 1419 | 1287 | 0.00079 |
| Claude Haiku 5.5 | latbench_native | 12 |  | 8.9 | 29.9 | 8.7 (12) | 10.7 | 866 | 1771 | 0.00097 |
| Claude Haiku 5.5 | latbench_solve | 12 |  | 1.4 | 2.5 | 1.4 (12) | 2.5 | 204 | 281 | 0.00016 |
| Claude Haiku 5.5 | native | 100 |  | 40.2 | 59.2 | – (0) | – | 894 | 1738 | 0.00096 |
| Claude Haiku 5.5 | render | 64 |  | 26.7 | 50.7 | 5.5 (1) | 5.5 | 1042 | 929 | 0.00057 |
| Claude Haiku 5.5 | solve_seeds | 100 |  | 33.0 | 53.6 | – (0) | – | 219 | 274 | 0.00016 |
| Claude Haiku 5.5 | writer_common | 40 |  | 2.5 | 4.2 | 1.5 (40) | 3.7 | 241 | 305 | 0.00018 |
| Claude Sonnet 5.5 | formalize | 102 |  | 35.9 | 56.2 | – (0) | – | 1433 | 815 | 0.01102 |
| Claude Sonnet 5.5 | latbench_native | 12 |  | 4.7 | 24.9 | 4.0 (12) | 6.6 | 866 | 567 | 0.00740 |
| Claude Sonnet 5.5 | latbench_solve | 12 |  | 2.0 | 2.5 | 2.0 (12) | 2.5 | 204 | 222 | 0.00263 |
| Claude Sonnet 5.5 | native | 100 |  | 36.2 | 57.3 | – (0) | – | 894 | 624 | 0.00802 |
| Claude Sonnet 5.5 | render | 64 |  | 25.5 | 50.9 | 2.5 (1) | 2.5 | 1042 | 239 | 0.00447 |
| Claude Sonnet 5.5 | solve_seeds | 100 |  | 33.3 | 53.5 | – (0) | – | 219 | 180 | 0.00224 |
| Claude Sonnet 5.5 | writer_common | 40 |  | 6.0 | 7.3 | 1.8 (40) | 2.4 | 241 | 181 | 0.00230 |
| gemini-3.5-flash-lite | formalize | 125 | {'429': 27, '503': 1} | 148.3 | 259.4 | 3.9 (43) | 9.7 | 1133 | 698 | 0.00000 |
| gemini-3.5-flash-lite | native | 100 |  | 145.3 | 280.4 | 3.1 (24) | 15.3 | 493 | 208 | 0.00000 |
| gemini-3.5-flash-lite | render | 64 |  | 160.2 | 272.9 | 2.8 (29) | 9.2 | 579 | 121 | 0.00000 |
| gemini-3.5-flash-lite | solve_seeds | 100 | {'503': 2} | 119.3 | 283.7 | 2.7 (27) | 17.5 | 105 | 158 | 0.00000 |
| gemini-3.5-flash-lite | writer_common | 36 |  | 81.8 | 214.1 | 2.6 (36) | 4.2 | 112 | 142 | 0.00000 |
| gemini-3.1-flash-lite | formalize | 121 | {'429': 22} | 147.3 | 266.6 | – (0) | – | 1088 | 522 | 0.00000 |
| gemini-3.1-flash-lite | latbench_native | 12 |  | 2.4 | 3.5 | 2.4 (12) | 3.5 | 478 | 214 | 0.00000 |
| gemini-3.1-flash-lite | latbench_solve | 12 |  | 2.5 | 3.7 | 1.9 (12) | 2.4 | 98 | 180 | 0.00000 |
| gemini-3.1-flash-lite | native | 100 | {'503': 5} | 133.2 | 266.1 | – (0) | – | 493 | 214 | 0.00000 |
| gemini-3.1-flash-lite | render | 64 | {'503': 5} | 132.1 | 230.9 | 1.6 (1) | 1.6 | 579 | 119 | 0.00000 |
| gemini-3.1-flash-lite | solve_seeds | 100 | {'503': 2} | 133.0 | 276.0 | – (0) | – | 105 | 134 | 0.00000 |
| gemini-3.1-flash-lite | writer_common | 40 |  | 16.7 | 31.6 | 1.7 (40) | 4.7 | 114 | 131 | 0.00000 |
| deepseek-v4.1-flash | formalize | 1 |  | 223.8 | 223.8 | – (0) | – | 1055 | 3875 | 0.00000 |
| deepseek-v4.1-flash | solve_seeds | 8 |  | 193.9 | 257.3 | – (0) | – | 165 | 407 | 0.00000 |
| nemotron-3-ultra | formalize | 11 |  | 141.5 | 239.9 | – (0) | – | 995 | 4084 | 0.00000 |
| nemotron-3-ultra | native | 11 |  | 159.6 | 274.0 | – (0) | – | 529 | 3996 | 0.00000 |
| nemotron-3-ultra | solve_seeds | 11 |  | 73.4 | 215.5 | – (0) | – | 123 | 541 | 0.00000 |
| glm-5.3-flash | formalize | 11 |  | 121.1 | 247.1 | – (0) | – | 945 | 2203 | 0.00000 |
| glm-5.3-flash | native | 8 | {'502': 1} | 190.9 | 246.7 | – (0) | – | 498 | 3930 | 0.00000 |
| glm-5.3-flash | solve_seeds | 11 |  | 98.5 | 229.5 | – (0) | – | 113 | 462 | 0.00000 |
| gemini_judge | judge | 368 | 100 | 3.6 | 203.2 | 905 | 133 | 0.00000 |
| opus | judge | 119 |  | 25.7 | 52.3 | 413 | 627 | 0.01470 |
| opus | opus_solve | 119 |  | 22.8 | 46.8 | 272 | 361 | 0.00831 |

### F. Judge agreement: Opus 5.5 vs gemini-3.5-flash-lite (items judged by both)

| Field | n | Cohen's κ | raw agreement | Opus positive rate | Gemini positive rate |
|---|---|---|---|---|---|
| valid | 112 | 0.31 | 71% | 62% | 87% |
| teacher | 112 | 0.35 | 71% | 59% | 84% |
| new | 112 | 0.42 | 70% | 46% | 74% |
| absurd | 112 | 0.12 | 79% | 21% | 6% |
| tail | 112 | 0.37 | 95% | 4% | 5% |
| neardupj | 112 | 0.40 | 69% | 54% | 27% |
| leakj | 112 | 0.39 | 97% | 4% | 1% |
| op | 112 | 0.21 | 77% | 73% | 95% |
| natural>=4 | 112 | 0.43 | 80% | 70% | 89% |


## 3. Throughput (from `logs/api/G1`) and projections for 10k accepted items per arm

**Clean API latency** (Table E0, and HTTP latency in Table E):

| Model | native p50 / p90 | solve p50 / p90 |
|---|---|---|
| Haiku 5.5 | 8.7 / 10.8 s (~1,770 output tokens incl. thinking, even at effort=low) | 1.4 / 2.5 s |
| Sonnet 5.5 | 4.0 / 6.8 s (~570 tokens) | 2.0 / 2.5 s |
| gemini-3.1-flash-lite | 2.4 / 3.5 s | 1.9 / 2.4 s |
| gemini-3.5-flash-lite (HTTP) | 3.1 / 15 s | 2.7 s |
| Opus judge | 25.7 / 52 s | — |
| Opus blind-solve | 22.8 / 47 s | — |

NIM, uncontended (5-seed run, before the cap), native p50: deepseek-v4.1-flash ≈ 260 s (217–620 s, ~1.8–6.2k tokens, 504s before streaming), nemotron-3-ultra ≈ 48 s (~4k tokens), glm-5.3-flash ≈ 96 s (~3.9k tokens).

**Provider ceilings (measured):**
- **Claude:** 24 in flight sustained 44.6 OK calls/min over 19.8 min of mixed tasks with **zero 429s**. The ceiling is therefore our cap, not the API. With 24 in flight the per-model generation ceiling is ≈ 24×60/p50: Haiku ≈ 165/min, Sonnet ≈ 360/min. Batch (−50%) removes RPM concerns at ≤24 h turnaround.
- **Gemini flash-lite (6 independent Google projects, one key each):**
  - Bursts reach ~70 OK calls/min per model (the 6 × 12 RPM cap), about 60–77k tokens/min in total, or **≈12k tokens/min per project**. After 2–3 such minutes every lane starts returning 429 RESOURCE_EXHAUSTED.
  - The first 429s hit long-output calls (formalize, then judge). Lanes recover after the 60 s cool-downs (3.5-lite: minutes 10–14 throttled, 15–19 back at 71–72/min, 20–25 throttled again).
  - This pattern matches a per-project rolling token/request limit, not a daily cap. An earlier note in this run said "≈135 req/key/day"; that is retracted.
  - **Sustainable rate** over a 25-minute backlog: **31 OK calls/min per model** (≈1,880 calls/h; ≈30k tokens/min in total, ≈5k tokens/min per project).
  - 429 counts in the log: 3.5-lite 126 (99 judge, 27 formalize), 3.1-lite 22 (formalize); 503 "high demand": 15.
- **NIM (1 key, 38 RPM):** the ceiling is min(38 RPM, 35 in flight / latency): deepseek ≈ 8 RPM, glm ≈ 22 RPM, nemotron 38 RPM. Observed 13.5 OK calls/min across the three models before NIM was stopped.

**Projection for 10k accepted arm-N items** (raw items = 10k / Opus-consistent yield; generation only, plus the verification bill if every raw item gets the Opus judge and blind-solve at the measured $0.0147 + $0.0083 per item):

| Generator | raw items | generation wall-clock | generation $ direct (Batch) | + Opus verification, direct (Batch) | wall-clock if Gemini also judges and solves on the same projects (3 calls per raw item) |
|---|---|---|---|---|---|
| **Claude Haiku 5.5** | 15.0k | **≈1.5 h** (165/min) | **$14 ($7)** | $345 ($172) | — |
| Claude Sonnet 5.5 | 25.0k | ≈1.2 h (360/min) | $200 ($100) | $575 ($288) | — |
| gemini-3.1-flash-lite | 25.0k | ≈13.4 h (31/min) | $0 | $575 ($288) | ≈40 h |
| gemini-3.5-flash-lite | 27.3k | ≈14.7 h (31/min) | $0 | $628 ($314) | ≈44 h |
| nemotron-3-ultra (no yield data) | ≥25k if its yield is ≤40% | ≥11 h generation alone at 38 RPM; ≈50 h once formalize/solve/judges share the key | $0 | — | — |
| deepseek-v4.1-flash (no yield data) | ≥25k | ≥52 h generation alone (≈8 RPM) | $0 | — | — |

**Arm Y (symbolic) at 10k accepted.** Formalization is per seed: Haiku at 100% pass costs $0.0008 per seed, so ≈$8 for the full 10.4k seed pool. Render throughput and price match native generation (Haiku $0.00057 per render). The real constraint is quality: render pass (94–95%) × Opus teacher-acceptable rendered text (21–26%) ≈ 20–25%. That means ≈40–50k render calls, or ≈5 h and $25–30 with Haiku, *if the operator/edit design stays as piloted*. It should not stay as is (§4.3).

## 4. Per-role discussion

### 4.1 Native generator → Claude Haiku 5.5 (effort=low)
- **Best on every quality column under the single Opus judge (Table B):** valid 67%, teacher-acceptable 63%, new (M2–M6) 75%, leak 0%. The one weak spot is absurd at 23%, versus 13–31% for the others.
- **Best answer correctness:** 97% agree with Opus blind-solve. Haiku spends ~1.7k thinking/output tokens even at effort=low, and it shows.
- **Sonnet 5.5 at effort=low** writes harder, multi-clause problems and does not verify them. 4/29 were critical (non-integer forced answers: "L = 14,4 trận", "n = 6,5 bạn"; contradictions), and one gave answer −97. Only 69% agree with Opus. Sonnet at medium effort was not tested at full scale (budget), so "Sonnet-medium might beat Haiku" stays open. It would cost ≈2× Sonnet-low per call.
- **Gemini flash-lite (both):** mostly reskins. Near-dup 53–67% and new (M2–M6) 38–58% (Opus). Operators are often ignored (op_followed: Opus 73% overall), and answer-correct is 53–60%. 3.1-lite is clearly better than 3.5-lite on tag fidelity (95% vs 75%) and on the Gemini-judge table. Measured for comparison only.
- **Passes the T1 §2.5 quality gate? Not yet.** The gate needs teacher-would-write ≥ 70%, new ≥ 60%, leak 0% and absurd ≤ 3%. Haiku has teacher 63% (Opus), new 75%, leak 0%, absurd 23%. Absurdity is the gap. The fixes are a prompt rule plus the existing `realistic` gate: integrality at every step, and no "gộp lại rồi chia đều" merges that make no sense. The localization card also pushed a few implausible settings, such as a 9 km lake at a school fair.

### 4.2 Formalizer → Claude Haiku 5.5
- 100/100 pass within 3 attempts and 99/100 at pass@1, with 2 secondary flags (hard-coded literal or unreachable given). Sonnet is equal (100%, 98% at pass@1, 0 flags) at 14× the price.
- The Gemini lites reach 94%. Their failures: wrong answers (4–5), schema ids (2), and 27 + 22 formalize calls throttled with 429.
- For the arm-Y **round trip**, T1 needs a *different family* from the generator/editor. If Haiku generates and edits, the blind back-formalizer should be gemini-3.1-flash-lite (94% pass on seeds, free) or Sonnet. Sonnet is the same family, so it is weaker as an independence check.
- Partial NIM: nemotron 11/11, glm 10/10, deepseek 1/1. They pass, but take 60–400 s per call.

### 4.3 Editor → Claude Haiku 5.5, but fix the edit lists first
- Tag fidelity is a solved problem for Claude and for 3.1-lite (94–95%). 3.5-lite invents or drops tags: 14 tag-id errors, including a literal `<q n="NEW_QUESTION">`.
- **Opus judged the rendered texts poorly for every editor:** teacher-acceptable 15–26% (n=13–19); fluency issues 53–84%; absurd 21–32%.
- Reading the texts, the causes are upstream of the editor:
  - **(i)** The seeds are machine-translated GSM8K and keep their stiffness. Minimal edits preserve it by design.
  - **(ii)** My simplified operators produce semantically odd edits. M3 "giảm giá 15%" applied to *Johnson's profit share*. M4 flips "+" to "−" inside a "tổng" clause. M2 states "1,3 giờ" for a TV total.
- **Recommendation:** before scale-up, implement T1's typed M3 template library (operators by qtype and unit, with a story-role check) and an "M4 sibling exists" check. Then re-run this 64-item test; it costs < $0.10 with Haiku.
- The renderer comparison itself does not separate Haiku from Sonnet (21% vs 26%, overlapping CIs), so pick Haiku on cost.

### 4.4 Solution writer → Claude Haiku 5.5
- Haiku: 97% [92–99] on seeds, 95% [83–99] on the common accepted set. Sonnet: 98% / 100%. 3.1-lite: 92% / 98%. 3.5-lite: 91% / 97% (35 of 40 completed before throttling).
- Haiku's p50 is 1.4 s and it costs $0.00016 per solve. The Sonnet gain is inside the CIs.
- Partial NIM on seeds: nemotron 10/11, glm 9/11, deepseek 8/8.

### 4.5 Judge → Opus 5.5 is the only validated judge
- **Gemini-lite vs Opus on 112 doubly-judged items:**
  - κ: valid 0.31, teacher 0.35, new 0.42, natural ≥ 4 0.43, tail 0.37, absurd 0.12.
  - Gemini-lite is systematically lenient. It accepts 87% as valid (Opus 62%) and 74% as new (Opus 46%), and it misses most absurdity (6% vs 21%).
  - Its free quota is also the first thing to throttle (99 judge 429s).
  - **It fails the role.** The ≥0.6 agreement bar of T1 §2.5 (stated for judge vs human) is not met even judge vs judge.
- **Opus judge:** $0.0147 per item with the cached rubric, plus $0.0083 per blind-solve. Spot-reading its notes, they are specific and they quote the offending span (examples in §6).
- **At scale, Opus on every raw item costs ≈$172 (Batch) per 10k Haiku-accepted items.** Two cheaper options, neither run here:
  - (a) Opus only as an audit (≈400 items per arm) plus a cheap judge calibrated against Opus. Haiku- or Sonnet-as-judge was not measured, and Haiku judging Haiku output is same-family.
  - (b) Batch Opus on everything.
- This is a user budget decision.
- **Nemotron as an extra judge: no evidence.** It was never run as a judge, and its uncontended native latency is ~48 s at ~4k tokens.

## 5. Caveats and open items

1. **Small paired sample.** n = 30 for the Opus-judged columns: 95% CI half-widths of ±15–18 pp. Gemini-judge columns use n ≈ 81–98. A second 30–60-seed Opus batch would cost ≈$1.4 per 30 seeds × 4 generators.
2. **No human ratings** (user decision). Opus is the reference. It also judged Claude items (same family), which the policy assigns to Gemini as primary. The Opus-only Table B is therefore a cross-check, and the self-family bias risk for Haiku/Sonnet in Table B is unmeasured.
3. **Answer agreement is lenient on ambiguous separators.** "1.500" counts as both 1500 and 1.5. Agreement means Opus = generator, not proven truth. I read all 10 Claude disagreements: in 5 Opus is right, and the other 5 problems are ill-posed (contradictory or non-integer). The generator answer was defensible in none.
4. **Mutation operators are simplified pilot versions** (§4.3), so the render pass is a mechanical check. Rendered-text quality comes from Opus on n ≤ 19 per editor.
5. **Gemini-judge coverage is incomplete** because of throttling: 19 Gemini-generated items, 5 Haiku and 7 Sonnet. None of these are in the Opus subsample.
6. **NIM:** partial rows only. No NIM judge or yield data; they cannot be ranked.
7. **The cost tracker reads list prices from usage, not the invoice.** Batch was not used.

## 6. Three best and three worst items per generator (Opus verdicts; seed index = Vi-GSM8K train index)

**Claude Haiku 5.5**
- Best — idx 2206 (M2): "Bác Khoa trồng một số cây cà chua, 2 cây cà tím và 4 cây ớt… một nửa số cây cà chua và một cây ớt bị chết. Mỗi cây còn lại cho 5 quả… tổng cộng 45 quả. Hỏi bác Khoa đã trồng bao nhiêu cây cà chua?" (8; Opus: correct inversion, all counts integral).
- Best — idx 5342 (M3): "chú Lực chuyển 8 con gà từ chuồng thứ hai sang chuồng thứ nhất… chuồng thứ hai còn nhiều hơn chuồng thứ nhất bao nhiêu con gà?" (8; a real mid-story step that changes both quantities).
- Best — idx 5630 (M3): "Hùng làm phụ việc… 20.000 đồng cho mỗi giờ… mua 3 chục trứng gà với giá 35.000 đồng một chục… Hỏi Hùng còn lại bao nhiêu tiền?" (245.000 đồng).
- Worst — idx 1569 (M1): "Trong khu hội chợ của trường có một cái hồ hình vuông, mỗi cạnh dài 9 km" (NL-ABSURD).
- Worst — idx 921 (M3): the library sends unborrowed new books "đi tái chế" (NL-ABSURD, fluency).
- Worst — idx 2975 (M6): an artificial join, "số câu hỏi trắc nghiệm bằng đúng số tẩy của Minh"; generator answer 16 vs the correct 13.5.

**Claude Sonnet 5.5 (effort=low)**
- Best — idx 669 (M2): notebooks, colour-pencil boxes and rulers with "+2/5 mỗi loại" → 140; asks for the initial notebooks (60; all intermediate counts integral).
- Best — idx 876 (M2): choir inversion, "có đúng 10 bạn tham gia hát đoạn cuối… Hỏi lớp có tất cả bao nhiêu bạn" (30).
- Best — idx 2509 (M4): mango harvest, "gấp đôi buổi sáng… ít hơn buổi chiều 20 kg… bao nhiêu tạ xoài?" (2,8).
- Worst — idx 6204 (M2): "số trận hòa bằng một nửa số trận thua, còn số trận thắng nhiều hơn số trận thua 20 trận" → 14,4 losses (critical).
- Worst — idx 7133 (M4): "Hỏi nhóm có bao nhiêu bạn?" → n = 6,5 (critical).
- Worst — idx 2120 (M3): "Thảo đem biếu ông Bảy 8 cái trong số bánh mình đã bán được", which is self-contradictory; generator answer −97.

**gemini-3.5-flash-lite**
- Best — idx 2771 (M2): "cứ sau đúng 5 phút chơi thì cậu ấy ghi được một điểm và cứ ghi đủ 8 điểm thì thắng một trận… tổng cộng bao nhiêu giờ?" (16; new, correct inversion).
- Best — idx 4522 (M3): "Mỗi căn hộ cần lắp 2 ổ khóa mới… 3 chìa khóa cho mỗi ổ khóa" (144; M3 applied correctly, but near-dup).
- Best — idx 1569 (M1): square-schoolyard run, "mất 3 phút để đi bộ hết quãng đường 100 mét" (12 phút; natural 5).
- Worst — idx 2206 (M2): the problem says "mua … 3 hộp sữa" and then asks how many boxes were bought (leak + contradiction, critical).
- Worst — idx 2975 (M6): unanswerable red vs other erasers ("không cho biết bao nhiêu chiếc… màu đỏ").
- Worst — idx 6237 (M2): the forced hourly wage is 100.714,29 đ (non-integer, conflicting tax clause).

**gemini-3.1-flash-lite**
- Best — idx 3218 (M4): "Tuấn chạy được số vòng gấp 3 lần số vòng của Nhi, còn Nhi… gấp 2 lần… Hiếu chạy được 4 vòng" (36; relation type changed).
- Best — idx 1068 (M5): fair tickets, "Khánh và My mua mỗi người 3 vé…, Đạt và Quỳnh… 5 vé… 10.000 đồng" (160.000).
- Best — idx 3306 (M5): "400 viên gạch với giá 2.000 đồng một viên và 5 bao xi măng với giá 90.000 đồng" (1.250.000; natural, but the operator was ignored).
- Worst — idx 6204 (M2): "42 trận thắng, số trận hòa bằng một phần ba số trận thua" → 13,5 losses (critical).
- Worst — idx 876 (M2): states "có đúng 12 người hát" and then asks for it under a contradictory condition (leak/unanswerable).
- Worst — idx 1298 (M3): bolts "mỗi giờ… 50.000 đồng" onto the seed's final answer (NL-TACKED, the R5 failure).

## 7. Reproduce
```
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.seeds          # seeds + operator schedule
E:\venv_tools\Scripts\python -u -m vi_math_verified_aug.pilot_gen.pipeline     # resumable; --gens, --opus-sample 30
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.latbench        # clean latency
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.judge_render    # Opus on rendered texts (subsample)
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.analyze --since 2026-11-07T16:53:00
E:\venv_tools\Scripts\python -m vi_math_verified_aug.pilot_gen.report_tables   # tables A–G -> report_tables.md
```
All API calls are cached in `logs/api/G1`, so a re-run is free unless prompts or params change. Nothing has been committed.
