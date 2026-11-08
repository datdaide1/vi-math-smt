# D1 — Arm N (direct LLM generation) at scale, VI + EN (2026-11-07/08)

**Status (06:00 +07, 08/11): FROZEN.**
- **VI:** pool r5. The 341-item Batch tail landed at 06:00 just before the planned cancel and was collected; its items are N− eligible but not Haiku-v3 screened.
- **EN:** pool r4, made of part 1 (S− seeds), part 2 (S+ seeds) and one retry on S+.
- **Selection:** D5 owns the final 3,000/arm selection in `data/train/final/`.

**Code:** `vi_math_verified_aug/scale_gen/`

| Module | Role |
|---|---|
| `run.py` | Driver: gen → solve → screens; resumable |
| `batchrun.py` | Message Batches runner and pending-batch ledger |
| `checks.py` | Parse/format gate, X1 agreement, `clean_trace` |
| `solver.py` | The **one shared solution writer** for arms N and Y |
| `screen.py` | The **common "+" screen** for N+ and Y+ |
| `lite.py` | Packed flash-lite screen. Not applied at freeze (§4) |
| `slots.py` | Shared `slots_v1` |
| `decontam.py`, `embed_modal.py` | X5 gate; bge-m3 on Modal |
| `finalize.py`, `finish.py` | Pool, train files, manifest |
| `report.py` | The tables in this report |
| `watchdog.sh` | Kills and resumes a stalled job |

**Tests:** `tests/test_scale_gen.py`, 16 pass.

**New prompts:**
- `prompts/native_generator_en.v3.md`, `solution_writer_en.v2.md`, `validity_judge_en.v3.md` (E1 R4 mirrors).
- `prompts/validity_lite.v1.md`.
- EN cards: `scale_gen/cards_en_v1.json`.

**Logs:** API calls in `logs/api/D1/{claude,gemini}.jsonl`; batch ledger in `data/train/N/_work/batches.jsonl`.

## 0. Answers

| Question | Answer (evidence below) |
|---|---|
| What is frozen | **VI** `data/train/N/vi/` (pool sha `73c466cd…0fff`): 13,500 slots (slots_v1 ranks 0–13,499), N− eligible 12,968 (9,402 seeds), **N+ 7,195**. **EN** `data/train/N/en/` (pool sha `087372f9…8120`): 4,395 slots, N− eligible 4,254, **N+ 2,325** (incl. 244 retries). Each dir has pool.jsonl (every generated item with labels), provisional train_10k_{plus,minus}.jsonl, SHA256SUMS and manifest.json. D5 owns the final selection. |
| D5 coverage (`selection_vi_v1*.json`) | **S−** (3,000 + 300 reserve): EN N− eligible on 2,903/3,000 S− seeds and 293/300 reserve. **S+** (2,414 seeds, provisional): VI N+ on 2,373; EN N+ on 2,318 (any attempt); **both: 2,286**. |
| Generator | Haiku 5.5 (`claude-haiku-5-5`), `native_generator[_en].v3`, effort low, adaptive thinking (G2's validated setting = config A; pilot in §2), max_tokens 6000 with one retry at 12,000. |
| Common "+" screen | Haiku-v3 only (`validity_judge[_en].v3`). Flash-lite was not applied because the Gemini quota ran out (lead decision 02:05). N+ = N− ∧ trace agrees with the generator answer ∧ Haiku-v3 pass. |
| Opus audit of VI N+ (80 random) | absurd **2.5% [0.7–8.7]**, teacher 78.8% [68.6–86.3], valid 83.8%, leak 1.2%, tail 0%, new (M2–M6) 87.3%, accept precision 77.5% [67.2–85.3]. 40 cascade rejects: accept 37.5%, absurd 32.5%. EN N+ quality comes from D5's joint gate audit (`validity_judge_en.v1`); D1 ran no EN Opus audit (budget). |
| **E1 R8 manipulation check** | "+" cascade rejection among screened N−, first attempts: **VI 22.9% [22.1–23.8]** (2,140/9,335) vs **EN 12.9% [11.6–14.3]** (308/2,389). Haiku-v3 absurd flag: VI 6.3% vs EN 1.6%; ¬teacher: 20.0% vs 9.8%. **Caveat:** EN was screened only on S+ seeds, which D5 selected for VI Y+ passes. On the 315 S+ seeds with an unbiased paired comparison (same slot screened in both languages, VI N unscreened before selection), pass rates are VI 87.3% vs EN 86.3%, McNemar p = 0.81. The population gap is therefore partly seed selection (§7). |
| Claude spend | **$17.24** (ledger, scope D1; incl. the $0.04 VI tail). Cap $21. |
| Modal spend | **$0.19** (sum of the `d1-n-decontam` app in `modal billing report`: 7 runs on L40S/A10 from the fallback list). Cap $1.5. |
| Blocking the training tranche | **Nothing from D1.** The VI and EN pools are complete for D5's S− and S+ lists. |

## 1. Pipeline (as run)

1. **Slots** (`slots.py`, created by D1 at 02:08 because it did not exist yet). Built from S3's frozen `data/seeds/pool_v1.jsonl`, sha256 `1b565411…6deca` (matches `data/seeds/SHA256SUMS` at 02:06): **9,649 seeds → 28,947 slots** in `data/train/slots_v1.jsonl` + `.meta.json`.
   - J = 3 slots per seed. op ~ π = {.20, .20, .25, .15, .10, .10}; M6 is GSM8K-only, renormalized for MATH (T1 §2.0).
   - Per-slot RNG `Random("20261107|seed|j")`. M6 partner by rendezvous hashing, so it stays stable if the pool changes. Card uniform over C01–C12.
   - `rank`: every j=1 slot first, then j=2, then j=3; within each, sha1 order. This is breadth over seeds first.
2. **Generation.** Haiku 5.5 via the Message Batches API, chunks of 1,000.
   - Each chunk is recorded in the batch ledger *before* results exist, so a restart re-attaches to it.
   - Every collected result is cached (`.cache/vimath/llm_cache.sqlite`). Identical requests are never paid twice. 13,145/13,500 provisional items were reused when the final slots arrived; 355 were regenerated because S3 changed the seed text or the M6 partner.
3. **Parse/format gate (= N−)** (`checks.parse_gen`):
   - JSON, a problem, and a single-number answer;
   - the op tag equals the slot op; length 40–1,600 characters;
   - leak regex; language check (VI word ratio);
   - multi-digit answer not present verbatim in the text (T1 arm-B "leak").
4. **Shared solution writer** (`solver.py`, lead/QA-T rule for both arms): Haiku 5.5, `solution_writer[_en].v2`, effort low, max_tokens 3000, Batch.
   - Training row = {question, solution = `clean_trace(trace)`, answer = the trace's own last \boxed}.
   - `clean_trace` drops trailing answer-only lines and unwraps inline \boxed, so the template's `Đáp số: \boxed{}` appears exactly once (0 multi-boxed rows in both train files).
5. **Agreement.** The trace answer must equal the generator answer (X1 grader `is_correct`, typed gold inferred from the generator answer).
6. **Common "+" screen** (`screen.py`): **Haiku-v3 only** (§4). `validity_judge[_en].v3`, Haiku 5.5, effort low, max_tokens 4000, rubric cached, Batch. pass = ¬absurd ∧ ¬unanswerable ∧ teacher_would_write (lab_v3). Run in rank order, stopped at ~7,000 passes.
7. **Decontamination** (X5 G4 rule) against every `data/eval/*.jsonl` (v1.1, VI and EN) + Vi-ExamMath. These are 13,655 targets, the same `targets()` S3 uses. A row is rejected on any of:
   - ≥ 1 shared 13-gram (exact or diacritics-stripped);
   - ≥ 50% templated 13-gram coverage;
   - bge-m3 cosine ≥ 0.85 same-language or ≥ 0.80 cross-language (Modal, GPU fallback `["L4","A10G","L40S"]`, `retries=0`);
   - MinHash 5-gram J ≥ 0.8 within the arm.
8. **Outputs** (`finalize.py`): `pool.jsonl` holds **every generated item** with its labels: seed_id, slot_id, rank, op, source, card, partner, seed_screens (S3 solvecheck incl. PENDING2), minus_eligible, verified (True/False/None), and per-gate flags.
   - `train_10k_{plus,minus}.jsonl` are provisional rank-order prefixes of equal size.
   - Also written: `SHA256SUMS`, and `manifest.json` (prompt sha256s, model ids, gen params, counts per gate, rejection reasons, deviations, slots/pool sha).

## 2. Pilot (200 items) and the generator config

On 100 provisional VI slots, 2 configs × the same slots (`data/train/N/_work/vi/gen.jsonl`, keys `vi|A|*` and `vi|B|*`):

| Config | Format OK | Solve agrees | Lite OK (3.5-lite) | Gen $/item (Batch) | Output tokens p50 |
|---|---|---|---|---|---|
| **A**: effort low, adaptive thinking (G2) | 97/100 | 95/97 | 79/97 | $0.00061 | ≈2,100 |
| B: effort low, thinking disabled | 82/100 | 67/82 | 56/82 | $0.00023 | 698 |

B saves money per raw item but loses it again on yield. Without thinking, Haiku often writes the problem, fails its own solve and returns an empty answer: 14/100 had `answer_not_number`. Per N+ item the costs are ≈ $0.0009 for A vs ≈ $0.0007 for B, at clearly lower quality, so **A was chosen** (lead approved 23:05). Measured over all 13,500 VI items, A costs gen $0.000647, solve $0.000104 and Haiku-v3 $0.000246 per item.

## 3. VI yields per gate (`report.py gates`, pool r5, final)

| Gate | Pass | Rejection reasons (count) |
|---|---|---|
| Generated (slots 0–13,499) | 13,500 | — |
| Format OK | 96.5% [96.2–96.8] (13,034/13,500) | answer_in_text 303, answer_not_number 97, wrong_language 43, length 42, leak_regex 5, parse_fail 4, op_mismatch 4 |
| Trace has \boxed | 99.97% (13,030/13,034) | includes the 341 traces of the Batch tail (submitted 02:29, landed 06:00) |
| Decontam pass | 99.5% [99.4–99.6] (12,968/13,030) | 13-gram 41, cosine 27, near-dup 1 |
| **N− eligible** | **12,968** (9,402 seeds) | — |
| Solve agrees, within N− | 97.4% [97.1–97.6] (12,629/12,968) | — |
| Haiku-v3 pass, within judged agreeing items | 80.0% [79.1–80.8] (7,195/8,996) | ¬teacher 20.0%, absurd 6.3%, unanswerable 5.6% |
| **N+ verified** | **7,195** (6,964 seeds) | 3,633 agreeing items not screened (incl. the 341-item tail). Screening ran in rank order until ~7,000 passes (lead cap), plus the 320 S+ seeds D5 asked for. |

- By operator (r1, n = 7,394 judged), the Haiku-v3 pass rates are M1 87%, M2 84%, M3 78%, M4 86%, M5 82%, **M6 39%** (185/473): compose problems are the hardest.
- The format failures that came from MATH seeds (abstract algebra) are mostly `answer_not_number`: the prompt asks for a single number.
- Lite (flash-lite) verdicts exist for the first 1,458 N− items only and are kept as information: lite_ok 80.7%, absurd 8.0%, unanswerable 8.8%.

## 4. Deviations from the brief (all lead-approved unless noted)

1. **Flash-lite not applied** (lead decision, 02:05). The Gemini free tier is **500 requests/day/project per model**. D1 probes on 07/11 at 22:47 and on 08/11 at 02:05 (logged as `event: quota_probe`) show:
   - 3.1-flash-lite exhausted on 6/6 keys;
   - 3.5-flash-lite on 3/6 keys;
   - 3.1-flash-lite-preview and flash-lite-latest on 5/6 keys;
   - 2.5-flash-lite returns 404.
   - D1 itself made only ≈200 successful Gemini calls; other workstreams used most of the quota.
   - The common "+" screen became Haiku-v3 only, for both arms. G4's calibration on 358 Opus-labelled items: absurd among accepts is 8% [5–12] for Haiku-v3 alone vs 6% [3–11] with lite. The packed lite screen (`validity_lite.v1`, 10 items per call) remains in the code.
2. **No Opus on cascade disagreements.** Budget: Opus judge+solve costs ≈ $0.011 per item on Batch. Opus was used only for a random audit: 80 verified + 40 rejected (§5).
3. **Pool size.** Lead decisions changed n_train from 10k to 4k to 3k per arm per language. The 20k generated pool was deferred as a costed top-up (§8). VI has 13,500 generated items, which is above the brief's 1.4× rule for 3k–7k N+.
4. **Solve agreement uses 1 sample**, not T1's 2–4 (budget). Agreement is 97.5%, so V1's extra samples would change little at this difficulty.
5. **Generator `answer_in_text` gate.** T1's arm-B leak rule is applied to N− and N+ alike. It is the largest format reject (303). Some of these are coincidental: the answer equals a given.
6. **D1-created shared artifacts.**
   - `data/train/slots_v1.jsonl`: the rule is in its `.meta.json` and `slots.py`.
   - `prompts/validity_judge_en.v3.md`: the E1 R4 mirror, made by one logged Opus call ($0.149) and reviewed.
   - `scale_gen.solver` and `scale_gen.screen` are imported by G4 for Y.
7. **Real-time API for critical-path chunks** (lead rule: use it when a Batch chunk sits > 30 min). Batch latency rose from 4–12 min (23:00) to 25 min–2 h+ (after 02:00). Real-time was used for:
   - the S+ Haiku-v3 screens (VI 320, EN ≈ 2,300);
   - the EN retry (gen, solve, screen);
   - 194 EN solves.
   Two slow batches were cancelled, both logged as `event: batch_cancel` in `logs/api/D1/claude.jsonl`:
   - `msgbatch_01BA1M…`: 558 of its 752 items had already succeeded;
   - `msgbatch_01W98a…`: the S− EN screen, cancelled on the lead's instruction.
   The VI 341-item solve tail (`msgbatch_01Gddy…`, submitted 02:29, 0 done at 05:45) had **ended (341 succeeded)** when the 06:00 cancel ran. So it was collected instead ($0.04), and those 341 items are N− eligible in VI r5. They are not Haiku-v3 screened (verified = None).
10. **Bug found and fixed at freeze (local, no API spend).** For the 355 VI slots regenerated after S3's pool_v1 changes, the old item's Haiku-v3 row was not invalidated; only the solve and lite rows were. That would have attached stale verdicts to the new items once their solve landed. 199 stale rows were tombstoned. The check: no cached Haiku-v3 response exists for the current text (`run.py` now also tombstones hv3 on regeneration).
    - Earlier pools were unaffected, because those items had no trace yet. VI r3 and r5 have identical verified sets: train_10k_plus sha `4510695b…` in both.
8. **EN retry on the next slot** for the 331 S+ seeds that have VI N+ but no EN N+ (lead GO). Retries are "+"-only, `attempt_no=2`, and excluded from R8. D1 retried a superset of the lead's 286 seeds, because no D5 file with the Y-en categories existed.
9. **Process incident.** At 23:21 the Gemini daily-quota exhaustion caught the lite loop, which re-queued failed packs on every tick. It appended 2.9M error rows (1.08 GB) before the 02:05 restart. The store was compacted to one row per key, the bloat deleted, and the loop fixed: failed rows are not retried in-process.

## 5. Opus audit (VI): `validity_judge.v2`, Opus 5.5 effort medium, Batch, $0.834

Random sample (seed 20261108) from the screened region: 80 N+ items and 40 items rejected by agree ∧ Haiku-v3. Data: `data/train/N/_work/vi/audit.jsonl`.

| Set | n | Opus accept* | teacher | valid | absurd | leak | tail | new (M2–M6) |
|---|---|---|---|---|---|---|---|---|
| **N+ verified** | 80 | **77.5% [67.2–85.3]** | 78.8% [68.6–86.3] | 83.8% [74.2–90.3] | **2.5% [0.7–8.7]** | 1.2% [0.2–6.7] | 0.0% [0–4.6] | 87.3% [76.9–93.4] (55/63) |
| Cascade-rejected | 40 | 37.5% [24.2–53.0] | 37.5% | — | 32.5% [20.1–48.0] | — | — | — |

\*accept = valid ∧ teacher ∧ ¬leak ∧ ¬tail (G2 definition).

- The gate thresholds were teacher ≥ 70%, new ≥ 60%, leak 0% and absurd ≤ 3%. N+ meets teacher, new and absurd (point estimates).
- On leak, 1 of 80 is flagged by Opus. Our regex and answer_in_text gates did not catch it.

## 6. Costs (Claude, scope D1; batch ledger `data/train/N/_work/batches.jsonl`)

| Stage / language | Batches | Items | $ |
|---|---|---|---|
| gen/vi (incl. 100 config-B pilot items, retries, 355 regenerations) | 35 | 14,189 | 9.371 |
| solve/vi | 20 | 13,120 (341 cancelled) | 1.371 |
| hv3/vi (Haiku-v3 screen; Batch + 320 real-time) | 12 | 8,942 + 320 | 2.327 + 0.151 |
| audit/vi (Opus) | 1 | 120 | 0.834 |
| gen/en (Batch: part 1 + part 2; real-time: 331 retries) | 2 | 4,064 + 331 | 1.449 + 0.265 |
| solve/en (Batch + real-time: 194 cancelled-batch items + retries) | 2 | 3,949 + ≈510 | 0.150 + 0.043 |
| hv3/en (real-time: the S+ screen and retries) | — | ≈2,610 | 1.084 |
| direct calls (1 config-B probe, EN judge-prompt translation by Opus) | — | 2 | 0.149 |
| **Total (ledger, scope D1)** | | | **17.20** |

- **Per-item unit costs.** Batch gen is $0.00065 (VI) vs $0.00036 (EN); EN uses fewer thinking tokens (p50 output 1,233 vs 2,106). Solve is $0.00010 vs $0.00004 on Batch. Haiku-v3 is $0.00025 on Batch and $0.00042 real-time.
- **Gemini:** ≈200 OK calls ($0, free tier).
- **Modal:** $0.19.

## 7. EN

### Mirror parity (E1 R1–R4)

- **Slots.** EN runs only on D5's selection.
  - Part 1 = `data/train/final/selection_vi_v1.json` (sha256 `218e9222…edd163`): `slots.N` of the 3,000 `seeds_minus` + 300 `reserve_minus`, all `<seed>:1`.
  - List file: `data/train/N/_work/en_slots_part1.txt`.
  - Same slot ids, same per-slot op, card id and M6 partner as VI. EN seed text = `pool_v1.en.problem`, the English original.
- **Models.** The same generator, solver and Haiku-v3 screen models and settings.
- **Prompts.** Translated 1:1 with the same structure, JSON keys and self-check keys. Word-count ratio EN/VI:
  - native_generator 0.96 (744/772);
  - validity_judge v3 0.96 (1,610/1,685);
  - solution_writer 0.85 (22/26 words; a one-sentence prompt).
- **Localization.** EN cards (`cards_en_v1.json`) keep the 12 card ids, adapted to a US setting with USD price ranges. Realism rules use US ranges.
- **Language-specific gate parts.** Replaced by English equivalents: leak regex, and the language check (VI-word ratio < 5% for EN).

### Part 1 (N− on S− seeds, no screen, as instructed), frozen 03:50

| Gate | EN part 1 (3,300 slots) | VI, all 13,500 slots (for comparison) |
|---|---|---|
| Format OK | 96.9% [96.2–97.4] (3,197/3,300) | 96.5% [96.2–96.8] |
| Format reject reasons | answer_in_text 96, answer_not_number 5, leak_regex 2, length 1, wrong_language 1 | answer_in_text 303, answer_not_number 97, wrong_language 43, length 42, … |
| Trace has \boxed | 100% (3,197/3,197) | 97.4% |
| Decontam pass | 99.97% (3,196/3,197) | 99.5% |
| **N− eligible** | **3,196**: 2,903/3,000 S− seeds, 293/300 reserve | 12,629 |
| Solve agrees (information only, no filter) | 97.8% (3,127/3,196) | 97.5% |
| Gen output tokens p50 | 1,233 | 2,106 |
| $/item gen + solve (Batch) | $0.00036 + $0.00004 | $0.00065 + $0.00010 |

### Part 2 (S+ seeds) and the retry, final EN r4 (05:44)

**Part 2 inputs.**
- List: `selection_vi_v1_plus_provisional.json` (sha `c9279a36…d7b7d3a`), 2,414 S+ seeds with `slots.N`.
- 1,650 of those slots already had an EN item from part 1; 764 were generated new.
- Haiku-v3 ran on every agreeing EN item of the S+ slots, real-time because Batch latency was 25–45 min.
- On the 320 S+ seeds whose VI N item was still unscreened, the VI Haiku-v3 screen passed 279/320.

**Retry (lead GO, 05:35).** One fresh EN item on the next slot (`<seed>:(j mod 3)+1`) for the 331 S+ seeds that have VI N+ but no EN N+.
- Funnel: format 316 → agree 306 → Haiku-v3 pass 244 (79.7% [74.9–83.9]).
- Rows carry `attempt_no=2` and `plus_only=true`. They never enter minus and never enter R8 rates.

| EN gate (first attempts, 4,064 slots) | Pass |
|---|---|
| Format OK | 97.2% [96.6–97.6] (3,949/4,064). Rejects: answer_in_text 106, answer_not_number 7, leak 2, length 1, language 1 |
| Trace has \boxed | 100% (3,949/3,949) |
| Decontam pass | 99.97% (3,948/3,949; 1 cosine hit) |
| **N− eligible** | **3,948** (3,869 seeds) |
| Solve agrees | 97.9% [97.4–98.3] (3,866/3,948) |
| Haiku-v3 pass (S+ slots, judged) | 90.2% [88.9–91.4] (2,081/2,307). absurd 1.6%, unanswerable 3.2%, ¬teacher 9.8% |
| **N+ verified** | 2,081 first attempt + 244 retry = **2,325** |

**S+ coverage (slots.N, plus retries).** EN N+ on 2,318/2,414 seeds; VI N+ on 2,373; **both on 2,286**.

### E1 R8 manipulation check: rejection by the "+" cascade, per language

| Population | VI | EN | Note |
|---|---|---|---|
| All screened N− items, first attempts | **22.9% [22.1–23.8]** (2,140/9,335) | **12.9% [11.6–14.3]** (308/2,389) | VI = rank-order sample (all j=1 slots first). EN = S+ seeds only, which D5 selected for a VI Y+ pass, so it is a biased, easier subset. |
| Haiku-v3 flags among judged agreeing items | absurd 6.3%, unanswerable 5.6%, ¬teacher 20.0% | absurd 1.6%, unanswerable 3.2%, ¬teacher 9.8% | Same caveat |
| **Paired, same slots**: the 315 S+ seeds whose VI N item was unscreened at selection, screened in both languages | pass 87.3% [83.2–90.5] | pass 86.3% [82.1–89.7] | McNemar: VI-only 35, EN-only 32, p = 0.81 |

**Reading.**
- The unpaired gap (≈10 pp more VI rejections) has the predicted direction: VI rejects more.
- On the only paired set with no selection on N outcomes, there is no language difference: −1 pp, CI about ±5 pp.
- That set is itself conditioned on a VI Y+ pass for the seed, so it is an easier subset.
- A clean population-level R8 estimate needs EN screening of a random slot sample. The lead decided not to spend on the S− EN items. About $0.40 of Haiku-v3 Batch on the 1,559 unscreened EN N− items would settle it. The successor can run it with `--hv3-only en:data/train/N/_work/en_slots_part1.txt`.

**Files.** `data/train/N/en/`: pool.jsonl (sha `087372f9…8120`), manifest.json, SHA256SUMS. The provisional train_10k_plus/minus hold 2,325 rows each in rank order; minus excludes retry rows. Superseded: part 1 r1 (`9875e3d7…`), r2 (`e5f60fbd…`), r3 (`130f2730…`).

## 8. Costed top-ups (for the successor)

Measured VI unit costs (Batch):
- per slot: gen $0.000647 + solve $0.000104 × 0.966 + Haiku-v3 $0.000246 × 0.912 = **$0.00097 per slot** with full screening;
- verified yield: **0.727 N+ per slot** (0.966 × 0.974 × 0.995 × 0.975 × 0.797);
- calls per slot: 1 gen + 0.97 solve + 0.91 Haiku-v3.

| Target (VI) | Extra slots | Extra calls | Extra $ |
|---|---|---|---|
| N+ = 10k | screen the 3,633 unscreened agreeing items + ≈260 new slots | ≈3.6k Haiku-v3 + ≈0.75k | **≈ $1.15** |
| N+ = 20k | ≈14.0k new slots (ranks 13,500–27,500; 28,947 available) + screen the 3,633 | ≈40k | **≈ $14.5** |
| 20k *generated* pool (N− only, no screen) | 6,500 new slots | ≈13k | **≈ $4.9** |

**EN unit costs** (Batch):
- per slot: gen $0.00036 + solve $0.00004 + Haiku-v3 $0.00025 × 0.95 ≈ **$0.00064 per slot**;
- yield ≈ 0.972 × 1.0 × 0.9997 × 0.979 × 0.90 ≈ **0.86 N+ per slot** (S+-biased; assume ≈ 0.75 on random slots).

**Top-ups per arm and language.** Arm N only; arm Y is G4/G5's. Each N+ target needs ≈ target/yield slots.

| Target | VI extra | EN extra (beyond the 4,395 EN slots; EN has no rank-prefix pool yet) |
|---|---|---|
| N+ = 10k (and N− = 10k from the same slots) | ≈ $1.1 (screen the 3,633 unscreened agreeing items, ≈ $0.9 on Batch, + ≈ 200 new slots) | ≈ 13.3k slots at 0.75 → ≈ 9k new slots ≈ **$5.8** (≈ 27k calls) |
| N+ = 20k | ≈ $14.5 (≈ 14k new slots, ≈ 40k calls) | ≈ 26.7k slots → ≈ 22k new ≈ **$14** (≈ 66k calls) |
| 20k *generated* pool, no screen | ≈ $4.9 (6.5k slots, 13k calls) | ≈ 15.6k new slots × $0.0004 ≈ **$6.3** (31k calls) |

All of these need ≈ 1–3 h wall-clock on Batch at the 08/11 latency (25–45 min per chunk), or ≈ 2× the $ on the real-time API (≈ 50–100 Haiku calls/min on this workspace before 429s).

## 9. How to resume (any interruption)

```
# VI or EN generation + solve + Haiku-v3 screen (resumes: rows are appended per item, un-collected batch ids in
# data/train/N/_work/batches.jsonl are re-attached, collected results are cached):
W=data/train/N/_work
bash vi_math_verified_aug/scale_gen/watchdog.sh $W/logs/gen_vi_final.log \
     -m vi_math_verified_aug.scale_gen.run gen --langs vi --n 13500 --cfg A --no-lite --target-common 7000
# EN on D5's lists (as run): part 1 + part 2, Haiku-v3 only on S+ slots, real-time screen
bash vi_math_verified_aug/scale_gen/watchdog.sh $W/logs/gen_part2.log \
     -m vi_math_verified_aug.scale_gen.run gen --langs vi,en --n 13500 --cfg A --no-lite \
     --lang-slots en:$W/en_slots_all.txt --hv3-only vi:$W/vi_hv3_unscreened_splus.txt,en:$W/en_slots_splus.txt \
     --target-common vi:99999,en:99999 --chunk 3300 --realtime hv3
# EN retry (next slot, real-time end to end):
bash vi_math_verified_aug/scale_gen/watchdog.sh $W/logs/gen_retry.log \
     -m vi_math_verified_aug.scale_gen.run gen --langs en --n 28947 --cfg A --no-lite \
     --lang-slots en:$W/en_slots_retry.txt --hv3-only en:$W/en_slots_retry.txt --target-common en:99999 \
     --realtime gen,solve,hv3
# decontam (Modal) + freeze (project .venv, numpy):
.venv\Scripts\python -m vi_math_verified_aug.scale_gen.finish --lang vi --cfg A --n 13500
.venv\Scripts\python -m vi_math_verified_aug.scale_gen.finish --lang en --cfg A --n 28947 \
     --slots-from $W/en_slots_all_r4.txt --retry-slots $W/en_slots_retry.txt
python -m vi_math_verified_aug.scale_gen.run audit --lang vi --n 13500 --n-plus 80 --n-rej 40
python -m vi_math_verified_aug.scale_gen.report {gates,audit,costs,examples}
```

Run only one D1 bulk process at a time. A second process collects the first one's batches, which is harmless (results are cached and charges are idempotent) but noisy.

## 10. 20 random VI examples (`report.py examples --lang vi`, seed 20261108; N− eligible items)

| # | slot | op | question | answer (trace) | gen answer | agree | Haiku-v3 | verified |
|---|---|---|---|---|---|---|---|---|
| 1 | 9eebbd0adf1a:1 | M3 | Trường tiểu học có hai khu đất trống hình vuông dùng làm sân tập cho câu lạc bộ bóng đá thiếu nhi. Khu lớn có cạnh 235 m. Khu nhỏ ban đầu có cạnh 221 m, nhưng sau đó một dải đất dọc theo mỗi cạnh bị rào lại để làm khu thi đấu riêng, làm cạnh khu nhỏ bị rút ngắn 8 m. Phần diện tích còn lại của khu lớn sau khi trừ khu nhỏ được chia thành các ô hình chữ nhật, mỗi ô có diện tích 14 m² để làm vị trí tập luyện. Hỏi có bao nhiêu ô tập luyện? | 704 | 704 | ✓ | fail(unansw)(¬teacher) | False |
| 2 | 04c8ecda8219:1 | M3 | Anh Quân làm sơn tại một xưởng nhỏ, mỗi ngày làm được cùng một số thùng sơn. Lúc đầu, anh làm trong 3 ngày thì xong 18 thùng sơn. Sau 3 ngày đó, xưởng thay một máy trộn mới nên mỗi ngày anh làm được thêm 3 thùng sơn so với trước. Hỏi anh Quân cần tổng cộng bao nhiêu ngày để làm xong 360 thùng sơn? | 41 | 41 | ✓ | pass | True |
| 3 | 40863d11a59d:1 | M3 | Hoa bỏ tiền xu vào một chiếc ống heo để dành mua quà tặng cô giáo Thu. Trong giờ đầu tiên, Hoa cho vào 20 đồng xu. Trong hai giờ tiếp theo, mỗi giờ Hoa cho vào 30 đồng xu. Sau giờ thứ hai, em trai Nam lén lấy 15 đồng xu để mua kẹo. Trong giờ thứ tư, Hoa cho vào 40 đồng xu. Trong giờ thứ năm, mẹ mượn 20 đồng xu trong ống heo. Hỏi sau giờ thứ năm, trong ống heo còn lại bao nhiêu đồng xu? | 85 | 85 | ✓ | pass | True |
| 4 | 099b7cf40f12:1 | M4 | Trên chu vi của một đường tròn có bảy điểm phân biệt, trong đó có hai điểm A và B. Người ta nối từng cặp điểm trong bảy điểm đó bằng một dây cung, mỗi cặp điểm chỉ nối một lần. Hỏi có bao nhiêu dây cung đi qua ít nhất một trong hai điểm A, B? | 11 | 11 | ✓ | pass | True |
| 5 | ae24c067fd06:1 | M5 | Sở Giao thông thử nghiệm hai tuyến xe buýt A và B trong ba tuần đầu sau khi mở tuyến. Cả hai tuyến đều có 40 lượt khách trong tuần đầu tiên. Số lượt khách trong tuần thứ hai và tuần thứ ba của mỗi tuyến lập thành cấp số nhân, nghĩa là số lượt khách mỗi tuần bằng số lượt khách tuần trước nhân với một công bội không đổi; công bội của tuyến A và của tuyến B khác nhau. Biết rằng số lượt khách tuần thứ hai của tuyến A nhiều hơn số lượt khách tuần thứ hai của tuyến B là 80 lượt, và hiệu số lượt khách tuần thứ ba của hai tuyến bằng 3 lần hiệu số lượt khách tuần thứ hai của hai tuyến. Tính tổng số lượt khách tuần thứ ba của hai tuyến. | 260 | 260 | ✓ | fail(¬teacher) | False |
| 6 | 485722d58119:2 | M5 | Tòa nhà chung cư nơi Sơn ở có nhiều tầng. Sơn sống ở tầng 16. Hải sống ở tầng có số bằng 1/4 số tầng của Sơn. Yến sống cách Hải hai tầng, và Yến ở tầng cao hơn Hải. Hỏi Yến và Sơn cách nhau bao nhiêu tầng? | 10 | 10 | ✓ | – | None |
| 7 | 29c677523cc3:1 | M4 | Mỗi ngày, ông Tám thu hoạch được 6 quả trứng vịt trong 9 ngày liền. Ông để lại một phần ba số trứng thu được cho gia đình ăn, số trứng còn lại được xếp đều vào các hộp, mỗi hộp 6 quả. Hỏi ông Tám xếp được bao nhiêu hộp trứng? | 6 | 6 | ✓ | pass | True |
| 8 | b5fe3996e84e:1 | M3 | Mẹ đưa cho Yến một số tiền để sửa sân và vườn nhà. Yến mua 40 viên gạch lát, giá 2.500 đồng một viên, và 5 bao xi măng, giá 90.000 đồng một bao. Khi bác thợ Hải kiểm tra thì thấy một bao xi măng bị ẩm, Yến trả lại bao đó và được hoàn lại 90.000 đồng. Sau đó, Yến mua 12 mét ống nước, giá 30.000 đồng một mét. Khi mọi việc đã xong, Yến còn lại 120.000 đồng. Hỏi lúc đầu mẹ đưa cho Yến bao nhiêu tiền? | 940000 | 940000 | ✓ | – | None |
| 9 | ab05692b8072:2 | M2 | Tam giác ABC có AB = 5 cm và BC = 6 cm. Hai con bọ cùng xuất phát từ đỉnh A, bò dọc theo chu vi tam giác theo hai hướng ngược nhau với cùng tốc độ. Chúng gặp nhau lần đầu tại điểm D nằm trên cạnh BC, với BD = 4 cm. Tính độ dài cạnh AC. | 7 | 7 | ✓ | – | None |
| 10 | d998967a2ea5:2 | M3 | Út đạp xe từ nhà ra thị xã. Trong 2 giờ đầu, Út đi với tốc độ trung bình 12 km/h. Sau đó, đường xấu nên Út đi tiếp 1,5 giờ với tốc độ trung bình 8 km/h thì đến nơi. Tâm đạp xe trên cùng quãng đường đó trong 4 giờ. Hỏi Tâm phải đạp xe với tốc độ trung bình bao nhiêu km/h để đi được quãng đường bằng quãng đường Út đã đi? | 9 | 9 | ✓ | – | None |
| 11 | ca8f91abe95b:1 | M1 | Ngọc và Huy cùng đi siêu thị vào cuối tuần để mua đồ dùng cho gia đình. Ngọc mua 15 hộp sữa và 20 gói mì. Huy mua số hộp sữa gấp đôi Ngọc, nhưng số gói mì chỉ bằng một nửa số gói mì của Ngọc. Huy mua nhiều hơn Ngọc bao nhiêu món hàng? | 5 | 5 | ✓ | pass | True |
| 12 | 3997f2a5cbc4:1 | M3 | Trong buổi tập của câu lạc bộ bóng đá thiếu nhi, Tuấn mang theo 1,5 kg kẹo sữa, Nhi mang theo 500 g kẹo đậu phộng và Hiếu mang theo 2 kg kẹo sữa. Trước khi bắt đầu tập, Hiếu tặng 400 g kẹo sữa trong túi của mình cho em trai đang chơi ở sân. Hỏi ba bạn còn tổng cộng bao nhiêu gam kẹo khi bắt đầu buổi tập? | 3600 | 3600 | ✓ | pass | True |
| 13 | 716224a650e7:1 | M4 | Vào cuối tuần, chị Ngọc thường ghé siêu thị mua sữa cho con. Tuần đầu tiên chị mua 2 hộp sữa. Từ tuần thứ hai trở đi, mỗi tuần chị mua nhiều hơn tuần trước đó 3 hộp sữa. Hỏi trong 4 tuần đầu tiên, chị Ngọc đã mua tổng cộng bao nhiêu hộp sữa? | 26 | 26 | ✓ | pass | True |
| 14 | f318dace077b:2 | M2 | Hằng dùng ba đoạn dây thép để rào một góc chuồng gà. Đoạn dây thứ hai dài 160 cm, đoạn dây thứ ba dài 1 m. Biết tổng chiều dài của cả ba đoạn dây là 5 m 40 cm. Hỏi đoạn dây thứ nhất dài bao nhiêu xăng-ti-mét? | 280 | 280 | ✓ | – | None |
| 15 | 4ed0dc1e8fd5:1 | M4 | Huấn luyện viên Lâm vẽ trên sân tập của câu lạc bộ một đa giác đều để bố trí các vòng chạy cho thiếu nhi. Biết rằng số đo mỗi góc trong của đa giác hơn số đo mỗi góc ngoài tương ứng 100 độ. Đa giác đó có bao nhiêu cạnh? | 9 | 9 | ✓ | pass | True |
| 16 | 458296b4dad8:1 | M3 | Tuấn bắt đầu giao báo ở tầng một của một tòa nhà chung cư. Anh đi lên 5 tầng rồi đi xuống 2 tầng. Sau đó, một khách hàng ở tầng trên báo bị thiếu tờ báo, nên Tuấn phải đi lên thêm 3 tầng để giao bù. Cuối cùng, anh lại đi lên 7 tầng và thấy mình cách tầng cao nhất của tòa nhà 9 tầng. Hỏi tòa nhà đó có bao nhiêu tầng? | 23 | 23 | ✓ | pass | True |
| 17 | 5c886f22c232:1 | M3 | Bác thợ Hải xây tường rào trong 5 ngày đầu. Ngày thứ nhất trời mưa to nên bác không xây được viên nào. Ngày thứ hai bác xây được 100 viên gạch, ngày thứ ba xây được 200 viên, ngày thứ tư xây được 400 viên và ngày thứ năm xây được 800 viên. Khi nghiệm thu, anh Sơn phát hiện 180 viên gạch trong số đã xây bị nứt nên phải dỡ bỏ, và các viên này không được tính là gạch xây hợp lệ. Hỏi trung bình mỗi ngày bác Hải xây được bao nhiêu viên gạch hợp lệ (không tính 180 viên đã dỡ bỏ)? | 264 | 264 | ✓ | pass | True |
| 18 | 6b851a425bb1:1 | M3 | Phúc được ông Tám, chủ một trang trại chăn nuôi, thuê làm việc với tiền công 40.000 đồng mỗi giờ. Phúc dành 8 giờ để sơn lại chuồng gà. Thời gian Phúc sửa hàng rào lâu gấp 3 lần thời gian sơn chuồng. Sau khi sửa xong hàng rào, ông Tám nhờ Phúc tưới vườn rau thêm 5 giờ. Cuối cùng, Phúc cắt cỏ quanh trang trại trong 6 giờ. Hỏi ông Tám đã trả cho Phúc bao nhiêu tiền? | 1720000 | 1720000 | ✓ | pass | True |
| 19 | 64a6491d13fb:2 | M4 | Long vừa tròn 12 tuổi và bắt đầu học đàn piano. Người bạn Trang kể với Long về quy tắc 10.000 giờ, theo đó sau 10.000 giờ luyện tập, người học có thể trở thành chuyên gia trong lĩnh vực của mình. Long muốn trở thành chuyên gia piano trước khi 20 tuổi, và Long tin rằng chỉ cần luyện tập đủ 60% số giờ của quy tắc này là đã đủ. Mỗi năm, Long luyện tập mỗi ngày từ thứ Hai đến thứ Sáu và nghỉ hai tuần để về quê. Hỏi mỗi ngày Long cần luyện tập bao nhiêu giờ? | 3 | 3 | ✓ | – | None |
| 20 | 46b929af7808:1 | M3 | Câu lạc bộ khoa học ban đầu có 100 học sinh, trong đó 65 học sinh học toán, 43 học sinh học vật lý và 10 học sinh học cả toán lẫn vật lý. Đầu năm học, câu lạc bộ nhận thêm 5 học sinh mới: 3 học sinh chỉ đăng ký học toán và 2 học sinh chưa đăng ký học môn nào trong hai môn trên. Hỏi sau khi nhận thêm 5 học sinh mới, có bao nhiêu học sinh của câu lạc bộ không học toán cũng không học vật lý? | 4 | 4 | ✓ | pass | True |

## 11. 20 random EN examples (`report.py examples --lang en`, seed 20261108; N− eligible items)

| # | slot | op | question | answer (trace) | gen answer | agree | Haiku-v3 | verified |
|---|---|---|---|---|---|---|---|---|
| 1 | a94f1cd70f5b:1 | M3 | Coach Martin runs three kids' soccer teams: Jack's team, Nora's team, and Henry's team. Jack's team and Nora's team have the same number of players. Henry's team has 12 players, which is half of the total number of players on Jack's team and Nora's team combined. Before the new season starts, 4 players move from Henry's team to Nora's team. How many players are on Nora's team now? | 16 | 16 | ✓ | – | None |
| 2 | 19af3b4a160c:1 | M1 | Liam has 15 pairs of sneakers he needs to clean before the school field day. If he has cleaned 40% of the individual shoes, how many more shoes does he need to clean? | 18 | 18 | ✓ | pass | True |
| 3 | 957cd06ce1af:1 | M2 | Chloe is stocking the shelves at the supermarket on Saturday morning. She has to move 17 cartons of milk from the back room to the cart, and she can carry the same number of cartons on every trip. Chloe makes the fewest possible trips, and it takes her exactly 6 trips to move all 17 cartons. How many cartons can Chloe carry on each trip? | 3 | 3 | ✓ | fail(absurd)(unansw)(¬teacher) | False |
| 4 | eedd713bf675:1 | M1 | Grace has $350 in her savings jar. Every time she visits the bakery, she buys a bagel for $3. She visits the bakery 5 times a month. How many dollars will be left in her savings jar after one year? | 170 | 170 | ✓ | pass | True |
| 5 | 173994100eca:1 | M3 | Coach Martin runs a kids' sports club with 56 members. The club has twice as many soccer players as basketball players, and 4 more basketball players than swimmers. Halfway through the season, 6 soccer players switch to the swim team. How many soccer players are in the club now? | 24 | 24 | ✓ | – | None |
| 6 | 4c5934d4dc40:1 | M2 | Ms. Chen drives a city bus and is paid $18 per hour. Last month she worked 10 days, and on each day she completed the same route 8 times. Each round trip took the same amount of time. Her pay for the month was $1,440. How many minutes does one round trip take? | 60 | 60 | ✓ | pass | True |
| 7 | fe0d79c18d6f:1 | M5 | Coach Martin keeps training cones in a bin at the gym. There are 7 yellow cones. There are 7 more orange cones than yellow cones, and there are 5 more green cones than orange cones. How many more green cones than yellow cones are in the bin? | 12 | 12 | ✓ | pass | True |
| 8 | 65518cfd5f82:1 | M1 | Mr. Price ordered 5 boxes of crayons at $6 each for his class, and he paid a $4 shipping fee to the delivery driver. If he paid with one $50 bill, how many dollars of change did he get back? | 16 | 16 | ✓ | pass | True |
| 9 | 472c52016853:1 | M1 | Uncle Ray grills a 40-ounce pork roast for the family picnic on the farm. The fire burns half of the roast. Of the part that is not burned, he eats 75%. How many ounces of pork did Uncle Ray eat? | 15 | 15 | ✓ | – | None |
| 10 | a3f02b78e42e:1 | M3 | Olivia had some holiday cards. She gave 6 of her cards to her cousins. Later that day, her grandpa gave her 10 more cards. Now Olivia has 27 cards. How many cards did Olivia have at the start? | 23 | 23 | ✓ | – | None |
| 11 | 46035fb919fa:1 | M2 | Ethan sold 20 apples at a stand near the bus stop to earn money for repairing his bike. His bike cost $80, and the repairs cost 25% of what the bike cost. After the repairs were paid for, 1/5 of the money he had earned was left over. How much did Ethan charge for each apple? | 1.25 | 1.25 | ✓ | – | None |
| 12 | 3b5628ab6bcf:1 | M1 | Olivia earns $3 for every holiday card she addresses for her grandfather's business. For every 12 cards she addresses, she also gets a $5 bonus. How much money will Olivia earn for addressing 36 cards? | 123 | 123 | ✓ | pass | True |
| 13 | a18546d3dfa8:1 | M3 | Ms. Reed works at the library. She has 2 dozen storybooks and 6 more reference books than storybooks. Before the library opens on Monday, she gives 8 storybooks to a school book drive. How many books does the library have now? | 46 | 46 | ✓ | fail(unansw)(¬teacher) | False |
| 14 | b91222ebfeb3:1 | M3 | Ms. Patel surveys her class of 30 students with two questions: whether they prefer dogs or cats, and whether they prefer movies or video games. Half of the students chose dogs and video games, and 10% of the students chose dogs and movies. After the survey, 4 of the students who chose dogs and movies transfer to another class. How many students in Ms. Patel's class now prefer dogs over cats? | 15 | 14 | ✗ | – | False |
| 15 | e979ec649a36:2 | M3 | Zoe had only read a handful of books before this summer. Her best friend, Kevin, reads all the time. During a two-week visit to Kevin's house, Zoe read 8 books along with Kevin at the library. Kevin also read 20 more books on his own during those two weeks. Then Kevin caught a cold and stayed home on 4 days he had planned to read alone at the library, so he did not read those books. In the end, the total number of books Kevin read during the two weeks was four times the number of books Zoe had read before her visit. How many books had Zoe read before her visit? | 7 | 6 | ✗ | – | False |
| 16 | b8e79db27756:1 | M1 | Ms. Reed, the librarian, receives 7 dozen used books every Monday and Thursday from donors. She gives 2 dozen books to the school across the street and 3 dozen books to a bookstore for its sale. On Saturday, she sets aside 5 dozen books for the summer reading program. By Sunday, she donates the remaining books to a shelter. How many books does she donate? | 48 | 48 | ✓ | pass | True |
| 17 | 9984965a0376:1 | M3 | Four friends ran a 4 x 100 relay race. Tyler ran the first leg and took twice as long as Abby. Abby ran second and took 10 seconds longer than Sam. Sam ran third and finished in 30 seconds. After the race, the officials added a 3-second penalty to Tyler's time for stepping over the lane line, so his official time was longer than the time on his stopwatch. Tiffany ran the last leg and finished in 7 seconds less than Tyler's official time. How many seconds did it take the team to finish the race? | 229 | 229 | ✓ | – | None |
| 18 | 7493025dda82:1 | M1 | Ryan starts his shift at the supermarket at 7:00 a.m. He stocks shelves in the bakery until 10:00 a.m., then goes to a safety training session with his team until 12:00 p.m. After the training, he works at the cash register for three more hours before heading home. How many hours does Ryan stay at the supermarket? | 8 | 8 | ✓ | pass | True |
| 19 | efba209c5a05:1 | M1 | Olivia has 7 more than 4 times as many holiday cards as Noah. Noah has a third as many holiday cards as Grandpa Joe. If Grandpa Joe has 15 holiday cards, how many holiday cards does Olivia have? | 27 | 27 | ✓ | pass | True |
| 20 | c753b0302ae9:1 | M1 | Ms. Brooks orders 4 pizzas for the 8 students in her class after a reading contest. Each pizza is cut into 6 slices. If every student gets the same number of slices, how many slices will each student get? | 3 | 3 | ✓ | pass | True |
