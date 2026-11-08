# D4 — hand-off package: skeleton and first pass (2026-11-07, ~23:00 UTC+7)

Brief: lead's D4 task + `DELIVER_BRIEF.md`; mid-task updates from the lead (commits allowed for `handoff/`; replace
COMMIT_PLAN with BUNDLE; ledger must cover every log format and reconcile to the Anthropic console).
Owned files: `handoff/**`, `tests/test_handoff_tools.py`. No API spend. No push.

## 1. What was built (`handoff/`)

| File | Content | How it stays current |
|---|---|---|
| `README.md` | P1–P3, C1–C4, arms, H1–H4; repo map; where every dataset/prompt/output/report lives; exact commands per stage (CLI sub-commands checked against argparse) | hand-edited; PENDING markers for Q1/S3/G5/D1/D5 |
| `RUNBOOK.md` | operator checklist around T2's queue/launcher: per-block costs (from `t2_queue.totals.json`), pre-flight, run, failure table, after-block steps | links `exec/T2_training_tranche.md` as source of truth |
| `ENV.md` | laptop env (uv.lock), the five extra venvs, Kaggle pins (S2 §5 + T1r lock), Modal images per app | hand-edited |
| `DATASHEETS/` | Gebru 7-section datasheets: `vi_exam_core_v1`, `eval_sets_v1.1`, `seed_pool_v1`, `train_pools` + README with the **Claude provenance** (model ids, prompt hashes, API-log locations) | PENDING fields filled 11-08 |
| `MODEL_CARD_TEMPLATE.md` + `tools/fill_model_cards.py` | HF-style card (base_model, adapter relation, data, provenance, training, eval table) filled from `jobs/outputs/**/manifest.json` + `metrics.json`; evals attached via `eval_of`; missing fields → `n/a`, never invented | `MODEL_CARDS/` (1 card now: base Qwen3-0.6B 4-shot) |
| `PROMPTS.md` + `PROMPTS.sha256` (`tools/prompt_catalog.py`) | 35 registry prompts: id, version, role, lang, status, sha256, calls per WS from the ledgers, **hash check vs logged `prompt_sha256`** (0 drifted); 6 legacy G1 prompts; 1 orphan ledger id | regenerate |
| `LEDGER.md` (`tools/ledger.py`) | Claude per WS from every log format (logged `cost_usd`/`cost` used as-is; X2 `cost_ledger.jsonl`; X3 batch results recomputed at Batch rate only because nothing was logged), cross-check vs `spend.sqlite`, console reconciliation line; Modal per app from the latest saved billing report; Kaggle quota from timelines | regenerate |
| `BUNDLE.md` + `tools/make_bundle.py` + `BUNDLE_MANIFEST.tsv` | data / runs / private bundles from frozen paths; plan (bytes), `--hash`, `--build --out <outside repo>` with per-member sha256 verify | plan only tonight (not run on big data) |
| `KNOWN_ISSUES.md` | unfinished work, bugs found, env, hygiene, pending decisions (incl. venvs to remove) | finalize 11-08 |
| `SECRETS_SCAN.md` (`tools/secrets_scan.py`) | paths only, never values | regenerate |
| `test_report.md` (`tools/run_light_tests.py`) | per-file pytest under a 300 MB Windows Job Object cap | regenerate |

Commits on `feat/sprint0` (not pushed): `0d0d117` tools, `24544b6` docs, `191cc46` generated snapshot, `e06ec15`
test report, `d2a10c7` + `7692d7b` count fixes. This report itself is left uncommitted for the lead.

## 2. Measured

**Tests** (`.venv/Scripts/python.exe handoff/tools/run_light_tests.py --alt "E:/venv_y1/Scripts/python.exe=<test_train_*.py>"`,
one process per file, cap 300 MB committed memory, `OPENBLAS_NUM_THREADS=1`): **882 passed, 1 failed, 2 skipped,
2 xfailed** over 31 files; `test_analysis_heavy.py` excluded (Modal only). Max peak 188.7 MB (train evalrun/sft
loop tests). Evidence: `handoff/test_report.md`.
- The failure: `test_common_prompts_config.py::test_repo_registry_has_all_role_templates` asserts every prompt is
  `language: vi`; English prompts now exist (`seed_solve_en`, `q1_*`, `validity_judge.v2/v3`, `*_en`). Test is stale,
  prompts are fine (KNOWN_ISSUES B-1, owner Y3).
- The six `test_train_*.py` files skip entirely in `.venv` (pyyaml missing); under `E:\venv_y1` they pass 74/74
  (B-2). Under the cap, `import numpy` with default OpenBLAS threads commits ≈ 250 MB (test_analysis 290.7 MB,
  test_decontam 289.7 MB on the first run → 96 / 43 MB with one thread) (B-4).

**Environment.** `uv lock --check` exit 0 (uv 0.10.5, 50 packages); `uv export … --all-extras --no-hashes
--no-emit-project` is identical to `requirements.lock.txt`. `uv sync --locked --all-extras --dry-run`: `.venv` would
uninstall 25 / install 33 packages — not applied (shared by running agents). Venvs listed for removal:
`E:\venv_tools` (77 pkgs), `E:\venv_y1` (60, torch 2.14.0+cpu), `E:\venv_dc` (44, torch 2.14.1+cu126), `E:\venv_mv`
(14), `E:\venv_stats` (21). Nothing installed, nothing deleted.

**Spend (snapshot 22:50).** Claude ledger total **$44.64** (G1 7.55, G2 6.50, G3 4.21, G4 4.15, Q1 9.62+, S3 0.59+,
X2 8.84, X3 1.73, plus new G5/D1/D5 calls as they land); at the console reading (≈ $43, 22:40) the ledger said
$43.22 (+0.5 %). STATUS.md's $32.5 omitted X2 and X3. Modal metered **$5.32** for 2026-11-07 (latest saved per-app
report 22:05; billed $0.00 against the credit). Kaggle quota **2.52 h** this week (last timeline reading 15:05Z).
Evidence: `handoff/LEDGER.md`.

**Secrets scan** (5,766 files; `.env*`, `.venv`, `.git` excluded; 16 `.env` secret values searched verbatim, never
printed): **0 hits in git-tracked files, 0 verbatim `.env` values anywhere.** All 682 shape hits are in 341 cached
loigiaihay pages under `data/vi_exam/raw/` (git-ignored, private bundle): the site's own embedded browser key,
1 distinct value. Evidence: `handoff/SECRETS_SCAN.md`.

**Bundle plan** (no hashing): data 338 files / 144 MiB (train pools not frozen yet), runs 177 / 9 MiB (no tranche
yet; smoke adapters and optimizer checkpoints excluded), private 1,596 / 441 MiB (raw HTML, third-party solutions and
corpus snippets, API ledgers with benchmark items). `--hash --only data` ran in seconds; `data/eval` 18/18 OK.

## 3. Blocked / pending
- Datasheet numbers for core_v1 (Q1), pool_v1 (S3), Y/N pools (G5, D1, D5) and model cards for the tranche (T2) —
  their data/reports did not exist at 23:00.
- `.gitignore` additions proposed to the lead (`*.bak`, `*.pem`, `*.key`, `kaggle.json`, `.modal.toml`) — not my file.
- User actions: keep the Claude provenance disclosure in any release (D-1); confirm the VN license reading (D-2); `paper_text.txt` is
  still in history (R-1).

## 4. Finalize on 11-08 (when resumed)
1. Regenerate: `ledger.py` (pass `--console-usd` with the user's console figure), `prompt_catalog.py`,
   `fill_model_cards.py`, `make_bundle.py --hash`, `secrets_scan.py`, `run_light_tests.py --alt …`.
2. Fill every [PENDING] field from the Q1/S3/G5/D1/D5/T2 reports and the frozen `SHA256SUMS`/`manifest.json`.
3. Mark finished rows from `jobs/queue/t2_state.json` in RUNBOOK; update KNOWN_ISSUES A.
4. Build the zips outside the repo (`make_bundle.py --build --out …`) at the 18:00 freeze; commit the regenerated
   `handoff/` files.

---

## 5. Finalization pass (2026-11-08, 08:30 → ~09:30 +07)

Inputs read: DELIVER_BRIEF (all "Changes" sections) and exec reports S3, Q1 (core_v1.1), D1, G4, G5, D5_selection,
D5_gate_audit, C3, R-GPU, QA-T, T1r, T2, R1 (08:18 refresh).

| Item | Done | Evidence |
|---|---|---|
| Datasheets | every [PENDING] filled with final counts and hashes: pool_v1 `1b565411…` (9,649 + dev 512), core_v1.1 `ceac95de…` (1,545; 1,367 VERIFIED; audit 1.3 % [0.16, 4.7]), Y/N pools (sha per pool), final matched sets n = 1,808 (8 sha256), secondary sets (`final_shared` 1,679, `final_natural` 2,734/2,325, `final_opmatched` 1,387/1,404); Claude provenance (model ids, prompt hashes, API logs); deviations (Gemini quota → Haiku-v3-only screen; Qwen3-32B-AWQ cycle model; n 10k → 1,808; seed-matched selection) | `handoff/DATASHEETS/*.md`; all `SHA256SUMS` re-verified OK with `sha256sum -c` |
| Model cards | 26 cards (base 4-shot + 25 adapters: VI s0–3 × 5 arms, EN s0 × 5 arms) from `jobs/queue/t2_state.json` + `jobs/outputs/t2-modal/*/manifest.json`; per-dataset pass@1/greedy with CIs; adapter location on Modal Volume `t1r-eval` + pull command; VI s3 cards show "no evaluation" | `handoff/MODEL_CARDS/`, `tools/fill_model_cards.py` (T2 mode, new test) |
| LEDGER | Claude ledger **$91.06** (D1 17.32, G5 18.46, Q1 15.36, X2 8.84, G1 7.55, G2 6.50, D5 4.40, G3 4.21, G4 4.15, S3 2.53, X3 1.73); console = source of truth, reconciliation line (final reading to be passed with `--console-usd`); Modal month-to-date **$28.78** per app (fresh read-only `modal billing report --for "this month"` at 08:35; the launcher's guard read $29.95 incl. in-flight); Kaggle 3.43 quota-h (last timeline) | `handoff/LEDGER.md`, `jobs/logs/D4/modal_billing_mtd_20261108T0835.json` |
| RUNBOOK | final platform (Modal L40S), state table, adapter pull, costs per cell, successor queue #0–#13 in the lead's priority order with costs | `handoff/RUNBOOK.md` §0, §5 |
| KNOWN_ISSUES | rewritten: A unfinished (U-1…U-12), B grader gaps G-1…G-4, C data defects (X2-SEG, Q1-NV, P-1…P-4), D bugs, E env, F hygiene, G decisions (D-1…D-8), H methodology (M-1…M-5) | `handoff/KNOWN_ISSUES.md` |
| Bundle | SPEC extended (data/audit, all of jobs/outputs incl. `t2-eval` preds, jobs/queue, bench/r_gpu results; pilot/smoke adapters and optimizer state excluded); `make_bundle.py --hash` over **3,082 files, 2.8 GB** (data 695 / 953 MiB · runs 789 / 275 MiB before the 25 adapters · private 1,598 / 1.6 GiB); zips NOT built | `handoff/BUNDLE_MANIFEST.tsv` (sha256 per file), `handoff/BUNDLE.md` |
| Tests | **930 passed, 1 failed, 2 skipped, 2 xfailed** over 38 files (7 yaml-dependent train files under `E:\venv_y1`), max 189 MB per process under the 300 MB cap; the failure is the known stale B-1 | `handoff/test_report.md` |

**Findings for the lead.**
1. `jobs/outputs/t2-eval/ev-*.jsonl` (140 MB), `jobs/outputs/t2-g6-eval/` (55 MB) and `bench/r_gpu/results/preds/`
   (40 MB) are **not git-ignored** (the `preds*.jsonl` pattern does not match `ev-*.jsonl`). Add them to `.gitignore`
   before committing T2/R-GPU outputs; they ship in the `runs` bundle.
2. `data/train/final/manifest_vi.json` records the N-VI pool as `cde1e8a3…`, but D1 rewrote the pool at 07:04
   (`73c466cd…`). Checked: all 3,616 selected N-VI rows are identical in the current pool (KNOWN_ISSUES P-1).
3. VI seed 3 (5 adapters) is trained but not evaluated (workspace guard hard stop) — ≈ $0.47 to finish.
4. The 25 adapters exist only on the Modal Volume `t1r-eval`; pull them (BUNDLE.md §Adapters) before building the
   `runs` zip and before the Modal account changes hands.

---

## 6. FINAL FREEZE (2026-11-08, ~09:20 → 09:50 +07, from HEAD `ffa8ae6`)

| Item | Result | Evidence |
|---|---|---|
| Ledger | Claude ledger **$91.06** vs Anthropic console **$91.40** (09:45, the user's reading) → −$0.34 (−0.4 %), reconciled; Modal month to date **$29.43** of $30 (fresh read-only billing at 09:27); Kaggle GPU quota 3.43 h (+ CPU kernels, which use no GPU quota) | `handoff/LEDGER.md`, `jobs/logs/D4/modal_billing_mtd_20261108T0927.json` |
| Model cards | **27**: 4-shot base, **0-shot base** (arms' template; VI gsm8k 30.3, math 26.8, Core 18.1), 25 adapters all evaluated (VI s0–3, EN s0) with the local and Volume adapter paths | `handoff/MODEL_CARDS/` |
| Adapters | 25 local adapters verified (safetensors header/size, base model, r = α = 32, LoRA, target modules vs the run manifest): **25/25 OK**, 25 distinct hashes, 77 MiB F32 each. Manifests record no final-adapter hash (only `lora_init_sha`) → `ADAPTERS.sha256` is the reference | `handoff/ADAPTERS.md`, `ADAPTERS.sha256`, `tools/verify_adapters.py` |
| Docs | README "Results" = R1 final (supported / not supported / caveats); KNOWN_ISSUES opens with the successor queue (EN s1 → 4-shot eval of adapters → length-matched retrain → 1.7B / second family → rest); RUNBOOK §0/§5 rewritten; BUNDLE adapters section | `handoff/*.md` |
| Bundle hash | 3,153 files, 4.7 GiB, sha256 per file | `handoff/BUNDLE_MANIFEST.tsv` |
| **Zips** | built and member-verified in `E:\MATH-REASONING-THESIS\handoff_bundle_20261108\` (+ `MANIFEST.sha256`, `BUNDLE_MANIFEST.tsv`); `sha256sum -c MANIFEST.sha256` → 3/3 OK | see table below |
| Tests | **931 passed, 1 failed (stale B-1), 2 skipped, 2 xfailed**, 38 files, ≤ 189 MB per process | `handoff/test_report.md` |
| Secrets | 8,217 files: 0 hits in tracked files, 0 verbatim `.env` values; 683 shape hits = 682 site key in cached pages + 1 base64 blob in an S3 log (both private) | `handoff/SECRETS_SCAN.md` |

| Zip | Files | Size | sha256 |
|---|---|---|---|
| `vimath-data-20261108.zip` | 704 | 243,359,894 B (232.1 MiB) | `d29cb60c85b88c430fb59acbe8d7163fe2000cb10cc4482cbfd933ac4e2ca686` |
| `vimath-runs-20261108.zip` | 851 | 1,899,701,080 B (1,811.7 MiB) | `953da06aa57e6b38893ba0213f83591bc9cf140d3750f4c32a5a76e1378ee876` |
| `vimath-private-20261108.zip` | 1,598 | 326,932,943 B (311.8 MiB) | `e7da7e0dd5180780d8a90c50535eef7a7f3dac1c7d5385007a67b57adf87ec20` |

Issue fixed during the build: the per-zip `.sha256` files were written with CRLF on Windows, so `sha256sum -c` could
not open them; rewritten with LF, and `make_bundle.py` now writes LF and a combined `MANIFEST.sha256` itself.
The `private` zip holds raw API logs with benchmark items and third-party text: never public.
