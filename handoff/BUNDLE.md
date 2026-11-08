# BUNDLE — delivering what is not in git

Git holds code, prompts, docs, small manifests and this package. Everything below is **git-ignored or too large for
git** and is delivered as zip files built by [`tools/make_bundle.py`](tools/make_bundle.py) from the frozen paths. The
file list with bytes and **sha256 of every file** is [`BUNDLE_MANIFEST.tsv`](BUNDLE_MANIFEST.tsv) (hashed 2026-11-08
with `make_bundle.py --hash`; the zips are built at the final freeze).

## Bundles

| Bundle | Contents (spec = `SPEC` in `make_bundle.py`) | Who may receive it | Size (2026-11-08) |
|---|---|---|---|
| `vimath-data-<tag>.zip` | `data/eval/` v1.1 (minus `_work/`); Vi-ExamMath public files (`items.jsonl`, `exams.jsonl`, `summary.json`, indices, QA queue, Claude keys); `data/vi_exam_qa/` (**core_v1.1**, v1, invalid sets, decisions, audits, batches); `data/seeds/` (**pool_v1**, dev_v1, drops, changelog, `_work` audit trail minus embeddings); `data/train/` (Y/N pools + `_work`, **final** matched sets, `final_shared`, `final_natural`, `final_opmatched`, `slots_v1`); `data/t1r/` (S0 + dev sets); `data/audit/` (D5 gate audit, R1 tables, diversity); `data/contam/` flags + scans (no corpus snippets); `data/pilot/` | successor, advisor (internal) | 695 files, 953 MiB |
| `vimath-runs-<tag>.zip` | everything under `jobs/outputs/` (run manifests, configs, train/dev logs, pre-flight, pack gates, post-run checks + loss curves, **item-level predictions** `t2-eval/ev-*.jsonl`, G6 sweep, GRPO smoke, parity/repeatability checks), **the 25 tranche adapters once pulled (below)**, `jobs/queue/` (queue + launcher state), `bench/r_gpu/results/`. Excluded: optimizer checkpoints (`ckpt/`, `optimizer.pt`, `adapter.pt`), smoke-test and Kaggle-pilot adapters | successor, advisor (internal) | 789 files, 275 MiB **before** the adapters (+ ≈ 25 × 81 MB ≈ 2.0 GB after) |
| `vimath-private-<tag>.zip` | `data/vi_exam/items_private.jsonl` (third-party worked solutions), `data/vi_exam/raw/` (cached HTML), fetch log, listings; `data/contam/**/corpus_snippets*` (third-party corpus text); **`logs/api/**`** (every prompt + response, incl. **benchmark/test items sent to Claude**; D1 alone is 571 MB) | successor only; **never public, never uploaded to HF/Zenodo** | 1,598 files, 1.6 GiB |

Not bundled: `.env*`, `.cache/` (LLM response cache + spend DB; rebuildable from `logs/api` via
`common/cache.rebuild_from_logs()`), venvs, `_scratch/`, legacy `data/{finetune,formalize_output,generate,translation,
test,...}` (Sept-2026 intermediates; the frozen sets record their hashes as `local_sources` / datacard `sources`).

## Adapters

The 25 tranche adapters (Qwen3-0.6B × {S0, Y+, Y−, N+, N−} × {VI s0–3, EN s0}) were pulled from the Modal Volume
**`t1r-eval`** (`/train/<run_id>/adapters/epoch3`) into `jobs/outputs/t2-modal/<run_id>/adapters/epoch3/`
(`adapter_model.safetensors` 77 MiB F32 + `adapter_config.json`; 1.9 GB in total; git-ignored) and ship in the `runs`
zip. `handoff/tools/verify_adapters.py` checked all 25 (safetensors header and size, base model, r = α = 32, LoRA,
target modules vs the run manifest): **25/25 OK**, 25 distinct hashes. The run manifests record no final-adapter hash
(only `lora_init_sha`), so `handoff/ADAPTERS.sha256` is the reference from now on. Load with PEFT on top of
`Qwen/Qwen3-0.6B-Base` (bf16), or serve with vLLM `--enable-lora` as `evalrun/modal_eval.py` does. Re-pull one:
`modal volume get t1r-eval /train/<run_id>/adapters/epoch3 jobs/outputs/t2-modal/<run_id>/adapters/epoch3`.

## Commands

```
# 1) plan (fast, no hashing): refreshes BUNDLE_MANIFEST.tsv with paths + bytes
.venv/Scripts/python.exe handoff/tools/make_bundle.py
# 2) plan + sha256 of every file (streams; 3,082 files, 2.8 GB hashed in ≈ 8 s on the laptop, warm cache)
.venv/Scripts/python.exe handoff/tools/make_bundle.py --hash
# 3) build + verify zips OUTSIDE the repo (refuses an --out inside it) — at the final freeze only
.venv/Scripts/python.exe handoff/tools/make_bundle.py --build --out D:/vimath_bundle --tag 20261108 --only data,runs
.venv/Scripts/python.exe handoff/tools/make_bundle.py --build --out D:/vimath_private --tag 20261108 --only private
```
Each zip contains `MANIFEST.sha256` (sha256 of every member); every member is re-read and checked after writing, and
`<zip>.sha256` is written next to the zip. Receiver: `sha256sum -c vimath-data-<tag>.zip.sha256`, unzip at the repo
root, then verify the inner `SHA256SUMS` files (`data/eval`, `data/seeds`, `data/vi_exam_qa/core_v1.1.sha256`,
`data/train/{Y,N}/*`, `data/train/final*/{Y,N}/*`).

## Transfer
Private channel only (the user's Google Drive / USB to the successor). No public upload until the release decisions in
`KNOWN_ISSUES.md` §G are taken (Vi-ExamMath licensing, public dataset licenses). The later public
layout (HF datasets + Zenodo) is specified in [`06_packaging.md`](../docs/research_2026-11/06_packaging.md) §3, §5.4.

## Git hygiene check for the bundle paths (2026-11-08; fixed by the lead in `54488f3`)
`data/`, `logs/api/`, `.cache/`, `*.safetensors`, `jobs/outputs/**/{ckpt,adapters}/`, `jobs/outputs/**/preds*.jsonl`
are ignored. **Not ignored:** `jobs/outputs/t2-eval/ev-*.jsonl` (38 files, 140 MB of predictions), `jobs/outputs/t2-g6-eval/`
(55 MB) and `bench/r_gpu/results/preds/` (40 MB) — add `jobs/outputs/t2-eval/`, `jobs/outputs/t2-g6-eval/` and
`bench/r_gpu/results/preds/` to `.gitignore` before committing T2/R-GPU outputs (they ship in the `runs` bundle).
