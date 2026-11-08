# LEDGER — compute and API spend

_Generated 2026-11-08 09:27 (UTC+7) by `handoff/tools/ledger.py` from files on disk. Re-run it after the last job finishes; do not hand-edit the tables._

## 1. Claude API (Anthropic) by workstream

**Source of truth for Claude spend is the Anthropic console** (workspace usage/cost page). The ledger numbers below are a reconstruction from our own logs, used to attribute spend to workstreams.

Source: `logs/api/<WS>/*.jsonl` (append-only ledgers, keys never logged). Cross-check: `.cache/vimath/spend.sqlite` (the `CostTracker` hard-cap database used by `common/llm.py`).

| WS | Claude USD (ledger) | spend.sqlite USD | Cap (DELIVER_BRIEF) | Models returned |
|---|---|---|---|---|
| D1 | $17.32 | $17.24 | $21 | claude-haiku-5-5, claude-opus-5-5 |
| D5 | $4.40 | $4.40 | $4.8 | claude-opus-5-5 |
| G1 | $7.55 | – (not via common/llm.py) | $10 | claude-haiku-5-5, claude-opus-5-5, claude-sonnet-5-5 |
| G2 | $6.50 | $6.50 | $8 | claude-haiku-5-5, claude-opus-5-5, claude-sonnet-5-5 |
| G3 | $4.21 | $4.21 | $6 | claude-haiku-5-5, claude-opus-5-5 |
| G4 | $4.15 | $4.15 | G4+G5 $27 | claude-haiku-5-5, claude-opus-5-5, claude-sonnet-5-5 |
| G5 | $18.46 | $18.46 | G4+G5 $27 | claude-haiku-5-5, claude-opus-5-5, claude-sonnet-5-5 |
| Q1 | $15.36 | $15.36 | $15.5 | claude-opus-5-5 |
| S3 | $2.53 | $2.53 | $4 | claude-haiku-5-5, claude-opus-5-5, claude-sonnet-5-5 |
| X2 | $8.84 | – (not via common/llm.py) | $8.84 approved | claude-opus-5-5 (X2 report; results files carry no model field) |
| X3 | $1.73 | – (not via common/llm.py) | – | claude-opus-5-5 |
| **Total** | **$91.06** | $72.86 | | |

**Reconciliation with the console:** 2026-11-07 ~22:40 the user read ≈ $43 in the console while this ledger said $43.22 (+0.5 %). Latest console reading ≈ $91.40 (2026-11-08 09:45 +07) vs ledger $91.06: difference $-0.34 (-0.4 %). A gap above ~$1 needs investigation (most likely: a Batch whose results were never collected, or calls made outside `common/llm.py`).

Rule: where a client logged `cost_usd`/`cost`, that value is used (it carries the Batch −50 % discount). Cost is recomputed from `usage` only where nothing was logged (X3 Batch results, at Batch rate). Recomputing G4/Q1 from usage at list price would overstate them, because part of their calls ran on Batch.

Claude budget: $100 (+ workspace header). Prices: list $/MTok as in `common/costs.py` (Haiku 5.5 0.10/0.50, Sonnet 5.5 2/10, Opus 5.5 4/20; Batch ×0.5).

## 2. All providers, per workstream × model

| WS | Provider | Model requested | Calls | Errors | Cache hits | In tok | Out tok | USD | First | Last |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | claude | `?` | 3 | 0 | 0 | 0 | 0 | $0.00 | 2026-11-07T22:20 | 2026-11-07T23:00 |
| D1 | claude | `claude-haiku-5-5` | 50,349 | 1,843 | 31 | 25,238,088 | 55,486,405 | $16.34 | 2026-11-07T15:36 | 2026-11-07T23:00 |
| D1 | claude | `claude-opus-5-5` | 121 | 0 | 0 | 56,069 | 76,844 | $0.98 | 2026-11-07T16:04 | 2026-11-07T19:48 |
| D1 | gemini | `gemini-2.5-flash-lite` | 6 | 6 | 0 | 0 | 0 | $0.00 | 2026-11-07T19:03 | 2026-11-07T19:03 |
| D1 | gemini | `gemini-3.1-flash-lite` | 44 | 27 | 0 | 19,068 | 11,256 | $0.00 | 2026-11-07T15:41 | 2026-11-07T19:04 |
| D1 | gemini | `gemini-3.1-flash-lite-preview` | 6 | 5 | 0 | 0 | 0 | $0.00 | 2026-11-07T19:03 | 2026-11-07T19:03 |
| D1 | gemini | `gemini-3.5-flash-lite` | 208 | 44 | 0 | 207,615 | 116,734 | $0.00 | 2026-11-07T15:45 | 2026-11-07T19:04 |
| D1 | gemini | `gemini-flash-lite-latest` | 6 | 5 | 0 | 0 | 0 | $0.00 | 2026-11-07T19:03 | 2026-11-07T19:03 |
| D5 | claude | `claude-opus-5-5` | 676 | 0 | 1 | 252,096 | 375,112 | $4.40 | 2026-11-07T15:43 | 2026-11-08T01:27 |
| G1 | claude | `claude-haiku-5-5` | 441 | 0 | 0 | 357,169 | 448,234 | $0.26 | 2026-11-07T16:16 | 2026-11-07T17:20 |
| G1 | claude | `claude-opus-5-5` | 359 | 0 | 0 | 151,819 | 187,650 | $4.53 | 2026-11-07T16:16 | 2026-11-07T17:20 |
| G1 | claude | `claude-sonnet-5-5` | 443 | 0 | 0 | 362,538 | 204,001 | $2.77 | 2026-11-07T16:16 | 2026-11-07T17:19 |
| G1 | gemini | `gemini-3.1-flash-lite` | 507 | 34 | 0 | 239,905 | 115,532 | $0.00 | 2026-11-07T16:50 | 2026-11-07T17:19 |
| G1 | gemini | `gemini-3.5-flash` | 186 | 73 | 0 | 56,949 | 179,635 | $0.00 | 2026-11-07T16:16 | 2026-11-07T16:50 |
| G1 | gemini | `gemini-3.5-flash-lite` | 947 | 130 | 0 | 575,591 | 185,729 | $0.00 | 2026-11-07T16:50 | 2026-11-07T17:18 |
| G1 | gemini | `gemini-3.8-flash` | 28 | 3 | 0 | 21,180 | 24,501 | $0.00 | 2026-11-07T16:17 | 2026-11-07T16:49 |
| G1 | nim | `deepseek-ai/deepseek-v4.1-flash` | 34 | 5 | 0 | 17,242 | 63,923 | $0.00 | 2026-11-07T16:17 | 2026-11-07T16:58 |
| G1 | nim | `nvidia/nemotron-3-ultra-550b-a55b` | 101 | 0 | 0 | 57,405 | 248,349 | $0.00 | 2026-11-07T16:16 | 2026-11-07T16:58 |
| G1 | nim | `z-ai/glm-5.3-flash` | 90 | 1 | 0 | 46,568 | 157,543 | $0.00 | 2026-11-07T16:16 | 2026-11-07T16:58 |
| G2 | claude | `claude-haiku-5-5` | 1,343 | 0 | 21 | 559,365 | 1,122,951 | $0.64 | 2026-11-07T10:35 | 2026-11-07T11:06 |
| G2 | claude | `claude-opus-5-5` | 350 | 0 | 0 | 125,114 | 164,252 | $3.92 | 2026-11-07T10:56 | 2026-11-07T11:10 |
| G2 | claude | `claude-sonnet-5-5` | 323 | 0 | 7 | 136,219 | 129,735 | $1.94 | 2026-11-07T10:46 | 2026-11-07T11:28 |
| G2 | gemini | `gemini-3-flash-preview` | 137 | 17 | 0 | 107,095 | 284,495 | $0.00 | 2026-11-07T11:10 | 2026-11-07T11:21 |
| G2 | gemini | `gemini-3.1-flash-lite` | 495 | 3 | 14 | 1,249,441 | 100,357 | $0.00 | 2026-11-07T10:35 | 2026-11-07T11:00 |
| G2 | gemini | `gemini-3.1-pro-preview` | 6 | 6 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:28 | 2026-11-07T10:28 |
| G2 | gemini | `gemini-3.6-flash` | 12 | 6 | 0 | 5,199 | 6,112 | $0.00 | 2026-11-07T11:21 | 2026-11-07T11:21 |
| G2 | gemini | `gemini-3.7-flash` | 19 | 13 | 0 | 965 | 1,222 | $0.00 | 2026-11-07T10:42 | 2026-11-07T10:56 |
| G2 | gemini | `gemini-3.8-flash` | 82 | 13 | 1 | 56,865 | 49,830 | $0.00 | 2026-11-07T10:29 | 2026-11-07T10:51 |
| G3 | claude | `claude-haiku-5-5` | 386 | 0 | 69 | 864,786 | 511,496 | $0.29 | 2026-11-07T11:10 | 2026-11-07T11:23 |
| G3 | claude | `claude-opus-5-5` | 338 | 0 | 2 | 117,751 | 166,513 | $3.93 | 2026-11-07T11:25 | 2026-11-07T14:08 |
| G3 | gemini | `gemini-3.1-flash-lite` | 486 | 57 | 24 | 743,836 | 200,914 | $0.00 | 2026-11-07T11:17 | 2026-11-07T14:10 |
| G4 | claude | `claude-haiku-5-5` | 552 | 0 | 18 | 336,991 | 648,232 | $0.36 | 2026-11-07T14:26 | 2026-11-07T15:21 |
| G4 | claude | `claude-opus-5-5` | 532 | 0 | 0 | 183,086 | 237,044 | $2.80 | 2026-11-07T14:45 | 2026-11-07T15:00 |
| G4 | claude | `claude-sonnet-5-5` | 346 | 0 | 0 | 142,952 | 136,962 | $1.00 | 2026-11-07T15:00 | 2026-11-07T15:18 |
| G4 | gemini | `gemini-3.1-flash-lite` | 204 | 6 | 0 | 186,381 | 113,250 | $0.00 | 2026-11-07T14:35 | 2026-11-07T15:22 |
| G5 | claude | `claude-haiku-5-5` | 52,213 | 115 | 6,706 | 30,548,815 | 45,898,092 | $16.68 | 2026-11-07T15:33 | 2026-11-07T22:47 |
| G5 | claude | `claude-opus-5-5` | 242 | 0 | 0 | 81,924 | 110,945 | $1.30 | 2026-11-07T15:58 | 2026-11-07T15:58 |
| G5 | claude | `claude-sonnet-5-5` | 75 | 0 | 0 | 30,322 | 29,565 | $0.49 | 2026-11-07T15:52 | 2026-11-07T15:53 |
| G5 | gemini | `gemini-3.1-flash-lite` | 79 | 50 | 0 | 29,818 | 16,939 | $0.00 | 2026-11-07T15:54 | 2026-11-07T18:23 |
| G5 | gemini | `gemini-3.5-flash-lite` | 139 | 50 | 1 | 101,454 | 66,620 | $0.00 | 2026-11-07T15:54 | 2026-11-07T18:23 |
| G5 | nim | `google/gemma-4-31b-it` | 122 | 50 | 0 | 67,432 | 51,052 | $0.00 | 2026-11-07T19:10 | 2026-11-07T19:24 |
| G5 | nim | `nvidia/nemotron-3.5-bolt-e4b` | 129 | 0 | 0 | 141,265 | 79,003 | $0.00 | 2026-11-07T19:05 | 2026-11-07T19:08 |
| G5 | nim | `z-ai/glm-5.3-flash` | 1,798 | 235 | 0 | 1,623,088 | 5,221,817 | $0.00 | 2026-11-07T19:16 | 2026-11-07T20:34 |
| Q1 | claude | `claude-opus-5-5` | 1,593 | 0 | 0 | 1,514,636 | 1,200,797 | $15.36 | 2026-11-07T14:36 | 2026-11-07T20:08 |
| S3 | claude | `claude-haiku-5-5` | 10,356 | 48 | 74 | 3,056,805 | 2,843,881 | $1.08 | 2026-11-07T14:38 | 2026-11-07T19:03 |
| S3 | claude | `claude-opus-5-5` | 60 | 0 | 0 | 22,108 | 33,316 | $0.81 | 2026-11-07T19:04 | 2026-11-07T19:05 |
| S3 | claude | `claude-sonnet-5-5` | 216 | 32 | 54 | 127,351 | 57,832 | $0.65 | 2026-11-07T14:39 | 2026-11-07T16:50 |
| S3 | gemini | `gemini-3.1-flash-lite` | 1,499 | 12 | 180 | 335,700 | 188,457 | $0.00 | 2026-11-07T14:40 | 2026-11-07T16:48 |
| S3 | gemini | `gemini-3.5-flash-lite` | 1,804 | 47 | 9 | 272,532 | 303,892 | $0.00 | 2026-11-07T15:49 | 2026-11-07T16:17 |
| W3 | gemini | `gemini-3.5-flash-lite` | 1 | 0 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:49 | 2026-11-07T10:49 |
| W3 | nim | `deepseek-ai/deepseek-v4.1-flash` | 1 | 0 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:49 | 2026-11-07T10:49 |
| W3 | nim | `nvidia/nemotron-3-ultra-550b-a55b` | 6 | 0 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:48 | 2026-11-07T10:58 |
| W5 | nim | `deepseek-ai/deepseek-v4.1-flash` | 10 | 1 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:44 | 2026-11-07T10:55 |
| W5 | nim | `nvidia/nemotron-3-ultra-550b-a55b` | 1 | 0 | 0 | 0 | 0 | $0.00 | 2026-11-07T10:44 | 2026-11-07T10:44 |
| X2 | claude | `claude-opus-5-5` | 1,585 | 0 | 0 | 605,105 | 762,758 | $8.84 | 20261107T103028 | 20261107T110355 |
| X3 | claude | `claude-opus-5-5` | 509 | 0 | 0 | 362,839 | 100,267 | $1.73 |  |  |

Gemini (`gemini-*`, free tier, 6 keys/projects) and NIM (free trial) are $0. Timestamps are UTC as logged.

## 3. Modal

Latest saved per-app report: `jobs/logs/D4/modal_billing_mtd_20261108T0927.json` (written 2026-11-08 09:27). Totals are metered (pre-credit) USD, summed over every object and interval of each app in that report (a `*billing_mtd*` report = month to date, `modal billing report --for "this month" --json`). Modal credit: $30/month (November 2026).

| Modal app | Metered USD |
|---|---|
| `t1r-eval` | 10.100 |
| `g5-vllm-cycle2` | 6.566 |
| `r-gpu-bench` | 4.549 |
| `g5-vllm-cycle` | 2.887 |
| `s2-student-pilot` | 1.675 |
| `y2-analysis` | 1.348 |
| `o1-throughput-bench` | 0.838 |
| `o1-throughput-bench-eval3` | 0.438 |
| `o1-throughput-bench-eval2` | 0.342 |
| `x5-embed` | 0.274 |
| `d1-n-decontam` | 0.255 |
| `g5-decontam` | 0.048 |
| `qa-t-nllprobe` | 0.043 |
| `s3-seed-embed` | 0.035 |
| `g3-embed` | 0.015 |
| `g4-embed` | 0.014 |
| `qa-t-tokcheck` | 0.001 |
| **Total (apps in this report)** | **29.43** |

Modal notes (2026-11-08): the launcher's workspace guard read **$29.95 month to date at ~08:20** (billed apps + its estimate of in-flight unbilled work) and hard-stopped (`jobs/queue/t2_state.json` `hard_stop`; guard limit $29.5 of the $30 credit). Billing lags by minutes, so the per-app sum above can trail the guard. By workstream: T1r/T2 = `t1r-eval` (tranche training + eval, G6 sweep, pilots); G5 = `g5-vllm-cycle*` + `g5-decontam` ($9.50 of its $9.5 cap; one timed-out call cost $2.89); R-GPU = `r-gpu-bench`; S2, O1, Y2, X5, S3, G3, G4, D1, QA-T as named.

To refresh (read-only, no spend): `E:\venv_tools\Scripts\modal.exe billing report --for "this month" --json` (save the output as `jobs/logs/D4/modal_billing_mtd_<timestamp>.json`, then re-run this script).

## 4. Kaggle GPU quota

Weekly quota 30 GPU-h (T4×2 bills 1 quota-hour per wall-hour, measured by S2). Last reading per kernel timeline:

| Timeline | Last reading (UTC) | gpu_quota_used_h (week) | Status |
|---|---|---|---|
| `jobs/logs/s2-pilot-v2_timeline.jsonl` | 2026-11-07T11:50:18Z | 2.52 | COMPLETE |
| `jobs/logs/t1r-runs-v1_timeline.jsonl` | 2026-11-07T16:30:02Z | 3.43 | COMPLETE |

The quota counter is weekly and cumulative across kernels; the largest recent value is the week's usage. Kernels launched without the timeline writer (e.g. D5's `datuni/d5-select-embed`, ≈3 min T4) are not in these files; Kaggle shows the exact weekly figure on the account's settings page.

## 5. Method notes and known gaps

- G1 and X2/X3 predate `common/llm.py`, so they are absent from `spend.sqlite`; their ledger costs are summed here.
- X2's request/result dumps are not re-costed; `logs/api/X2/cost_ledger.jsonl` is the per-batch cost of record.
- G2's three one-shot Gemini quota probes were not logged (G2 report); they were free-tier.
- W3/W5 are early (pre-brief) NIM/Gemini probes; $0. No Claude calls are in them.
- STATUS.md's "≈ $32.5" (22:35) omits X2 ($8.84) and X3 ($1.73); this table includes them.
- Kaggle: no API exposes remaining quota; the timeline value is read from the Kaggle kernel status at poll time.
