# Y3 — shared runtime layer `vi_math_verified_aug/common/` + prompt registry + environment lock (2026-11-07)

Inputs: `06_packaging.md` §5.2 (ledger fields, cache, prompts, env locking), `tech/T6_compute_automation.md` §6.2/§6.4
(job data model, error classes, API adapters), `tech/T1` §2 / `tech/T2` §2.3 (prompt roles), G1's
`pilot_gen/clients.py` (lessons only, no code reused). No API calls were made; every test mocks the transports.

## 1. What was built

| File | What it does |
|---|---|
| `common/llm.py` | One async client `LLMClient` for **Claude** (anthropic SDK, streaming, `anthropic-workspace-id` header from `ANTHROPIC_WORKSPACE_ID`, SDK retries off, **Message Batches API** via `client.batches.{submit,status,collect,run}`), **Gemini** (REST `…/v1beta/models/{m}:generateContent`, key pool `GEMINI_API_KEYS`, header `x-goog-api-key`, key never in the URL), **NIM** (OpenAI-compatible, **off** unless `enable_nim=True` / `VIMATH_ENABLE_NIM=1`, one lane at 38 RPM regardless of key count per T6 §4.1). `LLMRequest` / `LLMResponse` dataclasses, `parse_json`. |
| `common/ratelimit.py` | Lanes per (provider, key, model): token-bucket RPM, in-flight cap, 429 cool-down (`Retry-After` header or Gemini `RetryInfo.retryDelay`, else exponential), **adaptive token-rate (AIMD on tokens / rolling 60 s)**: on a 429 `tpm_limit = max(min_tpm, 0.7 × tokens seen in the window)`, +5 % per success, capped at a configured `tpm`. This targets G1's finding that Gemini 429s on long outputs act like a per-key token-rate limit. `LanePool` sends each request to the lane that can start soonest (round-robin tie-break). After a 429 it retries on another key. |
| `common/costs.py` | Price table (Claude list prices as recorded by G1; Gemini/NIM $0). `cost_usd` covers cache read/write and the batch ×0.5. `CostTracker` gives a **hard cap that holds under concurrency**: it reserves the worst case (input estimate + `max_tokens` × output price) under a lock before each call, settles to the actual cost after, and records each charge once per `charge_id` (`INSERT OR IGNORE`) in SQLite. It reads totals incrementally, so other processes' spend counts too. Cache hits never touch spend. This fixes **G1's over-count bug** (G1 summed ledger costs while also adding live costs). |
| `common/cache.py` | Content-addressed response cache in SQLite (WAL), keyed by `request_hash`. Memory use stays constant, unlike G1's load-all-logs approach. `rebuild_from_logs()` regenerates it from the JSONL ledger. Only successes are cached; the first writer wins. |
| `common/apilog.py` | Append-only JSONL ledger `logs/api/<WS>/<provider>.jsonl`. Each line is one `write` under a thread lock plus an OS lock on a sidecar `<file>.lock` (`msvcrt.locking` on Windows, `flock` elsewhere). Windows `PermissionError` is retried. Fields follow 06 §5.2: `ts_utc, ws, run_id, git_sha, provider, base_url, model_requested, model_returned, params, prompt_id, prompt_version, prompt_sha256, request, response_text, raw_response, finish_reason, usage, latency_ms, attempt, status, error_class, error, cost_usd, cache_hit, request_hash, item_id, split, lane, client_lib_version`. Every record is also scrubbed of all secret-looking `.env` values. |
| `common/env.py`, `paths.py`, `hashing.py` | `.env` reader (process env wins) and secret list. Default dirs: `logs/api/`, `.cache/vimath/`, `state/`, `prompts/`, each overridable by a `VIMATH_*_DIR` variable. `canonical_json` / `stable_hash` are the single identity serialisation. |
| `common/prompts.py` + top-level `prompts/` | Versioned registry: `prompts/<id>.v<k>.md`, with front matter `id/version/role/language/status/inputs/description/source`, `### SYSTEM` / `### USER` sections and `<<name>>` placeholders. The markers and placeholder syntax match G1's pilot prompts, so the winners port by adding front matter. Rendering is strict (missing or undeclared inputs fail). The sha256 is LF-normalised, so CRLF and LF checkouts hash the same. `status: template` refuses to render. **Six empty templates** are in place: `formalizer, editor, native_generator, solution_writer, validity_judge, answer_judge` (all `vi`, v1, with a draft `inputs` contract and a pointer to the G1 source file). See `prompts/README.md`. |
| `common/config.py` | Frozen `RunConfig(arm, lang, model, seed, size, kind, recipe_ver, data_ver, prompt_ver, id_extra, budget, notes)` with validation. `job_id()` = sha1 of the exact T6 §6.2 tuple `[kind, base, arm, n_train, seed, recipe_ver, data_ver, prompt_ver]`, with `id_extra` appended only when non-empty. `run_id()` = `<kind>-<arm>-<lang>-<model>-n<size>-s<seed>-<job_id[:10]>`. Budget and notes are not identity. Loads `.toml` (stdlib) or `.json`, and `.yaml` only if PyYAML is present. `write_resolved()` writes `config.json` into a run dir. |
| `common/jobs.py` | `JobLedger` (SQLite WAL), a pure library. `jobs` table with the T6 columns. Statuses are `PENDING, READY, SUBMITTED, RUNNING, NEEDS_RESUME, COMPLETE, FAILED, BLOCKED`, with an explicit `TRANSITIONS` map; COMPLETE is terminal. Each transition is a compare-and-set (`expect=`) inside `BEGIN IMMEDIATE`. `submit` is idempotent, `retry`, `refresh_ready` handles deps (a FAILED dep sets BLOCKED `dep_failed:`, and the job is released after the dep completes; budget-BLOCKED jobs are not auto-released). `platform` comes from `platform_for(kind, spec)`, so callers never choose it. `events` table + `ledger.events.jsonl` mirror, and a `budget` counter table. |

Behaviour worth knowing:
- **Routing guard (fail closed):** Gemini and NIM accept only `split in {"train","dev"}`. `test`, `benchmark` or an unset split raises `ForbiddenRoute` before any network call. Claude accepts any split.
- **Retries per error class:** rate_limit 8 (lane cool-down + TPM cut, another key if available); overloaded (529/503) 5, server 4, network 4, timeout 3 (exponential backoff with jitter); content (empty or blocked 200) 2; client (400/404/422) 1; auth (401/403) disables that key's lane, tries the other keys, then raises `AuthError`. API failures return `ok=False`. Budget, routing, unknown-price and auth errors raise.
- **Cache key** = sha256 of provider, model, system, messages, params (incl. `seed`, `extra`), prompt id/version/sha, `sample_idx`, `cache_tag`. `item_id` and `tag` are metadata, so identical text is one paid call. Use `sample_idx` for independent samples. Identical concurrent requests are single-flighted: one live call, the rest are served from the cache.
- **Unknown Claude model price raises** `UnknownPrice` instead of calling unpriced.

## 2. Evidence

```
$ .venv/Scripts/python.exe -m pytest tests/test_common_jobs.py tests/test_common_prompts_config.py \
      tests/test_common_ratelimit.py tests/test_common_apilog.py tests/test_common_llm.py -q -rs
..................................s..............................        [100%]
SKIPPED [1] tests\test_common_apilog.py:46: spawns 4 Python processes; opt in with VIMATH_MP_TESTS=1 (laptop RAM rule)
```
64 passed, 1 skipped. Run in the project `.venv`, which has no `anthropic`, so the lazy imports are exercised too.
Key assertions:
- 20 concurrent calls under a $0.50 cap: the refused ones never reach the API, `spent == Σ response costs ≤ cap`, and reserved returns to 0.
- Re-running the same 10 requests, in-process and from a fresh client on the same state dir, adds $0.
- A batch with cached and duplicate requests sends 3 of 5, is priced at 50 %, and collecting twice does not double-count. Batch results feed the cache.
- Gemini: 429 `RetryInfo` 37 s → cool-down and adaptive TPM engaged. The key appears only in the header, and the ledger shows `***REDACTED***` where the error echoed it.
- A test split is never sent to Gemini (`split` ∈ test/benchmark/None/TEST).
- `RunConfig.job_id() == jobs.job_id(kind, cfg.to_spec())`.
- 8 threads submitting the same job → exactly one created.
- 8 threads × 50 lines of 5 kB to one JSONL → 400 intact lines.

Real SDK wiring, checked with `E:\venv_tools` and no network:
```
$ E:/venv_tools/Scripts/python.exe -c "...ClaudeProvider('sk-ant-dummy-not-real','wrkspc_dummy').sdk..."
AsyncAnthropic wrkspc_dummy 0 anthropic-2.6.0          # workspace header set, SDK retries off
```
Lock:
```
$ uv lock            -> Resolved 50 packages ; Updated numpy -> v2.4.6, z3-solver v5.3.3.0 -> v5.1.0.0
$ uv lock --check    -> Resolved 50 packages in 1ms  (LOCK_OK)
$ uv export --format requirements.txt --all-extras --no-hashes --no-emit-project -o requirements.lock.txt
```

## 3. Environment

- `pyproject.toml` core deps are only what the package imports today: sympy, z3-solver, numpy, requests, beautifulsoup4, lxml, python-dotenv. I removed openai, pandas and tqdm because nothing in `vi_math_verified_aug/` imports them.
- Extras: `api` (anthropic, httpx2), `data` (pyarrow, huggingface-hub), `grader` (math-verify), `dev` (pytest, pytest-cov).
- `[tool.uv] constraint-dependencies` pin the libraries that shape results to the versions the `.venv` actually ran with: sympy 1.14.0, z3-solver 5.1.0.0, numpy 2.4.6, lxml 6.1.3, bs4 4.15.0. Without the pin, uv would have locked z3 5.3.3.
- **`uv.lock`** (universal, Windows + Linux) and **`requirements.lock.txt`** (pip export, all extras) are new. Install with `uv sync --locked --all-extras`, or `pip install -r requirements.lock.txt && pip install -e . --no-deps`. I did **not** run `uv sync`, because it would rewrite the shared `.venv` that other workstreams test in.
- `requirements.txt` is kept unchanged except for a header marking it **legacy** (`src/` + notebooks only). Deleting it is the user's pending decision.
- **GPU/Kaggle stack** (torch, transformers, peft, trl, unsloth, vllm, datasets) is deliberately **not** in the CPU lock. **Y1 owns** the Kaggle/Modal requirements: a Linux-x86_64 pin list or wheels dataset per T6 §1.3, plus `env.txt` per run per 06 §5.2. When it exists, link it from `pyproject.toml`'s comment.
- `.gitignore` now covers `.env*` (except `.env.example`), `logs/api/`, `logs/**/*.log`, `.cache/`, `state/`, `*.sqlite*`, `*.lock` (except `uv.lock`), and the uv/ruff/mypy caches. `data/` was already covered. `git ls-files -ci --exclude-standard` returns nothing, so no tracked file became ignored.

## 4. Migration notes for the other workstreams

1. **API calls.** Replace the ad-hoc clients (`pilot_gen/clients.py`, `llm_aug/clients.py`, `data_fix/nim_client.py`) with `LLMClient(ws="<WS>", budget_usd=..., run_id=...)`. Mapping from G1's `call(model_key, system, user, task, item, …)`: `LLMRequest(model=<id>, system=…, messages=[{"role":"user","content":user}], max_tokens=…, temperature=…, item_id=item, split="train"|"dev"|"test", tag=task, extra={"output_config":{"effort":…}})`. Sync code wraps `client.run(client.map(reqs))`.
2. **Set `split` on every request.** Gemini and NIM refuse anything else (fail closed). Test/benchmark items are Claude-only.
3. **NIM users (G1)** must pass `enable_nim=True`. Port G1's SSE streaming reader into `NimProvider.send` before heavy NIM use: the gateway 504s non-streamed calls after ~300 s.
4. **Old ledgers** (`logs/api/G1/*.jsonl`) are in G1's format and are **not** read as cache. That is intentional: re-reading them caused the over-count. To reuse G1 responses, write a one-off converter to the new record format, then run `ResponseCache.rebuild_from_logs`. Spend starts at $0 per `budget_scope`. To carry G1's spend, add one `CostTracker.charge("g1-carry-over", <usd>)`.
5. **Prompts.** Copy each G1 winner into `prompts/<id>.v1.md`: add front matter, set `status: active`, and keep the `### SYSTEM`/`### USER` + `<<name>>` syntax. Pass `RenderedPrompt` through `LLMRequest.from_prompt` so that id, version and sha reach the ledger. Never edit a used version; bump it. English-mirror prompts get their own ids (e.g. `validity_judge_en`).
6. **Runs and jobs (S2/orchestrator).** Build ids with `RunConfig(...).run_id()` / `.job_id()`. `infra/jobqueue.py` can move onto `JobLedger` (`submit(kind, cfg.to_spec(), deps=…)`, `transition(..., expect=...)`, `refresh_ready()`, `ready()`), with the scheduler loop kept outside the library.
7. **Defaults.** `VIMATH_API_LOG_DIR`, `VIMATH_CACHE_DIR`, `VIMATH_STATE_DIR` and `VIMATH_PROMPTS_DIR` relocate everything (e.g. on Kaggle). Prices can be overridden with `prices=` or `VIMATH_PRICES_JSON`.

## 5. Open items / caveats

- **Gemini lanes are per key**, as the brief specified. T6 §4.3 quotes Google: limits are "per project, not per API key", and multiplying projects counts as circumvention. If several keys share a project, the adaptive TPM will only partly discover the shared limit. The user should confirm one key per project, or lower `LaneConfig.rpm`.
- **Claude prices** are G1's recorded list prices (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10, Opus 5.5 $4/$20 per M tokens). Re-check them against the pricing page before a large run.
- The cross-**process** JSONL test is opt-in (`VIMATH_MP_TESTS=1`) and was not run, under the one-Python-process RAM rule. The OS lock path is the same one the 8-thread test exercises, but a multi-process run is still owed on a machine with spare RAM.
- Budget reservations are per process. Persisted spend is shared, but two processes can each reserve up to the remaining cap at the same moment, so the worst-case overshoot is one in-flight call per extra process.
- The prompt `inputs` contracts are drafts. G1 decides the final ones.
