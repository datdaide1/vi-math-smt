"""common — the shared runtime layer every workstream migrates onto (Y3).

Modules
-------
paths      repo-root-relative default locations (logs, cache, state, prompts); env overrides
hashing    canonical JSON + stable hashes (request hash, run_id, job_id)
env        .env reading without leaking secrets
apilog     append-only JSONL API ledger, safe under threads and processes (Windows too)
cache      content-addressed response cache (SQLite, WAL)
costs      price table, concurrency-safe cost tracker with a hard budget cap
ratelimit  per-(provider, key, model) lanes: token-bucket RPM + adaptive token-rate (TPM) backoff
llm        the one async LLM client: Claude (SDK + Message Batches), Gemini (REST key pool), NIM (off)
prompts    versioned prompt registry loader (top-level ``prompts/``)
config     typed run configs + deterministic run_id
jobs       minimal SQLite job ledger (T6 §6.2): idempotent ids, checked status transitions

Nothing here imports a provider SDK at import time: ``anthropic`` / ``httpx2`` load lazily, so the
module tree imports (and the tests run) in the plain project ``.venv``.
"""
