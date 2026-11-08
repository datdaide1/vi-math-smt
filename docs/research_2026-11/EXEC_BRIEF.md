# Execution brief — parallel workstreams (2026-11-07)

Read first: `docs/research_2026-11/00_SOLUTION.md`, then the tech reports your workstream cites
(`tech/T1..T6`). Goal: Q2 paper + internal hand-off; scope is NOT cut; quality first; free compute.

## Ownership (never edit files owned by another workstream)
| WS | Owns (create/edit only here) |
|---|---|
| W1 grader | `vi_math_verified_aug/eval/math_verify.py`, `tests/test_math_verify.py` (+ new `tests/grader_cases/`) |
| W2 infra + student pilot | `vi_math_verified_aug/infra/` (new), `jobs/` (new), `docs/research_2026-11/exec/W2_*.md` |
| W3 eval/seed repair | `vi_math_verified_aug/data_fix/` (new), `data/clean/` (new), `docs/research_2026-11/exec/W3_*.md` |
| W4 benchmark | `vi_math_verified_aug/vi_exam/`, `data/vi_exam/`, `docs/research_2026-11/exec/W4_*.md` |
| W5 generation pilot | `vi_math_verified_aug/llm_aug/`, `vi_math_verified_aug/symbolic/` (new), `data/pilot_gen/`, `docs/research_2026-11/exec/W5_*.md` |
Shared read-only: `data/` inputs, `.env`. Never print secrets. Do not git commit (the lead commits).

## Shared rules
- **NIM: ONE working key** (`NVIDIA_API_KEY` in `.env`; old keys are dead). Account limit 40 RPM total.
  Budget: **W5 ≤ 24 RPM, W3 ≤ 10 RPM, W4 ≤ 4 RPM**, others 0. Use a token bucket in your process.
  Do NOT send Vi-ExamMath (benchmark) items or Vi-GSM8K/Vi-MATH **test** items to NIM or Gemini.
- Callable NIM models (verified): `deepseek-ai/deepseek-v4.1-flash`, `nvidia/nemotron-3-ultra-550b-a55b`,
  `z-ai/glm-5.3-flash`, `google/gemma-4-31b-it`, `nvidia/nemotron-3.5-bolt-e4b`.
- Log EVERY API call as JSONL (ts, provider, model, params, request, raw response, latency) under
  `logs/api/<ws>/`. Prompts live as versioned files, never inline-only.
- Kaggle CLI works (`kaggle` 2.2.5 + `six`; token at `~/.kaggle/access_token`; account `datuni`;
  GPU quota 30 h/week, refresh 2026-11-14). Kaggle notebooks: **private** only. Only W2 pushes kernels.
- Modal: workspace `tran-hoang-dat-2312`; rates T4 $0.59/h, L4 $0.80/h. Only W2 touches Modal.
- Python: project `.venv` (3.11) for CPU work; install extra packages only into the scratchpad venv or a
  new env you document. Use `python -I` when reading downloaded files.
- Every claim/result in your report needs the command + output or file reference.
- Final message ≤250 words: what was built, what was measured, what is blocked.
