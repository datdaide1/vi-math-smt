# Datasheets (Gebru et al., "Datasheets for Datasets", CACM 2021)

One datasheet per dataset: motivation, composition, collection, preprocessing, uses, distribution, maintenance.
Final numbers and hashes as of 2026-11-08 (every `SHA256SUMS` re-verified on the laptop at 08:40).

| Datasheet | Dataset | Path | Version / key hash |
|---|---|---|---|
| [vi_exam_core_v1.md](vi_exam_core_v1.md) | Vi-ExamMath Core (C1 benchmark) | `data/vi_exam_qa/core_v1.1.jsonl` (+ `data/vi_exam/`) | **core_v1.1**, 1,545 items (1,367 VERIFIED), `ceac95de…` |
| [eval_sets_v1.1.md](eval_sets_v1.1.md) | Frozen eval sets (VI + EN parallel) | `data/eval/` | **v1.1**, `SHA256SUMS` 18/18 |
| [seed_pool_v1.md](seed_pool_v1.md) | Seed pool + dev set + S0 anchor sets | `data/seeds/`, `data/t1r/` | **pool_v1**, 9,649 seeds, `1b565411…` |
| [train_pools.md](train_pools.md) | Arms Y± / N± × VI / EN: pools, final matched sets (n = 1,808), secondary sets | `data/train/{Y,N}/`, `data/train/final*/` | `select_v1` (07:03) |

## Provenance of Claude outputs (applies to every dataset that contains Claude-written text)

- **What is Claude-written:** all Y and N training rows (problems, renders, solutions; Haiku 5.5), the screens/judges
  (Haiku 5.5, Opus 5.5 audits), 253 MTA-vi re-translated seeds (Haiku/Sonnet 5.5), Vi-ExamMath keys and adjudications
  (Opus 5.5), 9 re-translated eval solutions and the eval-set review decisions (Opus 5.5).
- **Model ids** (requested and returned, from the ledgers): `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`.
  Per-workstream breakdown: `handoff/LEDGER.md` §1–2. Prompts with sha256: `handoff/PROMPTS.md`.
- **API logs:** `logs/api/<WS>/claude.jsonl` (one line per call: ts, model requested/returned, params, prompt
  id/version/sha256, full request and response, usage, cost, request hash; keys never logged). Batch ledgers in each
  workstream's data dir. These logs go in the **private** bundle (they contain benchmark items sent to Claude).
- **Disclosure:** the Y and N training sets, and every model card trained on them, state that the text was generated
  by Claude models; this provenance is kept in every release (KNOWN_ISSUES D-1).

## Other providers
- Gemini flash-lite (free tier): solve-checks and screens on **training** seeds/items only; no Gemini text is training
  data in the final sets (the legacy Vietnamese translations of GSM8K/MATH are attributed to Gemini 2.5 Flash in
  06 §4.3 — UNVERIFIED).
- Qwen3-32B-AWQ (self-hosted on Modal): Y cycle-check verdicts only.
- NVIDIA NIM: G1 pilot and G5 cycle-model validation only; no NIM text in any frozen dataset.

Licensing background for the public datasets and exam sources: [`06_packaging.md`](../../docs/research_2026-11/06_packaging.md) §4
(not legal advice; points marked UNVERIFIED there are still unverified).
