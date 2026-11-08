# vi_math_verified_aug

Implementation package for the thesis *"Does Symbolic Verification Still Earn Its
Keep?"* — a controlled comparison of symbolic-verified vs. LLM-based math-data
augmentation for small models in Vietnamese, plus **Vi-ExamMath**, a
contamination-controlled Vietnamese exam benchmark.

- **Why / what:** [`../RESEARCH_PLAN.md`](../RESEARCH_PLAN.md)
- **How (task backlog):** [`../EXECUTION.md`](../EXECUTION.md)
- **Provenance of every reported number:** [`../RESULTS_PROVENANCE.md`](../RESULTS_PROVENANCE.md)

The legacy pipeline stays under [`../src/`](../src/) and is not imported here; a
few pieces are ported deliberately (GSM8K SMT formalizer, Vietnamese translations).

## Setup

```bash
py -3.11 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[dev]"
python -c "import vi_math_verified_aug; print(vi_math_verified_aug.__version__)"
pytest
```

Put a free `NVIDIA_API_KEY` in `.env` (see `.env.example`) — it is the only key
required (NIM hosts the generator). The heavy ML stack (Unsloth / vLLM / torch)
installs separately, per `RESEARCH_PLAN.md` §6.3.

## Layout

| path | role | task |
|---|---|---|
| `eval/math_verify.py` | the single accuracy definition | T0.2 |
| `eval/run_eval.py` | config-driven eval harness | T0.3 |
| `verify/uniqueness.py` | the uniqueness gate + N2 audit (a C3 result) | T1.1 |
| `verify/round_trip.py` | round-trip gate for S1 | T1.5 |
| `symbolic_aug/` | S1 operators M0 / O1 / O2 | T1.2, T1.6 |
| `informalize/` | symbolic → Vietnamese + leakage filter | T1.3, T1.4 |
| `llm_aug/` | S2/S3 generation | T0.6, T2.3 |
| `vi_exam/` | Vi-ExamMath: scrape → segment → verify | T1.8, T2.9, T2.10 |
| `scripts/` | SFT, per-arm builders, grid runner | T0.4, Sprint 2/4 |
| `modal_app.py` | L4 SFT for the 1.7B runs only | T0.5 |
