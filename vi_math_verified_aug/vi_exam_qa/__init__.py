"""Q1: Vi-ExamMath answer-key QA (LLM judges replace the human raters of T5 §3.1 stages 4–5).

Pipeline (report: ``docs/research_2026-11/exec/Q1_benchmark_qa.md``):

1. ``keys prepare``      (.venv, grader 3.0.2)  re-type every existing key, re-check X2's agreements with the grader,
                                                pick the items that need adjudication and the audit sample.
2. ``llm run solve2``     (E:\\venv_tools)       second independent blind Opus solve, effort high (Batches API).
3. ``keys grade-solve2``  (.venv)               grade solve #2 against the other keys.
4. ``llm run adjudicate`` (E:\\venv_tools)       Opus adjudicator → typed gold, validity, confidence.
5. ``llm run audit``      (E:\\venv_tools)       150-item random audit, fresh blind re-solve.
6. ``finalize``           (.venv)               statuses, audit comparison, contamination flags → core_v1.jsonl.
   ``llm run audit_judge``                       LLM judge for audit mismatches, then ``finalize`` again.

Inputs are read-only (X2 owns ``data/vi_exam/``, X5 owns ``data/contam/``). Outputs: ``data/vi_exam_qa/``.
Test items go to Claude only (``split="benchmark"`` on every request; the client refuses Gemini/NIM for it).
"""
