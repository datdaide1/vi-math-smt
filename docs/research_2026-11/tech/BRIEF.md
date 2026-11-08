# Technical research brief (shared by agents T1–T6), 2026-11-07

**Project.** Vietnamese math reasoning for small open LLMs (≤~4B). Target: a Q2-level paper + a complete
internal hand-off package (data, models, code, prompts). Scope must NOT be cut. Free compute only: laptop
(RTX 3050 Ti 4 GB, 7.7 GB RAM), Modal $30/month credit, Kaggle (T4×2 / P100, now allowed), Colab free,
NVIDIA NIM free API (40 RPM), $100 Claude API. No grants, no lab GPU. The user wants the whole thing done
as fast as possible ("ideally 1 day") at full capacity, without cutting scope.

**Working research question (may be challenged):** for SFT/RL of small models in Vietnamese, how do
translated data, natively generated Vietnamese data, and symbolic/program-verified generated data compare —
and does the ranking change on native Vietnamese exam problems (new benchmark Vi-ExamMath: grade-10 entrance +
HSG exams, temporal split) vs translated benchmarks (Vi-GSM8K, Vi-MATH)?

**What failed before (measured, see content_review/R3–R5):** the old pipeline formalized GSM8K/MATH into
SMT-LIB, then "mutated" by wrapping the ANSWER expression in random arithmetic ((a+k)·k, a+2h, a·(a+1)), then
had an LLM verbalize it. Result: 0% genuinely new problems, 84% "old story + bolted-on arithmetic tail"
("Lan có 2 kẹo, cho anh 1 cái và cộng thêm 0,5"), 15.5% real-world nonsense, 58% leak variable names/answer.
SMT programs were mostly straight-line arithmetic (Z3 used as a calculator); only 42% (GSM8K) / 17% (MATH)
expose story quantities as named variables. "Verified" certified a number, not a valid problem.

**Rules for every agent.**
1. Do NOT be bound by the old plan or prior memory — research what is newest and best (2025–2026, up to
   today). Prefer methods with reported numbers, ideally at ≤4B scale or low-resource languages.
2. Every claim needs evidence (arXiv id/URL opened, docs page, or a command you ran). Mark anything not
   opened as [UNVERIFIED]. Use WebSearch (load via ToolSearch "select:WebSearch") and the web-fetch agent.
3. Output TECHNICAL content, not process: concrete algorithms, data structures, objective functions,
   pseudo-code, hyperparameters, decision rules, thresholds — each compared against ≥2 alternatives in a
   table, with ONE recommended choice and its justification (evidence + expected effect size + cost).
4. Say how it fits the free compute, and what it costs in GPU-hours / API calls.
5. Write ONLY your own report file in `docs/research_2026-11/tech/`. Do not edit other project files.
   Scratch files go in the session scratchpad. Final message: ≤250-word summary.
