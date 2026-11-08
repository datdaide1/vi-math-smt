"""vi_math_verified_aug — symbolic vs. LLM math-data augmentation for small models,
with a contamination-controlled Vietnamese exam benchmark (Vi-ExamMath).

See ``RESEARCH_PLAN.md`` for the design and ``EXECUTION.md`` for the task backlog.

Subpackages
-----------
ir            lightweight typed-quantity / operation-DAG IR (S1 operators)
symbolic_aug  S1 operators: constrained constant substitution, structural (O1/O2)
llm_aug       S2/S3 generation clients + drivers
informalize   symbolic -> natural Vietnamese, with a leakage filter
vi_exam       Vi-ExamMath build: scrape -> segment -> solver-verify
verify        Z3 helpers, the uniqueness gate, the round-trip gate
eval          math_verify, the eval harness, the data-quality panel, decontamination
scripts       SFT + per-arm dataset builders + the training-grid runner
"""

__version__ = "0.0.1"
