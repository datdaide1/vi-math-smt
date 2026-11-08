"""X3 frozen evaluation sets (Vietnamese + English mirror).

Pipeline (run from the repo root, in this order):

1. ``fetch``         pinned HF sources -> ``data/eval/_sources/`` (+ manifest with revision and sha256)
2. ``build_base``    Vi-GSM8K-Platinum (1,209) and Vi-MATH alg/prealg (2,058) + MATH-500 subset (206), deterministic
                     fixes only, English golds attached and alignment-checked -> ``data/eval/_work/``
3. ``review_queue``  translation-defect queue (R1/R2 patterns, number-multiset EN<->VI, glossary hits)
4. ``opus_fix``      Claude Opus proposal + self-solve + independent judge (Batch API) -> ``data/eval/_work/opus/``
5. ``freeze``        apply auto-accepted fixes, draw the 1,000-item primary sample, write VI + EN sets with
                     datacards and ``SHA256SUMS``

Design: ``docs/research_2026-11/tech/T5_evaluation.md`` §2.1, ``exec/E1_english_mirror.md`` §5.6,
``04_data_eval.md`` §2.2-2.3, content reviews R1/R2.  Report: ``docs/research_2026-11/exec/X3_eval_sets.md``.
Written from scratch; ``data/clean/`` and ``data_fix/`` (earlier stopped agent) are not imported.
"""
