"""Vi-ExamMath collection entry points (X2, 2026-11). This module only documents the pipeline order.

  1. ``python -m vi_math_verified_aug.vi_exam.discover``        listings + monthly sitemaps -> exam_index.jsonl
  2. ``python -m vi_math_verified_aug.vi_exam.collect all``     exam pages + Wayback CDX     -> raw/, first_seen.jsonl
  3. ``python -m vi_math_verified_aug.vi_exam.build``           offline parse/segment/tag/keys -> items*.jsonl
  4. ``python -m vi_math_verified_aug.vi_exam.claude_key …``    blind Claude key (Batch)      -> claude_keys.jsonl
     then re-run step 3 to merge the keys.

Modules: ``crawl`` (polite cached fetcher), ``page`` (DOM -> blocks), ``segment`` (question tree + solution
alignment), ``tagging``, ``answers`` (provisional key), ``symkey`` (SymPy key + answer comparison).
"""
