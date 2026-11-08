"""S3 — the clean, frozen seed pool shared by every generation arm (T1 §2.0).

Pool = Vi-GSM8K train (7,473) ∪ Vi-MATH train algebra (1,744) ∪ prealgebra (1,205), each row aligned to its English
original (parallel ``en`` fields for the English mirror E1). ``seed_id = sha1(f"{source}||{index}")[:12]`` with
``source`` in ``vi_gsm8k_train | vi_math_train_algebra | vi_math_train_prealgebra`` (the G1/G2/G3 pilots' definition
for GSM8K); DEV = ``int(seed_id, 16) % 20 == 0`` (5 % by hash; pilot and calibration only).

Stages (``python -m vi_math_verified_aug.seeds.build <stage>``; every stage is resumable and writes data/seeds/_work/):

    align       load + align VI/EN, seed ids, DEV flag                         -> _work/aligned.jsonl
    detect      deterministic repairs + defect triggers (R1/R2 patterns)       -> _work/detect.jsonl
    solvecheck  blind solve of every VI seed (Gemini flash-lite, EN control)   -> _work/solve.jsonl
    mta         MTA-vi re-translation of damaged rows (Haiku -> Sonnet)        -> _work/mta.jsonl
    gold        typed gold (grader 3.0.2 infer_gold) + EN/VI agreement         -> _work/gold.jsonl
    decontam    X5 G4 gate + review rule vs data/eval/* and Vi-ExamMath        -> _work/decontam.jsonl
    translationese  Opus naturalness / teacher-acceptability on 60 seeds       -> _work/translationese.jsonl
    freeze      pool_v1.jsonl, dev_v1.jsonl, datacard, changelog, SHA256SUMS

Report: docs/research_2026-11/exec/S3_seed_pool.md.
"""
