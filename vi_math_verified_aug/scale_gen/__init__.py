"""D1: arm N (direct LLM generation, CoT-Self-Instruct style with a localization card) at scale, VI + EN.

Stages (``python -m vi_math_verified_aug.scale_gen.run <stage> ...``):
    slots      data/train/slots_v1.jsonl from S3's frozen data/seeds/pool_v1.jsonl (shared with arm Y / G4)
    gen        Haiku 5.5 generation via the Message Batches API (resumable, chunked, budget-guarded)
    solve      Haiku 5.5 blind solve (solution_writer[_en].v2) -> answer agreement (V1-style screen)
    lite       gemini flash-lite packed validity/absurd screen (validity_lite.v1), training items only
    audit      Opus 5.5 (validity_judge.v2) on an audit sample + cascade disagreements (Batch)
    decontam   13-gram (exact / no-diacritics / templated) locally + bge-m3 cosine on Modal vs data/eval + Vi-ExamMath
    select     N- / N+ pools, size-matched 10k subsamples, SHA256SUMS, manifest.json
    report     tables for exec/D1_N_scale.md
"""
