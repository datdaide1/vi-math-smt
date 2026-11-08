"""G3: end-to-end pilot of the symbolic arm (arm Y) on 60 new Vi-GSM8K train dev seeds.

    seeds      60 seeds (Random(20261108)) from the 5% dev pool minus R1-corrupt / G1 / G2 seeds; J=3 slots each
    common     LLM client (ws G3, $ cap), prompt rendering, resumable JSONL stores under data/pilot/symbolic/
    editor     the editor-facing view of an X4 edit list (tagged text, numbered sentences, M6 <ref>) + tag check
    gates      leak regex, answer-in-text, chrF, 5-gram Jaccard dedup
    pipeline   stages 1-5: formalize (Haiku) -> mutate (X4) -> render (M1 deterministic, M2-M6 Haiku editor)
               -> cycle check (blind gemini-3.1-flash-lite back-formalization) -> common gates
    judge      stage 6: Opus 5.5 judge + blind solve; second judge for kappa
    report     tables for docs/research_2026-11/exec/G3_symbolic_pilot.md
"""
