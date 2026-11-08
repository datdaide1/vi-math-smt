"""G4: second iteration of the symbolic arm (arm Y), paired with G3 on the same 60 seeds / 180 slots.

    common     LLM client (ws G4, $ cap), prompts, stores under data/pilot/symbolic_v2/
    localize   IR-level localization (USD -> đồng rescale, re-executed), localization cards, IR view for R2
    roles      operator fixes: M6 dropped; deterministic story-role filters for M3/M4/M5 + Haiku story check
    pipeline   mutate (fixed ops) -> E (edit only, G3 editor.v3) -> R1 (edit + fluency) / R2 (free render from IR)
               -> tag check -> cycle check (gemini-3.1-flash-lite, blind) -> gates
    judge      Opus 5.5 (validity_judge.v2) + Sonnet 5.5 (validity_judge.v3) + Opus blind solve, via Message Batches
    report     tables for docs/research_2026-11/exec/G4_symbolic_v2.md
"""
