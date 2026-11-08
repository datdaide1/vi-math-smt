"""G5: arm Y v3 + scale (VI first, then the EN mirror from the same IR).

    common     ws G5 client ($11.9 cap), call_many (direct or Message Batches), stores
    ops        v3 operators: M6 dropped; M3/M4 constrained at mutation time (entity role, causal direction); pi_v3
    irview_en  English blueprint of a program for the EN renderer
    pipeline   seeds -> formalize -> mutate (+ story check) -> render (VI R1b / R2b, EN) -> cycle (blind) -> gates
               -> validity gate (Haiku validity_judge.v3) -> export
    calib      validity-gate calibration on G4's 358 Opus/Sonnet-labelled items
    evalrun    Opus judge + blind solve on a run (Batch)
"""
