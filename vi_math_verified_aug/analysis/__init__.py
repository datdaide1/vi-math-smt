"""Statistical analysis pipeline (workstream Y2; T5 §2–§3, E1 §5.5).  Report: docs/research_2026-11/exec/Y2_analysis.md

Modules
-------
schema        input JSONL schema -> dense run x item table (EvalData)
contrasts     declarative contrasts + the pre-registered library (H1 P2 lang x verif, H2 Y vs N, H3 verif, H4 P3)
bootstrap     two-way cluster bootstrap (items within dataset x seeds within lang x arm x model), vectorised
multiplicity  Holm, BH-FDR, TOST, the decision rule (GLMM and bootstrap must agree)
intervals     Wilson / Clopper-Pearson / Beta intervals, item-bootstrap slice CIs, exact McNemar
glmm          cell-means GLMM: R export (glmm_lme4.R, lme4 on a Kaggle/Modal CPU), statsmodels VB stand-in
irt           2PL MAP (numpy), item flags, Core/Hard split, temporal-split difficulty tools
variance      seed-variance (sigma_run, sigma_run x eval) moment estimator for the pilot rule
power         T5 App. A and E1 App. A power simulations + pilot rule
synth         synthetic predictions with planted effects (tests, dry runs)
run           CLI: analyze / export / synth
modal_jobs    off-laptop jobs (heavy tests, coverage simulation, lme4 and VB checks) on Modal CPU

Only numpy + pandas are needed locally; scipy/statsmodels/R are used only inside the Modal image.
"""
