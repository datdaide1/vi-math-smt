"""C3 literature and in-repo baselines for error rates of synthetic math data (D5, fetched 2026-11-07).

Every row quotes the number exactly as the source reports it (``quote``) and says how it maps to our error classes.
``comparable`` = how directly the number can be set next to a G3/G4 rate:
  "operator-matched"  same kind of transformation, human-checked validity of the transformed question
  "pipeline"          a whole-pipeline invalid/filter rate, different rubric
  "filter-proxy"      share removed by an automatic filter (not a measured error rate)
  "not reported"      the source gives no number (recorded so the gap is explicit)
"""
from __future__ import annotations

LIT: list[dict] = [
    # ------------------------------------------------------------------ operator-matched (GSM-Plus, human-checked)
    {"id": "GSM-Plus", "cite": "Li et al., GSM-Plus, ACL 2024, arXiv:2402.19255 (v2), §3.2 + App. B.2 Table 7",
     "url": "https://arxiv.org/html/2402.19255v2", "generator": "GPT-4", "checker": "human annotators (5, IAA α=0.567)",
     "comparable": "operator-matched",
     "rows": [
         {"what": "variations revised by humans (all 8 types)", "value": 18.85, "unit": "% revised",
          "quote": "human annotators revised 18.85% of the variations produced by GPT-4", "ours": "≈ not-valid, all ops"},
         {"what": "Numerical Substitution pass rate", "value": 91.51, "unit": "% pass", "ours": "M1 value resample"},
         {"what": "Adding Operation pass rate", "value": 75.75, "unit": "% pass", "ours": "M3 insert step"},
         {"what": "Reversing Operation pass rate", "value": 29.76, "unit": "% pass", "ours": "M2 inverse (FOBAR-style)"},
         {"what": "Integer-decimal-fraction Conversion pass rate", "value": 84.24, "unit": "% pass", "ours": "—"},
         {"what": "Distractor Insertion pass rate", "value": 88.25, "unit": "% pass", "ours": "—"},
         {"what": "Problem Understanding (paraphrase) pass rate", "value": 97.49, "unit": "% pass", "ours": "—"},
     ]},
    # ------------------------------------------------------------------ cycle-consistency (MathCAMPS)
    {"id": "MathCAMPS", "cite": "Mishra, Poesia, Goodman, arXiv:2407.00900 v3 (COLM 2025), App. F",
     "url": "https://arxiv.org/html/2407.00900v3", "generator": "GPT-4 (symbolic structure -> word problem)",
     "checker": "authors' manual check of 245 random problems", "comparable": "operator-matched",
     "rows": [
         {"what": "unfaithful to the symbolic structure before the cycle check", "value": 12.2, "unit": "% (30/245)",
          "quote": "we identified 30 word problems (12.2%) that were not faithful to the original symbolic structure",
          "ours": "answer-unfaithful (Opus blind solve != program), pre-cycle"},
         {"what": "faithful among cycle-consistent survivors", "value": 97.7, "unit": "% (210/215)",
          "quote": "Out of the remaining 215 problems, 210 (97.7%) were judged as faithful",
          "ours": "Opus agrees with program on cycle-passed items"},
     ]},
    # ------------------------------------------------------------------ teacher-style validity of generated MWPs
    {"id": "MATHWELL", "cite": "Christ, Kropko, Hartvigsen, arXiv:2402.15861 v5 (2024), Table 3",
     "url": "https://arxiv.org/html/2402.15861v5", "generator": "GPT-4 Turbo / GPT-3.5 / Llama-2 / MATHWELL",
     "checker": "expert annotators, 250 generations per model", "comparable": "pipeline",
     "rows": [
         {"what": "GPT-4 Turbo meets all criteria (solvable ∧ accurate ∧ appropriate)", "value": 78.8, "unit": "% MaC",
          "ours": "≈ teacher-acceptable (arm N, no seed)"},
         {"what": "GPT-4 Turbo solvable", "value": 94.8, "unit": "%", "ours": "1 − unanswerable"},
         {"what": "MATHWELL (Llama-2-70B fine-tuned) meets all criteria", "value": 74.8, "unit": "% MaC", "ours": "≈ teacher"},
         {"what": "Llama-2 meets all criteria", "value": 62.4, "unit": "% MaC", "ours": "≈ teacher"},
     ]},
    {"id": "ElemMWP", "cite": "Ariyarathne et al., arXiv:2506.05950 v2 (Mar 2026), Tables 5/10",
     "url": "https://arxiv.org/html/2506.05950v2", "generator": "Llama-2-7B-chat, Zephyr-7B",
     "checker": "human reviewers (~520 MWPs per round)", "comparable": "pipeline",
     "rows": [
         {"what": "unsolvable, Llama-2 zero-shot", "value": 6.92, "unit": "% (100 − Solvable)", "ours": "unanswerable"},
         {"what": "unrealistic, Llama-2 zero-shot", "value": 4.62, "unit": "%", "ours": "absurd"},
         {"what": "unsolvable, fine-tuned Llama-2 5-shot + solvability", "value": 15.83, "unit": "%", "ours": "unanswerable"},
     ]},
    # ------------------------------------------------------------------ question-validity audits of synthetic corpora
    {"id": "MathQ-Verify", "cite": "Shen et al., Let's Verify Math Questions Step by Step, arXiv:2505.13903 v2, §3.1–3.2",
     "url": "https://arxiv.org/html/2505.13903v2", "generator": "NuminaMath-style synthesis, 5 generator models",
     "checker": "≥2 domain experts per question", "comparable": "pipeline",
     "rows": [
         {"what": "GPT-4o raw synthetic questions incorrect (authors' preliminary experiment)", "value": 30.0,
          "unit": "% (upper bound, 'fewer than 30%')", "quote": "(with fewer than 30% incorrect samples)",
          "ours": "not-valid (contradiction/completeness ≈ absurd+unanswerable)"},
         {"what": "ValiMath composition (curated to ~1:3, NOT a population rate)", "value": 39.5,
          "unit": "% (848/2,147)", "ours": "—"},
     ]},
    {"id": "RV-Syn", "cite": "RV-Syn, arXiv:2504.20426, §4.5 Table 4 (GPT-4o judge, 1,000 samples per corpus)",
     "url": "https://arxiv.org/html/2504.20426", "generator": "various corpora", "checker": "GPT-4o (not human)",
     "comparable": "pipeline",
     "rows": [
         {"what": "problem error rate, ScaleQuest", "value": 5.6, "unit": "%", "ours": "not-valid (problem)"},
         {"what": "problem error rate, NuminaMath", "value": 0.6, "unit": "%", "ours": "not-valid (problem)"},
         {"what": "problem error rate, Jiuzhang3.0", "value": 2.4, "unit": "%", "ours": "not-valid (problem)"},
         {"what": "solution error rate, Mammoth2", "value": 10.6, "unit": "%", "ours": "answer wrong"},
         {"what": "solution error rate, RV-Syn (verifiable)", "value": 1.4, "unit": "%", "ours": "answer wrong"},
     ]},
    # ------------------------------------------------------------------ filter proxies and tolerance
    {"id": "CoT-Self-Instruct", "cite": "Yu et al., CoT-Self-Instruct, arXiv:2507.23751, Table 1",
     "url": "https://arxiv.org/html/2507.23751", "generator": "Qwen3/Llama synthetic prompts", "checker": "Answer-Consistency filter",
     "comparable": "filter-proxy",
     "rows": [
         {"what": "verifiable-reasoning prompts kept by Answer-Consistency (2,926/5,000)", "value": 58.5, "unit": "% kept",
          "quote": "not all generated synthetic instructions are well-defined and answerable", "ours": "N-arm self-consistency yield"},
     ]},
    {"id": "OpenMathInstruct-2", "cite": "Toshniwal et al., arXiv:2410.01560, §1 and §2.2.3",
     "url": "https://arxiv.org/html/2410.01560", "generator": "Llama3.1-405B", "checker": "automatic filters + 20-example manual check",
     "comparable": "filter-proxy",
     "rows": [
         {"what": "data flagged by low-quality filters", "value": 6.0, "unit": "% to 12%",
          "quote": "The proportion of data filtered by the different methods ranges from 6% to 12%", "ours": "—"},
         {"what": "SFT tolerance to incorrect solutions (no degradation at ≥256K)", "value": 20.0, "unit": "% incorrect",
          "quote": "robust to the presence of up-to 20% low-quality data", "ours": "context for gate thresholds"},
     ]},
    # ------------------------------------------------------------------ not reported (explicit gaps)
    {"id": "Yeo-2602.19187", "cite": "Yeo et al., Adaptive Problem Generation via Symbolic Representations, arXiv:2602.19187, §3.5, §4.4",
     "url": "https://arxiv.org/html/2602.19187", "generator": "LLM edits SymPy/SMT code vs NL", "checker": "GPT-5-mini filter",
     "comparable": "not reported",
     "rows": [
         {"what": "valid-problem rate Sym vs NL", "value": None, "unit": "not reported",
          "quote": "Both pipelines produce similar rates of valid problems", "ours": "—"},
         {"what": "SMT renderings", "value": None, "unit": "qualitative only",
          "quote": "rendering SMT problems into natural language often results in unnatural-sounding word problems",
          "ours": "fluency"},
         {"what": "diversity, mean pairwise cosine distance Opt-Sym vs Opt-NL", "value": 0.37, "unit": "vs 0.07", "ours": "near-dup"},
     ]},
    {"id": "GSM-Symbolic", "cite": "Mirzadeh et al., arXiv:2410.05229, §3.1", "url": "https://arxiv.org/html/2410.05229",
     "generator": "hand-written templates", "checker": "10 samples/template manual + ≥2 models answer correctly",
     "comparable": "not reported",
     "rows": [{"what": "template/instance error rate", "value": None, "unit": "not reported",
               "quote": "10 random samples per template are reviewed manually", "ours": "—"}]},
    {"id": "MetaMath", "cite": "Yu et al., MetaMath, arXiv:2309.12284 (ICLR 2024)", "url": "https://arxiv.org/abs/2309.12284",
     "generator": "GPT-3.5 (AnsAug, Rephrasing, SV, FOBAR)", "checker": "answer filter only", "comparable": "not reported",
     "rows": [{"what": "invalid-question rate of SV/FOBAR", "value": None, "unit": "not reported in the paper", "ours": "—"}]},
]

# In-repo, measured by reading (PROTOCOL.md), same taxonomy as C3 -> directly comparable labels, different judge (human-
# style reading by an agent, not the Opus v2 judge).
INREPO: list[dict] = [
    {"id": "R5 legacy pipeline (answer-expression mutation + informalize, gpt-oss-120b)",
     "src": "docs/research_2026-11/content_review/R5_augmented.md §0 (n=200 read)",
     "rates": {"tail (NL-TACKED)": "84.0% [78.3–88.4]", "leak": "58.0% [51.1–64.6]", "absurd": "15.5% [11.1–21.2]",
               "near-dup (identity op)": "13.0% [9.0–18.4]", "teacher would write": "10.5% [7.0–15.5]",
               "genuinely new": "0.0% [0.0–1.9]"}},
    {"id": "X4 legacy operators in the new IR (structural)",
     "src": "tests/test_symbolic_rules.py::test_legacy_operators_are_answer_tails_on_every_fixture (re-run by D5: 12 passed)",
     "rates": {"rejected as answer tail": "1,008/1,008 (6 ops × {plain, in-place} × 28 fixtures × 3 constants)"}},
    {"id": "MetaMathQA (EN source errors, seen through MetaMathQA-vi)",
     "src": "docs/research_2026-11/content_review/R6_other_sets.md §3 (n=150 read)",
     "rates": {"problems inherited from the EN source": "10.0% (15/150)", "translation major+critical": "18.7% [13.2–25.7]",
               "NL-UNANSW": "1.3%", "SOL-WRONG": "3.3%"}},
]
