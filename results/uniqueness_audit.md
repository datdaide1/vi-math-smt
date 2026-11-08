# Uniqueness-gate audit — base SMT formalizations (T1.1, a C3 result)

Source: `data\finetune\mine\train_vietnamese_final.jsonl` · n = **14870** · date 2026-09-09

`classify(smt)` drops the answer-defining assert, asks Z3 whether the remaining relational constraints still pin a unique value, and flags the answer being written in as a literal / irrational.

## Overall

| bucket | n | % | meaning |
|---|---:|---:|---|
| **derived** (≥1 relational, unique) | 10607 | 71.3 | a non-vacuous computation from the pinned inputs |
| &nbsp;&nbsp;└ **derived_strict** (≥2 relational) | 6270 | 42.2 | N2's definition — "genuinely derive it through ≥2 relational constraints" |
| **hardcoded** | 1401 | 9.4 | answer value / an irrational written into the program |
| **unknown** | 2862 | 19.2 | unsat / timeout / parse error / under-determined / no relational step |

**N2 ballpark reproduced:** derived_strict **42.2%** vs. the audit's "~40% genuinely derive"; hardcoded **9.4%** vs. "15.9%" (this classifier is precision-first — the π/√ decimal-approximation cases, ~another 3–5%, are counted here as `hardcoded:messy_constant` only when the irrational is a leaf pin; borderline symbolic-answer cases land in `unknown`).

## By subject (derived% / hardcoded%)

| subject | n | derived % | hardcoded % |
|---|---:|---:|---:|
| algebra | 1742 | 67.1 | 3.7 |
| arithmetic | 7473 | 73.4 | 7.6 |
| counting_and_probability | 768 | 89.1 | 3.3 |
| geometry | 832 | 62.0 | 17.2 |
| intermediate_algebra | 1269 | 63.4 | 17.1 |
| number_theory | 855 | 67.1 | 12.4 |
| prealgebra | 1203 | 78.0 | 8.2 |
| precalculus | 728 | 60.3 | 25.0 |

GSM8K (`arithmetic`) is the healthy branch (73% derived / 8% hardcoded); the non-linear MATH subjects — precalculus (25% hardcoded), geometry/intermediate_algebra (17%), number_theory (12%) — are where QF_LRA can't faithfully express radicals / π / digits / combinatorics, matching RESEARCH_PLAN §2.2 N2. This is why S1's seed scope is GSM8K + algebra/prealgebra only.

## Reason breakdown

| reason | n |
|---|---:|
| `derived:unique_1rel` | 4337 |
| `derived:unique_2rel` | 2773 |
| `derived:unique_3rel` | 1978 |
| `unknown:no_relational` | 1414 |
| `unknown:underdetermined` | 939 |
| `derived:unique_4rel` | 779 |
| `hardcoded:answer_pinned_literal` | 575 |
| `hardcoded:answer_is_pinned_var` | 521 |
| `unknown:no_answer` | 434 |
| `derived:unique_5rel` | 378 |
| `hardcoded:messy_constant` | 305 |
| `derived:unique_6rel` | 139 |
| `derived:unique_7rel` | 91 |
| `unknown:no_answer_decl` | 40 |
| `derived:unique_8rel` | 31 |
| `derived:unique_9rel` | 30 |
| `unknown:solve_parse_error` | 27 |
| `derived:unique_10rel` | 22 |
| `derived:unique_11rel` | 12 |
| `derived:unique_15rel` | 5 |
| `derived:unique_14rel` | 4 |
| `unknown:no_asserts` | 4 |
| `unknown:solve_sat` | 4 |
| `derived:unique_12rel` | 4 |
| `derived:unique_18rel` | 4 |
| `derived:unique_13rel` | 4 |
| `derived:unique_19rel` | 3 |
| `derived:unique_21rel` | 2 |
| `derived:unique_17rel` | 2 |
| `derived:unique_20rel` | 2 |
| `derived:unique_23rel` | 1 |
| `derived:unique_25rel` | 1 |
| `derived:unique_16rel` | 1 |
| `derived:unique_50rel` | 1 |
| `derived:unique_22rel` | 1 |
| `derived:unique_49rel` | 1 |
| `derived:unique_29rel` | 1 |

Full machine-readable report + 20 examples/bucket: `results/uniqueness_audit.json`.