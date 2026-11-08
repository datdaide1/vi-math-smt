# Datasheet — frozen evaluation sets v1.1 (Vietnamese + English mirror)

Builder: X3 (`vi_math_verified_aug/eval_sets/`), report
[`exec/X3_eval_sets.md`](../../docs/research_2026-11/exec/X3_eval_sets.md). Files: `data/eval/` (each set has
`<name>.jsonl` + `<name>.datacard.json`; `SHA256SUMS` covers all 18 files; `sha256sum -c` → 18/18 OK; re-running
`freeze` reproduces SHA256SUMS byte for byte). Status: **FROZEN v1.1** (2026-11-07).

## 1. Motivation
Parallel Vietnamese/English test sets with identical ids and order, so that (a) Vietnamese accuracy is measured on
cleaned translations with typed golds and (b) the English mirror (P2/H1) compares the same items across languages.

## 2. Composition
| Set (VI / EN, same ids and order) | Rows | Role |
|---|---|---|
| `vi_gsm8k_platinum` / `en_gsm8k_platinum` | 1,209 | full GSM8K-Platinum (co-primary; EN = secondary) |
| `vi_math_algprealg` / `en_math_algprealg` | 2,058 | MATH test algebra 1,187 + prealgebra 871 |
| `vi_math500_algprealg` / `en_math500_algprealg` | 206 | MATH-500 subset of the above (124 + 82) |
| **`vi_primary_1000` / `en_primary_1000`** | **1,000** | **frozen primary sample**: 600 GSM8K-Platinum (stratified by solution length) + 400 MATH alg/prealg (stratified subject × level, all 206 MATH-500 forced in); seed 20261107 |
| `en_math500_full` | 500 | English MATH-500, all subjects; literature comparison only |

- Row fields: `id` (`gsm8k-test-<idx>`, `math-test-<subject>-<pos>`), `question`, `gold {answer_canonical,
  answer_type, locale[, accept_also]}`, `gold_raw`, `solution`, `subject`, `level`, `in_math500`,
  `math500_unique_id`, `platinum_status`, `source_index`, review fields.
- Gold types in `vi_primary_1000`: int 900, rational 39, expr 19, decimal 15, radical 10, interval 7, tuple 4,
  text 3, choice 2, set 1. Golds come from the **English** sources (Platinum `####`, MATH-500 `answer`, last
  `\boxed{}` of the MATH solution), typed once by `math_verify.infer_gold`; never extracted from Vietnamese text.
- Review: 36 items auto-fixed, 29 cleared after Opus review, 0 still flagged in the primary sample (11 flagged items
  remain in the full sets and are excluded from the primary sample). `math-test-algebra-0333` carries `accept_also`.

## 3. Collection process
- English upstream pinned by HF revision: `madrylab/gsm8k-platinum@e762492`, `openai/gsm8k@740312a`,
  `EleutherAI/hendrycks_math@21a5633` (algebra, prealgebra test), `HuggingFaceH4/MATH-500@6e4ed1a`; raw-file and JSONL
  sha256 in `data/eval/_sources/manifest.json` and in each datacard.
- Vietnamese text: the project's earlier translations (`data/test/gsm8k/gsm8k_test_vietnamese.jsonl`,
  `data/test/math/{algebra,prealgebra}_vi.jsonl`, hashes in each datacard `local_sources`). Translator of that legacy
  step: per `06_packaging.md` §4.3 the existing Vietnamese translations were made with Google Gemini 2.5 Flash
  **[UNVERIFIED for the test files specifically — confirm from the Sept-2026 logs]**. Quality audited in
  `content_review/R1`, `R2`.

## 4. Preprocessing / cleaning / labeling
Alignment checks (GSM8K local EN test = `openai/gsm8k` 1319/1319; Platinum → GSM8K index 1209/1209 exact), JSON-escape
repair for MATH (`\right`, `\times`, literal `\n`), translation-defect review queue (R1/R2 patterns, numeral multiset
EN↔VI), Opus 5.5 proposals/solves/judgements for flagged items (`opus_fix`, Message Batches, $1.73, logs
`logs/api/X3/`), 9 Platinum-revised GSM8K solutions re-translated by Opus with checks. Full list: each datacard's
`transformations` and `changelog`.
**Models used:** `claude-opus-5-5` only. Test items were never sent to Gemini or NIM.

## 5. Uses
Evaluation only. **Never train on these items** (decontamination gates in S3/G5/D1 use bge-m3 cos < 0.85 VI↔VI,
< 0.80 VI↔EN, templated 13-gram coverage < 0.5 against every set here). Known external contamination: a
re-translated GSM8K test exists in `vinhnq29/vietnamese-elementary-math`; MetaMathQA-vi contains MATH train/test
near-twins (X5 §4.1–4.2) — do not use those corpora for training without the gate.

## 6. Distribution and licenses
| Upstream | License | Note |
|---|---|---|
| GSM8K (`openai/gsm8k`) | MIT | keep notice |
| GSM8K-Platinum (`madrylab/gsm8k-platinum`) | **[check the HF card before release]** | derived from GSM8K (MIT) |
| MATH (`hendrycks/math`, via `EleutherAI/hendrycks_math`) | MIT | keep notice |
| MATH-500 (`HuggingFaceH4/MATH-500`) | subset of MATH; **[check the HF card]** | |
| Vietnamese translations + typed golds + review fields (ours) | proposed CC BY 4.0 (06 §4.5) | disclose machine translation + Opus fixes |
Internal: bundle `data` (`BUNDLE.md`).

## 7. Maintenance
Versioned (v1.0 → v1.1 on 2026-11-07, changelog in every datacard). Any change → new minor version, item-level diff,
new SHA256SUMS. Grader version used for gold typing: 3.0.1 (sympy-fallback) — the current grader is 3.0.2; self-match
is 100 % on every frozen gold (X1).
