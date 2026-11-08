# 05 — Repository audit (state on 2026-11-07, branch `feat/sprint0` @ `e84af51`)

Read-only audit. Nothing in the project was modified except this file. All helper scripts live in the
session scratchpad (`…/scratchpad/{inv,n1,n1b,n1c,mv,mv2,uq,uq2,wz,wz3,wz9}.py`). "§" numbers below
refer to sections of this report.

---

## 0. Verdict (one paragraph)

The repository is **two projects in one tree**: (a) a legacy Vietnamese SMT pipeline (`src/`, `notebooks/`,
~1.4 GB of untracked `data/`) whose outputs are **not shippable** — the augmented corpus is dominated by
answer-only mutations with ugly answers and formal-artifact leakage, the fine-tunes were buggy, and the
legacy "augmented SFT+DPO" prediction file contains **rows whose reasoning is a fixed template plus the
gold answer string copied in** (§4.5); and (b) a new package `vi_math_verified_aug/` that is **~10 %
built**: 3 real modules (`math_verify`, `uniqueness`, `z3_utils`, 542 lines) + 23 stub modules that raise
`NotImplementedError`. Tests pass (54/54), no secrets are in git history, but dependencies are
unpinned, there are no model artifacts at all, ~45 % of the data on disk is duplicate/intermediate, and
several data assets have unclear or non-commercial licenses. **Keep** the new evaluation/verification
core, the translated seed/test sets (after license review) and the GSM8K formalizer; **fix**
`math_verify` (two confirmed false-positive bugs); **discard** the legacy augmented corpus, legacy
trained-model results and the junk intermediates.

---

## 1. Method (what was run)

| step | command (abridged) | result |
|---|---|---|
| tracked files | `git ls-files \| wc -l` | 138 tracked files; `git status --short` clean |
| dir sizes | `du -sh data _scratch results …` | data 1.4 G, _scratch 3.7 M, src 1.3 M, vi_math_verified_aug 134 K, results 36 K, docs 0 |
| data inventory | `scratchpad/inv.py` (walk `data/`, count JSON/JSONL records, first-record keys, VI-diacritic language heuristic, md5) | 155 files, table §2 |
| tests | `.venv/Scripts/python -m pytest -q -p no:cacheprovider` | **54 passed in 1.8 s** |
| compile | `python -m compileall -q src notebooks vi_math_verified_aug _scratch` | rc=0 (syntax OK everywhere) |
| secrets | `git log --all -p -G'<pat>'` for `nvapi-…`, `sk-…`, `hf_…`, `AIza…`, `ghp_…` | **0 hits in history**; 0 hits in tracked tree (`git grep`) |
| uniqueness audit re-run | `python -m vi_math_verified_aug.verify.uniqueness --audit --out <scratch>` | identical to committed report (§4.2), 90 s |
| N1 re-measure | `scratchpad/n1*.py` on `data/finetune/augmented/train_augmented_final.jsonl` | §4.1 |
| legacy results re-score | `scratchpad/wz*.py` through `eval/math_verify.is_correct` | §4.4–4.5 |

---

## 2. Data inventory (`data/`, entirely gitignored — `.gitignore:3`)

Language = majority of Vietnamese diacritics in the first record. "Producer" = script that writes the
file (found by `grep -rl <filename> src notebooks _scratch`). "Plan" = referenced in
RESEARCH_PLAN/EXECUTION/README. Hash = first 10 hex of md5.

### 2.1 Source / seed / test sets

| path | MB | records | schema (first record) | lang | producer | plan | notes / verdict |
|---|---:|---:|---|---|---|---|---|
| `generate/gsm8k/train.json` / `train.jsonl` | 4.3 / 4.2 | 7,473 / 7,473 | question, answer, ground_truth (+index) | en | `src/convert/preprocess_gsm8k.py` | indirect | upstream GSM8K (MIT). json≡jsonl content → keep one |
| `generate/gsm8k/test.json` | 0.8 | 1,319 | question, answer, ground_truth | en | same | — | upstream |
| `generate/MATH/train/*.json` (7 files) | 7.2 | 7,500 | index, problem, level, type, solution, subject, ground_truth | en | `src/convert/download_datasets.py` | indirect | upstream MATH (MIT) |
| `generate/MATH/test/*.json` (7 files) | 4.3 | 5,000 | problem, level, type, solution, subject | en | same | — | upstream |
| `translation/gsm8k/gsm8k_train_vietnamese.jsonl` | 5.1 | 7,473 | index, question, answer, ground_truth | vi | `src/translate/gsm8k/translate_gsm8k_train.py` (Gemini 2.5 Flash via proxy, l.8-10) | §2.1 | **seed asset — keep** (license note §2.4) |
| `translation/gsm8k/gsm8k_test_vietnamese.{json,jsonl}` | 0.9+0.9 | 1,319 | question, answer, ground_truth | vi | `translate_gsm8k.py` | — | **byte-identical** to `test/gsm8k/*` (hash a492eebc29 / 9ae3be1d98) → duplicate |
| `translation/MATH/train/*_vi.jsonl` (7) | 7.4 | 7,501 | index, problem, level, type, solution, subject, ground_truth | vi | `src/translate/math/translate_math_train.py` (gemini-2.5-flash-lite via proxy, l.10-11) | §2.1 | precalculus has 747 rows vs 746 source (1 stray row) |
| `translation/MATH/train/geometry_vi.jsonl.bak` | 1.3 | 870 | same | vi | manual | — | differs from live file → **junk** |
| `translation/hungarian_exam_vn.jsonl` | 0.01 | 33 | Question | vi | **none** | — | orphan; upstream `keirp/hungarian_national_hs_finals_exam` license unverified |
| `test/gsm8k/gsm8k_test_vietnamese.{json,jsonl}` | 0.9+0.9 | 1,319 | question, answer, ground_truth | vi | copy | §2.1 | **Vi-GSM8K test — keep one format** |
| `test/math/*_vi.jsonl` (7) | 4.5 | 5,000 | problem, level, type, solution, subject | vi | `translate_math.py` | §2.1 | **Vi-MATH test — keep** |
| `test/math.zip` | 1.1 | (7 files) | — | vi | manual | — | same 7 files; 6/7 byte-equal sizes, precalculus differs by 2 bytes → **duplicate** |
| `translation_test_output/svamp/svamp-vi.jsonl` | 0.3 | 1,000 | ID, Body, Question, Answer, Type, Equation | vi | `translate_svamp.py` | §2.1 | Vi-SVAMP (MIT) — keep |
| `translation_test_output/asdiv/asdiv-validation-vi.jsonl` | 0.7 | 2,305 | body, question, answer, solution_type, formula | vi | no producer in repo | §2.1 | **CC BY-NC 4.0** (DATA_SOURCES.md:15) — NC-only |
| `test/metaMATH/metaMathQA-395K_json_vi.jsonl` | **824.1** | 395,000 | original_question_vi/en, query_vi/en, response_vi/en, type | vi+en | **none** (downloaded; RESEARCH_PLAN.md:747 names 5CD-AI HF set) | S5 arm | 58 % of `data/`. Don't redistribute — re-download by HF id + revision; translator/license of the 5CD-AI copy not recorded |

### 2.2 Legacy formalization / merge outputs

| path | MB | records | schema | producer | verdict |
|---|---:|---:|---|---|---|
| `formalize_output/gsm8k/gsm8k-smt-nvidia-FULL.jsonl` | 4.3 | 7,473 | index, question, smt_lib, z3_score, z3_error, retries | `src/formalize/gsm8k/formalize_gsm8k_nvidia_langgraph.py` (gpt-oss-120b, l.27) | keep (S1 seed SMT) |
| `formalize_output/gsm8k/gsm8k-backup.jsonl` | 4.3 | 7,473 | same | — | **identical hash d48f053b51 → delete** |
| `formalize_output/math_cleaned/*.jsonl` (7) | 4.4 | 7,500 | index, subject, ground_truth, smt_lib, status, error, solver, attempts | `formalize_math_nvidia.py` | intermediate; MATH SMT mostly not trustworthy (§4.2) |
| `formalize_output/math_cleaned-copy/*.jsonl` (7) | 4.5 | 7,704 | same | manual | older version with extra rows (e.g. precalculus 892 vs 746) → **junk** |
| `formalize_output/math-fail.jsonl` | 0.2 | 369 | same | `remedy_math_failures.py` | intermediate |
| `finetune/gsm8k/gsm8k_train_vietnamese_smt.jsonl` | 7.1 | 7,473 | index, question, answer, ground_truth, smt | `src/merge/gsm8k/gsm8k_complete.py` | intermediate |
| `finetune/gsm8k/gsm8k_train_vietnamese_final.{json,jsonl}` | 7.9+7.5 | 7,473 | index, problem, smt, level, type, solution, subject, ground_truth | `gsm8k_complete.py` | json≡jsonl duplicate |
| `finetune/math/math_train_vietnamese_final.{json,jsonl}` | 10.8+10.4 | 7,397 | same | `math_complete.py` | json≡jsonl duplicate |
| `finetune/mine/train_vietnamese_final.{json,jsonl}` | 18.5+17.9 | 14,870 | same | `src/merge/merge_all.py` | **the base seed** (input to T1.1 audit, `uniqueness.py:124`); keep jsonl only |

### 2.3 Legacy augmentation, training data, predictions

| path | MB | records | schema | producer | verdict |
|---|---:|---:|---|---|---|
| `finetune/augmented/phase1_validated_complete.jsonl` | 27.7 | 14,870 | …, validation_status, classification, smt_code, z3_answer | `src/mutation_informalize/phase1_validate.py` | validation_status: FIXED 10,807 / GOOD 2,745 / **UNFIXABLE 1,318**; classification TRIVIAL 7,328 / WEAK 4,158 / GOOD 2,745 / BROKEN 639 |
| `finetune/augmented/phase2_mutated_smt.jsonl` | 21.8 | 34,022 | source_index, subject, level, smt_code, z3_answer, z3_status, mutation_id | `phase2_mutate.py` | intermediate |
| `finetune/augmented/phase3_informalized.jsonl` | 63.6 | 27,684 | …problem, solution, answer, status, feedback, attempts, mutation_strategy | `phase3_informalize.py` | raw: ok 19,495 / failed 8,189 — intermediate |
| `finetune/augmented/phase3_informalized_clean.jsonl` | 62.0 | 27,024 | same | `tools/cleanup_phase3.py` | ok 19,750 / failed 7,274 — intermediate |
| `finetune/augmented/train_augmented_final.jsonl` | 60.6 | 34,620 | index, subject, level, problem, solution, answer, smt_code, source, mutation_strategy, status | `phase4_merge.py` | 14,870 original + 19,750 augmented; **51 exact-duplicate problems**; quality §4.1 → **discard** |
| `finetune/augmented/analysis_plots/*.png` (9) | 0.4 | — | — | `phase3_analyze.py` | discard |
| `finetune/processed/sft_data.jsonl` | 48.0 | 34,620 | text | **no producer in repo** | every row starts with literal `<s>` (34,620/34,620) → N3 double-BOS; discard |
| `finetune/processed/dpo_data.jsonl` | 1.7 | 545 | prompt, chosen, rejected | **no producer, no consumer** (notebook 03 is Online DPO) | orphan → discard |
| `finetune/processed/train_cot_only{,_clean}.jsonl` | 13.1 / 14.8 | 14,870 each | text, index, subject, level | `notebooks/00_prepare_data.py` (clean only) | non-clean superseded; both Mistral `[INST]` format → discard |
| `result/wizardMATH/**` (8 files) | 13.8 | 1,319 + 5,000 | problem, …, model_reasoning, extracted_answer | external run | keep as raw anchor predictions (§4.4) |
| `evaluate/evaluate_result/WizardMATH/**` (5) | 11.5 | 5,000 + summaries | … | `src/evaluate/wizardmath/*` | derived from above, old scorer → regenerate |
| `result/original-sft/**` (8) | 8.6 | 1,319 + 4,998 | …, model_reason, model_answer, format_ok | `notebooks/04_inference_base.py` | buggy fine-tune (N3) → archive, never cite |
| `result/augmented-sft-dpo/**` (4 jsonl + 6 png) | 13.2 | 5,000 / 1,319 / 2,305 / 1,000 | id, problem, …, model_reason, model_answer | `notebooks/05_inference_augmented.py` (nominally) | **integrity problem §4.5 → discard, never cite** |
| `evaluate/img/**`, `formalize_output/img/**`, `merge_img/**` | ~8 | PNGs | — | analysis scripts | regenerable figures → discard |
| `arms/.gitkeep`, `eval/.gitkeep`, `vi_exam/.gitkeep` | 0 | — | — | T0.1 | placeholders; no new-pipeline data exists yet |

Outside `data/`: `_scratch/` holds `analysis_errors.csv` (0.97 MB), `evaluation_results.csv` (2.0 MB),
`gsm8k_smt_output*.json(l)`, `z3_evaluation_results.json`, and `_scratch/vi_exam/loigiaihay_exams.json`
(261 KB, 13 exam pages). `results/` tracks only `uniqueness_audit.{md,json}`.

### 2.4 Duplicates, junk and redistribution flags

**Byte-identical or near-identical duplicates (≈ 60 MB):** `gsm8k-backup.jsonl` ≡ `gsm8k-smt-nvidia-FULL.jsonl`;
`translation/gsm8k/gsm8k_test_vietnamese.*` ≡ `test/gsm8k/*`; `test/math.zip` ≈ `test/math/`; five `.json`/`.jsonl`
pairs with the same records (`gsm8k_train_vietnamese_final`, `math_train_vietnamese_final`, `train_vietnamese_final`,
`gsm8k_test_vietnamese` ×2, `generate/gsm8k/train`).
**Stale copies / junk:** `math_cleaned-copy/`, `geometry_vi.jsonl.bak`, `train_cot_only.jsonl`, `phase3_informalized.jsonl`
(raw), `dpo_data.jsonl`, all PNG folders.
**Unknown provenance (no producer script in repo):** `metaMathQA-395K_json_vi.jsonl`, `sft_data.jsonl`, `dpo_data.jsonl`,
`hungarian_exam_vn.jsonl`, `asdiv-validation-vi.jsonl`, `result/augmented-sft-dpo/**`.

**Redistribution — cannot ship without review (not legal advice):**

| item | issue | evidence |
|---|---|---|
| Vi-ASDiv (+ `asdiv_results.jsonl`) | CC BY-NC 4.0 → non-commercial only; can't go into an MIT/CC-BY bundle | DATA_SOURCES.md:15, :28-33 |
| All Vietnamese translations (GSM8K/MATH/SVAMP/ASDiv) | made with Gemini 2.5 Flash / Flash-Lite **through a third-party proxy (`beeknoee/…`)**; translator model, proxy and date not recorded in any datacard | `src/translate/gsm8k/translate_gsm8k_train.py:8-10`, `src/translate/math/translate_math_train.py:10-11`, `math_fix_issues.py:83-84` |
| `metaMathQA-395K_json_vi` | machine-translated third-party copy; translator and license not recorded | no producer; RESEARCH_PLAN.md:747 |
| `hungarian_exam_vn.jsonl` | upstream license not checked; not in DATA_SOURCES.md | — |
| Scraped loigiaihay.com content (`_scratch/vi_exam/*`) | site-authored solutions are the site's copyrighted text; no robots/ToS check in the scraper | `_scratch/vi_exam/scrape_loigiaihay.py:10-26` |
| `src/mutation_informalize/reference/paper_text.txt` (93 KB, **tracked and pushed to origin**) | full text of a third-party paper (Li et al., "Neuro-Symbolic Data Generation for Math Reasoning") | `git rev-list --objects --all` largest blob; header line 1 |
| DATA_SOURCES.md | lists only 4 datasets; omits MetaMathQA-vi, Hungarian exam, loigiaihay, and every LLM used for generation | DATA_SOURCES.md:10-15 |

---

## 3. Code inventory

### 3.1 `vi_math_verified_aug/` (the new package)

`wc -l`: 933 lines total including tests. `grep -c NotImplementedError` → 23 stub modules.

| module | lines | status | runs? | tests | notes |
|---|---:|---|---|---|---|
| `eval/math_verify.py` (T0.2) | 272 | **real** | yes | `tests/test_math_verify.py` (44 cases, pass) | **2 confirmed false-positive bugs + gaps**, see below |
| `verify/uniqueness.py` (T1.1) | 196 | **real** | yes; `--audit` reproduces committed report exactly | `tests/test_uniqueness.py` (10 cases, pass) | caveats below; default source path is cwd-relative (`:124`) |
| `verify/z3_utils.py` | 74 | **real** | yes | indirect only | `is_unique` blocks only the first model — correct for the stated use |
| `__init__.py` + 9 subpackage `__init__` | 18 / 1 each | docstrings | yes | — | — |
| `eval/{run_eval,data_quality,decontam}.py`, `ir/typed_dag.py`, `symbolic_aug/{constant_sub,structural}.py`, `informalize/{generate,leakage_filter}.py`, `llm_aug/{clients,generate}.py`, `verify/round_trip.py`, `vi_exam/{scrape,segment,verify}.py`, `scripts/{build_s0..s6,run_grid,sft}.py` | 9 each | **stub** (`_not_implemented` raising `NotImplementedError`, e.g. `scripts/build_s1.py:1-9`) | import only | none | 23 modules |
| `modal_app.py` | 7 | docstring only | — | — | T0.5 not started |
| `configs/`, `configs/prompts/` | — | `.gitkeep` only | — | — | **no prompts are versioned anywhere in the new package** |

So of the EXECUTION.md backlog only T0.1, T0.2, T1.1 are done (EXECUTION.md:58-65); T0.6 and T1.8 are
"in progress" but exist only as `_scratch/` prototypes.

**`math_verify` defects found by probing** (`scratchpad/mv.py`, `mv2.py`):

| input (pred, gold) | expected | got | cause |
|---|---|---|---|
| `"0.125"`, `"125"` | False | **True** (false positive) | `_collapse_number_seps` (`math_verify.py:152-158`) treats any `.ddd` as a thousands separator → `0.125`→`0125` |
| `"(1,2)"`, `"1.2"` | False | **True** (false positive) | decimal-comma rule `:157` rewrites tuple `(1,2)` → `(1.2)` |
| `"0.125"`, `"1/8"` / `"\boxed{\frac{1}{8}}"`, `"0.125"` / `"2.500"`, `"2.5"` / `"3.000"`, `"3"` | True | False | same root cause |
| `"(1,2)"`, `"(1, 2)"` | True | False | spacing changes which rule fires |
| `"\pm 2"`, `"-2"` | — | sign dropped (`:136`) | `\pm` deleted |
| `"50%"`, `"0.5"` | policy | False | % stripped without scaling (`:132`) |
| any gold that is text (`\text{Evelyn}`, `\text{(C)}`, `\text{chẵn}`) | True vs itself | **False** | `\text{}` stripped to empty (`:129`) → `_atoms_equal("", …)` is False (`:234`) |

Impact on Vi-MATH test (5,000 golds via `extract_answer(solution)`): **64 golds (1.3 %) cannot match even
themselves**; 126 golds are no-space tuples/lists exposed to the comma bug; 4 have 3-decimal values.
None of these cases are in the 44-case test file.

**`uniqueness` caveats** (`scratchpad/uq.py`, `uq2.py`):
- `classify` never compares the Z3 answer with the ground truth. Re-check on the 10,607 "derived"
  programs: **609 (5.7 %) yield an answer that `is_correct` does not match to `ground_truth`**
  (566 of them MATH; upper bound, some are formatting).
- Trivial wrappers count as "derived": `x=42; answer=(* x 1)` → `derived:unique_1rel`. The 1-relation
  bucket is 4,337 rows (29 % of all), so **"derived 71.3 %" is an upper bound**; `derived_strict` (≥2
  relations) is the defensible number.
- EXECUTION.md T1.1 asks for a QF_NRA/QF_NIA → SymPy fallback (EXECUTION.md:196-198) — not implemented;
  acceptance asked for 20 hand-checked flags, done note says 15 (EXECUTION.md:208).

### 3.2 Tests

`tests/` = `conftest.py` (5 lines), `test_math_verify.py` (94), `test_uniqueness.py` (52). 54 tests, all
pass, 1.8 s. No coverage of `z3_utils` directly, of `audit()`, of any legacy code, or of the scraper.
No CI config in the repo.

### 3.3 Legacy `src/` (51 tracked .py + 22 `*.outputs.md` run logs)

| area | what it does | runs? | issues |
|---|---|---|---|
| `src/convert/` | download GSM8K/MATH, preprocess | needs `datasets` (**not in .venv**) | cwd-relative paths |
| `src/translate/` | Gemini translation of train/test/SVAMP | needs `google-generativeai` (missing) or proxy key | proxy endpoint; prompts inline in code |
| `src/formalize/gsm8k/` | LangGraph ReAct agent, gpt-oss-120b via NIM, Z3 tool | needs `langgraph`, `langchain_*` (missing) | **reusable** — the one stage the plan keeps (RESEARCH_PLAN.md §2.1) |
| `src/formalize/math/` | MATH → SMT with repair | openai only | output often vacuous (§4.2) |
| `src/mutation_informalize/` | phase1–4: validate, mutate, informalize, merge | sibling-style imports (`from smt_parser import …`, 11 files) → only runs with cwd = that folder | N1 operators (`mutation_engine.py:36-71`), no-op `(- (+ e k) k)` at `:27` |
| `src/merge/`, `src/analysis/`, `src/evaluate/` | merges, stats, 3 divergent scorers | needs matplotlib/seaborn (missing) | superseded by `eval/math_verify.py` |

Overall: **35 / 51** src files use `".."`-relative paths (cwd-dependent); only 14 have a `__main__`
guard (others execute on import); `requirements.txt` deps `datasets, google-generativeai, langchain-core,
langgraph, langchain-openai, matplotlib, seaborn` are **not installed** in `.venv` (checked by import).
The legacy pipeline is not runnable end-to-end from this checkout.

### 3.4 `notebooks/` (7 Kaggle scripts exported as `.py`)

No `.ipynb` files → no stale output cells (good). All paths hard-coded to `/kaggle/input/...`
(`01_sft_train_base.py:25-26`, `04_inference_base.py:30-33`, `05_inference_augmented.py:36-41`, …).
N3 bugs confirmed in code: `MAX_SEQ_LENGTH = 2048` (`01:33`, `02:39`), `repetition_penalty=1.3` +
`no_repeat_ngram_size=4` (`04:237-238`, `05:257-258`), literal `<s>[INST]` in prompts (`00:172`, `04:40`,
`05:48`). Base model `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`. ML stack versions are pinned only in
inline `!pip install` cells (requirements.txt:20-22 comment). **No trained adapters/checkpoints exist
locally** (`find . -name '*.safetensors' -o -name adapter_config.json …` → none); they lived in Kaggle
inputs only.

### 3.5 `_scratch/` (gitignored, 36 files)

- `vi_exam/scrape_loigiaihay.py` (154 lines) — single-source prototype: one index page
  (`:14-17`), `requests`+BeautifulSoup, keeps MathJax `\(..\)`, segments on `Câu|Bài` headers (`:43`),
  regex "final answer" (`:60-63`), 1 s politeness delay, contact email in UA. No robots.txt/ToS check,
  no caching of raw HTML, no tests, writes next to itself (`:11`).
  **Measured yield** (`loigiaihay_exams.json`): 13 exam pages (9 Hà Nội + 4 provinces), 96 segments,
  72 with solution, **60 figure-free with solution (62.5 %)** — matches EXECUTION.md:282 "~62 %"; 50 also
  have a regex answer. But the regex answer is unreliable: the first sample in `sample_core_triples.md`
  lists `x = 9` (a *given*) and `x > 0, x ≠ 1` (a domain) as "final answers", and segments are
  multi-part (1), 2), 3) incl. proofs) — **"segment" ≠ "single-answer item"**, so true Core yield is
  lower than 62 %. `year` comes from the page title regex (`:72`) and is noisy (5/13 pages say 2026).
- `nim_smoke.py` — NIM smoke test of `deepseek-ai/deepseek-v4-pro-0813`; hard-coded absolute `.env`
  path (`:4`) and **prints the first 7 chars of the API key** (`:11`). Its output is not saved, so the
  T0.6 "smoke-tested" claim has no artifact.
- 13 `patch_*.py`, `build_v2_notebook.py`, `manage_final_pipeline.py`, `formalize_v5_manager.py`,
  `temp_generate_one.py` — one-off notebook patchers for the legacy pipeline (models seen:
  `beeknoee/deepseek-chat-v3.1`, `claude-sonnet-4-6`, `qwen-3-235b…`, `gpt-5-nano`). → discard.

---

## 4. Re-measured audit claims

### 4.1 N1 — augmented samples have ugly answers / leaked artifacts → **holds qualitatively, numbers are definition-sensitive**

Corpus: `train_augmented_final.jsonl`, 19,750 augmented rows (the memory/plan say "19,495": that is the
*raw* phase-3 ok count; the merged corpus uses the cleaned file with 19,750 — RESEARCH_PLAN.md:165 should
be corrected). Augmented rows are **MATH-only** (algebra 4,184, prealgebra 3,606, int. algebra 3,605, …;
0 GSM8K), i.e. the legacy augmentation was applied exactly where the SMT is least faithful.

| measure | original (14,870) | augmented (19,750) |
|---|---:|---:|
| non-integer numeric answer | 0.8 % | **63.6 %** |
| denominator > 12 | 0.2 % | 51.4 % |
| denominator > 12, excluding k/100 decimals | 0.1 % | 21.5 % |
| denominator ≥ 1000 (float-to-fraction artefacts like `26763185307/16000000000`) | 0.0 % | 12.5 % |
| problem text contains the English token `answer` (e.g. `\text{answer}= (A-B)+2h`) | ~0 % | **32.2 %** (6,360) |
| snake/`h_12345`-style identifiers in problem | 0.2 % | 1.7 % |
| answer string appears verbatim in problem | 0.8 % | 1.3 % |
| any of the leak signals | 0.9 % | **33.7 %** |

Plan says 40.7 % ugly / 16.3 % leak / seed 1.6 % (RESEARCH_PLAN.md:167-168). Any reasonable definition
puts ugly answers between 21 % and 64 % (vs ≤ 0.8 % seed), and leakage is **worse** than claimed
(≥ 33.7 % with a simple regex, mostly a literal `answer` variable in the statement). The no-op operator is
confirmed at `mutation_engine.py:27`; all three operators wrap only `answer_expr` (`:36-71`).
**Claim holds; replace the exact percentages with a scripted, versioned definition.**

### 4.2 N2 / T1.1 uniqueness audit → **reproduced exactly; "15.9 % hardcoded" does not hold**

Re-run output equals `results/uniqueness_audit.md`: n = 14,870; derived 71.3 %, derived_strict 42.2 %,
hardcoded 9.4 %, unknown 19.2 %; per-subject identical (precalculus 25.0 % hardcoded, GSM8K 7.6 %).
- "~40 % genuinely derive" → holds (42.2 % strict; GSM8K 50.4 %, MATH 33.9 % strict).
- "15.9 % hard-code the answer" → **not reproduced** (9.4 %); EXECUTION.md marks T1.1 done although its
  own acceptance ("reproduces … 15.9 % hardcoded ballpark", EXECUTION.md:200-201) is only half met.
- Plan claim "GSM8K formalization 7,473/7,473 … genuinely works" (RESEARCH_PLAN.md:154; README.md:51
  "14,870/14,870 verified SMT") is contradicted: GSM8K is 73.4 % derived / 7.6 % hardcoded / ~19 % unknown,
  and phase-1 labels 1,318 base rows UNFIXABLE yet all 14,870 entered training.
- New: 5.7 % of "derived" programs disagree with the ground truth (§3.1).

### 4.3 N3 — fine-tune bugs → **holds** (code references in §3.4; 34,620/34,620 SFT rows start with `<s>`).

### 4.4 WizardMath anchor (60.3 Vi-GSM8K / 21.8 Vi-MATH) → GSM8K holds, MATH shifts

Re-scored through `math_verify.is_correct` on the full model output: **Vi-GSM8K 60.2 %** (60.5 % on the
`extracted_answer` field), **Vi-MATH 24.9 %** (+3.1 pt vs 21.8, scorer-dependent; and `math_verify` still
has the false-positive bugs above, so this number is provisional).

### 4.5 NEW — legacy `result/augmented-sft-dpo/` predictions contain gold-answer rows

Re-scoring gives implausible numbers for a 4-bit Mistral-7B SFT+DPO: Vi-MATH **53.1 %** (65.3 % via its
`model_answer` field) vs. 11.1 % for the base SFT model and 24.9 % for WizardMath-7B. Inspection
(`scratchpad/wz3.py`, `wz9.py`):
- **185 / 5,000** MATH rows have `model_reason` equal to one fixed sentence ("Để giải bài toán này, ta tiến
  hành các bước logic. Đáp án cuối cùng là: \boxed{…}"); **all 185 end with `\boxed{<ground_truth>}`
  verbatim**, including **53 where `ground_truth` is a truncated extraction artefact** (e.g. `\frac{1`,
  `6\sqrt{3`) that no model would produce on its own. 117 of the 185 are Level 5.
- Control: 0 / 4,998 such rows in `result/original-sft/`.
- 1,154 / 5,000 rows have a `model_answer` field that disagrees with the `\boxed{}` in `model_reason`.
- No script in the repo (src, notebooks, _scratch) contains that template sentence; the producer is
  unknown.

Whatever the cause (fallback filler, patching script, merge bug), **this file and every number derived
from it (README/paper tables, `data/result/augmented-sft-dpo/plots/*`) must not be cited or shipped**.
The remaining 4,815 rows have unverifiable provenance too.

### 4.6 Memory file status

`project-audit-findings.md`: N1 qualitatively true / numbers definition-dependent and the 19,495
denominator is wrong; N2 half-true (40 % ✓, 15.9 % ✗ → 9.4 %); N3 true; reusable-assets claim "GSM8K
formalization real, 7473/7473" is overstated (≈ 73 % derived); WizardMath 60.3 ✓ / 21.8 ≈ 24.9.

---

## 5. Reproducibility hygiene

| check | status | evidence |
|---|---|---|
| `.env` tracked? | **No** — ignored; only `.env.example` ever committed | `.gitignore:15-17`; `git check-ignore -v .env`; `git log --all --name-only` |
| secrets in history | **NO** for `nvapi-`, `sk-`, `hf_`, `AIza`, `ghp_` (0 hits); no hard-coded key literals in .py | §1 |
| local `.env` | holds 7 keys incl. 3 legacy (`GEMINI_API_KEY`, `NVIDIA_API_KEYS`, `BEEKNOEE_API_KEY`) | names only inspected |
| key exposure risk | `_scratch/nim_smoke.py:11` prints key prefix | — |
| personal data in tracked files | Windows user path in `src/formalize/gsm8k/formalize_gsm8k_nvidia_langgraph.outputs.md:18`, `analyze_math_results.outputs.md:27`; `@hus.edu.vn` in `.env.example:13`, EXECUTION.md:164 | `git grep` |
| dependency pinning | **none**: `pyproject.toml` 10 unversioned deps; `requirements.txt` unversioned and lists a different (legacy) set; no lockfile | files |
| installed (de-facto) versions | Python 3.11.7 (Anaconda base), z3-solver 5.1.0.0, sympy 1.14.0, openai 3.10.0, numpy 2.4.6, pandas 3.0.5, pytest 9.1.1 | `pip freeze` |
| Python version consistency | README badge says 3.10+ (README.md:7) vs `requires-python >=3.11` | — |
| ML stack | unpinned; `!pip install` in Kaggle cells; no Modal image yet | §3.4 |
| large files in git | none (largest blob 93 KB `paper_text.txt`; pack 836 KB) | `git count-objects -vH` |
| notebook state | `.py` exports only; no outputs committed | §3.4 |
| data versioning | none: 1.4 GB untracked, no checksums, no `INVENTORY.md` (T0.8 open), RESULTS_PROVENANCE.md has 0 rows | files |
| random seeds | legacy `random.seed(42)` at import in 5 modules; LLM calls not seeded (except planned NIM `seed=42`) | `grep random.seed` |
| prompts versioned | legacy prompts inline in code; new `configs/prompts/` empty | §3.1 |
| CI | none | — |
| branches | `main` (dbfecbb) is 10+ commits behind `dev`/`feat/sprint0`; all pushed to `origin` | `git branch -vv` |

---

## 6. KEEP / FIX / DISCARD

Effort in person-days (pd) for one person familiar with the repo.

| item | decision | reason | effort |
|---|---|---|---|
| `vi_math_verified_aug/verify/{uniqueness,z3_utils}.py` + tests | **KEEP + small FIX** | reproducible, deterministic, tested; add GT-match check, treat 1-relation trivial wrappers as non-derived, add `--src` relative to repo root | 0.5 pd |
| `vi_math_verified_aug/eval/math_verify.py` + tests | **FIX** | single scorer is the right design, but 2 false-positive bugs (`0.125`≡`125`, `(1,2)`≡`1.2`), text answers unscorable (64 Vi-MATH golds), `\pm` and `%` policies; add ≥20 regression cases from §3.1 and run against all 5,000 Vi-MATH golds (self-match must be 100 %) | 1–1.5 pd |
| Package skeleton (`pyproject`, subpackage layout, `RESULTS_PROVENANCE.md`) | **KEEP** | sound structure | — |
| 23 stub modules + `modal_app.py` | **KEEP as placeholders** (they are the backlog) | no content to judge | (the whole plan) |
| `results/uniqueness_audit.{md,json}` | **KEEP + FIX text** | numbers reproduce; reword "N2 ballpark reproduced" (15.9 % not reproduced) | 0.1 pd |
| Vi-GSM8K test, Vi-MATH test, Vi-SVAMP test (one format each) | **KEEP** (eval only) | needed as anchors; MIT upstream; record translator/model + date | 0.5 pd (datacard) |
| Vi-ASDiv test | **KEEP for eval, exclude from any MIT/CC-BY release** | CC BY-NC 4.0 | 0.1 pd |
| Vi GSM8K + MATH train translations (`finetune/mine/train_vietnamese_final.jsonl`) | **KEEP as seed** (pending license review) | central to S0/S1; 1,318 UNFIXABLE / 1,401 hardcoded rows must be filtered, not shipped as "verified" | 0.5 pd |
| GSM8K SMT (`gsm8k-smt-nvidia-FULL.jsonl`) + `src/formalize/gsm8k/` | **KEEP, port into package** | best legacy asset; port off LangGraph-only deps, pin, add derived-only filter | 2 pd |
| `metaMathQA-395K_json_vi.jsonl` | **KEEP locally, don't ship** | S5 arm; replace with a download script pinned to HF id + revision | 0.3 pd |
| WizardMath raw predictions | **KEEP as anchor input** | re-score with fixed `math_verify` (T0.7) | 0.2 pd |
| `_scratch/vi_exam/scrape_loigiaihay.py` | **FIX → port to `vi_exam/scrape.py`** | idea works (62 % figure-free segments) but single source, no robots/ToS check, no raw-HTML cache, unreliable answer regex, multi-part segments; needs per-source adapters + license field per item | 4–6 pd (T1.8 + T2.9) |
| `_scratch/nim_smoke.py` | **FIX → `llm_aug/clients.py` (T0.6)** | remove key-prefix print and abs path; save smoke output as artifact | 1–2 pd |
| Legacy `src/mutation_informalize/` | **DISCARD code paths for operators; KEEP as reference only** | N1 operators are the problem being replaced; architecture idea reused per plan | — |
| `train_augmented_final.jsonl`, `phase*` files, `sft_data.jsonl`, `dpo_data.jsonl`, `train_cot_only*.jsonl`, analysis PNGs | **DISCARD** | quality §4.1, double-BOS, orphan/unknown producers | 0 (delete/archive) |
| `result/augmented-sft-dpo/**` and any number derived from it | **DISCARD — never cite** | gold-answer rows §4.5 | 0 (but audit the draft paper, see gaps) |
| `result/original-sft/**`, `evaluate/**` | **ARCHIVE** | buggy fine-tune (N3), old scorers | 0 |
| Duplicates (`gsm8k-backup`, `translation/gsm8k/*test*`, `math.zip`, `*.json` twins, `math_cleaned-copy/`, `.bak`) | **DISCARD** | §2.4 | 0.1 pd |
| `src/{convert,translate,merge,analysis,evaluate}` | **ARCHIVE** (keep in git history / `legacy/` tag) | not runnable from checkout, cwd-relative, superseded | 0.2 pd |
| `notebooks/*.py` | **ARCHIVE**; reuse nothing but the Kaggle→Modal lesson | Mistral, `[INST]`, N3 bugs, `/kaggle` paths | — |
| `_scratch/patch_*.py`, build/manage scripts, CSVs | **DISCARD** | one-off patches | 0 |
| `src/mutation_informalize/reference/paper_text.txt` | **REMOVE from tree (and consider history rewrite)** | third-party paper full text pushed to GitHub | 0.2 pd |
| README.md (legacy-pipeline claims, 3.10+ badge, "14,870 verified") | **FIX / rewrite** | contradicts §4.2, §4.5 | 0.5 pd |

---

## 7. Gaps to be shippable (data + models + code + prompts)

1. **Integrity first.** Trace where the §4.5 rows came from; check whether `E:\CONFERENCE\conference_paper_IEEE_en.tex`
   (RESEARCH_PLAN.md:147) or any slide/thesis text uses augmented-SFT-DPO numbers; withdraw them.
2. **Correct the scorer** (`math_verify` bugs) before any new number is produced; freeze it with a
   regression suite (incl. all 5,000 Vi-MATH golds self-matching).
3. **Implement the pipeline**: 23 stub modules = everything from S1 operators to the training grid
   (EXECUTION.md T0.3–T6.x). Nothing in `data/arms/`, `data/vi_exam/`, `results/runs/` exists yet.
4. **Models:** zero checkpoints/adapters exist locally or are referenced by a stable ID. Need an HF repo (or
   release asset) per trained model with config, seed, data commit, and a model card.
5. **Prompts:** none versioned in the new package (`configs/prompts/` empty); legacy prompts are inline
   in code. Every generation/informalization/solver prompt must be a file with an ID logged in outputs.
6. **Dependency pinning:** a lockfile (`uv.lock`/`pip-tools` `requirements.lock`) for the core package; a
   separate pinned ML image (Modal) for training/inference; make README, pyproject and requirements.txt
   agree (drop or move legacy deps to `requirements-legacy.txt`).
7. **Data packaging:** `INVENTORY.md` with checksums (T0.8), a DATACARD per released dataset, download
   scripts for upstream/third-party data (MetaMath-vi, GSM8K, MATH) instead of redistributing them,
   train/test decontamination report (exact-match train∩test is 0 today; near-dup not checked).
8. **Licensing:** per-subset license table (ASDiv NC; Gemini/proxy translation provenance; DeepSeek/NIM
   generation provenance; loigiaihay/official exam content); remove `paper_text.txt`; add MetaMath-vi, Hungarian exam,
   exam sites and all generator models to DATA_SOURCES.md.
9. **Provenance:** RESULTS_PROVENANCE.md has 0 rows; every reported number needs run id + data hash.
10. **Run-ability:** one entry point per stage (`python -m vi_math_verified_aug.scripts.build_s1 …`) with
    repo-root-relative paths, a `make`/`justfile` recipe for the full pipeline, and CI running `pytest`.
11. **Hygiene leftovers:** strip local user paths from tracked `*.outputs.md`; remove key-prefix printing;
    merge `feat/sprint0` → `dev` → `main` once green (main is 10+ commits behind).
12. **Human-gated items** still open per EXECUTION.md: MOS raters (T3.2), 2 annotators for Vi-ExamMath QA
    (T3.3), advisor license sign-off (T1.8).
