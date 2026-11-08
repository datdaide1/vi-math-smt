# T4: Cross-lingual and low-resource techniques for Vietnamese math reasoning

Agent T4 · 2026-11-07 · brief: `tech/BRIEF.md`. Measured defects to design against: `content_review/R1_vi_gsm8k.md`,
`R2_vi_math.md` and `R6_other_sets.md`.

**Evidence tags.** **[O]** means I opened the paper, page or model card in this session and the number comes from
it. **[S]** means the number comes from a search-result summary only, and is therefore **[UNVERIFIED]**. **[CMD]**
means a command I ran in this repository. **[R1/R2/R6]** means the project's own hand-checked content reviews.
**[EST]** means my own estimate, with its basis stated.

**Resource update from the coordinator, applied throughout.** The user reports:
- the NIM free API is available with several keys;
- Gemini free-tier keys are available;
- $100 of Claude credit is available;
- Kaggle T4 GPUs are available;
- Modal L4 time costs $0.80/h, so GPU time is scarce.

---

## 0. TL;DR: one decision per question

| # | Decision | Choice | Main evidence | Cost per 10k items |
|---|---|---|---|---|
| D1 | What to protect before MT | **Mask math, code and annotations; leave prose numbers unmasked but check them**. Mask LaTeX math, `[asy]`, `<<calc>>` and `\boxed{}`. Translate `\text{}` contents separately. Prose numbers are verified by multiset. | LaTeXTrans: 97% one-pass compile with placeholders [S]. R2/R6 macro and asy corruption (5,048 rows with translated macros, 71% of asy rows) | $0, CPU |
| D2 | Unit of translation | **One item per call, plain text with tags, never an LLM-emitted JSON array** | Root cause of 211 `\right`→CR rows and of the precalculus off-by-one is the old JSON batch format [CMD, §2] | — |
| D3 | Translator | **NIM `z-ai/glm-5.3-flash`** (fast, MIT). Fix it by a 60-item bake-off against `deepseek-v4.1-flash` and `gemma-4-31b-it`. Escalate to Claude Sonnet/Opus Batch only for items that fail twice. | 02_models: GLM is fast and fluent, DeepSeek and Gemma are slow on NIM. MGSM-Rev2 and mAceReason use an LLM self-refine loop [O] | ≈39k NIM calls, $0, ≈5 h wall with 3 keys. Escalation ≈ $2–4 |
| D4 | Terminology | **Triggered glossary.** Inject only the entries whose English trigger fires, add construction rules for comparatives, then check deterministically for required and forbidden Vietnamese terms. | R6: dozen→chục in 44% of dozen rows; R2: congruent→đồng dạng in 20–37% of occurrences | $0 |
| D5 | Quality gate | **Layered**: G0 structural, G1 numbers + glossary, G2 **solve-check against gold**, G3 **LLM MQM-span QE from a different model family**, G4 MetricX-24-QE as a tie-break | LLM span QE agrees with humans at F1 ≈0.55, against 0.28 for xCOMET-XXL [O 2605.24904]. Relational errors keep every number, so neural QE and number checks miss them [R1] | in D3 |
| D6 | Thresholds | **Calibrate on the 1,050 hand-labelled items** already in `content_review/annotations/`. Target recall ≥0.90 on major/critical errors and flag rate ≤15%. | No published CometKiwi/MetricX cut-off transfers across languages; percentile or auto thresholds are the norm [S] | ≈2k calls once |
| D7 | Repair | **Self-refine with explicit feedback** (failed checks + error spans), at most 2 rounds. Then escalate once. Then drop the item (train) or send it to a human (test). | mAceReason: up to 5 rounds, most accepted after ≤1 [O]. MGSM-Rev2 loop: retranslate → verify → re-solve [O] | in D3 |
| D8 | Solutions of translated problems | **Regenerate Vietnamese solutions** with the shared teacher, accepting a solution only when its answer matches the gold. Translated English solutions are kept as an ablation. | Holds solution style constant across C2 arms. The solve-check output G2 already produces these traces | ≈0 extra |
| D9 | Localization | **Not a training factor.** Train with light localization: Vietnamese names, currency word "đô la", **units transliterated and never converted**. Run a 300-item name/currency **evaluation probe**. | Entity swaps change accuracy by 0.3–5.9 pp, more for small models [S 2503.18018]. Localized versus translated: up to −9 pp for GPT-4o-mini, ≤ +3.6 pp from fine-tuning [O 2508.14913]. R1/R2: unit conversion *caused* gold errors | ≈600 calls |
| D10 | Reasoning language | **Vietnamese CoT in the main grid**, with Language Consistency (LC) reported. **One ablation, EN-CoT** (Vietnamese question → English reasoning → Vietnamese answer), on the best arm at 0.8B and 2B. Code-switched CoT is skipped. | Forcing target-language thinking costs accuracy, e.g. R1-1.5B MGSM 28.6→21.8 [O 2505.22888]. EN-Think SFT 66.7 vs Native-Think 61.5 [O MMATH]. M-Thinker 1.5B reaches vi LC 99% at no accuracy loss, but needs RL [O] | 2 extra SFT runs |
| D11 | Transfer method | **Translate-train (SFT) + RLVR on gated Vietnamese problems.** Translate-test is an **evaluation reference row only**. Layer-swap merging is a stretch goal. | 3B: RL on non-English translated data 66–71.5 vs RL-en 62.7 MGSM avg; SFT differences ≤2.4 pp [O 2509.23657]. Layer swap tested only at 8B [O] | — |
| D12 | Tokenizer | **No vocabulary adaptation.** Measure fertility as a covariate and scale `max_new_tokens` by the measured ratio. | ~150k multilingual vocab gives near-English Vietnamese fertility [S, NeurIPS 2026]. Expansion needs continued pre-training, which is out of budget | CPU minutes |
| D13 | Fair evaluation | Use verified test v1.1 with the residual error bound. Add an English-solvable-subset control, an **alternative-translation probe** (300 items, second MT family), a VN-number-format-aware grader, and keep Vi-ExamMath off NIM and Gemini-free. | MGSM-Rev2: a 17.6 pp FR gap fell to 1.2 pp after data and regex fixes [O]. MT benchmarks bias absolute scores 1–4.6 pp but keep rankings [O 2605.24904] | ≈1k calls + 1 T4-h |

---

## 1. Evidence table

### 1.1 Translating math data, quality estimation and benchmark translation errors

| Method / finding | Paper, date | Numbers | Languages | Scale | Tag |
|---|---|---|---|---|---|
| Placeholder masking of LaTeX commands and environments before LLM MT | LaTeXTrans, arXiv 2508.18791 (Aug 2025) | En→Zh: 97% one-pass compile success, <0.5 format errors per document | en→zh and others | whole papers | [S] |
| Mask math spans → MT → restore | LaTeXMT, EMNLP 2025 demo | "state-of-the-art MT may break LaTeX structures"; LGPL code | — | — | [S] |
| Prompt-only LaTeX protection, then LLM-graded self-refine for ≤5 rounds, discard if still rejected | mAceReason-Math, 2603.10767 (Mar 2026) | Source cleaning: 4% removed, 11% fixed. 97% of a 100-item pilot rated acceptable or excellent by native speakers. <0.1% problematic in the 190-item test. 36% received stylistic post-edits. **No COMET, no back-translation** | 14 languages, **no vi** | >10k per language; Claude Sonnet 4 as translator and grader | [O] |
| GPT-4 MT → 3-model cross-verification → certified humans | MMATH, 2505.19126 (May 2025) | Cross-model revision changed ~15% of items; humans then changed ~3% | 10 languages **incl. vi** | 374 problems per language | [O] |
| Semi-automatic QA of a translated math benchmark: flag items most models fail → retranslate → verify facts → re-solve | "Mind the Gap… or Not?" (MGSM-Rev2), 2511.05162 (Nov 2025) | GPT-5 FR 80.0 vs EN 97.6, gap **17.6 pp → 1.2 pp** after fixes. A number-format-aware regex alone took FR 80→90. Bengali digits 45.2→86.4 | MGSM languages (no vi) | 250 items × 11 languages | [O] |
| Effect of translation errors on translated-benchmark accuracy | Thellmann et al., 2605.24904 (May 2026; ACL 2026) | An item with ≥1 target-side error scores −5.6 to −7.5 pp (−6.1 to −10.7 pp on English-solved items). +10% erroneous items costs ≈0.6–0.8 pp. Rankings are preserved (ρ≈1.0), absolute scores fall 1–4.6 pp. **LLM span QE: GPT-5.2 F1≈0.55 vs xCOMET-XXL 0.28** | 9 EU languages, DeepL MT | 225 human-annotated segments | [O] |
| Machine vs human translation of benchmarks | MultiLoKo, 2504.10356 | MT undervalues scores, rank correlation 0.97 | 31 languages | — | [S] |
| QE-based data selection for LLM fine-tuning | 2512.11388 (Dec 2025) | CometKiwi-selected data beats random and TF-IDF. **No universal threshold**: papers use auto thresholds or top-k% | low-resource MT | — | [S] |
| MetricX-24 Hybrid-Large (QE mode supported) | HF card | Apache-2.0, mT5-large. Scores 0–25, lower is better. WMT24 average 0.705 (hybrid) / 0.683 (QE) | multilingual | ~1.2B [EST from mT5-large] | [O] |
| CometKiwi-22 | HF card | **CC-BY-NC-SA, gated**. Score 0–1. Vietnamese covered | — | infoxlm-large | [O] |
| XCOMET-XL | HF card | ~3.5B, **CC-BY-NC-SA, gated**. Outputs error spans. Vietnamese covered | — | 3.5B | [O] |
| TranslateGemma 4B/12B/27B (Jan 2026; report 2601.09012) | HF card + blog | 4B: WMT24++ MetricX 5.32 (Gemma-3-4B: 6.97), COMET 81.6. **2K-token input limit; fixed prompt format, so no glossary instructions**. 55 languages (the WMT24++ set, which includes en→vi_VN [UNVERIFIED per-language]). Gemma terms, gated | 55 languages | 4B fits a T4 | [O card] / [S report] |
| EnViT5 (VietAI, MTet) | HF / arXiv 2210.05610 (2022) | PhoMT test en→vi 45.5 BLEU (Google Translate in 2022: 39.9). T5-base, short context, no LaTeX handling | en↔vi | ~275M | [S] |
| Hunyuan-MT-7B / HY-MT1.5 | 2509.05209 | 33 languages, reportedly incl. vi; claims 30/31 WMT25 first places | — | 7B | [S] |

### 1.2 Localization and cultural adaptation

| Finding | Paper | Numbers | Languages | Tag |
|---|---|---|---|---|
| Swapping cultural entities (names, foods, places, currency) in GSM8K, numbers unchanged | Karim et al., 2503.18018 (Mar 2025) | Accuracy drop 0.3% (Claude 3.5 Sonnet) to **5.9% (Llama-3.1-8B)**. **Smaller models drop more.** Currency errors are 4.0% of errors | EN, 6 cultures | [S] |
| LLM localization (names, organisations, currency), with fallback to the plain translation when difflib < 0.8 | Azime et al., 2508.14913 (Aug 2025) | Translated vs localized: **up to −9 pp** (GPT-4o-mini). Fine-tuning on localized data: small changes (e.g. +3.6 pp Kinyarwanda, Gemma-2-9B) | 18 African languages, no vi | [O] |
| Unit conversion during translation breaks the gold | R1 6922, R2 algebra 387 and 1000, c&p 444; R6 26489 | yard→mét, gallon→lít and "cubic yards"→"mét khối" while the gold stayed in imperial units | vi | [R1/R2/R6] |
| Name/digit perturbation variance on MGSM | MGSM-Pro, 2601.21225 | 5 instantiations per item. Low-resource languages are more sensitive to digit changes | MGSM languages | [S] |

### 1.3 Reasoning language (think in English, in Vietnamese, or mixed)

| Finding | Paper | Numbers | Scale / languages | Tag |
|---|---|---|---|---|
| Forcing the thinking trace into the query language (prompt hack) | Qi et al., 2505.22888 (EMNLP-F 2025) | **R1-Distill-1.5B MGSM: match 54→86%, acc 28.6→21.8 (−6.8)**. 7B: 50.5→48.9. 14B: 68.1→63.9. AIME collapses at every size (7B 17.4→8.3). Post-training on 100 target-language traces (7B, TE): MGSM acc 31.7→**5.5** | 1.5–32B, 11 languages, no vi | [O] |
| EN-Think vs Native-Think SFT | MMATH, 2505.19126 | Qwen2.5-32B-Instruct + 3k examples: **EN-Think 66.72, Native-Think 61.46**, EN-SFT 62.38 (base 29.41). Answer-in-target prompt: 7B acc 58.44→56.13, answer LC 47→56 | 32B (no ≤4B SFT), 10 languages incl. vi | [O] |
| Vietnamese accuracy of small reasoning models | MMATH | R1-Distill-Qwen-1.5B: **vi 36.89**, en 45.41. 7B: vi 58.53, en 63.90. The 7B answers vi questions mostly in English | 1.5B/7B | [O] |
| RL with a language-consistency reward plus cross-lingual thinking alignment | M-Thinker, 2510.07300 (ACL 2026) | **vi is held out (not trained).** 1.5B vi: base LC 8.2 / acc 36.5. Naive GRPO: LC **0** / acc 47.7. M-Thinker: LC 99.1 / acc 37.7. 4B (Qwen3-4B-Thinking): base LC 0 / acc 77.8, M-Thinker LC 99.3 / acc **73.3 (−4.5)**. Cold-start SFT alone at 1.5B: acc 12.8 | 1.5B/4B/7B. 8×H20 for RL plus a 2×8 H20 judge | [O] |
| RL vs SFT, English vs non-English (translated) training data | Huang et al., 2509.23657 (Sep 2025) | **Qwen2.5-3B**, MGSM average: base 31.8. SFT-en 45.2, SFT-non-en 46.3–47.6. **RL-en 62.7, RL-non-en 66.0–71.5.** MMath500: RL-zh 61.3 vs RL-en 55.7 | **3B**, 10 languages, translated with Qwen3-30B-A3B | [O] |
| Translate-to-English at inference (translate-test), selective | Luar, 2606.02465 (Jun 2026) | Qwen3-4B average: base 44.9, **full translation 55.4**, Luar 54.4 with 20% translator calls. Gains are concentrated in low-resource languages (zu 8→88.5). For high-resource languages the gain is small (ar 89.3→91.2, ko 89.6→90.9) | 4B, 10 languages, no vi; GPT-4.1 as translator | [O] |
| Test-time scaling of English-centric reasoners | Yong et al., 2505.05408 | "quote-and-think" pattern. Scaling benefits appear at ≳3B. Recommend reasoning in high-resource languages | s1.1 models | [S] |
| Language-Mixed CoT (English anchor + target language) | Son et al., 2510.04230 (Korean) | **+18.6 pts** average for small and mid models. Mixed beats monolingual CoT. Yi-Sang data: 3.7M traces | 4B–35B, ko | [O abstract] |
| Enforcing monolingual decoding in a bilingual reasoner | 2507.15849 | −5.6 pp on math | zh/en | [S] |
| Progressive code-switching SFT → RL | PCS, 2607.00485 (Jul 2026) | 30% of steps code-switched at cold start (from search). The page gives no numbers | 5 languages | [O abstract] / [S] |
| Vietnamese SLM test-time scaling | Vi-S1K, 2604.17794 (Apr 2026) | Qwen3-1.7B. SFT: +77% explanation quality. CoT+self-consistency beats ReAct. Data localized with Gemini 2.5 Flash-Lite | 1.7B, vi | [O abstract] |

### 1.4 Cross-lingual transfer and merging

| Method | Paper | Numbers | Scale | Tag |
|---|---|---|---|---|
| Layer swapping: bottom 5 and top 2 layers from a language expert, middle from a math expert, no extra training | Bandarkar et al., 2410.01335 (ICLR 2025) | Llama-3.1-8B MGSM avg/max: sw 24.8 base → swap 32.4/36.4; te 12.0→22.7; bn 29.2→37.1 (math expert alone 38.3); ja 33.6→38.5 (math expert 42.7). **Only 8B tested**; experts trained on 30–40k samples each | 8B | [O] |
| Question alignment (translate questions only) then English-response SFT | QAlign, 2401.07817 | +9.6 / +11.3 pp over MathOctopus (translated-GSM8K SFT) at 7B / 13B | 7–13B | [S] |
| SFT on translated responses + RL with a privileged pairwise judge | SP3F, 2601.18722 (Jan 2026) | Claims to beat fully post-trained models with <1/8 of the data; the page gives no numbers | — | [O abstract] |

### 1.5 Tokenizer

| Finding | Source | Numbers | Tag |
|---|---|---|---|
| Vietnamese "tokenization tax" | NeurIPS 2026 poster 164760 | Fertility ratio vs English: 0.83× (VinaLLaMA) to 1.93× (Mistral-7B), driven by diacritics. **~150k multilingual vocabulary is enough for near-English fertility.** A 65k English-centric vocabulary is worse than a 32k one | [S] |
| FLORES vi fertility | HF dataset `Helmo21/asia-fertility` | Qwen3 ≈1.28, Gemma-2 ≈1.20 | [S] |
| Vocabulary expansion needs continued pre-training | SambaLingo 2404.05829; 2407.05841 | Lower fertility after expansion plus continued pre-training (Llama-2) | [S] |

### 1.6 Hosted services and their roles in the stack

| Service | Facts | Role | Tag |
|---|---|---|---|
| Gemini API, unpaid tier | Free-tier models include `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`. **The docs do not publish free-tier RPM/RPD; they point to AI Studio**, so the limits are [UNVERIFIED] | Use Gemini only as a **QE judge on public-source items**. It does not author training text and never sees Vi-ExamMath | [O] |
| NIM trial | Hosted endpoint; inputs leave our control (`02_models` §2) | Public GSM8K/MATH translation only; unreleased native test items are not sent | [O via 02_models] |
| Claude | Opus 5.5 Batch ≈ $0.014 per translation, $0.015 per repair; Sonnet 5.5 Batch ≈ $0.007 (`02_models` §8.2) | Fine for escalation only. Translating 10k items with Opus would cost ≈ $140 | [O via 02_models] |

---

## 2. Root causes of the measured defects, and what removes each

[CMD] `src/translate/math/translate_math_train.py` sends **batches of items as a JSON array** to
`gemini-2.5-flash-lite` through a local proxy, and asks the model to return a JSON array. When `json.loads`
fails, it applies `re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', raw)`. That regex deliberately **leaves `\r \t \f \b \n` un-doubled**, so
`\right`, `\times`, `\text`, `\frac`, `\theta`, `\neq` and `\nu` decode as control characters. The script also hand-patches only
`\begin`, `\bf` and `\frac`.

| Measured defect | Rate | Cause | Countermeasure in §3 |
|---|---|---|---|
| `\right`→CR+`ight`, `\times`→TAB (R2) | 211 train rows | LLM-emitted JSON plus the escape regex above | D2: tagged plain text, the program writes JSON. G0: no control characters and a round-trip check |
| Precalculus off-by-one, duplicates and gaps (R2 §4.1), GSM8K 220–225 shift (R1) | 165 rows / 8 rows | Batch output count ≠ input count, re-joined by position | D2: one item per call, ID carried in the program, never in the LLM output |
| Literal `\n`, dropped `}` in `\boxed{`, `$` unescaped (R2) | 7+22 / 10 / 25 rows | Same JSON path, plus no structural check | Masking `\boxed{…}` and math. G0 brace and `$` balance |
| `[asy]` edited or translated (R2: 36 rows; R6: 71% of asy rows) | — | Code exposed to the translator | Mask `[asy]…[/asy]` byte-exact. Only `label("…")` strings are translatable sub-segments |
| LaTeX macros translated (`\đóng hộp`, `\góc`, `\giây`) (R6) | 5,048 rows | Math not masked | Mask all math. G0: every `\cmd` in the output must exist in the source |
| dozen→chục, median→trung bình, congruent→đồng dạng, monic→đơn thức, common ratio→tỉ số chung, counterclockwise→chiều kim đồng hồ | 44% / 40% / 20–37% of occurrences | No terminology control | D4 triggered glossary + G1 forbidden/required-term check (§4) |
| "10 of the remaining" → "10 … còn lại" (R1, 4/4 occurrences wrong) | 100% of occurrences | EN quantifier scope rendered as a definite description | Construction rule CR-2 + G1 pattern gate + G2 solve-check |
| "N more X than Y" → "nhiều hơn N X" (anchor dropped) | 202 calque rows (2.7%); SVAMP 8 rows; ASDiv 4 rows | Comparative calque | CR-1 anchor rule + G1 + G3 |
| "6 more than twice" → "nhiều hơn 6 lần" | train 2597 | Additive relation rendered as multiplicative | CR-3 + G2 |
| Imperial→metric swap | R1 6922; R2 387, 1000; c&p 444; R6 26489 | The model "localized" units | Policy: units transliterated, never converted (G1 unit lexicon check) |
| `54,321` left with an English comma in Vietnamese prose | 42 MATH rows; ASDiv 3 | No number-format policy | Prose number normalizer (§3.6) + grader accepts both formats |

---

## 3. MATH-TRANSLATION ALGORITHM (MTA-vi)

### 3.1 Data structures

```python
@dataclass
class Item:
    id: str                    # stable source id, e.g. "gsm8k/train/2597"; never round-tripped through an LLM
    src: dict                  # {"problem_en", "solution_en", "gold"}; gold is a canonical string from eval/math_verify.py
    masked_en: str             # English text with placeholders
    spans: dict[str, str]      # "M3" -> "$\\frac{1}{2}$", "A1" -> "[asy]...[/asy]", "C2" -> "<<4*2=8>>", "B1" -> "\\boxed{18}"
    subsegs: dict[str, str]    # "T1" -> "apples"  (\text{} payloads and asy label strings that DO need translation)
    triggers: list[str]        # glossary entry ids + construction rules fired on the English text
    vi: dict                   # {"problem", "solution"} after unmasking
    checks: dict               # G0..G4 results, with failure details
    status: str                # ACCEPT | REPAIR_r | ESCALATED | DROP | HUMAN
    prov: dict                 # mt_model, judge_model, solver_model, prompt_sha, glossary_version, round, timestamp
```

### 3.2 Masking

The grammar is applied in a fixed order, longest constructs first:

```python
MASK_ORDER = [
  ("A", r"\[asy\].*?\[/asy\]"),                      # Asymptote: byte-exact. label("...") strings become subsegs
  ("E", r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}"),       # display environments
  ("D", r"\$\$.*?\$\$|\\\[.*?\\\]"),                 # display math
  ("I", r"\\\(.*?\\\)|(?<!\\)\$(?:\\\$|[^$])+?(?<!\\)\$"),  # inline math; skip escaped \$ amounts
  ("C", r"<<[^<>]*>>"),                              # GSM8K calculator annotations
  ("B", BALANCED(r"\\boxed\{")),                     # \boxed{...} with balanced braces (parser, not regex)
  ("U", r"https?://\S+"),
]

def mask(text):
    spans, subsegs = {}, {}
    for tag, pat in MASK_ORDER:
        for m in finditer_nonoverlapping(pat, text):
            body = m.group(0)
            # \text{...}, \mbox{...}, \textbf{...} and asy label("...") holding real words -> translatable sub-segment
            body, subs = extract_text_payloads(body, keep_if=lambda s: is_unit_abbrev(s) or not has_word(s))
            subsegs.update(subs)                                   # e.g. T1 -> " apples"
            k = f"{tag}{len(spans)+1}"
            spans[k] = body
            text = text.replace(m.group(0), f"<{k}/>", 1)
    return text, spans, subsegs
```

**Placeholder form.** Use self-closing tags such as `<M3/>`. LLMs preserve HTML/XML tags in translation, and the
form cannot collide with LaTeX. Numbers in prose (`Lan has 12 apples`) are **not masked**. Vietnamese needs them
in its word order with classifiers, and the measured number errors are rare: 12 of 7,473 rows in GSM8K [R1].
Instead, G1 checks the number multiset.

### 3.3 Glossary and construction rules

The glossary is applied as triggered entries, not a full dump. Each entry has the form
`(id, en_trigger_regex, context_regex|None, vi_required, vi_forbidden, note, source)`. The seed list is in §4.
Construction rules (CR) are English regexes with a required Vietnamese template:

| CR | English trigger | Vietnamese template (what must be preserved) | Forbidden rendering | Source of defect |
|---|---|---|---|---|
| CR-1 anchor | `(\d+|<\w+/>) (more|fewer|less) \w+ than` | "A có nhiều hơn **B** N X". The referent comes between *hơn* and the quantity, or use "A có … , nhiều hơn số … của B là N" | "A có nhiều hơn N X" with no referent (reads as "> N") | R1 3661, test 382; R6 595, 625, 1434 |
| CR-2 scope | `(\d+|\w+) of the remaining` | "N **trong số** X còn lại" | "N X còn lại" (= "the remaining N") | R1 130, 520, 1801, 2911 |
| CR-3 additive-of-multiple | `(\d+) (more|less) than (twice|\w+ times)` | "nhiều/ít hơn k lần số … của B là N" (= kB ± N) | "nhiều hơn N lần" | R1 2597; R6 364168 |
| CR-4 direction | `(\w+) less than|lighter|heavier|subtracted from|divides` | Preserve the direction: "nhẹ hơn", "lấy a trừ đi b", "a là ước của b" / "b chia hết cho a" | Reversed comparatives and reversed divisibility | R6 391354, 1315; R2 c&p 323 |
| CR-5 strictness | `strictly|at least|at most|no more than` | "(thực sự) nhỏ hơn", "ít nhất", "nhiều nhất / không vượt quá" | "đáng kể" | R1 529 |
| CR-6 fraction-multiplier | `\d+/\d+ times (more|as)` | Flag for G3 and a human. The English itself is ambiguous | — | R6 213083 |

Only fired entries go into the prompt, at most 15 glossary lines plus fired CRs. This keeps the prompt short and avoids
priming irrelevant terms.

### 3.4 Translate (per item, plain text)

```text
SYSTEM: Bạn là biên dịch viên toán học Anh→Việt cho sách giáo khoa Việt Nam.
USER:
Quy tắc:
1. Giữ NGUYÊN mọi thẻ dạng <M3/>, <A1/>, <C2/>, <B1/> — không thêm, bớt, đổi thứ tự trong cùng câu nếu không cần.
2. Không đổi đơn vị đo, không quy đổi tiền tệ, không đổi bất kỳ con số nào.
3. Thuật ngữ bắt buộc: {fired glossary lines: "dozen → tá (KHÔNG dùng 'chục')", ...}
4. Cấu trúc bắt buộc: {fired CR templates with one VI example each}
5. Văn phong đề thi Việt Nam, tự nhiên; số thập phân trong văn xuôi dùng dấu phẩy.
Trả về đúng hai khối:
<problem>...</problem>
<solution>...</solution>
NGUỒN:
<problem>{masked_en_problem}</problem>
<solution>{masked_en_solution}</solution>
SUBSEGMENTS (dịch từng dòng, giữ khóa): T1: apples | T2: ...
```

Use temperature 0 and seed 42. Parse with a tolerant tag regex, then unmask. The program builds the JSON with
`json.dumps(ensure_ascii=False)`. Assert that `json.loads(json.dumps(x)) == x` and that no character in `[\x00-\x08\x0b\x0c\x0e-\x1f]` is present.

### 3.5 The gate (accept only when every layer passes)

```python
def gate(it):
    g = {}
    # G0 structural, deterministic
    g["G0"] = all([
        set(placeholders(it.vi_masked)) == set(it.spans),          # placeholder bijection
        all(it.vi.count(s) == it.src_text.count(s) for s in it.spans.values()),  # restored spans byte-exact
        no_control_chars(it.vi), braces_balanced(it.vi), dollars_balanced(it.vi),
        latex_cmds(it.vi) <= latex_cmds(it.src_text),               # no new or translated macros (\đóng hộp, \giây)
        en_residue_ratio(strip_masks(it.vi)) < 0.05,                # langid / English stopword share
        0.8 <= vi_en_char_ratio(it) <= 2.2,                         # R2 used <0.7 / >2.5 for outliers; tighter here
    ])
    # G1 numbers + glossary + units, deterministic
    g["G1"] = all([
        num_multiset(norm_vn(prose(it.vi.problem))) == num_multiset(norm_en(prose(it.src.problem_en))),
        # norm: spelled numbers -> digits, "tá" -> 12 when EN has "dozen", VN separators, %, fractions
        not any(forbidden_hit(e, it) for e in it.triggers),         # e.g. EN "dozen" & VI "chục" -> FAIL
        all(required_hit(e, it) for e in it.triggers if e.required),
        all(cr_template_ok(cr, it) for cr in it.fired_CRs),         # CR-1..CR-5 VI regexes; CR-6 -> route to G3
        unit_lexicon(it.vi) == unit_lexicon_mapped(it.src),         # yard stays yard, never mét
    ])
    # G2 solve-check: an answer-space round trip, more informative than back-translation for math
    s_vi = solver.solve(it.vi.problem, k=1)                         # NIM deepseek-v4.1-flash, temperature 0
    if not equal(s_vi, it.src.gold):
        s_vi2 = solver.solve(it.vi.problem, k=1, temperature=0.6)
        s_en  = solver.solve(it.src.problem_en, k=1)                # English control
        g["G2"] = "PASS" if equal(s_vi2, it.src.gold) else ("UNINFORMATIVE" if not equal(s_en, it.src.gold) else "FAIL")
    else:
        g["G2"] = "PASS"
    it.vi_trace = s_vi.trace                                        # reused as the regenerated VI solution (D8)
    # G3 LLM MQM-span QE, judge from a different family than the translator
    spans = judge.mqm(src=it.src, mt=it.vi,
                      categories=["quantity-relation", "quantifier-scope", "direction", "unit", "term",
                                  "omission/addition", "question-target", "latex", "fluency"])
    g["G3"] = not any(s.severity in ("major", "critical") for s in spans)
    # G4 neural QE (MetricX-24-hybrid-large, QE mode, Kaggle T4) on masked text; lower is better
    g["G4"] = metricx_qe(strip_masks(it.src_text), strip_masks(it.vi)) <= TAU4
    if all([g["G0"], g["G1"], g["G2"] != "FAIL", g["G3"]]) and not g["G4"]:
        g["G3b"] = judge2.mqm(...)            # tie-break with a second judge family; accept if clean
    return g
```

**Model roles.** Each role uses a different model family:

| Role | Model |
|---|---|
| Translator T | `glm-5.3-flash` |
| Judge J1 | `nemotron-3-ultra-550b-a55b` |
| Solver S | `deepseek-v4.1-flash` |
| Judge J2 | Gemini `gemini-3.8-flash` (free tier, public items only), or Claude Sonnet 5.5 Batch for eval sets |

If the step-0 bake-off picks DeepSeek as T, swap the T and S roles.

**Threshold calibration (D6).** This runs once and needs about 2k calls.
1. Run G0–G4 on the **1,050 hand-labelled items** in `content_review/annotations/*.jsonl`. These are the existing
   translations, labelled major or critical in 56 cases (GSM8K 7, MATH 12, MetaMath 32, SVAMP 3, ASDiv 2; [CMD]).
   Exclude the `origin=source` inherited errors from the positives.
2. Choose J1's severity mapping and `TAU4` as follows:
   - **recall ≥ 0.90** on translation-caused major/critical errors (about 45 positives, so the recall CI is about ±9 pp);
   - **flag rate ≤ 15%** on items labelled clean or minor;
   - set `TAU4` at the 90th percentile of MetricX-QE on clean items.
3. Report the confusion matrix per gate. This doubles as a C3 result: which automatic QE catches math-meaning errors.

**Quality level required.** This is a design rule, not a literature result, because no paper gives a translated-SFT
noise tolerance for math.
- **Eval sets:** residual major+critical ≤ 1%. Verify it on a fresh blind 150-item draw: 0/150 gives a Wilson upper bound of 2.5%, so use 300 items to reach 1.3%.
- **Training data:** residual major+critical ≤ 2%. Translated-arm noise must stay below the 1–3 pp between-arm effects the thesis expects. Raw MetaMathQA-vi has 18.7% and Vi-GSM8K train has 3.3%; both fail this target, while the target is reachable after the pipeline.
- Thellmann's slope (+10% erroneous items costs ≈0.7 pp of accuracy) means 1% residual error biases evaluation by <0.1 pp.

### 3.6 Number-format normalizer (prose only)

Text inside math masks is never touched. In prose, convert `1,234`→`1.234` (thousands) and `3.5`→`3,5` (decimal),
following SGK convention. Ambiguous `54,321` is resolved from the English: a thousands comma in English becomes
`54.321`. Store the gold canonically, without separators. `eval/math_verify.py` must accept both `3,5` and `3.5`,
and `1.234` and `1,234`, with the item's source locale as the tie-breaker. In MGSM-Rev2, fixing the regex alone moved French accuracy 80→90 [O].

### 3.7 Repair loop and escalation

```python
def translate_item(it, max_rounds=2):
    it.masked_en, it.spans, it.subsegs = mask(it.src_text)
    it.triggers = fire_glossary(it.src_text); it.fired_CRs = fire_CR(it.src_text)
    it.vi = unmask(T.translate(prompt(it)), it.spans, it.subsegs); g = gate(it)
    r = 0
    while not accepted(g) and r < max_rounds:            # self-refine with EXTERNAL feedback, not self-critique
        fb = feedback(g)    # e.g. "placeholder <M4/> missing"; "EN 'dozen' -> VI 'chục' forbidden, use 'tá'";
                            # "numbers EN {12,3} vs VI {10,3}"; "solver got 18 on VI, gold 30; EN solved 30";
                            # J1 spans: [("10 yêu tinh còn lại", "quantifier-scope", "critical")]
        it.vi = unmask(T.post_edit(masked_src=it.masked_en, masked_vi=mask_like(it.vi), feedback=fb), ...)
        g = gate(it); r += 1
    if not accepted(g):
        it.vi = ESCALATE(it, fb)       # eval set: Claude Opus 5.5 Batch; train: Claude Sonnet 5.5 Batch, else J1's model as translator
        g = gate(it)
    it.status = "ACCEPT" if accepted(g) else ("HUMAN" if it.is_eval else "DROP")
    log(it, g)                          # every failure reason -> C3 error taxonomy
    return it
```

Back-translation is **not** a gate. R1 shows the damaging errors keep every number and read fluently, and an
EN→VI→EN round trip through LLMs tends to "repair" them silently. Solve-check (G2) and the span QE (G3) target the
actual failure.

### 3.8 Repairing the existing translations (instead of re-translating everything)

```text
Step 1  deterministic fixes (R2 §5, R1 §6, about 1 h of scripting, $0):
   CR/TAB/FF/BS -> backslash escapes only before a command stem (imes, ext, ight, heta, frac, ...);
   literal "\n" -> newline (exclude \neq \nu \not ...); close \boxed{; re-escape \$; copy English [asy] back verbatim;
   rebuild precalculus from the content map (drop 7 duplicates, add 6 missing rows, swap 478/479, fix 42 levels);
   re-order c&p 355-357; GSM8K rows 220-225, 2577, 3077 -> retranslate; "#### 9.000" -> canonical gold.
Step 2  treat each existing Vietnamese row as an MT candidate and run G0+G1 (free), then G2 on every row.
Step 3  run G3 only on: G1 fail, any CR/glossary trigger, G2 FAIL, or the MetricX top decile.
Step 4  failures enter the §3.7 repair loop, with the existing VI as the first draft (post-edit, not from scratch).
Step 5  eval sets: Claude adjudicates every item still flagged, then a human (R1: 2-4 h; R2: about 2 h).
        Freeze as Vi-GSM8K-test-v1.1 and Vi-MATH-test-v1.1 with a changelog and a re-estimated error bound.
MetaMathQA-vi (395k): do NOT repair. If a MetaMath arm is needed, retranslate a controlled 20-40k query subset
through MTA-vi and regenerate solutions (D8). R6 measured 18.7% major+critical in the raw file.
```

### 3.9 Cost per 10k items (new translation)

Basis for the token assumptions: GSM8K-sized items, with MATH items about 1.5× larger.

| Stage | Engine | Calls per 10k | Tokens (in / out) | $ | Wall time |
|---|---|---|---|---|---|
| Translate | NIM `glm-5.3-flash` | 10,000 | ≈12M / 8M | 0 | 4.2 h at 40 RPM per key |
| G2 solve-check (second sample on miss ≈15%, plus English control) | NIM `deepseek-v4.1-flash` | ≈12,500 | ≈6M / 20M | 0 | 5.2 h; latency-bound at ~35 s per call, so ≥25 concurrent requests |
| G3 MQM-span QE | NIM `nemotron-3-ultra` | 10,000 | ≈15M / 3M | 0 | 4.2 h |
| Repair (≈20% of items × 1.3 rounds) + re-gate | T + S + J1 | ≈7,800 | — | 0 | 3.3 h |
| G4 MetricX-24-QE | Kaggle T4 (or Modal L4) | 10k pairs | — | $0 Kaggle (≤$0.40 on L4) | ≤0.5 GPU-h [EST] |
| Tie-break J2 | Gemini free (public items only) | ≈1,000 | — | 0 | RPD limit [UNVERIFIED] |
| Escalation (≈3% of items) | Claude Sonnet 5.5 Batch (train) / Opus 5.5 Batch (eval) | ≈300 | — | ≈$2 / ≈$4.5 | ≤24 h batch |
| **Total** | | **≈40k NIM calls** | | **≈$2–5** | **≈17 key-hours.** About 5.5 h wall with 3 keys, provided the 40 RPM limit applies per key and not per account [UNVERIFIED] |

**Alternative translator, with no API calls.** TranslateGemma-4B on a Kaggle T4 (fp16, vLLM) produces about 8M output tokens per 10k items, which is about 2–3 T4-hours [EST], or ≈$2 on Modal L4. Drawbacks: a 2K-token
context, and a fixed prompt format with no glossary instructions, so CR-1..CR-5 cannot be injected. Use it as the
**second family for the alternative-translation probe** (D13), not as the primary translator.

**Existing-data repair.** This covers the ≈21.3k rows of Vi-GSM8K train/test and Vi-MATH train/test. It needs
about 21k G2 calls + about 4k G3 calls + about 1.5k items through the repair loop, which is ≈ 30k NIM calls
(≈12.5 key-hours). Claude adjudication of the eval-set flags is ≈300–450 calls ≈ $5–7. These figures are
consistent with the $25–30 allowed for eval repair in `02_models` §8.4.

---

## 4. Vietnamese math glossary seed (60 entries)

**Columns.** *Req* is the required Vietnamese term. *Forbid* is the forbidden rendering (G1 fails if *Forbid* appears
while the trigger fired).

**Sources.**
- **R1, R2, R6:** a measured defect in this project.
- **W:** the Vietnamese Wikipedia page, opened this session:
  - **W-CSN** [Cấp số nhân](https://vi.wikipedia.org/wiki/Cấp_số_nhân)
  - **W-ĐT** [Đa thức](https://vi.wikipedia.org/wiki/Đa_thức)
  - **W-TV** [Số trung vị](https://vi.wikipedia.org/wiki/Số_trung_vị)
  - **W-TG** [Tam giác](https://vi.wikipedia.org/wiki/Tam_giác)
  - **W-NTCN** [Số nguyên tố cùng nhau](https://vi.wikipedia.org/wiki/Số_nguyên_tố_cùng_nhau)
- **Wk:** [vi.wiktionary "tá"](https://vi.wiktionary.org/wiki/tá).
- **SGK [UNVERIFIED]:** standard Vietnamese textbook usage that I did not open a page for. A Vietnamese math teacher must confirm these before glossary v1.0 is frozen.

| # | EN trigger (context) | Req (VI) | Forbid | Source |
|---|---|---|---|---|
| 1 | dozen | tá (1 tá = 12) | chục | R6 (1,582 rows), Wk |
| 2 | half a dozen | nửa tá | nửa chục | R6, Wk |
| 3 | median (statistics: data, scores, list) | số trung vị | trung bình | R6 303780, W-TV |
| 4 | median (triangle, segment from a vertex) | đường trung tuyến | trung vị | W-TG |
| 5 | mean / average | trung bình cộng | — | W-TV |
| 6 | mode (statistics) | mốt | kiểu, chế độ | W-TV ("số yếu vị"); SGK [UNVERIFIED] |
| 7 | range (statistics) | khoảng biến thiên | phạm vi | W-TV |
| 8 | quartile / interquartile range | tứ phân vị / khoảng tứ phân vị | — | W-TV |
| 9 | congruent (figures, segments, angles) | bằng nhau (hai tam giác bằng nhau) | đồng dạng | R2 (11 test problems), W-TG |
| 10 | non-congruent | không bằng nhau | không đồng dạng | R2 (4 counting problems) |
| 11 | similar (figures) | đồng dạng | — | W-TG |
| 12 | congruent (modular, ≡, mod) | đồng dư | đồng dạng, bằng nhau | SGK [UNVERIFIED] |
| 13 | common ratio | công bội | tỉ số chung | R2 (26 rows), W-CSN |
| 14 | geometric sequence / progression | cấp số nhân | dãy hình học | W-CSN |
| 15 | arithmetic sequence | cấp số cộng | dãy số học | W-CSN (see also) |
| 16 | common difference | công sai | hiệu chung | SGK [UNVERIFIED] |
| 17 | relatively prime / coprime | nguyên tố cùng nhau | tương đối nguyên tố | R2 (37/126 rows), W-NTCN |
| 18 | greatest common divisor / gcd | ước chung lớn nhất (ƯCLN) | — | R2, W-NTCN |
| 19 | least common multiple / lcm | bội chung nhỏ nhất (BCNN) | — | R2 |
| 20 | monic (polynomial) | (đa thức) monic / có hệ số cao nhất bằng 1 | đơn thức | R2 (11 rows), W-ĐT |
| 21 | monomial | đơn thức | — | W-ĐT |
| 22 | degree (of a polynomial) | bậc | độ | W-ĐT |
| 23 | degree (angle, °) | độ | bậc | W-ĐT / context |
| 24 | root / zero (of a polynomial or equation) | nghiệm | gốc, số không | W-ĐT |
| 25 | coefficient | hệ số | — | W-ĐT |
| 26 | leading coefficient | hệ số cao nhất | hệ số dẫn đầu | SGK [UNVERIFIED] |
| 27 | constant term | hệ số tự do / số hạng tự do | — | SGK [UNVERIFIED] (W-ĐT: "phần tử hằng") |
| 28 | slope (of a line) | hệ số góc | độ dốc | R2; SGK [UNVERIFIED] |
| 29 | y-intercept | tung độ gốc | — | SGK [UNVERIFIED] |
| 30 | kite (quadrilateral) | hình diều / tứ giác có hai cặp cạnh kề bằng nhau | hình thoi | R2; [UNVERIFIED] |
| 31 | rhombus | hình thoi | — | SGK [UNVERIFIED] |
| 32 | trapezoid | hình thang | — | SGK [UNVERIFIED] |
| 33 | counterclockwise | ngược chiều kim đồng hồ | (theo) chiều kim đồng hồ | R2 (16 rows) |
| 34 | clockwise | theo chiều kim đồng hồ | ngược chiều | R2 |
| 35 | improper fraction | phân số không thực sự | phân số không đúng quy cách | R2 |
| 36 | remainder | số dư | phần còn lại | R6 |
| 37 | factor / divisor (of an integer) | ước | nhân tố | R6 |
| 38 | prime factor | ước nguyên tố | — | R6 / SGK [UNVERIFIED] |
| 39 | factor (verb, polynomial) | phân tích thành nhân tử | — | SGK [UNVERIFIED] |
| 40 | cross product | tích có hướng | tích chéo | R6 |
| 41 | hypotenuse | cạnh huyền | — | W-TG |
| 42 | altitude / height (triangle) | đường cao | độ cao | W-TG |
| 43 | angle bisector / perpendicular bisector | đường phân giác / đường trung trực | — | W-TG; R6 Hungarian ("trung trực vuông góc" is redundant) |
| 44 | centroid / orthocenter | trọng tâm / trực tâm | — | W-TG |
| 45 | incircle / circumcircle | đường tròn nội tiếp / ngoại tiếp | — | W-TG, R2 |
| 46 | isosceles / equilateral / right triangle | tam giác cân / đều / vuông | — | W-TG |
| 47 | palindrome | số đối xứng (palindrome) | leaving the English word alone | R2; [UNVERIFIED] |
| 48 | "N of the remaining X" | N trong số X còn lại | N X còn lại | R1 (4/4 occurrences wrong), CR-2 |
| 49 | "N more X than Y" | nhiều hơn Y N X / … hơn số … của Y là N | nhiều hơn N X (no referent) | R1, R6, CR-1 |
| 50 | "N more than k times" | nhiều hơn k lần số … là N | nhiều hơn N lần | R1 2597, CR-3 |
| 51 | strictly (less/cheaper) | (thực sự) nhỏ hơn / rẻ hơn | đáng kể | R1 529, CR-5 |
| 52 | "b subtracted from a" | lấy a trừ đi b | b trừ từ a / a được trừ từ b | R6 ASDiv 1315, CR-4 |
| 53 | "a divides b" | a là ước của b / b chia hết cho a | a chia hết cho b | R2 c&p 323, CR-4 |
| 54 | yard / square yard / cubic yard | yard / yard vuông / yard khối | thước, mét, mét vuông, mét khối | R1 6922, R2 387/1000, R6 26489 |
| 55 | foot (feet) / inch / mile | foot / inch / dặm | mét, thước, cm, km; leaving "miles" in English | R1 test 445, R2 |
| 56 | gallon / quart / ounce / pound (weight) | gallon / quart / ounce / pound | lít, gam, kg | R2 c&p 444, R6 391354 |
| 57 | penny / nickel / dime / quarter (coin) | đồng 1 xu / 5 xu / 10 xu / 25 xu | leaving the coin name in English | R1 X-CULTURE |
| 58 | $N (prose) | N đô la, or `\$N` inside math | an unescaped `$` | R2 (25 rows) |
| 59 | twice / k times as many | gấp đôi / gấp k lần | — | R1 |
| 60 | LaTeX and asy keywords: `\boxed \angle \sec \parallel \right \times \text`, asy `pair dir label dot real` | **never translated** (masked) | `\đóng hộp`, `\góc`, `\giây`, `thư mục(`, `nhãn(`, `dấu chấm(` | R6 (5,048 + 2,797 rows) |

Ship the glossary as `glossary_vi_math.yaml` (versioned). Each entry stores `en_regex`, `context_regex`, `req`, `forbid`
and `source`. The `forbid` lists make G1 a pure string check.

---

## 5. Reasoning language and localization as experimental factors

### 5.1 What the evidence says at ≤4B

- **Forcing Vietnamese thinking with prompts or a little SFT hurts small models.**
  - R1-Distill-1.5B on MGSM loses 6.8 pp [O 2505.22888].
  - 100-example target-language SFT collapsed a 7B model to 5.5% on MGSM (Telugu).
  - M-Thinker's cold-start SFT took the 1.5B model on vi from 36.5 to 12.8.
- **Getting consistent Vietnamese CoT without loss takes RL with a language-consistency reward.** M-Thinker at 1.5B on held-out vi reached LC 8→99 with acc 36.5→37.7. At 4B, acc fell 77.8→73.3. That RL used 8×H20 plus a 2×8 H20 judge, far beyond this project's compute.
- **English thinking is the accuracy-optimal SFT target.** MMATH found EN-Think +5.3 over Native-Think (32B, 3k examples, includes vi). The penalty is answer-language drift: the 7B model answers vi questions in English.
- **SFT on translated data barely differs by language.** At 3B, SFT averages are 45.2 for English against 46.3–47.6 for non-English. **RL on target-language translated data beats RL-en by 3.3–8.8 pp** [O 2509.23657]. So translated Vietnamese problems are most valuable in the RLVR stage, provided they are gated (§3.5), because RL rewards only the answer and a broken problem statement poisons the reward.
- **Translate-test matters for low-resource languages and little for high-resource ones.** Luar gave +1.3 to +1.9 for ar and ko but +80 for zu, at 4B. Vietnamese is mid/high-resource for Qwen-family models: R1-Distill-1.5B scores vi 36.9 vs en 45.4 on MMATH. So translate-test is expected to give a modest gain [EST] and belongs only as a reference row.
- **Code-switched or language-mixed CoT** showed +18.6 for Korean models of 4B and up [O abstract]. That required a 3.7M-trace corpus, and PCS evidence is at 8B. Nothing exists at ≤2B or for vi.

### 5.2 Conditions and whether they are worth the compute

| Condition | Put it in? | Why | Expected effect [EST from the cited numbers] | Cost |
|---|---|---|---|---|
| **VI-CoT** (Vietnamese question, reasoning and answer). Default for every C2 arm | **Yes, main grid** | Matches the product and the thesis claim, and is the only condition comparable across arms | baseline | in grid |
| **EN-CoT** (Vietnamese question, English reasoning, Vietnamese final line "Đáp án: …"). Uses the original English GSM8K/MATH solutions directly, so no translation of solutions is needed | **Yes, one ablation**: best arm × {0.8B, 2B}. Report Acc, answer-LC and tokens | Tests the strongest literature claim at sizes where nobody has measured vi | +2 to +7 pp accuracy (MMATH +5.3; inverse of Qi's −6.8 at 1.5B). Answer-LC drops unless the final line is templated. About −25% output tokens (VI fertility ≈1.3×) | 2 SFT runs (≈ 2 grid cells) |
| Mixed / code-switched CoT (30% of steps in Vietnamese) | **No** | No ≤4B or vi evidence, needs custom trace construction, and adds a third level to every analysis | unknown | — |
| M-Thinker-style LC + alignment RL | **No** (cite as future work) | Needs a judge model during RL plus multi-iteration RL. Beyond Modal $30 and Kaggle | LC ≈99% at ≈ ±1 pp at 1.5B | ≫ budget |
| Translate-test (MT the Vietnamese test into English, solve in English) | **Evaluation reference row only**, for the best checkpoint per scale | Cheap. Shows whether Vietnamese-language understanding is the bottleneck | small for vi [EST] | ≈1.3k + 0.5k MT calls once |
| Localization as a training factor | **No** | Fine-tuning gains from localized data are small (≤3.6 pp, 2508.14913). Converting units or currency changes numbers and risks the gold (R1/R2 swaps) | ≤1 pp [EST] | — |
| **Localization probe** (300 Vi-GSM8K-test items: English names → Vietnamese names under a consistent map; "$" → "đô la"; no number changes) | **Yes, evaluation only** | The literature shows small models are the most sensitive (−5.9 pp at 8B). Cheap translationese/robustness evidence for C3 | 0–6 pp drop for small students | ≈600 calls (rewrite + G1/G3), $0 |

**Language consistency metrics (all runs).**
- `LC_think` and `LC_answer`: the share of the trace and the final line detected as Vietnamese by fastText-lid on text with math masked.
- `LC&Acc`, as in M-Thinker.
- Report these next to accuracy, so that "accuracy bought with English" is visible.

### 5.3 Tokenizer (D12)

Do not adapt the tokenizer. Reasons:
- Every candidate student (Qwen3.5 and Gemma-4) has a large multilingual vocabulary. The ~150k rule [S] says Vietnamese fertility is then near English.
- Vocabulary expansion only pays off with continued pre-training, which costs GPU-days.

Instead, measure on CPU once per student:
- tokens per word on Vi-GSM8K test vs GSM8K test;
- tokens per gold solution.

Set `max_new_tokens(VI) = ceil(ratio × max_new_tokens(EN))`, so that Vietnamese CoT is not truncated more often than
English CoT. Truncation would masquerade as a reasoning deficit. Report tokens per correct answer as an efficiency
covariate in the EN-CoT ablation.

---

## 6. Fair evaluation: native vs translated (D13)

1. **Verified translated sets.** Report on Vi-GSM8K-test-v1.1 and Vi-MATH-test-v1.1, built per §3.8, together with the
   residual error bound from a blind 300-item draw. Rankings survive translation noise but absolute numbers drop
   1–4.6 pp [O 2605.24904]. So VI-vs-EN *gaps* are reported only on the verified sets, and only on the
   **English-solvable subset** as a control: items each model solves in English.
2. **Grader.** Use a VN-locale-aware number parser (decimal comma, thousands dot), strip `\text{độ}` and other
   `\text{}` units, and take the gold from the English source (R2 §5). MGSM-Rev2 shows that a parser difference
   alone can be worth 10 pp.
3. **Alternative-translation probe.** Take 300 stratified Vi-GSM8K test items plus 300 Vi-MATH-500 items.
   - Produce translation B with a second family: TranslateGemma-4B on a T4, with masking, plus G0–G3.
   - Score every model on A and B.
   - If any paired |Acc_A − Acc_B| exceeds its McNemar 95% interval, translation variance is material, and C2 effects
     smaller than it must be reported as within translation noise.
   - Cost: ≈1 T4-hour + ≈1.2k NIM gate calls.
4. **Translationese probe.**
   - Compare each model's rank on Vi-ExamMath-Core (native) against its rank on Vi-GSM8K-v1.1 (translated), using Kendall τ, and compare accuracy deltas between the C2 arms.
   - The research question needs exactly this: whether the arm ranking flips on native items. Compute it on bootstrap pairs of items.
   - Native-vs-translated gaps show up most on generative tasks when the model is strong [S 2406.12822], so expect the effect at 2–4B more than at 0.8B.
5. **Contamination.** Vi-ExamMath items never go to NIM or to the Gemini free tier (§1.6). Solver QA for
   Vi-ExamMath uses Claude (Batch) or self-hosted models only, in line with `02_models` §8.

---

## 7. Alternatives compared (one choice per decision)

| Decision | Option A | Option B | Option C | **Choice** and justification |
|---|---|---|---|---|
| D1 Protection | Prompt-only "keep LaTeX". This is the old pipeline and mAceReason's approach | Mask math, code and annotations; translate `\text{}` payloads as sub-segments; leave prose numbers unmasked but check them | Also mask every number | **B.** A gave 211 + 5,048 corrupted rows here (R2, R6). C breaks Vietnamese classifier and word order, and number errors are already rare (12/7,473), with G1 catching them. LaTeXTrans reports 97% compile success with placeholders [S] |
| D2 Unit and format | Batched JSON array from the LLM (old) | One item per call, tagged plain text, the program writes JSON | Batched tagged text | **B.** It removes both the escape corruption and the misalignment classes by construction. Cost: more calls, but NIM is free |
| D3 Translator | NIM `glm-5.3-flash` | TranslateGemma-4B (local T4) | Claude Opus 5.5 Batch | **A**, confirmed by a 60-item bake-off against DeepSeek-v4.1-flash and Gemma-4-31B. $0, fast, follows glossary and tag instructions. B cannot take a glossary (fixed prompt, 2K context), so it is used only for the alternative-translation probe. C costs ≈ $140 per 10k with Opus, so it is reserved for escalation. EnViT5 (2022, T5-base) cannot handle LaTeX or long items and is rejected |
| D4 Terminology | None | Full glossary in every prompt | Triggered entries + CR templates + deterministic forbid/require check | **C.** Defects are term-specific (§2). B dilutes the prompt; only C is verifiable |
| D5 QE gate | Neural QE only (MetricX, CometKiwi) | Back-translation + English-side comparison | Layered G0 → G1 → G2 solve-check → G3 LLM MQM spans → G4 MetricX tie-break | **C.** The damaging errors are fluent and number-preserving (R1), so neural QE is weak at the span level (xCOMET 0.28 vs LLM 0.55 F1 [O]). G2 tests exactly what training needs, namely that the problem text supports the gold. MetricX is Apache-2.0, while CometKiwi/xCOMET are NC-SA and gated |
| D6 Thresholds | Literature fixed cut-off | Percentile (keep top X%) | Calibrated on the 1,050 labelled items to recall ≥0.9 at a ≤15% flag rate | **C.** No transferable cut-off exists [S]. C also yields a publishable QE-vs-human table for C3 |
| D7 Repair | Retranslate from scratch | Self-refine with external feedback (≤2 rounds) → escalate → drop or human | Human post-edit of every flag | **B**, following the mAceReason and MGSM-Rev2 practice. Humans only for eval-set residue |
| D8 Solutions | Translate the English solution | Regenerate a VI solution with the shared teacher, accepted only if the answer equals the gold | Both, as SFT augmentation | **B** for the translated arm, so the C2 arms differ only in problem source, not solution style. A is kept as an ablation, and it is also the EN-CoT condition's data |
| D9 Localization | None | Light: names, currency word; units transliterated | Full: convert units and VND amounts | **B** for training text, plus a 300-item evaluation probe. C changes numbers and caused gold errors (R1/R2/R6). Localization gains from fine-tuning are ≤3.6 pp [O] |
| D10 Reasoning language | VI-CoT only | VI-CoT main + EN-CoT ablation at 0.8B/2B | + code-switched + LC-RL | **B.** The strongest effect in the literature is the +5–7 pp EN advantage, never measured for vi at ≤2B. C lacks ≤4B evidence and is beyond the compute budget |
| D11 Transfer | Translate-test as the method | Translate-train SFT + RLVR on gated VI problems; translate-test as a reference row | Layer-swap merge (VI language expert ⊕ EN math expert) | **B.** RL-non-en beats RL-en by 3.3–8.8 pp at 3B [O]. C is only validated at 8B and its layer ranges do not transfer; it is a stretch goal (one extra SFT run for a VI-instruction expert plus a CPU merge) |
| D12 Tokenizer | Vocabulary expansion + continued pre-training | Measure-only + token-budget scaling | Switch to a VI-native model (Sailor2/VinaLLaMA) | **B.** A needs GPU-days. C is already covered by the optional Sailor2 control in `02_models` |
| D13 Eval fairness | Raw translated sets | Verified v1.1 + English-solvable control + VN-aware grader | B + alternative-translation probe + native-vs-translated rank test | **C.** ≈1 T4-hour + ≈1.2k calls. It directly answers the research question's "does the ranking change on native items" |

---

## 8. Execution order and compute (fits the free stack)

1. **Day 0, about 2 h.**
   - Write `glossary_vi_math.yaml` (§4) and the masker/unmasker with unit tests. The tests should cover R2's worst rows: ia-798 `\right`, test ia-760 `\n`, asy test pc-52, and `\$18.09`.
   - Write G0/G1.
2. **Day 0, about 3 h.**
   - Run the bake-off of 3 translators on 60 items: 20 GSM8K with CR triggers, 20 MATH with LaTeX-heavy text, 20 MetaMath with dozen/median/asy.
   - Run threshold calibration on the 1,050 labelled items (≈2k NIM calls, plus MetricX on a Kaggle T4 in about 10 min).
3. **Day 1.**
   - Repair the existing data per §3.8 (≈30k NIM calls; 3 keys ≈ 4–6 h wall).
   - Run Claude Batch adjudication of the eval-set flags (≈$5–7).
   - Do 4–6 h of human review, then freeze v1.1.
4. **Day 1–2.**
   - Translate the new train items needed by the translated arm through MTA-vi. At ≈40k calls per 10k items, budget about 5.5 h wall per 10k with 3 keys.
   - Run the RLVR pool through the same gate.
5. **Evaluation-only probes:** localization (≈600 calls), alternative translation (≈1 T4-h), translate-test row
   (≈1.8k calls), and fertility measurement (CPU).

**GPU use.** Total GPU use by this report's methods is ≈2–4 T4-hours (MetricX, plus TranslateGemma for the probe),
which costs $0 on Kaggle or ≈ $2–3 on Modal L4. Claude spend is ≈ $10–15 of the $100. All MT, QE and solve calls run on NIM.

**Risks.**
- NIM model retirement mid-run. Store `prov.mt_model` per item. MTA-vi is model-agnostic; re-run the bake-off if a model goes EOL.
- The free-tier limits for Gemini are unverified, so J2 falls back to Claude Sonnet Batch at ≈ $0.007 per call.
- SGK-term entries marked [UNVERIFIED] need a one-hour teacher check before glossary v1.0.
