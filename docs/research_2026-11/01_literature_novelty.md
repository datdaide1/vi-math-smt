# 01 — Literature & Novelty Re-check (as of 2026-11-07)

**Scope.** Re-validate the novelty of the plan in `RESEARCH_PLAN.md` (§0, §1, §3, §14; written Sept 2026)
against work published through early Nov 2026, and propose re-framed research questions that are novel,
feasible on free compute (4 GB RTX 3050 Ti laptop + free hosted APIs + small Modal credit), and suitable for
a complete hand-off research package.

**Method.** (1) arXiv full-text listing searches sorted by date (`arxiv.org/search`, queries: "Vietnamese
mathematics", "Vietnamese math", "Vietnamese reasoning", "Vietnamese exam", "Vietnamese benchmark LLM",
"Vietnamese synthetic data", "low-resource language mathematical reasoning", "math problem synthesis
verified", "symbolic data augmentation mathematical reasoning"); (2) general web search (standard +
extended); (3) Hugging Face dataset search ("vietnamese math", sorted by modified); (4) ACL Anthology via
web search; (5) every paper in the table below had its arXiv abstract page (and, where noted, HTML full
text) opened in this session.

**Evidence caveats (read before relying on numbers).**
- Pages were read through a summarizing fetch tool. Abstract-level claims are reliable; numbers quoted from
  full-text HTML (marked "HTML") came from a summary of the full text and should be re-checked against the
  PDF before being cited in a paper.
- arXiv listing search is keyword-based; a paper that never uses the words "Vietnamese"/"math" together in
  its metadata could be missed. Non-arXiv venues (VLSP 2026 proceedings, Vietnamese journals) are only
  partially covered (see §6).
- Anything not opened this session is marked **[UNVERIFIED]**.

---

## 1. Key papers (opened/verified this session)

Legend — Rel.: relevance to this thesis. **C1** = Vi-ExamMath benchmark; **C2** = symbolic-vs-LLM augmentation
comparison; **C3** = failure analysis of naive mutation; **NEW** = bears on a re-framed RQ (§5).

### 1A. Math data synthesis & verification

| # | Paper | arXiv / URL | Date (v1; venue) | One-line finding | Rel. |
|---|---|---|---|---|---|
| A1 | Li et al., *Neuro-Symbolic Data Generation for Math Reasoning* | [2412.04857](https://arxiv.org/abs/2412.04857) | Dec 2024; NeurIPS 2024 | SMT-LIB formalize→mutate→informalize; solver guarantees validity; LLaMA-2/Mistral beat prior SOTA. | S1 lineage (C2/C3) |
| A2 | Seegmiller et al., *FLAMES: fine-grained analysis of the data synthesis pipeline* | [2508.16514](https://arxiv.org/abs/2508.16514) | Aug 2025; EMNLP 2025 | Controlled study of 10 synthesis strategies + 6 quality-control strategies; **at fixed budget, problem coverage matters more than keeping only reliably-solved problems**; complexity-increasing agents best. Qwen2.5-Math-7B. | **Partial scoop of RQ-B** (EN, 7B) |
| A3 | Yeo et al., *Adaptive Problem Generation via Symbolic Representations* | [2602.19187](https://arxiv.org/abs/2602.19187) | Feb 2026; preprint | Seeds (GSM8K/MATH) → SymPy code → LLM modifies code → SymPy computes answer → LLM renders NL; compared with **NL-only LLM modification** baselines. Students Qwen2.5-1.5B/3B-Instruct, **RLVR (PPO/GRPO)**. HTML: 1.5B GRPO seed 51.4 → Opt-Sym 54.7; at 3B NL ≈ Sym (Opt-NL 63.1 vs Opt-Sym 62.5 PPO); **SMT-LIB renderings are unnatural and perform worse than SymPy and NL**; Sym diversity 0.37 vs NL 0.07. | **Closest scoop of C2 and part of C3** (EN, RL, not SFT) |
| A4 | Li et al., *LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters!* | [2502.07374](https://arxiv.org/abs/2502.07374) | Feb 2025 | Long-CoT SFT (Qwen2.5-32B): training on samples with **incorrect answers costs only 3.2 pts**; breaking step structure hurts a lot. | Undercuts value of a correctness certificate (C2) |
| A5 | Chandra et al., *Shape of Thought: When Distribution Matters More than Correctness* | [2512.22255](https://arxiv.org/abs/2512.22255) | Dec 2025 | Synthetic CoT ending in **wrong** answers beats human-written CoT on MATH/GSM8K; distribution closeness matters; 1.5B–9B (Qwen/Llama/Gemma). | Same; also argues translated human data may be "far" from model distribution (NEW) |
| A6 | Wang et al., *RV-Syn* | [2504.20426](https://arxiv.org/abs/2504.20426) | Apr 2025; EACL 2026 | Function library → executable computational graphs → back-translate to problems; verifiable; beats prior synthesis. | Code-verified synthesis precedent |
| A7 | Yu et al., *MathAgent: adversarial evolution of constraint graphs* | [2604.11188](https://arxiv.org/abs/2604.11188) | Apr 2026; ACL 2026 Findings | Constraint-graph blueprints → NL instantiation; **1K samples beat LIMO/s1K** across 8 benchmarks, 10 models. | Structured synthesis precedent |
| A8 | Zhao et al., *PromptCoT 2.0* | [2509.19894](https://arxiv.org/abs/2509.19894) | Sep 2025 | EM-loop prompt synthesis; Qwen2.5-7B SFT **only on synthetic prompts beats human-written/hybrid data**. | LLM synthesis is strong (C2) |
| A9 | Shen et al., *Let's Verify Math Questions Step by Step* (MathQ-Verify) | [2505.13903](https://arxiv.org/abs/2505.13903) | May 2025 | Question-validity verification; ValiMath (2,147 human-verified); +25 F1 over direct verification. | Ill-posed synthetic questions (C2 data-quality panel) |
| A10 | Duan et al., *VeriGeo* | [2606.14176](https://arxiv.org/abs/2606.14176) | Jun 2026 | Seed-based rewriting is "weakly verifiable"; 3-stage numerical/analytical/global verification; raw generations often fail. | Verification-vacuity theme (C3) |
| A11 | Huang et al., *From Accuracy to Robustness: Rule- and Model-based Verifiers in Math* | [2505.22203](https://arxiv.org/abs/2505.22203) | May 2025; EMNLP 2026 | Rule-based verifiers → false negatives on equivalent formats; model-based verifiers → false positives / reward hacking. | Answer-checking design (C1 scoring, arms S2/S3) |
| A12 | Liu et al., *CompassVerifier* | [2508.03686](https://arxiv.org/abs/2508.03686) | Aug 2025 | Lightweight verifier model + VerifierBench for multi-format answers. | Candidate free answer-matcher for C1 |

### 1B. Small models (≤4B): SFT / distillation / RLVR, data efficiency

| # | Paper | arXiv / URL | Date (v1; venue) | One-line finding | Rel. |
|---|---|---|---|---|---|
| B1 | Li et al., *Small Models Struggle to Learn from Strong Reasoners* | [2502.12143](https://arxiv.org/abs/2502.12143) | Feb 2025 (v3 Nov 2025) | **≤3B models don't reliably benefit from long-CoT / large-teacher distillation**; short CoT and Mix Distillation work better. | Trace-format design for 0.6–2B |
| B2 | Zeng et al., *SimpleRL-Zoo* | [2503.18892](https://arxiv.org/abs/2503.18892) | Mar 2025; COLM 2025 | Zero-RL on 10 bases incl. Qwen2.5-0.5B/1.5B; format reward + difficulty control matter. | RL feasibility on tiny models |
| B3 | Kim et al., *RL vs. Distillation: Accuracy and Capability* | [2505.14216](https://arxiv.org/abs/2505.14216) | May 2025 | RLVR raises pass@1, not pass@k; distillation improves capability only when new knowledge is introduced. | SFT-vs-RL framing |
| B4 | Shao et al., *Spurious Rewards* | [2506.10947](https://arxiv.org/abs/2506.10947) | Jun 2025 | Random rewards give Qwen2.5-Math-7B +21.4 on MATH-500 (vs +29.1 true reward); effect is Qwen-specific. | RL gains on Qwen are unreliable evidence |
| B5 | Wu et al., *Reasoning or Memorization? Unreliable RL Results due to Data Contamination* | [2507.10532](https://arxiv.org/abs/2507.10532) | Jul 2025; AAAI 2026 | Qwen2.5 contaminated on popular benchmarks; on a leakage-free generator only accurate rewards help. | **Motivates contamination-clean eval (C1)** |
| B6 | Wang et al., *RL with One Training Example* | [2504.20571](https://arxiv.org/abs/2504.20571) | Apr 2025 | 1 example RLVR: Qwen2.5-Math-1.5B MATH-500 36.0→73.6. | Cheap RL is possible but (cf. B4/B5) contamination-sensitive |
| B7 | Wang et al., *Tina: Tiny Reasoning Models via LoRA* | [2504.15777](https://arxiv.org/abs/2504.15777) | Apr 2025 | LoRA-RL on 1.5B, ~US$9 total; LoRA mainly teaches reasoning format. | Free-compute RL feasibility |
| B8 | Xu et al., *VibeThinker-1.5B* | [2511.06221](https://arxiv.org/abs/2511.06221) | Nov 2025 | Diversity-exploring distillation + MaxEnt RL; 1.5B AIME24 80.3; US$7.8k. | SOTA small-model ceiling; leaderboard row |
| B9 | Xu et al., *Phi-4-Mini-Reasoning* | [2504.21233](https://arxiv.org/abs/2504.21233) | Apr 2025 | 3.8B: distilled mid-training → SFT → rollout-DPO → RLVR. | Small-model recipe reference |
| B10 | Ye et al., *LIMO* | [2502.03387](https://arxiv.org/abs/2502.03387) | Feb 2025; COLM 2025 | Few curated demos elicit reasoning if pretraining has the knowledge. | Data-efficiency framing |
| B11 | Muennighoff et al., *s1* | [2501.19393](https://arxiv.org/abs/2501.19393) | Jan 2025 | s1K (1k traces) + budget forcing on 32B. | Vi-S1K lineage |
| B12 | Armandpour et al., *Unmasking On-Policy Distillation* | [2605.10889](https://arxiv.org/abs/2605.10889) | May 2026 | No single OPD configuration is universally effective; depends on student capacity and task. | Distillation design caution |
| B13 | Qwen3 Technical Report | [2505.09388](https://arxiv.org/abs/2505.09388) | May 2025 | 0.6B–235B; small models built by leveraging flagship knowledge; 119 languages. | Base models |
| B14 | Qwen3.5-0.8B-Base (model card) | [HF](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | Feb–Mar 2026 | Exists; Apache-2.0; 201 languages; **natively multimodal** (vision encoder); Qwen3.5 small series = 0.8B/2B/4B/9B + Base. | **Plan's base-model choice (Qwen3-0.6B/1.7B) is one generation old** |

### 1C. Multilingual / low-resource math reasoning

| # | Paper | arXiv / URL | Date (v1; venue) | One-line finding | Rel. |
|---|---|---|---|---|---|
| C1 | Barua et al., *Long Chain-of-Thought Reasoning Across Languages* | [2508.14828](https://arxiv.org/abs/2508.14828) | Aug 2025; ICLR 2026 | 9 languages (zh, fr, ja, af, th, lv, mr, te, sw — **no vi**). En-CoT scales better than target-CoT; HTML: **SFT on machine-translated gold traces > target-language traces distilled from an LRM** (Qwen3-8B-Base, MATH-500 avg 78.3 vs 72.8). Evaluated on translated benchmarks. | **NEW (translated vs native data)** |
| C2 | Qi et al., *When Models Reason in Your Language* | [2505.22888](https://arxiv.org/abs/2505.22888) | May 2025; EMNLP 2025 Findings | Forcing thinking in the user's language improves readability but costs accuracy; 100-example post-training reduces mismatch. | Reasoning-language choice |
| C3 | *PolyMath* | [2504.18428](https://arxiv.org/abs/2504.18428); [HF](https://huggingface.co/datasets/Qwen/PolyMath) | Apr 2025; NeurIPS 2025 | 18 languages **incl. vi**, 500 items/lang, 4 difficulty levels; parallel expert-checked translations. | Existing (translated) Vietnamese eval |
| C4 | Dementieva et al., *PluraMath* | [2607.05992](https://arxiv.org/abs/2607.05992) | Jul 2026 | Extends PolyMath to 18 underrepresented languages; persistent gap; 27 LRMs. | Multilingual eval landscape |
| C5 | Han et al., *DeepSEA: Multilingual Evaluation Benchmark for SEA* | [2610.26155](https://arxiv.org/abs/2610.26155) | **19 Oct 2026** | 12 SEA languages incl. **vi**; math task = **MGSM professionally translated, 250 items/lang** (HTML); small models degrade on low-resource; pivoting through English helps reasoning. | **Invalidates plan §1.4 "MGSM has no Vietnamese split"**; another translated vi eval |
| C6 | Ogundepo et al., *Improving Multilingual Math Reasoning for African Languages* | [2505.19848](https://arxiv.org/abs/2505.19848) | May 2025 | HTML: Llama-3.1-8B, 30k each: **translated OpenMathInstruct 51.4 > native-language synthetic 39.4**; combined 57.7. **Evaluated on AfriMGSM = human-translated GSM8K**, GPT-4o judge. | **Key evidence for NEW RQ (translated-eval confound)** |
| C7 | Dhaliwal et al., *English is Not All You Need* | [2604.13286](https://arxiv.org/abs/2604.13286) | Apr 2026 | 220 SFT runs on parallel translated mixtures (≤8B): adding non-English languages helps, incl. English. | Multilingual SFT mixtures |
| C8 | Mofakhami et al., *Data Mixing for In-Language Reasoning* (Tiny Aya L2-Thinker) | [2609.10445](https://arxiv.org/abs/2609.10445) | Sep 2026 | 3.35B model reasons in prompt language >93% across 60 langs via data mixing; needs strong English reasoning base. | Reasoning-language for small models |
| C9 | Lim et al., *Native Multilingual CoT in Low-Resource SEA Languages* (OSCD) | [2608.00533](https://arxiv.org/abs/2608.00533) | Aug 2026 | "Cross-lingual collapse" to English; translator-in-the-loop distillation, up to 3.2× in native SEA languages (languages not listed in abstract). | SEA native reasoning |
| C10 | Enevoldsen et al., *Multilingual GSM-Symbolic* | [2610.03367](https://arxiv.org/abs/2610.03367) | Oct 2026 | 30k item-matched QA in 15 languages from symbolic templates; size/resource/typology explain 92% of between-language variance. | Template-based multilingual eval (vi membership unknown) |
| C11 | Xu et al., *MGSM-Pro* | [2601.21225](https://arxiv.org/abs/2601.21225) | Jan 2026; IJCNLP-AACL 2026 | 5 digit/name instantiations per MGSM item; low-resource languages drop sharply on digit variants. | Robustness slice design |
| C12 | Dipta et al., *GanitLLM* (Bengali) | [2601.06767](https://arxiv.org/abs/2601.06767) | Jan 2026; ACL 2026 Findings | Qwen3-4B, SFT → Curriculum-GRPO; +8/+6 on Bn-MGSM/MSVAMP; Bengali reasoning tokens 14%→88%. | **Closest "small model for one low-resource language" analogue** |
| C13 | Kishanthan et al., *LLMs for Math Education in Sinhala and Tamil* | [2602.14517](https://arxiv.org/abs/2602.14517) | Feb 2026; ITHET 2026 | Parallel set **independently authored** (not translated) to avoid translation confounds; complex reasoning degrades. | Native-authored eval precedent |
| C14 | Cho et al., *Knowing When to Teach* (TOPSD) | [2611.08179](https://arxiv.org/abs/2611.08179) | **5 Nov 2026** | Teacher drift in on-policy self-distillation appears earlier in lower-resource languages. | Very recent; distillation in LRLs |
| C15 | Yao et al., *Data Contamination Can Cross Language Barriers* | [2406.13236](https://arxiv.org/abs/2406.13236) | Jun 2024; EMNLP 2024 | Training on translated test sets inflates scores and evades overlap-based detection. | **Translated Vi-GSM8K etc. may be cross-lingually contaminated** |

### 1D. Vietnamese math benchmarks / datasets / models

| # | Paper / resource | arXiv / URL | Date | What it is (verified) | Overlap with C1 |
|---|---|---|---|---|---|
| D1 | *VNHSGE* | [2305.12199](https://arxiv.org/abs/2305.12199) | May 2023 | THPT graduation exam dataset, >19k questions, 9 subjects (MCQ). | THPT, MCQ |
| D2 | *V-Math* | [2509.12251](https://arxiv.org/abs/2509.12251) | Sep 2025; **v1 only** | Abstract: an **agentic framework** (question generator / solver / tutor) for THPT math; dataset details not in abstract. | THPT; tutoring-oriented, not a contamination-controlled benchmark |
| D3 | *Bridging the Reasoning Gap in Vietnamese with SLMs via Test-Time Scaling* (Vi-S1K / Vi-Elementary-Bench) | [2604.17794](https://arxiv.org/abs/2604.17794) | Apr 2026; FJICAI; **v1 only, no follow-up found** | Qwen3-1.7B; Vi-S1K localized by Gemini 2.5 Flash-Lite; elementary math; **LLM-as-judge** scoring (4.05/5 base); SFT "+77% explanation quality"; CoT+SC > ReAct. | Elementary; translated traces; judge-scored |
| D4 | *Measuring the Partial-Credit Gap* (THPT-Ladder) | [2608.18336](https://arxiv.org/abs/2608.18336) | Aug 2026 | 632 items, 21 official exams, **11 subjects**, Part-II true/false items under the 2025 rubric vs >1M human candidates. | Not math-specific; T/F |
| D5 | *VMMU* (ViExam) | [2508.13680](https://arxiv.org/abs/2508.13680) | Aug 2025 (v-latest Sep 2026); AACL 2026 Findings (per Anthology result) | ~2.5k multimodal MCQ, 7 domains incl. math. | Multimodal MCQ |
| D6 | Nguyen et al., *SiRC / ViMath-Bench, ViMath-InstructCode* | [ACL 2024.conll-1.20](https://aclanthology.org/2024.conll-1.20/) | 2024; CoNLL | CoT+code instruction tuning for Vietnamese math, <10B open models; ViMath-Bench (size/levels not in abstract; search snippet says 8.4K from 3 ed-platforms [UNVERIFIED detail]). Shares an author (Van-Vinh Nguyen) with D3. | Elementary/word problems; **same VNU-UET group as D3 → monitor** |
| D7 | DeepSEA (= C5) | [2610.26155](https://arxiv.org/abs/2610.26155) | Oct 2026 | Professionally translated **Vietnamese MGSM (250)**. | Translated grade-school |
| D8 | PolyMath-vi (= C3) | [HF](https://huggingface.co/datasets/Qwen/PolyMath) | 2025 | 500 translated vi items, 4 levels. | Translated, multi-level |
| D9 | HF "vietnamese math" datasets (18 listed, sorted by modified) | [HF search](https://huggingface.co/datasets?search=vietnamese%20math&sort=modified) | to Oct 2026 | Newest: `thang0102/vietnamese-math-qa` (**empty**), `hongduc05/vietnamese-math` (77.2k, MetaMath-style types GSM_AnsAug/FOBAR…, appears machine-translated, no card), `quin210/vietnamese-math-grpo-1k`, `ura-hcmut/Vietnamese-MATH` (12.5k), 5CD-AI translations. **None is native grade-10/HSG with solutions.** | No overlap |
| D10 | arXiv listings, Jul 2025 → 2 Nov 2026 ("Vietnamese reasoning", "…exam", "…benchmark LLM", "…synthetic data", "…mathematics") | arxiv.org/search | — | Only math-relevant Vietnamese items: D2, D3, D7. Others are legal, medical, VQA, agent, admin-reform (ViAdminShift 2611.03171), tutoring (DeepEdu-v1 2609.31568). **No Vietnamese math data-augmentation paper; no grade-10/HSG benchmark.** | — |

### 1E. Benchmark-construction practice (2025–2026)

| # | Paper | arXiv / URL | Date (venue) | One-line finding | Rel. |
|---|---|---|---|---|---|
| E1 | Pyeon et al., *LLMs on the 2026 Korean CSAT Math: Zero-Data-Leakage* | [2511.18649](https://arxiv.org/abs/2511.18649) | Nov 2025 | Digitized 46 items **within 2 h of release**; GPT-5 family 100, several >97, **gpt-oss-20B 95.7**. | **Temporal-split recipe; frontier saturation warning for C1-Core** |
| E2 | Dekoninck et al., *MathArena as an Evaluation Platform* | [2605.00674](https://arxiv.org/abs/2605.00674) | May 2026 | Static benchmarks insufficient; continuously refreshed comps; GPT-5.5 98% on USAMO 2026. | Refresh/temporal design |
| E3 | Cuclea et al., *RoMathExam* (Romanian exams 1895–2025) | [2604.16392](https://arxiv.org/abs/2604.16392) | Mar 2026; AIED 2026 | 10,592 problems, provenance JSON schema, topic tags, embeddings for variant dedup; LLM-agreement "solution-complexity" difficulty proxy (r>0.72). | **Direct template for C1 (national-exam archive genre already exists in other languages)** |
| E4 | Chandak et al., *Answer Matching Outperforms Multiple Choice* | [2507.02856](https://arxiv.org/abs/2507.02856) | Jul 2025 | Free-form + reference answer matching ≈ inter-human agreement; MCQ has shortcuts; reference-free LLM-judge aligns poorly. | Justifies free-response + reference matching (vs D3's judge, D1/D5's MCQ) |
| E5 | Vendrow et al., *Do LLM Benchmarks Test Reliability?* (platinum benchmarks) | [2502.03461](https://arxiv.org/abs/2502.03461) | Feb 2025 | Pervasive label errors hide failures; curated "platinum" versions. | Solver/human double-check of keys (C1) |
| E6 | Hofmann et al., *Fluid Language Model Benchmarking* | [2509.11106](https://arxiv.org/abs/2509.11106) | Sep 2025; COLM 2025 | IRT ability estimates increase validity; adaptive selection cuts variance; 50× fewer items on MMLU. | IRT discriminativeness analysis for C1 |
| E7 | Maia Polo et al., *tinyBenchmarks* | [2402.14992](https://arxiv.org/abs/2402.14992) | Feb 2024; ICML 2024 | ~100 curated items suffice to estimate MMLU performance. | Small benchmarks can be valid if curated |

**Seen only in search snippets, not opened → [UNVERIFIED]:** majority-vote pseudo-label studies
(arXiv 2609.18587, 2605.24998, 2608.06296), T³RL tool-verified voting (2603.02203), RESTRAIN (2510.02172),
AgenticMath ablation (2510.19361), S³-R1 "unverified data performs worse" (2605.01248), "Learning When to
Translate" Luar (2606.02465), ReasonXL (2604.12378), Multi-Aspect Question Reformulation (ICLR 2026,
2601.20614), GSM-Symbolic (2410.05229, cited in plan), MCLM, MetaMath/DART-Math/MathGenie (plan §1.1),
Qwen3.5 small-series release date 2 Mar 2026 (news sites only).

---

## 2. What the literature says, by theme (synthesis)

**2.1 Synthesis & verification.** The 2025–26 consensus is that *problem coverage, difficulty and diversity*
dominate, and *trace-level correctness matters much less than assumed* for SFT: FLAMES (A2) finds coverage >
solution reliability at a fixed budget; A4 finds wrong-answer traces cost ~3 pts; A5 finds wrong-answer
synthetic CoT beats human CoT. Structured/executable synthesis (A6, A7) wins mainly via *controllable
structure and diversity*, not via a correctness certificate. The one direct symbolic-vs-NL comparison (A3)
finds symbolic (SymPy) modification gives more diverse problems and modest RLVR gains at 1.5B, parity at 3B,
and **SMT-LIB renderings are unnatural and worse**. Verifier reliability is itself a research problem (A11):
rule-based checks under-credit, model-based checks get hacked.
→ **Prediction for the plan's C2:** LLM augmentation (S2/S3) ≥ symbolic (S1) on downstream accuracy; the
"verification premium" (S1 vs S4, S2 vs S3) will be small. The plan's hypothesis space is therefore largely
pre-answered in English.

**2.2 Small models.** ≤3B students learn worse from long-CoT strong-teacher traces (B1) → short CoT is the
right default at 0.6–2B. RLVR on tiny models is cheap (B6, B7: ~$9 LoRA-RL) but its gains on Qwen are
confounded by contamination and spurious-reward effects (B4, B5) — **which makes a contamination-controlled
native benchmark scientifically valuable as an instrument**, independent of augmentation. Qwen3.5 small
(0.8B/2B, Base available, 201 languages, B14) supersedes Qwen3-0.6B/1.7B as the obvious 2026 base.

**2.3 Multilingual.** Three converging findings: (i) reasoning in English is usually more accurate than in the
target language (C1, C2, C5), though small-model in-language reasoning can be trained (C8, C12);
(ii) **translated training data beats natively generated/distilled target-language data** (C1: 78.3 vs 72.8;
C6: 51.4 vs 39.4); (iii) **those comparisons were evaluated on translated benchmarks** (C1: translated
MATH-500/AIME/MMLU-ProX; C6: AfriMGSM, human-translated GSM8K). Native-authored evaluation exists only as
eval-only studies (C13, E3). Translated test sets can also be cross-lingually contaminated (C15).
→ **Open gap:** nobody has tested whether "translated > native" survives evaluation on *native-authored*
problems. This gap is exactly where the thesis already has assets (translated corpora + a native exam
benchmark in construction).

**2.4 Vietnamese.** No new Vietnamese math benchmark or Vietnamese math-training paper appeared Sept–Nov 2026
on arXiv; the only new Vietnamese math evaluation data are *translated* (DeepSEA-vi MGSM, Oct 2026;
PolyMath-vi, 2025). Grade-10 entrance / HSG free-response remains unclaimed.

**2.5 Benchmark practice.** Best practice in 2026: free-response + reference answer matching (E4), label
auditing (E5), provenance + dedup schema (E3), temporal post-cutoff splits (E1, E2), IRT-based
discriminativeness/validity (E6, E7). Frontier models are saturating national school exams (E1: ≥95% even
for a 20B open model), so a grade-10 Core split will be **discriminative for small/mid models only**.

---

## 3. Plan statements that are now outdated or need correction

| Plan location | Statement | Status (evidence) |
|---|---|---|
| §1.4 | "MGSM has **no Vietnamese split**." | **Outdated.** DeepSEA (2610.26155, Oct 2026) provides a professionally translated Vietnamese MGSM (250); PolyMath has vi (500). Both are *translated*, which strengthens the native-vs-translated framing. |
| §1.4 | "There is **no** verified/symbolic math-data-augmentation study for any low-resource language." | Still true as far as found, but the English symbolic-vs-NL comparison now exists (A3, 2602.19187) and must be cited. |
| §1.4 table / §14 | V-Math = "MCQ + free-response, human worked solutions, agentic solver". | Abstract only confirms an **agentic generator/solver/tutor framework**; dataset composition **[UNVERIFIED]** this session — re-check PDF before writing related work. |
| §1.4 table | Vi-S1K: "monitor scoop risk". | No v2 / follow-up found (2026-11-07). Note it is **LLM-judge-scored** (D3), a concrete point of methodological contrast. |
| §0 / §6.0 | Base models Qwen3-0.6B/1.7B-Base. | Qwen3.5-0.8B/2B-Base exist (B14); reviewers in 2027 will ask why not. 2B exceeds the "≤1.7B" hard rule — decision needed (0.8B on laptop is likely feasible with QLoRA; multimodal wrapper may complicate tooling — **[UNVERIFIED]** Unsloth/vLLM support). |
| §1.3 | "A final-answer check … catches most errors." | Too strong; see A11 (verifier FN/FP) and A9 (ill-posed questions). |
| §5.5 | "Any outcome is publishable." | Weaker now: a null/"LLM wins" result is *predicted* by A2–A5, so it is less citable. |

---

## 4. Novelty assessment of the old contributions

### C1 — Vi-ExamMath (grade-10 entrance + HSG, free-response, solver-verified, provenance)
**Verdict: STILL NOVEL (narrowly), but no longer sufficient on its own for more than a regional resource paper.**
- *Still novel:* every Vietnamese math resource found is THPT (D1, D2, D4, D5), elementary (D3, D6), or
  translated (D7, D8, D9). No grade-10/HSG free-response benchmark on arXiv, HF, ACL Anthology search, or
  GitHub search (§1D, D10). No Vietnamese benchmark uses solver-checked answer keys or reference answer
  matching (D3 uses an LLM judge; D1/D4/D5 are MCQ/T-F).
- *Weakening factors:* (a) the national-exam-archive genre is now common in other languages (E3 Romanian,
  E1 Korean, C13 Sinhala/Tamil), so "first Vietnamese X" is a thin claim; (b) frontier saturation — E1 shows
  ≥95% on a harder national exam for a 20B open model, so Core will discriminate only small/mid models;
  (c) DeepSEA/PolyMath now give Vietnamese evaluation coverage (translated) that reviewers will ask to compare
  against.
- *What would make it strong:* use C1 as an **instrument for a scientific question** — (i) a pre/post
  training-cutoff temporal split (2026 exams vs ≤2024) to *measure* contamination (E1, B5, C15), and
  (ii) a native-vs-translated gap measurement against PolyMath-vi / DeepSEA-vi / Vi-GSM8K.

### C2 — Symbolic-verified vs LLM augmentation for SFT of ≤1.7B models in Vietnamese
**Verdict: WEAKENED (not dead).**
- *Scoop evidence:* A3 (2602.19187) already runs symbolic-space vs NL-space LLM problem modification for
  1.5B/3B students (English, RLVR) and reports SMT-LIB renderings are unnatural and worse; A2 (FLAMES)
  already compares 6 quality-control strategies and finds coverage > verified reliability; A4/A5 show trace
  correctness matters little for SFT. The headline answer to RQ-A/RQ-B is thus predictable.
- *What remains unclaimed:* SFT (not RL), sub-1B, low-resource language, native-exam evaluation, and the
  translate-train arm (S5). Of these, **only the low-resource + native-evaluation axis is a genuine gap**;
  the symbolic-vs-LLM axis alone is incremental.
- *Structural problem:* S1 covers only linear arithmetic/algebra (plan N2), so S1-vs-S2 confounds
  "verification" with "coverage" — the very factor FLAMES identifies as dominant.

### C3 — Failure analysis of naive answer-expression mutation (+ vacuous SMT verification)
**Verdict: STILL NOVEL in specifics, NARROW in impact.**
- No paper found documenting (i) answer-expression-only mutation producing ugly answers / leaked formal
  artifacts, or (ii) hard-coded-answer tautologies in SMT formalizations of GSM8K/MATH (targeted search for
  autoformalization + hard-coded/vacuous verification returned only adjacent work: vacuous-statement filters in
  "Don't Trust: Verify" ICLR 2024 [UNVERIFIED, snippet only], VeriGeo A10).
- *Pre-empted at the claim level:* A3 already reports SMT-LIB renderings are unnatural and underperform;
  A10 calls seed rewriting "weakly verifiable". So the *general* lesson is known; C3's value is the
  **measured audit methodology** (15.9% hard-coded answers; 40.7% ugly answers; 16.3% leakage) as a
  cautionary section/appendix and as a reusable "verification-vacuity" checker. Not a standalone paper above
  workshop level.

---

## 5. Candidate research questions (novel as of Nov 2026)

**Ranking criterion (user constraint, 2026-11-07):** the target is **Q2** (strong journal or ACL/EMNLP
Findings-tier) and **scope must not be cut**. Each framing is therefore judged on whether it can clear a Q2
bar on *novelty* (the answer is not already predicted by §1) and *rigor* (multiple model families/sizes,
seeds, matched budgets, proper statistics, contamination control, human validation). The safest/smallest
option is not preferred. Each RQ below lists **"What it needs to reach Q2"**.

**Q2 bar used here (derived from what the accepted comparators did):** FLAMES (A2, EMNLP) = 10 strategies,
controlled pipeline; *English is Not All You Need* (C7) = 220 SFT runs; *Long CoT Across Languages* (C1, ICLR)
= 9 languages × several stages; GanitLLM (C12, ACL Findings) = new data + model + multi-benchmark eval;
Multilingual GSM-Symbolic (C10) = joint statistical modelling of determinants. I.e. Q2 work in this area has
(i) a controlled factorial design, (ii) ≥2 model families, (iii) significance testing beyond a single seed,
(iv) a released resource, and (v) an analysis that explains *why*, not just a leaderboard.

### RQ-1 (recommended) — "Does the translated-data advantage survive native evaluation?"
*For SFT of small (≤2B) models in Vietnamese, how do translated English corpora, LLM-native-generated
Vietnamese data, and symbolic-verified data compare — and does the ranking change when evaluated on
native-authored exam problems instead of translated benchmarks?*
- **Design sketch:** arms at matched N: T = translated (MetaMathQA-vi / OpenMathInstruct-vi, already
  available), N = LLM-generated *natively in Vietnamese*, seeded from a **native** pool disjoint from the
  benchmark (older exams, textbook exercises, `hllj/vi_math_problem_crawl`), S = symbolic-verified (existing
  pipeline, restricted to its coverage domain), M = T+N mix, plus S0 floor. Eval on two axes:
  *translated* (Vi-GSM8K, PolyMath-vi, DeepSEA-vi MGSM) vs *native* (Vi-ExamMath Core, with ≤2024 vs 2026
  temporal split). Primary statistic: the **arm × eval-type interaction**. Verification (checked vs
  unchecked N) as one ablation only.
- **Novelty evidence:** the two papers that compared translated vs native/distilled training data (C1
  2508.14828; C6 2505.19848) both evaluated on translated benchmarks and both found translated wins; no
  study with native-authored evaluation found (§2.3); no Vietnamese math training study at all (D10).
  Translated test sets also risk cross-lingual contamination (C15), which the native temporal split controls.
- **Feasibility:** high. Translated corpora + Vi-GSM8K already in repo; generator on free NIM; QLoRA SFT of
  0.6–0.8B on the laptop, 1.7–2B on Modal/free notebook; eval is CPU-scored via `math_verify`. Reuses S1/S5
  arms already planned → minimal re-plan.
- **Risk:** (a) statistical power — a 300–400-item native set gives ~±4.8 pp 95% CI at 40% accuracy
  (computed: SE = √(0.4·0.6/400) ≈ 2.45 pp), so only interactions ≳5 pp are detectable; mitigate with ≥3 seeds,
  paired bootstrap/McNemar, and the 2B model. (b) Sub-2B signal may be flat (B1). (c) "native-generated" data
  quality may be poor (C6 audit found unnatural phrasing) — that is itself a measured result.
- **Expected outcome:** either "translated wins everywhere" (confirms C1/C6 with a cleaner instrument — still
  a useful negative for LRL practice) or "the advantage shrinks/flips on native items" (a citable, surprising
  finding). Findings-tier (ACL/EMNLP Findings, COLING) if the interaction is significant; otherwise a strong
  regional paper (VLSP/RIVF/KSE) + resource.
- **What it needs to reach Q2:**
  - *Experiments:* full factorial arms {S0, T (translated), N-checked, N-unchecked, S1 symbolic-verified, S4
    symbolic-unverified, M = T+N} at matched N, **plus a data-efficiency axis** (≥3 budgets, e.g. 2k/5k/10k+)
    so the arm × eval interaction is shown not to be a single-budget artifact.
  - *Models:* ≥2 sizes × ≥2 families — e.g. Qwen3.5-0.8B-Base and Qwen3.5-2B-Base (or Qwen3-0.6B/1.7B) **plus a
    non-Qwen control** (e.g. Gemma-3-1B or Llama-3.2-1B; **[UNVERIFIED]** Vietnamese adequacy), because Qwen
    results are contamination/spurious-reward-prone (B4, B5). ≥3 seeds per cell.
  - *Baselines:* the arms above + released corpora as external references (5CD-AI MetaMathQA-vi,
    `ura-hcmut/Vietnamese-MATH`, Vi-S1K if released) + zero-shot base/instruct rows + a strong small reference
    (DeepSeek-R1-Distill-Qwen-1.5B / VibeThinker-1.5B) on all eval sets.
  - *Evaluation:* translated side = Vi-GSM8K, Vi-MATH(subset), PolyMath-vi, DeepSEA-vi MGSM; native side =
    Vi-ExamMath Core (≥400) + Hard, each split ≤2024 vs 2025–26 (temporal). Reference answer matching +
    `math_verify` only (E4, A11). An **English control** (same arms on English GSM8K/MATH-500, plan RQ-E) to show
    the effect is low-resource-specific.
  - *Analyses:* (1) mixed-effects logistic regression `correct ~ arm × eval_type × size + (1|item) + (1|seed)`
    as the primary test (handles item difficulty and the ~400-item power limit better than per-set accuracy);
    (2) IRT on the native benchmark to report discrimination and to compare arms on latent ability (E6);
    (3) mechanism analysis — *why* translated data helps/hurts on native items: translationese/register
    distance (perplexity of each eval set under each fine-tuned model, embedding distance train↔eval), answer
    format/unit conventions, Vietnamese-specific problem schemas (e.g. "giải bài toán bằng cách lập phương
    trình"); (4) contamination analysis (temporal gap; n-gram/embedding overlap of every training corpus with
    every eval set; cross-lingual check à la C15); (5) the data-quality panel of plan §5.4 regressed on utility
    (RQ-C); (6) human evaluation of naturalness/validity of ≥200 items per generated arm with ≥3 raters and
    Krippendorff's α.
  - *Release:* benchmark + 3 matched Vietnamese training corpora + adapters + prompts + code + datacards.

### RQ-2 — Reasoning language for ≤2B models on Vietnamese math
*Train on identical problems with English-CoT vs Vietnamese-CoT vs mixed traces; evaluate on native Vietnamese
exams (answer in Vietnamese).*
- **Novelty evidence:** C1 (no vi; ≥1.5B only in scaling), C2, C5 (English pivot helps; vi not reported
  separately), C8 (3.35B in-language), C12 (Bengali 4B) — none at ≤2B for Vietnamese with native eval.
- **Feasibility:** high (same pipeline; translating traces via free API). **Risk:** incremental in a crowded
  2026 sub-area (C1, C2, C8, C9, C12, C14 all appeared within 15 months); likely answer ("English CoT is more
  accurate, Vietnamese CoT is more readable") is predicted by C1/C2. **Outcome:** a solid section, weak
  standalone paper.
- **What it needs to reach Q2 (standalone):** ≥3 trace-language conditions (En / Vi / mixed or
  "En-think, Vi-answer") × ≥2 sizes × ≥2 families × ≥3 seeds; native eval (Vi-ExamMath) *and* translated eval;
  language-fidelity metrics (share of Vietnamese reasoning tokens, as in C12) alongside accuracy; a readability
  /faithfulness human study; and a second SEA language to generalize. Even then the expected answer is
  largely predicted (C1, C2, C5) → **unlikely to clear Q2 alone**; it **does** add Q2-grade depth as a full
  factor crossed with the RQ-1 arms (not cut — run on all primary arms if compute allows, otherwise on T, N and M).

### RQ-3 — Verification-signal ablation at fixed generator (refined C2)
*Same LLM generator; labels from none / self-consistency vote / LLM-judge / program execution (PoT) / SMT-solver
(where expressible); matched N and matched difficulty.*
- **Novelty evidence:** no head-to-head of labeling signals for *synthetic problem* SFT data found in search
  (snippet-level evidence only); but FLAMES (A2) compared 6 QC strategies in English and A4/A5 predict small
  effects. Novel only in language/scale.
- **Feasibility:** medium-high. **Risk:** high probability of a null result already predicted by A2/A4/A5; the
  SMT arm's coverage restriction confounds it. **Outcome:** a well-powered null is a section, not a headline.
- **What it needs to reach Q2 (standalone):** a coverage-matched design (restrict *all* arms to the
  SMT-expressible subset, then repeat on the full distribution without the SMT arm) so verification is not
  confounded with coverage; measured label precision/recall of every verifier against a human-adjudicated
  gold subset (≥500 items); difficulty-matched resampling; ≥2 families × ≥2 sizes × ≥3 seeds; and a
  label-noise dose-response curve (inject 0/10/25/50% wrong labels) to locate where verification starts to
  matter for small models. Without the dose-response and coverage control it is a FLAMES replication in
  Vietnamese → **below Q2 alone**; **inside RQ-1** it supplies the "does the certificate matter?" analysis
  (C2's original question) and should be kept at full rigor, not cut.

### RQ-4 — Benchmark-centred: contamination and translation gaps measured with a native exam benchmark
*Build Vi-ExamMath with pre/post-cutoff temporal split, solver-checked keys and IRT; measure (i) contamination
gap (old vs 2026 exams, matched difficulty), (ii) translated-vs-native gap (PolyMath-vi / DeepSEA-vi /
Vi-GSM8K vs Vi-ExamMath) across ~15 open models of 0.5–70B.*
- **Novelty evidence:** temporal zero-leakage evaluation exists for Korean (E1); exam archives exist for
  Romanian (E3); none for Vietnamese grade-10/HSG; no Vietnamese measurement of translated-vs-native gap.
- **Feasibility:** highest (eval-only; free endpoints). **Risk:** saturation for frontier models (E1); the 2026
  exam cohort is small (~1 year × N provinces ≈ 100–200 items) → noisy contamination estimate; it is an
  evaluation paper, so the "model" deliverable is weak. **Outcome:** solid resource + analysis paper
  (LREC-COLING / regional venues; Findings possible if the gaps are large).
- **What it needs to reach Q2 (standalone):** Core ≥400 + Hard ≥200 with full provenance; solver-certified
  keys + 2-annotator QA with κ; ≥15 models spanning 0.5B–frontier and ≥4 families; temporal split with
  difficulty-matched old/new strata (otherwise "new exams are harder" confounds the contamination gap);
  translated-vs-native gap with item-level matching (topic/grade band) across PolyMath-vi / DeepSEA-vi /
  Vi-GSM8K; IRT calibration (discrimination, information curves) showing which split is discriminative for
  which model tier; an error taxonomy of solver-verification failures (a C1 result) and of model failures by
  Vietnamese-specific problem schema. Feasible and plausibly Q2 at LREC-COLING/journal level, but as a pure
  benchmark it is in a genre that already exists elsewhere (E1, E3) → **borderline Q2 alone**; it is the
  mandatory instrument for RQ-1 and should be built at this full spec.

### Comparison

| Criterion | RQ-1 translated vs native data (+C1 as instrument) | RQ-2 reasoning language | RQ-3 verification ablation | RQ-4 benchmark + gaps |
|---|---|---|---|---|
| Novelty (Nov 2026) | **High** — explicit untested confound in C1/C6 | Low–medium (crowded) | Low (FLAMES, A4/A5) | Medium (genre exists elsewhere) |
| Prior result predicts answer? | No — two plausible outcomes | Mostly (C1, C2) | Yes (null likely) | Partly (saturation predictable) |
| Reuses existing assets | **Yes** (S1, S5, Vi-GSM8K, translations, scraper) | Partly | Yes | Yes (scraper) |
| Compute | SFT grid ~10–15 runs × 3 seeds at ≤2B (laptop + ~$10 Modal) | +2× runs on one arm | ~10 runs × 3 seeds | Eval-only |
| Main risk | Power at ~400 items; flat sub-2B signal | Incrementality | Null result | Saturation; small 2026 cohort |
| Hand-off package | Data (benchmark + 3 corpora) + models + code + prompts | Models + prompts | Data + models | Benchmark + leaderboard |
| Publishable floor / ceiling | Regional+resource / **Findings** | Section / workshop | Section / workshop | Regional / Findings-low |
| **Can clear Q2 alone?** | **Yes**, at the spec above (both outcomes informative) | Unlikely (answer predicted by C1/C2/C5) | Unlikely (FLAMES replication risk) unless coverage control + dose-response | Borderline (genre exists: E1, E3) |
| **Q2 contribution when combined** | Headline | Adds a full factor (language of reasoning × data source) | Supplies the certificate/verification analysis (original C2) | Mandatory instrument + standalone resource |

### Recommendation
**Adopt RQ-1 as the headline question, with RQ-4's benchmark built at full spec as its required instrument
(C1 stays the first deliverable), and RQ-2 and RQ-3 kept as full factors of the same factorial design — no
component of the original scope (C1 Core+Hard, C2's symbolic/verification arms, C3) is dropped; they are
re-ordered under a question the literature has not answered.** C3 becomes the methods section that
justifies the fixed symbolic arm and contributes a reusable verification-vacuity audit.

Indicative grid at Q2 spec (rough count, compute feasibility to be validated in the compute/re-plan doc —
**not verified here**): 7 arms × 4 models (2 sizes × 2 families) × 3 seeds = 84 SFT runs; + data-efficiency
(2 extra budgets × 3 key arms × 2 models × 3 seeds) = 36; + reasoning-language factor (1 extra condition × 3
arms × 2 models × 3 seeds) = 18; + English control (3 arms × 1 model × 3 seeds) = 9 → **≈150 short-CoT QLoRA
runs at ≤2B**, plus eval-only leaderboard of ≥15 models.

Justification:
1. It is the only option where the 2026 literature *does not already predict the answer*: the
   translated-data advantage (C1 78.3 vs 72.8; C6 51.4 vs 39.4) was measured exclusively on translated
   benchmarks, and translated test sets can be contaminated in ways overlap detection misses (C15).
2. It converts C1 from a "first Vietnamese X benchmark" (thin claim given E1/E3/C13) into a *scientific
   instrument* with a falsifiable use, and uses the temporal split (E1/B5) to keep the instrument clean.
3. It keeps C2's investment alive — S1 (symbolic) and S5 (translate-train) are already arms in plan §5.3 —
   while moving the headline away from the axis A3/A2/A4/A5 have largely answered.
4. It is feasible on the stated hardware: SFT only at ≤2B, translated data already in hand, free generator.
5. Every outcome yields a hand-off-able package (native benchmark + three matched Vietnamese training corpora
   + trained adapters + prompts + code), and a significant interaction would be a Findings-tier result.

Design consequences to carry into the re-plan (not decided here): (a) a native, benchmark-disjoint seed pool
is required for arm N (today's seed set is translated GSM8K+MATH, plan §5.2); (b) short-CoT traces for ≤2B
(B1); (c) consider Qwen3.5-0.8B/2B-Base (B14) and a non-Qwen control (B4: Qwen-specific artifacts);
(d) score with reference answer matching + `math_verify`, never reference-free LLM judging (E4, A11).

---

## 6. Open questions / not verified

1. **V-Math dataset composition** (format, size, solutions) — abstract does not state it; PDF not read.
2. **Language lists** of Multilingual GSM-Symbolic (2610.03367), MGSM-Pro (2601.21225), OSCD (2608.00533),
   TOPSD (2611.08179), Tiny Aya L2-Thinker (2609.10445), *When Models Reason in Your Language* (2505.22888) —
   whether Vietnamese is included was not confirmed; if any includes vi with native data, RQ-2's novelty drops further.
3. **DeepSEA data release/licence** and its Vietnamese-specific scores — not reported in the HTML summary.
4. **Non-arXiv Vietnamese venues** (VLSP 2026 proceedings, RIVF/KSE 2026, Vietnamese university journals) were
   only touched via web search; VLSP 2026 lists a physics-reasoning and a numerical-QA task, no math task seen.
   A grade-10/HSG benchmark published only in a Vietnamese journal could exist undetected.
5. **Vi-S1K group (VNU-UET, D3/D6)** — no follow-up found, but conference-only outputs (FJICAI, KSE) were not
   searchable; contact or monitor before submission.
6. **Majority-vote vs ground-truth label quality** for synthetic problems — only snippet-level evidence
   (2609.18587, 2603.02203, 2510.02172); not opened.
7. **Tooling for Qwen3.5 small Base** (multimodal architecture) under QLoRA on 4 GB and under vLLM — not checked.
8. **Frontier/mid-model scores on Vietnamese grade-10 items** — saturation is inferred from Korean CSAT (E1),
   not measured; the plan's discriminativeness gate (§4.6) should be run before committing the training grid.
9. Whether **"native-generated" Vietnamese synthetic data** can be produced at acceptable naturalness by the
   free generator — C6 found native synthetic data unnatural for Yoruba; Vietnamese is higher-resource, but
   this must be piloted (≈200 items, MOS rating) before the full grid.
10. Numbers quoted from HTML full text (A3, C1, C5, C6) were obtained through a summarizer and must be
    re-checked against PDFs before citation.
