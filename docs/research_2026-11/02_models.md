# 02 — Model choices: re-validation (2026-11-07)

**Scope.** Re-checks the model choices in `RESEARCH_PLAN.md` §6.0 (rows A–J) and §7 against evidence
available on 2026-11-07. The lens is a **Q2-level paper**: strong current baselines, more than one model
family, more than one size. The hand-off is internal. Public-release notes are brief (§4.3).

**Evidence rules.** Every claim has a source: a fetched URL, Hugging Face API output, or a command run on the
researcher's laptop. Scripts and raw outputs are in the scratchpad (see §7). **[UNVERIFIED]** marks a claim
from memory or from a source that could not be checked. **[ESTIMATE]** marks a projection.

---

## 0. Bottom line

1. **The planned generator no longer exists.** `deepseek-ai/deepseek-v4-pro-0813` on NVIDIA NIM returns
   **HTTP 410 Gone**: *"has reached its end of life on 2026-09-14T08:00:00Z"*. The planned weak
   generator/solver `meta/llama-3.1-8b-instruct` is also **410 Gone (EOL 2026-08-26)**. So are
   `deepseek-v4-pro`, `deepseek-v4-flash`, `llama-3.3-70b-instruct`, `gpt-oss-120b` and `qwen3.5-397b-a17b`
   (§2.2, `nim_probe.json`). Rows A, B, C, D and I of §6.0 must be re-assigned.
2. **Only the NVIDIA key in `.env` is filled in.** `GROQ_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`,
   `HF_TOKEN` and `NVIDIA_API_KEYS` are empty (length 0, checked without printing values). Groq returned 401,
   Gemini 403 and DeepSeek 401. The "fallback chain NIM → DeepSeek direct → Groq" in §6.0 does not exist
   today. Everything below that was tested live was tested on NIM.
3. **Recommended stack (details and reasons in §5):**

| Role (§6.0 row) | Pick | Fallback |
|---|---|---|
| **A/B Generator** (S1 informalization, S2/S3 generation, formalization drafts) | **`deepseek-ai/deepseek-v4.1-flash`** (NIM). MIT weights. | `nvidia/nemotron-3-ultra-550b-a55b` (NIM, OpenMDW-1.1) |
| **D LLM solver/judge** (round-trip check, different family from the generator) | **SymPy (authority) + `nvidia/nemotron-3-ultra-550b-a55b`** (NIM; fastest strong model, 10/10) **+ `z-ai/glm-5.3-flash`** (NIM, MIT, 10/10) | swap in DeepSeek if the roles are swapped |
| **C Weak generator** (generator-strength robustness) | **`nvidia/nemotron-3.5-bolt-e4b`** (NIM, Gemma-4-E4B post-train, ~4.5B effective) | `google/gemma-4-31b-it` (NIM, Apache-2.0), a "mid" generator |
| **I Difficulty-reference solver** | **Self-run frozen `Qwen3.5-2B`** (vLLM, greedy). Bolt-E4B saturated the smoke items | `nvidia/nemotron-3.5-bolt-e4b` |
| **Claude ($100 credit): benchmark/eval-side only** (new, §8) | **`claude-opus-5-5` via the Batch API**: Vi-ExamMath item QA, answer-key audit, eval-set translation check and repair, and audit-only quality judging of samples. **Never** to write SFT training data in this plan (§8.4) | `claude-sonnet-5-5` (half the price) |
| **E Students (fine-tuned)**, primary family | **Qwen3.5-Base: 0.8B + 2B** (+ **4B** as a third scale point) | **Qwen3-Base 0.6B / 1.7B / 4B** if the day-1 pilot gate in §5.2 fails |
| **E Students**, second family | **Gemma-4-E2B (base)** | Llama-3.2-1B (gated; weak Vietnamese) |
| **E Students**, Vietnamese-adapted control (optional) | **Sailor2-1B** (Apache-2.0, Qwen2.5-derived) | — |
| **F/G Leaderboard baselines (not fine-tuned)** | Qwen3.5-4B, Gemma-4-E4B-it, OpenMath-Nemotron-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-Math-1.5B-Instruct, Sailor2-3B-Chat, Arcee-VyLinh, Nemotron-3.5-Bolt-E4B, plus the teachers above | — |

4. **Biggest risk: hosted models disappear fast.** `deepseek-v4-pro-0813` was created on HF on 2026-08-13 and
   was EOL on NIM on 2026-09-14, about one month later. Any hosted generator can vanish mid-study. Mitigations
   are in §6: freeze the generation corpus early, log model ID + date + full request, and only use
   open-weight generators.
5. **Claude ($100 credit) goes to the evaluation side, not to training data.** Spend the $100 with Opus 5.5
   via the Batch API (50% off) on Vi-ExamMath QA, eval-set translation repair and audit-only judging: ≈ 7k
   judge/translate calls (§8.2, §8.4). Claude is also the judge used for the held-out test items, which are
   kept off NIM and Gemini (§8.3).

---

## 1. Hardware facts (commands run 2026-11-07)

| Item | Value | Command |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, **4096 MiB** | `nvidia-smi --query-gpu=name,memory.total --format=csv` |
| CPU | Intel Core i5-11400H, **6 cores / 12 threads**, 2.70 GHz | `Get-CimInstance Win32_Processor` |
| RAM | **7.71 GB** (8,277,110,784 bytes) | `Get-CimInstance Win32_ComputerSystem` |
| Disk free | E: 100 GB, C: 67 GB | `df -h` |
| Project venv | Python 3.11.7; `openai 3.10.0`, `requests 2.34.2`, `sympy 1.14.0`. **No torch/transformers/tokenizers** | `pip list` |

What this means for model choice. 7.7 GB of system RAM is the real local limit. A bf16 merge of a 4B model
(~9 GB) does not fit in RAM. A bf16 2B merge (~4.5 GB) barely fits. Unsloth's own VRAM table says
Qwen3.5-0.8B bf16 LoRA needs ~3 GB and 2B needs ~5 GB, and Gemma-4-E2B LoRA needs 8–10 GB (§3.4). So **only
the smallest student can train locally, and only as a dev check.** The infra report (`03_training_infra.md`)
reaches the same conclusion: the laptop is a dev box, and the grid runs on Modal (L4 $0.222/h, $30/month
credit).

---

## 2. Generator / teacher / judge — live verification

### 2.1 What is reachable for free (lists fetched 2026-11-07, `api_models_raw.json`)

| Provider | Status with this repo's keys | Evidence |
|---|---|---|
| **NVIDIA NIM** (`integrate.api.nvidia.com/v1`) | **200, 84 models listed.** About half of them return 404 "Function … Not found for account" when called (listed but not entitled), or 410 Gone. Real callable set below. | `list_api_models.py`, `nim_probe.py` |
| Groq | **401 Invalid API Key** (key empty in `.env`) | same |
| Gemini (AI Studio) | **403 PERMISSION_DENIED** (key empty) | same |
| DeepSeek direct | **401** (key empty). The "5M free tokens" grant in `.env` comments is **[UNVERIFIED]**. The pricing page lists `deepseek-flash` (now served by V4.1-Flash-1017) and `deepseek-v4-pro`, and does not mention a free tier ([pricing page](https://api-docs.deepseek.com/quick_start/pricing)). | same |
| HF Inference router | Public list returned 126 models without a token. **Calls need `HF_TOKEN`, which is empty.** Free credit amount **[UNVERIFIED]**. | same |
| OpenRouter | Public list: 460 models, **16 at $0** (e.g. `nvidia/nemotron-3-ultra-550b-a55b:free`, `google/gemma-4-31b-it:free`, `liquid/lfm-2.5-2.6b:free`). Needs an account; none was created. | same |

### 2.2 NIM probe — which IDs actually answer (`nim_probe.json`, prompt "7 cộng 5 bằng bao nhiêu?")

| Model | Result |
|---|---|
| `deepseek-ai/deepseek-v4-pro-0813` (**planned generator**) | **410 Gone, EOL 2026-09-14** |
| `deepseek-ai/deepseek-v4-pro`, `deepseek-v4-flash` | 410 Gone, EOL 2026-08-07 |
| `meta/llama-3.1-8b-instruct` (**planned weak model**), `meta/llama-3.3-70b-instruct` | 410 Gone, EOL 2026-08-26 |
| `meta/llama-4-maverick-17b-128e-instruct`, `qwen/qwen3.5-397b-a17b` | 410 Gone, EOL 2026-07-27 |
| `openai/gpt-oss-120b` | 410 Gone, EOL 2026-09-03 |
| `moonshotai/kimi-k2.6`, `nvidia/nemotron-3-super-120b-a12b`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `aisingapore/sea-lion-7b-instruct`, `mistralai/mistral-7b-instruct-v0.3`, `nv-mistralai/mistral-nemo-12b-instruct`, `ibm/granite-3.0-8b-instruct`, `nvidia/nemotron-nano-3-30b-a3b`, `microsoft/phi-3.5-moe-instruct`, `zyphra/zamba2-7b-instruct`, `nvidia/llama-3.1-nemotron-ultra-253b-v1`, `nvidia/llama-3.1-nemotron-70b-instruct`, `mistralai/mistral-large-2-instruct` | **404 "Not found for account"**: listed in `/v1/models` but not callable with this key |
| `deepseek-ai/deepseek-v3.2`, `qwen/qwen3-235b-a22b` | 404 page not found |
| `openai/gpt-oss-20b` | timed out (90 s) |
| `mistralai/mistral-large-4-1t-a52b` | answered once in 76 s, then timed out at 90 s (very slow) |
| **`deepseek-ai/deepseek-v4.1-flash`, `z-ai/glm-5.3-flash`, `nvidia/nemotron-3-ultra-550b-a55b`, `minimaxai/minimax-m3.1-flash`, `stepfun-ai/step-5-preview`, `google/gemma-4-31b-it`, `nvidia/nemotron-3.5-lightning-30b-a3b`, `nvidia/nemotron-3.5-bolt-e4b`, `google/diffusiongemma-26b-a4b-it`, `poolside/laguna-xs-2.1`** | **OK** (all answered "12") |

**Lesson:** `/v1/models` is not a reliable list of what this account can call. Probe before you plan.

### 2.3 Weights and licences of the callable teachers (HF API, `hf_teacher_info.json`)

| NIM ID | HF repo (created) | Size (total params, HF safetensors) | Weights licence | Vietnamese in card `language` tag |
|---|---|---|---|---|
| deepseek-v4.1-flash | `deepseek-ai/DeepSeek-V4.1-Flash` (2026-09-10); newer `-1017` (2026-10-17) | 763 B | **MIT** | not tagged |
| nemotron-3-ultra-550b-a55b | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (2026-06-03) | 561 B (MoE, 55B active) | **OpenMDW-1.1** (permissive, no restriction on outputs; [licence](https://openmdw.ai/license/1-1/)) | not in its 12-language list |
| glm-5.3-flash | `zai-org/GLM-5.3-Flash` (2026-08-25) | 321 B | **MIT** | en, zh only |
| step-5-preview | `stepfun-ai/Step-5-Preview` (2026-09-20) | 604 B | StepFun Model License 1.0 ([LICENSE](https://huggingface.co/stepfun-ai/Step-5-Preview/blob/main/LICENSE.txt)) | not tagged |
| minimax-m3.1-flash | `MiniMaxAI/MiniMax-M3.1-Flash` (2026-10-17) | 220 B | minimax-community | not tagged |
| mistral-large-4-1t-a52b | `mistralai/Mistral-Large-4-1T-A52B` (2026-09-30) | ~1 T (52B active) | Mistral Ecosystem Open-Weight (modified MIT) | 39-language tag incl. vi |
| gemma-4-31b-it | `google/gemma-4-31B-it` (2026-03-11) | 31 B | **Apache-2.0** | not tagged (card: "35+ languages, pre-trained on 140+") |
| nemotron-3.5-lightning-30b-a3b | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (2026-08-01) | 31.6 B (3B active) | OpenMDW-1.1 | en/es/fr/de/it/ja |
| nemotron-3.5-bolt-e4b | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` (2026-10-05) | 8.0 B total (Gemma-4-E4B, ~4.5B effective) | OpenMDW-1.1 (+ Gemma 4 Apache-2.0) | **34-language tag incl. vi** |

All callable teachers are **open-weight**. That matters for reproducibility. When NIM retires a model, the
weights stay on HF, and someone else can re-host them.

### 2.4 Smoke test — Vietnamese grade-10 word problems

**Set-up** (`smoke_teachers.py`). NIM, `temperature=0`, `top_p=1`, `seed=42`, `max_tokens=8192`. A Vietnamese
system prompt ("giáo viên toán THPT … đặt đáp số cuối cùng trong `\boxed{}`"). Two rounds:

- **Easy (4 items)**: rectangle + linear system (gold **187**), parabola max (**21**), profit quadratic
  (**42**), 4-digit even numbers from {0..5} (**156**).
- **Hard (6 items, exam-style traps)**: parameter m so a quadratic is > 0 ∀x, count in [-10;10] (**4**; Δ'<0
  ⇔ m²+m−6<0 ⇔ −3<m<2); probability of drawing all three colours, a+b of 270/495 = 6/11 (**17**); reflection
  minimum of MA+MB on Ox (**2.8**); fence against a wall with a divider, 3x+y=120 (**1200**); no two girls
  adjacent, 4!·C(5,3)·3! (**1440**); right triangle from a regular 12-gon, 60/220 = 3/11 → a+b (**14**).
  All golds were derived by hand (working shown here) before running.

Scoring re-parses the last `\boxed{}` (`summarize.py`). It accepts `\frac`, `a/b`, Vietnamese decimal
comma (`2{,}8`) and thousands dot (`1.440`). Every wrong answer was also read by hand. **Fluency** was judged
by reading the outputs. `vi_diacritic_ratio` (share of letters with Vietnamese diacritics; LaTeX letters
lower it) and `cjk_chars` (Chinese characters leaking in) are mechanical red-flag checks only.

**Results** (`smoke_summary.json`; 2026-11-07, 09:00–10:40 local; 3–4 parallel workers, always under 40 RPM)

| Model (NIM) | Easy (4) | Hard (6) | Median latency, easy / hard (s) | Max latency (s) | Median completion tokens (easy / hard) | Vietnamese fluency (read by hand) | Notes |
|---|---|---|---|---|---|---|---|
| **nvidia/nemotron-3-ultra-550b-a55b** | **4/4** | **6/6** | **14.8 / 9.4** | 29 | 819 / 863 | Fluent, correct terms (*khe hở*, *quy tắc nhân*, A₅³) | Fastest strong model |
| **deepseek-ai/deepseek-v4.1-flash** | **4/4** | **6/6** | 38.1 / 95.8 | 157 | 457 / 613 | Fluent, textbook THPT style | Concise. Slowest of the strong models today |
| **z-ai/glm-5.3-flash** | **4/4** | **6/6** | 26.5 / 17.5 | 40 | 522 / 595 | Fluent, structured "Bước 1…" | — |
| stepfun-ai/step-5-preview | 4/4 | 6/6 | 20.1 / 22.2 | 133 | 858 / 680 | Fluent | Not used as a generator (§4.2) |
| minimaxai/minimax-m3.1-flash | 4/4 | 3/3 (run stopped) | 57.3 / 35.1 | 112 | 634 / 748 | Fluent | Largest prompt overhead (371 vs ~130 prompt tokens) |
| google/gemma-4-31b-it | 4/4 | 5/5 answered, **1 timeout (600 s)** | 194 / 131 | **411** | 479 / 465 | Fluent. Uses decimal comma `2,8` | Very slow on NIM today |
| mistralai/mistral-large-4-1t-a52b | 4/4 | not run (stopped for time) | 38.8 / — | 60 | 1418 / — | Fluent | Probe timed out once at 90 s |
| nvidia/nemotron-3.5-lightning-30b-a3b | 4/4 | 5/6 | 18.0 / 18.2 | 132 | 1500 / 1281 | Fluent | h4: **hit max_tokens 8192** (reasoning loop) |
| **nvidia/nemotron-3.5-bolt-e4b** (~4.5B eff.) | **4/4** | **6/6** | **5.0 / 6.3** | 13 | 427 / 550 | Fluent, careful case analysis | Even the "weak" 4B model is at ceiling on these items |

What the smoke test shows, and what it does not:
- **Availability and Vietnamese quality: confirmed.** Every model that answered wrote fluent, idiomatic
  Vietnamese math prose using THPT conventions. No Chinese characters leaked (`cjk_chars = 0` everywhere).
  No English sentences appeared in the outputs read.
- **It cannot rank the teachers.** All strong models scored 10/10 and **Bolt-E4B also scored 10/10**. These
  grade-10 items are below the ceiling of every candidate. **A ranking needs a 100-item Vi-GSM8K + 50-item
  Vi-ExamMath-Hard pilot** before the generator is locked (§6).
- **Latency differs a lot (≈10 s to >400 s per item), and that matters for planning.** Nemotron-3-Ultra and
  GLM are fast. DeepSeek-V4.1-Flash and Gemma-4-31B were slow today.
- Raw outputs: `smoke_A/B/C.jsonl`, `hard_D/E/F/G.jsonl` (field `content`).

### 2.5 Limits of the free hosted APIs (fetched 2026-11-07)

| Provider | Free limits | Fit for this project | Notes |
|---|---|---|---|
| **NVIDIA NIM trial** (`integrate.api.nvidia.com/v1`) | Limits are per model and **not published** ("we do not publish those", NVIDIA staff; [forum](https://forums.developer.nvidia.com/t/request-for-nvidia-build-api-rate-limit-increase-40-rpm-200-rpm/377433)). The researcher's account is 40 RPM (`.env` note; consistent with forum reports). | **Used** for generation and judging of training-side items. | **Unreleased Vi-ExamMath test items are not sent through NIM** (contamination control). Access to any model can end at any time (EOL cases in §2.2). |
| Groq ([rate limits](https://console.groq.com/docs/rate-limits), [models](https://console.groq.com/docs/models)) | Page shows Developer-plan base limits, e.g. `qwen/qwen3.8-27b` and `gpt-oss-20b/120b` at 30 RPM, 1K RPD, 8K TPM, 200K TPD. Free-tier numbers are not shown separately. | Not checked in detail (no key). | Production models are only Llama-3.1-8B, Llama-3.3-70B and gpt-oss-20b/120b. Preview: Qwen3.8-27B, MiniMax-M2.7. Needs a key the user must create. |
| Gemini (AI Studio) | Free limits not published on the page ([rate limits](https://aistudio.google.com/docs/rate-limits)). | **Not used for generation.** | Fine as an occasional judge on training-side items; not as a data generator. |
| OpenRouter `:free` ([limits](https://openrouter.ai/docs/api-reference/limits)) | 20 RPM, **50 RPD** with < 10 credits purchased (1,000 RPD with ≥ 10 credits, which costs money) | Backup only. | 50 RPD is too low for corpus generation. Usable as a backup judge for Nemotron-3-Ultra or Gemma-4-31B. |
| Cerebras ([rate limits](https://inference-docs.cerebras.ai/support/rate-limits)) | Free trial: `gpt-oss-120b` and `qwen-3.8-27b` at 5 RPM, 1M TPD | Not checked | Too slow for generation. |
| GitHub Models | **Retired 2026-07-30** ([docs](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models)) | — | Not available. |
| SambaNova, Mistral La Plateforme free tier | Doc pages returned 404 | **[UNVERIFIED]** | Not pursued further (user scope note). |

---

## 3. Students (≤ ~4B) — survey

### 3.1 Survey table (HF API metadata: `hf_student_info.json`, `hf_student_info2.json`)

"Params" is the HF safetensors total. Math numbers are **base-model few-shot** scores from the Qwen3 Technical
Report (Tables 7–8, [arXiv:2505.09388](https://arxiv.org/html/2505.09388v1)) unless marked "it" (instruct /
thinking, from the model card).

| Model | Params | HF created | Licence (gated?) | Base + instruct? | Math evidence | Vietnamese evidence | Verdict |
|---|---|---|---|---|---|---|---|
| **Qwen3-0.6B-Base** | 0.596 B | 2025-04-28 | Apache-2.0 | yes / yes | GSM8K 59.59, MATH 32.44, MGSM 30.99 | 119 languages (Qwen3 TR abstract); MMMLU 50.16, INCLUDE 34.26 | Fallback primary |
| **Qwen3-1.7B-Base** | 1.72 B | 2025-04-28 | Apache-2.0 | yes / yes | GSM8K 75.44, MATH 43.50, MGSM 50.71 | MMMLU 63.27, INCLUDE 45.57. Vietnamese elementary-math study uses it as the base ([arXiv:2604.17794](https://arxiv.org/abs/2604.17794)) | Fallback primary |
| **Qwen3-4B-Base** | 4.02 B | 2025-04-28 | Apache-2.0 | yes / yes (incl. 2507) | GSM8K 87.79, MATH 54.10, MGSM 67.74 | MMMLU 71.42, INCLUDE 56.29 | Fallback scale point |
| **Qwen3.5-0.8B-Base** | 0.873 B (incl. ViT) | 2026-02-28 | Apache-2.0 | yes / yes | No official base numbers. it: PolyMATH 8.2 (thinking) | **201 languages** (card). it: MMMLU 44.3, MMLU-ProX 34.6, INCLUDE 40.6 | **Primary, small** |
| **Qwen3.5-2B-Base** | 2.27 B (incl. ViT) | 2026-02-28 | Apache-2.0 | yes / yes | Base GSM8K **71.00** (4-shot CoT, partial run, third-party Ascend report via search, **weak evidence**). it (thinking): HMMT Feb25 22.9, PolyMATH 26.1 | it: MMMLU 63.1, MMLU-ProX 52.3, INCLUDE 55.4, WMT24++ 45.8 | **Primary, mid** |
| **Qwen3.5-4B-Base** | 4.66 B (incl. ViT) | 2026-02-27 | Apache-2.0 | yes / yes | it: HMMT Feb25 74.0, HMMT Nov25 76.8, PolyMATH 51.1 | it: MMMLU 76.1, MMLU-ProX 71.5, INCLUDE 71.0 | **Primary, scale point** (cloud) |
| Qwen3.6 / Qwen3.8 | 27 B+ only | 2026-04 / 2026-08 | Apache-2.0 (27B) | — | — | — | No ≤4B sizes exist (HF author listing) |
| **Gemma-4-E2B** (base) | 5.12 B total / **2.3 B effective** | 2026-03-02 | **Apache-2.0, not gated** | yes / yes | it: AIME 2026 37.5, MMLU-Pro 60.0 | it: MMMLU 67.4; "pre-trained on 140+ languages" | **Second family** |
| Gemma-4-E4B | 8.0 B / 4.5 B eff. | 2026-03-02 | Apache-2.0 | yes / yes | it: AIME 2026 42.5 | it: MMMLU 76.6 | Leaderboard baseline |
| Gemma-3-1B-pt | 1.00 B | 2025-02-20 | Gemma licence, **gated (manual)** | yes / yes | **GSM8K 2.20, MATH 3.66, MGSM 1.74** | MMMLU 26.57 | **Reject** (floor effect) |
| Gemma-3-4B-pt | 4.30 B | 2025-02-20 | Gemma, gated | yes / yes | GSM8K 43.97, MATH 26.10, MGSM 33.11 | MMMLU 59.62 | Superseded by Gemma 4 |
| Llama-3.2-1B / 3B | 1.24 / 3.21 B | 2024-09-18 | Llama 3.2, gated | yes / yes | — | vi **not** among its 8 official languages (card tag) | Weak third-family option |
| SmolLM3-3B(-Base) | 3.08 B | 2025-06/07 | Apache-2.0 | yes / yes | — | 8 languages, **no vi** | Reject |
| Phi-4-mini-instruct / -reasoning | 3.84 B | 2025-02 / 04 | MIT | **no base** | — | 24-language tag, **no vi** | Reject (no base) |
| Granite-4.2-3b | 3.66 B | 2026-08-07 | Apache-2.0 | instruct | — | 12 languages, **no vi** | Reject |
| Granite-4.0-(h-)1b-base | 1.46–1.63 B | 2025-10-07 | Apache-2.0 | yes | — | not tagged | Reject |
| MiniCPM5-1B / 2B (Base) | 1.08 / 2.52 B | 2026-05 / 08 | Apache-2.0 | yes / yes | — | en, zh only | Reject |
| NVIDIA-Nemotron-3-Nano-4B | 3.97 B | 2026-03-07 | Nemotron Open Model Licence | instruct | — | en only | Reject |
| LFM2.5-2.6B(-Base) | 2.70 B | 2026-07/08 | LFM Open 1.0 (Apache-based, $10M revenue cap) | yes / yes | AIME25 51.9 (third-party BenchLM) | 16 languages incl. **vi** | Option (non-Transformer hybrid; tooling risk) |
| tiny-aya-base / -global | 3.35 B | 2026-02-13 | **CC-BY-NC-4.0**, gated (auto) | yes / yes | — | 67 languages incl. **vi** | Internal-only option; NC blocks release |
| **Sailor2-1B** (base) / -Chat | 0.99 B | 2024-12-03 | Apache-2.0 | yes / yes | — | vi in 12-lang tag. **Vietnamese perplexity 7.52 vs 26.86 for Qwen2.5-0.5B** (Sailor2 paper Table 16, [arXiv:2502.12982](https://arxiv.org/html/2502.12982)); Belebele-vie 36.78 | **Optional vi-adapted control** |
| Sailor2-3B / -Chat | 3.58 B | 2024-12 / 2025-02 | Apache-2.0 | yes / yes | — | as above | Leaderboard |
| SeaLLMs-v3-1.5B(-Chat) | 1.54 B | 2024-07 | SeaLLMs licence (custom) | yes / yes | — | vi | Leaderboard only |
| Arcee-VyLinh | 3.40 B | 2024-10-29 | Apache-2.0 | instruct only (Qwen2.5-3B) | — | vi-only | Leaderboard |
| vinallama-2.7b(-chat) | 2.77 B | 2023-12 | Llama 2 | yes / yes | — | vi | Outdated |
| Vistral-7B-Chat | 7.29 B | 2024-01-14 | AFL-3.0, gated | chat | — | VMLU 50.07 ([vmlu.ai](https://vmlu.ai/leaderboard)) | > 4B; leaderboard only |
| Qwen2.5-Math-1.5B(-Instruct) | 1.54 B | 2024-09-16 | Apache-2.0 | yes / yes | — | en only | Leaderboard baseline |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78 B | 2025-01-20 | MIT | reasoning only | AIME24 26.8, AIME25 21.4 ([OpenMath-Nemotron card](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)) | **VMLU 33.36** (vmlu.ai) | Leaderboard baseline |
| OpenMath-Nemotron-1.5B | 1.54 B | 2025-04-22 | CC-BY-4.0 | math-SFT only | **AIME24 61.6, AIME25 49.5** (CoT pass@1) | en only | Strongest small-math baseline |
| OpenReasoning-Nemotron-1.5B | 1.54 B | 2025-07-15 | CC-BY-4.0 | reasoning | — | en | Leaderboard option |
| Nemotron-Research-Reasoning-Qwen-1.5B | 1.78 B | 2025-05-28 | **CC-BY-NC-4.0** | reasoning | — | en | Avoid (NC) |
| DeepScaleR-1.5B-Preview | 1.78 B | 2025-01-29 | MIT | reasoning | — | en | Optional |
| **Nemotron-3.5-Bolt-E4B** | 8.0 B / ~4.5 B eff. | 2026-10-05 | OpenMDW-1.1 | instruct (from Gemma-4-E4B-it) | Smoke test §2.4 | 34-language tag incl. **vi** | Leaderboard + weak generator (NIM-hosted) |

Vietnamese-specific leaderboard check: [vmlu.ai](https://vmlu.ai/leaderboard) lists **no Qwen3.5, Gemma 3/4
or VyLinh** entries. The ≤8B rows are 2024–early-2025 models (e.g. Qwen2.5-7B-Instruct 57.51, Vistral-7B-Chat
50.07, R1-Distill-Qwen-1.5B 33.36). A web search found **no published Vietnamese math scores for Qwen3.5 or
Gemma 4**. MGSM has no Vietnamese split **[UNVERIFIED from memory; MGSM's 10 languages are bn, de, es, fr,
ja, ru, sw, te, th, zh]**. **So the paper's own Vi-GSM8K / Vi-ExamMath zero-shot rows will be the first
Vietnamese numbers for these bases. That is a small contribution in itself, and it is also why the pilot
in §5.2 is needed.**

### 3.2 Tokenizer efficiency on Vietnamese

Not measured. That would need `tokenizers` plus downloading tokenizer files, and the venv has neither.
Evidence found:

- Vocab sizes (HF `config.json`): **Qwen3.5 248,320**; Qwen3 151,936; Sailor2 151,936 (same as Qwen2.5);
  Gemma 4 **262K** (card).
- A NeurIPS 2026 paper on Vietnamese tokenization ([neurips.cc/virtual/2026/164760](https://neurips.cc/virtual/2026/164760))
  reports fertility from 0.83× (VinaLLaMA) to 1.93× (Mistral-7B). It finds that ~150K multilingual tokens is
  enough for near-English Vietnamese fertility, and that vocabulary composition matters as much as size. It
  does not report Qwen or Gemma numbers.
- Indirect live data point: for the **same Vietnamese prompt**, NIM-reported prompt tokens are in
  `smoke_summary.json` (`median_prompt_tokens`). Gemma-4-31B (262K vocab) and Nemotron-3.5-Bolt-E4B (Gemma 4
  tokenizer) can be compared with DeepSeek / GLM / MiniMax there. Chat-template overhead differs by model, so
  treat this as indicative only.
- **Action:** add a 20-line fertility script to the day-1 pilot (FLORES-200 vie_Latn devtest, tokens per
  word for Qwen3, Qwen3.5, Gemma 4 and Sailor2). It runs on CPU in seconds.

### 3.3 Architecture and tooling caveats (these decide trainability)

| Family | Caveat | Source |
|---|---|---|
| **Qwen3.5** (0.8B/2B/4B) | Hybrid **Gated DeltaNet + Gated Attention** (0.8B config: 18 `linear_attention` + 6 `full_attention` layers). Ships with a **vision encoder** (12-layer ViT) and is tagged image-text-to-text. Needs **transformers v5**. Unsloth: **"QLoRA (4-bit) not recommended"** for Qwen3.5 → bf16 LoRA. VRAM: 0.8B ≈ 3 GB, 2B ≈ 5 GB, 4B ≈ 10 GB. Custom Mamba Triton kernels compile slowly, "especially on T4". Unsloth RL needs `fast_inference=False`. vLLM support from 0.17 (current 0.33.0 per `03_training_infra.md`). A community card reports that **FA2 + sample packing drove the loss to 0** on Qwen3.5-9B. Verify packing on day 1. | [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B), [config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/raw/main/config.json), [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) |
| **Qwen3** | Plain dense Transformer. Mature everywhere (Unsloth, TRL, vLLM, QLoRA). Official base few-shot scores published. | Qwen3 TR |
| **Gemma 4 E2B/E4B** | Per-Layer Embeddings: 5.1B total for 2.3B effective. Ships audio (~300M) and vision (~150M) encoders. Unsloth: E2B LoRA needs **8–10 GB** VRAM, E4B LoRA 17 GB (E4B QLoRA recommended instead). Cloud only (T4 16 GB / L4 24 GB OK). | [Gemma-4-E2B card](https://huggingface.co/google/gemma-4-E2B), [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train) |
| Sailor2-1B | Qwen2-architecture (48 layers, untied embeddings). Mature tooling. | [config.json](https://huggingface.co/sail/Sailor2-1B/raw/main/config.json) |

### 3.4 Disagreement with `03_training_infra.md`

The infra report picks **Gemma-3-1B-pt** as the second family. The Qwen3 Technical Report (Table 8) gives
Gemma-3-1B base **GSM8K 2.20 / MATH 3.66 / MGSM 1.74**. A data-augmentation comparison on a near-zero base
measures format learning, not reasoning, and a reviewer will flag the floor effect. It is also gated under
the Gemma licence. **Gemma-4-E2B** is Apache-2.0, not gated, from the current generation, and much stronger
(it: AIME-2026 37.5). It costs more VRAM (8–10 GB LoRA), so it needs an L4 rather than a T4/laptop. On the
Modal prices in the infra report (L4 $0.222/h), that cost is acceptable. **[ESTIMATE]** about 1.5–2× the
GPU-hours of a 1.7B run, because the effective compute is 2.3B but the PLE tables are loaded.

---

## 4. Licences

### 4.1 Students (weights we fine-tune and might share)

Apache-2.0 / MIT, no gate: **Qwen3, Qwen3.5, Gemma 4, Sailor2, SmolLM3, Granite, MiniCPM5, Phi-4-mini, R1-Distill
(MIT)**. CC-BY-4.0 (attribution): OpenMath/OpenReasoning-Nemotron. Restricted: Gemma 3 (Gemma terms, gated),
Llama 3.2 (Llama licence, gated), LFM (revenue cap), SeaLLMs (custom), **tiny-aya and
Nemotron-Research-Reasoning (CC-BY-NC)**. All of the recommended students are Apache-2.0.

### 4.2 Teacher models used to generate training data (internal use)

Weight licences of the candidate generators:

- **DeepSeek-V4.1-Flash (MIT), GLM-5.3-Flash (MIT), Nemotron-3-Ultra / 3.5 family (OpenMDW-1.1), Gemma-4-31B
  (Apache-2.0):** open weights; these are the candidate generators.
- **Step-5-Preview (StepFun 1.0):** **not used as a generator.**
- **MiniMax (minimax-community), Mistral Large 4 (modified MIT):** **not used as generators.**

### 4.3 Note on later public release (brief, as requested)

- **NIM-generated datasets:** ask the group's supervisor before release.
- **Gemini:** **do not generate releasable data with it**.
- **StepFun / MiniMax / Mistral-Large-4 outputs:** keep them out of any released data (above).
- **tiny-aya, Nemotron-Research-Reasoning (CC-BY-NC):** never as a student whose fine-tune is released.
- **Claude outputs:** record every Claude role (model id, prompt, date) in the datasheet.
- **Contamination:** **keep held-out Vi-ExamMath test items off every hosted API** until the benchmark is
  published. Run hosted-model evaluation on test items only once, at the end, and log it.

---

## 5. Recommendation and justification

### 5.1 Teachers

- **Generator (rows A/B): `deepseek-ai/deepseek-v4.1-flash`.** Reasons:
  1. It is the direct successor of the planned DeepSeek-V4-Pro line, so the S1 informalization prompts carry over.
  2. 10/10 in the smoke test, with fluent and concise THPT-style Vietnamese (lowest median output length
     among the strong models, so cheaper SFT targets).
  3. MIT weights (763B on HF), so the generator stays re-hostable after NIM EOL.
  4. A widely known frontier family, which reviewers accept as a "strong generator".

  Cost: it was slow today (median 38–96 s), so run ~40–60 concurrent requests under the 40-RPM cap.
  **[ESTIMATE]** 30k items ≈ 12.5 h at 40 RPM. **Lock it only after the §6 pilot.** If DeepSeek is not
  clearly better than Nemotron-3-Ultra there, switch the two roles: Nemotron as generator, DeepSeek as judge.
- **Independent solver / judge (row D): `nvidia/nemotron-3-ultra-550b-a55b`.** It is a different family from
  the generator (NVIDIA hybrid Mamba-MoE vs DeepSeek MoE), so its agreement is real evidence and not
  self-agreement. It scored 10/10 and was the fastest strong model (median 9–15 s), which matters because
the judge runs on every generated item. Its weights are OpenMDW-1.1, with no restriction on outputs. GLM-5.3-Flash
(MIT, 10/10, a third family) is the second LLM solver for majority vote. SymPy stays the primary check, as
in the plan.
- **Weak generator (row C): `nvidia/nemotron-3.5-bolt-e4b`.** It replaces the retired Llama-3.1-8B. It is
  the only small (~4.5B effective, Gemma-4-E4B-based) model callable on NIM. It is fast (median 5–6 s) and
  writes fluent Vietnamese. It is ~100× smaller than the generator, so "weak vs frontier" is a clear
  contrast. Weights are open (OpenMDW). Caveat: it scored 10/10 on the smoke items, so it is "weaker", not
  "weak". The paper should report its measured Vi-GSM8K / Vi-ExamMath accuracy next to the generator's
  instead of assuming the gap.
- **Difficulty-reference solver (row I): a frozen small open model that we run ourselves.** Proposed:
  `Qwen3.5-2B` (instruct, non-thinking, greedy) on vLLM, on the same Modal L4 used for eval. Bolt-E4B is
  likely to saturate easy items (10/10 here), which would flatten the difficulty labels. A self-run model
  also cannot be retired by a provider. Hosted alternative: Bolt-E4B.
- **Fallbacks:** generator → Nemotron-3-Ultra (then GLM-5.3-Flash, MIT). Judge → GLM-5.3-Flash. Weak → a
  mid-strength `gemma-4-31b-it` (Apache-2.0) if Bolt-E4B is retired. Do **not** depend on Groq / Gemini /
  DeepSeek-direct: no keys exist.

### 5.2 Students — and the day-1 pilot gate

**Primary family: Qwen3.5-Base 0.8B and 2B, plus 4B as a scale point.**
1. **Current.** Qwen3.5 (Feb 2026) is the newest Qwen line with ≤4B sizes. Qwen3.6/3.8 have none.
2. **Licence.** Apache-2.0, not gated.
3. **Multilingual.** 201 languages and a 248K vocabulary. The instruct versions are far ahead of
   same-size alternatives on multilingual benchmarks (2B-it MMMLU 63.1; 4B-it 76.1).
4. **Q2 lens.** Three sizes in one family gives a scaling axis. A reviewer in 2027 would see Qwen3 (Apr 2025)
   as one generation behind.

**Second family: Gemma-4-E2B.** It has a different tokenizer (262K), a different architecture (PLE,
sliding-window) and a different pre-training corpus. That is the minimum a reviewer will ask for before
accepting that "S1 vs S2" generalises beyond Qwen. Apache-2.0, current. (§3.4 explains why not Gemma-3-1B.)

**Optional Vietnamese-adapted control: Sailor2-1B.** It asks whether a Vietnamese-adapted base changes the
augmentation conclusion. It is cheap (1B, mature tooling).

**Day-1 pilot gate.** Qwen3.5 has more tooling risk than Qwen3 and no official base math numbers, so lock
it only if all of these pass on one L4 session (≈1–2 h **[ESTIMATE]**):
1. Unsloth/TRL bf16 LoRA on Qwen3.5-0.8B-Base and 2B-Base runs with **packing on**, and the loss behaves
   normally. Check the "loss → 0" report.
2. A vLLM ≥0.17 (0.33 today) merged-adapter eval runs through `eval/math_verify.py`.
3. Few-shot Vi-GSM8K (500 items) base accuracy: Qwen3.5-2B-Base ≥ Qwen3-1.7B-Base, and Qwen3.5-0.8B-Base ≥
   Qwen3-0.6B-Base. Run the same for Gemma-4-E2B.
4. Tokenizer fertility on FLORES vie (§3.2), recorded for the paper.

If 1 or 2 fails → use **Qwen3-Base 0.6B / 1.7B / 4B** as the primary family. They have published base
numbers and mature tooling, so the paper loses nothing in rigour. The second family (Gemma-4-E2B) is kept
either way.

**Not fine-tuned (leaderboard rows F/G):** Qwen3.5-4B (it), Gemma-4-E4B-it, OpenMath-Nemotron-1.5B (strongest
small math model: AIME24 61.6), DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-Math-1.5B-Instruct, Sailor2-3B-Chat,
Arcee-VyLinh, Nemotron-3.5-Bolt-E4B, and the hosted teachers (Nemotron-3-Ultra, DeepSeek-V4.1-Flash,
GLM-5.3-Flash, Gemma-4-31B).

### 5.3 What changes in `RESEARCH_PLAN.md` §6.0 (for the owner to apply; this report does not edit it)

| Row | Old | New |
|---|---|---|
| A, B | deepseek-v4-pro-0813 (NIM) — **EOL 2026-09-14** | deepseek-v4.1-flash (NIM) |
| C | llama-3.1-8b-instruct (NIM) — **EOL 2026-08-26** | nemotron-3.5-bolt-e4b (NIM) |
| I | llama-3.1-8b-instruct (NIM) — **EOL** | self-run frozen Qwen3.5-2B (vLLM); hosted fallback Bolt-E4B |
| D | v4-pro + llama-3.1-8b + optional Groq | SymPy + nemotron-3-ultra-550b-a55b + glm-5.3-flash (two NIM families besides the generator) |
| new | — | Claude Opus 5.5 (Batch) for benchmark QA, eval-set translation repair and audit judging (§8). Not for training data. |
| E | Qwen3-0.6B/1.7B-Base | Qwen3.5-0.8B/2B(/4B)-Base + Gemma-4-E2B (+ Sailor2-1B); Qwen3-Base as the gated fallback |
| F/G | as listed | add Gemma-4-E4B-it, Qwen3.5-4B, OpenMath-Nemotron-1.5B, Nemotron-3.5-Bolt-E4B, hosted teachers |
| `.env` comments | "fallback chain NIM → DeepSeek → Groq" | That chain is empty (keys blank). Either the user fills the keys, or the plan must not rely on them. |

---

## 6. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| **Hosted model retired mid-study** | 8 NIM models in §2.2 already EOL; v4-pro-0813 lasted ~1 month | Generate the full corpus in one short window (≈ days). Log `model`, date, `seed`, the full request and the raw response. Keep a frozen JSONL. Only open-weight generators, so a reviewer can re-host. Re-probe weekly (`nim_probe.py`). |
| **Silent model drift under one ID** | DeepSeek's own API now routes `deepseek-flash` → V4.1-Flash-**1017**. The NIM `deepseek-v4.1-flash` snapshot is unspecified **[UNVERIFIED which snapshot]** | Record the response `model` field and the date. Treat "NIM deepseek-v4.1-flash, Nov 2026" as the version. |
| **NIM limits not published / account-specific** | NVIDIA staff: limits "vary for each model, and we do not publish those". Half of the listed models were not entitled for this account | Throttle to ≤ 35 RPM. Probe first. Keep a fallback model ready. |
| **NIM latency is high and variable** | Smoke test: some calls took 75–130 s on simple items | Run parallel workers under the RPM cap. Budget [ESTIMATE] ~1.5–3× the old "1–3 h" for 15–30k items. |
| **Teacher self-agreement inflates verification** | — | Judge from a different family (Nemotron) + SymPy as the authority. |
| **Smoke test too small to rank teachers** | 10 items. The top models are all at or near ceiling | Treat §2.4 as availability + fluency evidence only. Before locking the generator, run a 100-item Vi-GSM8K + 50 hard-item pilot on the top 3. |
| **Qwen3.5 tooling** (hybrid kernels, no QLoRA, packing bug report) | §3.3 | Day-1 pilot gate (§5.2) with the Qwen3 fallback. |
| **Gemma-4-E2B memory** (5.1B total) | Unsloth 8–10 GB | Train on L4. Not on laptop/T4 with packing. |
| **Floor effects / weak Vietnamese in small students** | Gemma-3-1B GSM8K 2.20; no Vietnamese math numbers published for any 2026 small model | Pilot few-shot Vi-GSM8K before the grid. Drop any base under ~15% few-shot **[ESTIMATE threshold]**. |
| **Test-set contamination via hosted APIs** | Inputs to hosted APIs leave our control | Keep Vi-ExamMath test items off hosted APIs until release. |
| **Missing keys** | `.env` audit (§0) | The user decides whether to create Groq/HF/DeepSeek keys. Nothing in the recommended stack needs them. |
| **Claude outputs leaking into training data** | Plan keeps Claude on the evaluation side (§8.4) | Keep Claude-written text out of every SFT/RL file. Tag provenance per field. |
| **Weak model is not weak enough** | Bolt-E4B 10/10 on the smoke items | Measure the actual generator gap on the pilot. If it is < ~15 points on Vi-GSM8K, use a smaller self-run model (e.g. Qwen3.5-0.8B-it) as the weak generator **[ESTIMATE threshold]**. |

---

## 7. Files (scratchpad: `C:\Users\tranh\AppData\Local\Temp\claude\E--MATH-REASONING-THESIS-vi-math-smt\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad\models\`)

| File | What |
|---|---|
| `hf_list_authors.py`, `hf_search.py`, `hf_vi.py`, `hf_model_info.py` (+ `student_ids*.txt`, `teacher_ids.txt`) | HF API queries. Outputs: `hf_authors_listing.json`, `hf_student_info.json`, `hf_student_info2.json`, `hf_teacher_info.json` |
| `list_api_models.py` → `api_models_raw.json` | Model lists from NIM, Groq, Gemini, DeepSeek, HF router and OpenRouter (keys never printed) |
| `nim_probe.py` → `nim_probe.json`, `nim_probe.log` | Callability probe for 37 NIM IDs (account ID redacted) |
| `smoke_teachers.py` | Smoke test (easy set; `SET=hard` for the hard set) |
| `smoke_A/B/C.jsonl`, `hard_D/E/F/G.jsonl` (+ `.log`) | Raw per-call results, including full model outputs |
| `summarize.py` → `smoke_summary.json` | Re-scoring + per-model summary |
| `test_parse.py` | Unit check of the answer parser |
| `per_model_compact.jsonl` | Compact per-model summary (reproduced in Appendix A) |
| `claude_cost_estimate.py` → `claude_cost_estimate.txt` | Claude $100 budget arithmetic (§8.2) |

---

## 8. Claude (Anthropic API) — the user's $100 credit

No live calls were made: there is no Anthropic key in `.env`. Everything here comes from the official docs,
fetched 2026-11-07.

### 8.1 Models and prices ([pricing page](https://platform.claude.com/docs/en/about-claude/pricing))

| Model ID | Input $/MTok | 5-min cache write | 1-h cache write | Cache hit | Output $/MTok | **Batch** input / output |
|---|---|---|---|---|---|---|
| `claude-opus-5-5` | $4 | $5 | $8 | $0.20 (0.05× input) | $20 | **$2 / $10** |
| `claude-sonnet-5-5` | $2 | $2.50 | $4 | $0.10 (0.05× input) | $10 | **$1 / $5** |
| `claude-haiku-5-5` (prompts ≤ 100K tokens) | $0.10 | $0.125 | $0.20 | $0.01 | $0.50 | **$0.05 / $0.25** |

- **Batch API:** "a 50 percent discount on both input and output tokens". Up to 100,000 requests or 256 MB
  per batch. "most batches completing within 1 hour", and a batch expires after 24 h. Results are kept for 29
  days. Opus 5.5, Sonnet 5.5 and Haiku 5.5 are all supported ([batch docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)).
- **Caching stacks with Batch:** "These multipliers stack with other pricing modifiers, including the Batch
  API discount". For batches, the docs advise the 1-hour cache, because batches can take longer than 5
  minutes.
- **Tokenizer:** Claude 4.7+ models "produce approximately 30 percent more tokens for the same text" than
  older Claude models. Vietnamese with diacritics is likely to be at the high end **[UNVERIFIED]**. Before
  any spend, run `count_tokens` on 20 real items to replace the assumptions below.
- **Thinking:** on Opus 5.5, thinking cannot be disabled (effort is the only control; default `medium`). It
  is billed as output. The budgets below include thinking tokens.

### 8.2 What $100 buys **[ESTIMATE]** (`claude_cost_estimate.py`)

Per-item token assumptions (Claude tokens): **generate** = 0.7k uncached + 1.5k cached prefix in, 1.5k
visible + 1.0k thinking out. **Judge/verify** = 1.8k + 1.2k cached in, 0.3k + 0.7k out. **Translate** =
0.7k + 1.2k cached in, 0.9k + 0.3k out. **Repair a flagged translation** = 1.5k + 1.2k cached in, 0.7k + 0.5k
out.

| Role | Opus 5.5, standard | **Opus 5.5, Batch** | Sonnet 5.5, Batch | Haiku 5.5, Batch |
|---|---|---|---|---|
| Generate a Vietnamese problem + solution (~1.5k out) | $0.053/item → 1,883 items | **$0.027 → 3,766** | $0.013 → 7,532 | $0.0007 → ~150k |
| Judge / verify one item | $0.027 → 3,644 | **$0.014 → 7,288** | $0.007 → 14,577 | $0.0003 → ~289k |
| Translate one item | $0.027 → 3,698 | **$0.014 → 7,396** | $0.007 → 14,792 | $0.0003 → ~293k |
| Repair a flagged translation | $0.030 → 3,306 | **$0.015 → 6,613** | $0.008 → 13,227 | $0.0004 → ~262k |

The numbers move roughly linearly with output length. Doubling thinking roughly halves the item counts.

### 8.3 Where Claude outputs go in this plan

- Claude is used to **evaluate, QA, translate or repair evaluation and benchmark items**.
- Claude verdicts are kept **audit-only** (measured and reported, not used to select training data).
- No training file, and no released fine-tune, contains Claude-written text in this plan.
- A benchmark whose items Claude helped QA or translate records Claude's role in the datasheet (model id,
  prompt version, date).
- The held-out Vi-ExamMath test items are sent to Claude for QA and are kept off NIM and Gemini.

### 8.4 Where Claude adds the most quality per dollar

Bulk generation is the wrong use. NIM already does it for $0 with open-weight, permissively licensed models
(§2). $100 of Opus Batch is only ~3.8k items, against
the ~30k the plan needs.

The scarce, high-value roles are on the **evaluation side**. There the free hosted models either carry a
contamination risk (inputs leave our control) or share a family with the generator:

| Use (priority order) | Why Claude here | Model | Volume | **[ESTIMATE]** cost |
|---|---|---|---|---|
| 1. **Vi-ExamMath QA:** solve each item independently, check the official answer key, flag ambiguous or ill-posed statements and OCR/LaTeX errors | Test items must not go to NIM (contamination). Needs the strongest independent solver. Directly raises benchmark validity, which is what reviewers scrutinize | Opus 5.5, Batch, effort `high` | ~2,000 items × 1 solve+audit call (~2× judge tokens) | ≈ $55 |
| 2. **Eval-set translation check + repair** (Vi-GSM8K test, Vi-MATH-500): flag mistranslations, repair the flagged ones | Translation errors in eval sets silently cap accuracy (an N-series audit item). Eval data never trains a model | Opus 5.5 Batch for flagging. Repair only the ~10–20% flagged **[ESTIMATE]** | ~1,800 checks + ~300 repairs | ≈ $25 + $5 |
| 3. **Audit-only quality judging** of a stratified sample of each augmentation arm (S1/S2/S3 data-quality panel, §5.4): correctness, fluency, pedagogy | An independent fourth-family judge strengthens the C2 data-quality claims. Audit-only | Sonnet 5.5 Batch | 4 arms × 300 items | ≈ $8 |
| 4. Reserve | Re-runs, prompt iteration, `count_tokens` calibration | — | — | ≈ $7 |

**In the stack (§0):** Claude Opus 5.5 (Batch, 1-h cache on the shared rubric prefix) becomes the
**benchmark QA, eval-translation repair and audit judge**. It writes nothing that enters a training file.
Generation stays on NIM.

---

## Appendix A — raw smoke-test summary (`per_model_compact.jsonl`)

```json
{"model": "deepseek-ai/deepseek-v4.1-flash", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 38.1, "max_latency_s": 41.1, "median_completion_tokens": 457, "median_prompt_tokens": 177, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "deepseek-ai/deepseek-v4.1-flash", "set": "hard", "n": 6, "api_ok": 6, "correct": 6, "median_latency_s": 95.8, "max_latency_s": 156.9, "median_completion_tokens": 613, "median_prompt_tokens": 170, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "google/gemma-4-31b-it", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 194.0, "max_latency_s": 395.3, "median_completion_tokens": 479, "median_prompt_tokens": 124, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "google/gemma-4-31b-it", "set": "hard", "n": 6, "api_ok": 5, "correct": 5, "median_latency_s": 131.2, "max_latency_s": 410.6, "median_completion_tokens": 465, "median_prompt_tokens": 122, "cjk_chars_total": 0, "truncated": 0, "errors": ["APITimeoutError: Request timed out."]}
{"model": "minimaxai/minimax-m3.1-flash", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 57.3, "max_latency_s": 112.2, "median_completion_tokens": 634, "median_prompt_tokens": 371, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "minimaxai/minimax-m3.1-flash", "set": "hard", "n": 3, "api_ok": 3, "correct": 3, "median_latency_s": 35.1, "max_latency_s": 53.5, "median_completion_tokens": 748, "median_prompt_tokens": 368, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "mistralai/mistral-large-4-1t-a52b", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 38.8, "max_latency_s": 59.6, "median_completion_tokens": 1418, "median_prompt_tokens": 132, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3-ultra-550b-a55b", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 14.8, "max_latency_s": 29.0, "median_completion_tokens": 819, "median_prompt_tokens": 135, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3-ultra-550b-a55b", "set": "hard", "n": 6, "api_ok": 6, "correct": 6, "median_latency_s": 9.4, "max_latency_s": 20.8, "median_completion_tokens": 863, "median_prompt_tokens": 131, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3.5-bolt-e4b", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 5.0, "max_latency_s": 6.3, "median_completion_tokens": 427, "median_prompt_tokens": 120, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3.5-bolt-e4b", "set": "hard", "n": 6, "api_ok": 6, "correct": 6, "median_latency_s": 6.3, "max_latency_s": 12.8, "median_completion_tokens": 550, "median_prompt_tokens": 116, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3.5-lightning-30b-a3b", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 18.0, "max_latency_s": 132.3, "median_completion_tokens": 1500, "median_prompt_tokens": 135, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "nvidia/nemotron-3.5-lightning-30b-a3b", "set": "hard", "n": 6, "api_ok": 6, "correct": 5, "median_latency_s": 18.2, "max_latency_s": 81.3, "median_completion_tokens": 1281, "median_prompt_tokens": 131, "cjk_chars_total": 0, "truncated": 1, "errors": []}
{"model": "stepfun-ai/step-5-preview", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 20.1, "max_latency_s": 34.1, "median_completion_tokens": 858, "median_prompt_tokens": 163, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "stepfun-ai/step-5-preview", "set": "hard", "n": 6, "api_ok": 6, "correct": 6, "median_latency_s": 22.2, "max_latency_s": 133.1, "median_completion_tokens": 680, "median_prompt_tokens": 156, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "z-ai/glm-5.3-flash", "set": "easy", "n": 4, "api_ok": 4, "correct": 4, "median_latency_s": 26.5, "max_latency_s": 36.2, "median_completion_tokens": 522, "median_prompt_tokens": 128, "cjk_chars_total": 0, "truncated": 0, "errors": []}
{"model": "z-ai/glm-5.3-flash", "set": "hard", "n": 6, "api_ok": 6, "correct": 6, "median_latency_s": 17.5, "max_latency_s": 40.3, "median_completion_tokens": 595, "median_prompt_tokens": 127, "cjk_chars_total": 0, "truncated": 0, "errors": []}
```
(`mean_vi_diacritic_ratio` is 0.21–0.28 for every model and is omitted here; it is in the file.)
