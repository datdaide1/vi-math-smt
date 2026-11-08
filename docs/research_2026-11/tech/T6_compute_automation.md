# T6 — Maximum free throughput and full automation (2026-11-07)

**Question.** Can the full Q2 plan run "in 1 day" on free compute, with Claude driving Kaggle, Modal, the
laptop and NIM from one CLI? If not, what is the fastest honest schedule?

**Evidence keys.**
- `[CLI]` = a command run in this session (kaggle 2.2.5 / modal 1.6.3 installed into the scratchpad venv
  `scratchpad/t6/kenv`, no credentials).
- `[SRC]` = the installed package source code.
- `[DOC]` = an official page I opened.
- `[COORD]` = measured by the coordinator on the user's own Modal workspace.
- `[UNVERIFIED]` = not confirmed from an official source.
- `[ESTIMATE]` = my arithmetic. All arithmetic is in `scratchpad/t6/feasibility.py`, output in
  `feasibility_out.txt`. Treat every compute figure as **±2×**.

---

## 0. Bottom line

1. **One day is not possible. It misses by about 10×, even with every free source running in parallel.**
   - The full plan needs **218–299 L40S-equivalent GPU-hours**: 230 SFT + 30 GRPO runs + evaluation.
   - The first 24 hours can supply about **22.5 L40S-eq-h**: all of Modal's $30, one day of Kaggle quota, and
     the laptop. That is **8–10 %** of the need.
   - Nothing that is free closes that gap.
2. **The prices in `03_training_infra.md` §4–§6.6 are wrong by 3,600 s/h.**
   - The pricing page lists prices **per second**. Report 03 copied them as per hour: L4 is
     $0.000222/s = **$0.80/h**, not $0.222/h `[DOC modal.com/pricing]`.
   - This matches `modal billing rates` `[COORD]`.
   - So $30/month buys **≈12.9 L40S-h** (with CPU and RAM included), not 55.
   - The plan therefore costs **$509–698** at Modal list prices. That is 17–23 monthly cycles if Modal were
     the only source. Every Modal number in 03 §5–§6.6 has to be multiplied by 3.6.
3. **Modal's own docs now say a payment method is required.** "Using a GPU requires having a valid payment
   method on file" `[DOC modal.com/docs/guide/gpu]`. Also: "you must have a payment method on file in order to
   use Modal" `[DOC …/guide/billing]`.
   - This conflicts with the project rule "never add a payment method".
   - **User decision D1:** run a 1-minute T4 smoke job.
     - If it is refused, either add a card with a spend limit (§2.2) or drop Modal GPUs. Dropping them makes
       the schedule 24–33 weeks instead of 18–24.
4. **Binding constraints, in order** (central estimates):

   | Rank | Constraint | Time |
   |---|---|---|
   | 1 | GPU at $0 (Kaggle's 30 h/week quota + Modal's $30/month) | **18–24 weeks (≈4–5.5 months)** |
   | 2 | Human annotation: 223 person-hours | ≈12 days with 3 people × 6 h/day |
   | 3 | NIM generation: ≈305k calls at 40 RPM | **5.3 days** |
   | 4 | Laptop: the 0.8B block only | ≈10 days |

   Items 2–4 run in parallel with item 1, so **only the GPU constraint is on the critical path.**
5. **Fastest honest schedule at $0:**
   - About **3 weeks** for everything except GPU training: pilots, benchmark QA, generation, human ratings.
   - **About 4–5.5 months** for the GPU grid. The ±2× band is 9–49 weeks.
   - Even with an unlimited budget, the Modal Starter plan's **10-GPU concurrency** makes the GPU phase
     ≥22–30 h of wall-clock. NIM and the human raters would still take about 2 weeks. So "1 day" is not
     reachable at any budget without more raters and a higher NIM tier.
6. **Automation is feasible.** Kaggle (CLI 2.2.5) and Modal (SDK 1.6.3) can both be driven end-to-end from
   the laptop. Colab cannot (§3). The user does a one-time setup of about 30 minutes (§1.5, §2.5). After
   that, one orchestrator (§6) runs everything.

---

## 1. Kaggle automation (verified from the CLI source)

### 1.1 What the CLI can do — `kaggle` 2.2.5 (latest on PyPI, `pip index versions kaggle` `[CLI]`)

**Packaging bug.** 2.2.5 imports `six` but does not declare it. A clean install crashes with
`ModuleNotFoundError: No module named 'six'` `[CLI]`. Install it as `pip install kaggle==2.2.5 six`.

| Need | Command (exact syntax from `--help` `[CLI]`) | Notes |
|---|---|---|
| Push + run a job | `kaggle kernels push -p DIR [-t SECONDS] [--accelerator ACC] [--no-run]` | `-t` "limit the run time … global maximum time will not be exceeded" |
| Poll | `kaggle kernels status OWNER/SLUG[/VERSION]` | Server enum `KernelWorkerStatus`: QUEUED, RUNNING, COMPLETE, ERROR, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, NEW_SCRIPT `[SRC kagglesdk/kernels/types/kernels_enums.py]` |
| Live log | `kaggle kernels logs -f OWNER/SLUG` | streams like `tail -f` |
| Get outputs | `kaggle kernels output OWNER/SLUG[/VERSION] -p PATH [--file-pattern REGEX] -o` | version-addressable |
| Get code back | `kaggle kernels pull OWNER/SLUG -p PATH -m` | `-m` regenerates the metadata |
| Upload data/code/wheels | `kaggle datasets create -p DIR [-r zip]` (private by default); `kaggle datasets version -p DIR -m MSG [-d]` | `-d` deletes old versions |
| Upload model weights | `kaggle models create`, `kaggle models variations create`, `kaggle models variations versions create OWNER/MODEL/FRAMEWORK/SLUG -p DIR` | |
| **GPU quota** | `kaggle quota` → used / remaining / total / refreshAt for GPU and TPU | `[SRC quota_view_cli → get_accelerator_quota_statistics]`. Lets the orchestrator read the real quota instead of guessing |

**`kernel-metadata.json` fields** `[SRC kernels_push]`, `[DOC github.com/Kaggle/kaggle-cli/docs/kernels_metadata.md]`:

- `id` (`owner/slug`), `title` (≥5 characters; it must slugify to `id`), `code_file`.
- `language` (`python`), `kernel_type` (`script` | `notebook`).
- `is_private` (default true).
- `enable_gpu`, `enable_tpu`, `enable_internet`.
  - Source default for `enable_internet` is true; the docs say false. **Always set it explicitly.**
- `machine_shape`.
- `dataset_sources` (`user/slug`), `kernel_sources` (`user/slug`),
  `model_sources` (`user/model/framework/variation/version`), `competition_sources`.
- `docker_image_pinning_type` ∈ {`original`, `latest`} `[SRC valid_push_pinning_types]`, plus `docker_image`.

**Machine shape.**
- The CLI docs list `NvidiaTeslaT4` (T4 ×2), `NvidiaL4`, `NvidiaL4X1`, `NvidiaTeslaA100`, `NvidiaH100`,
  `NvidiaRtxPro6000`, `TpuV5E8`, `TpuV6E8`. They also warn: "Some of these are only available to
  participants of specific competitions, and some are only available to Kaggle admins" `[DOC kernels.md]`.
- **P100 is retired.** The CLI source says "'nvidiateslap100' … Sessions run on the default GPU
  (NvidiaTeslaT4) instead" `[SRC _RETIRED_ACCELERATORS]`.
- So the free shape is **T4 ×2**. Whether a free account can get `NvidiaL4` is **[UNVERIFIED]**. Probe it
  once with `--accelerator NvidiaL4` and log `nvidia-smi`.

### 1.2 Quota, session caps, concurrency

| Item | Value | Evidence |
|---|---|---|
| Weekly GPU quota | "30 hours or sometimes higher depending on demand and resources", weekly reset | Official Kaggle GPU-tips text as quoted by fastero.com (2026-10-31). kaggle.com/docs is JS-rendered and did not load → **[UNVERIFIED-official]**. Measure with `kaggle quota`. |
| Session cap | 12 h for CPU/GPU, 9 h for TPU | same source; the CLI hard-codes `_DEFAULT_WAIT_TIMEOUT = 12*60*60  # A Kaggle notebook runs for at most 12 hours` `[SRC]` |
| Concurrent GPU sessions | Old official notice: 1 interactive + 2 batch ("commit") GPU sessions; a recent forum post reports "Maximum interactive GPU session count of 1 reached" | **[UNVERIFIED]**. The orchestrator probes it: push 2 GPU kernels and check whether the second stays QUEUED |
| T4 ×2 accounting | One session-hour is reportedly charged once for both GPUs | **[UNVERIFIED]**, secondary source (03 §6.6) |
| Phone verification needed for GPU and internet | widely reported | **[UNVERIFIED-official]** |
| Accounts | one per person | Plan with **one account** |

**Usable Kaggle GPU throughput.**
- Run 2 independent jobs per T4×2 session (one per GPU), with 30 session-h/week. That gives **≤60 T4-h/week
  ≈ 9.0 L40S-eq-h/week** `[ESTIMATE]`.
- In the first 24 h: 24–30 session-h ⇒ **48–60 T4-h** (7.2–9.0 L40S-eq-h).
- The T4 has no bf16. Training is **fp16**.
  - For RL this is arguably an advantage: arXiv:2510.26788 reports "simply reverting to FP16 effectively
    eliminates this mismatch" (BF16 training–inference mismatch in RL) `[DOC arXiv abs]`.
  - Gemma-family fp16 overflow is a known risk class **[UNVERIFIED for Gemma-4]** → it needs a pilot gate (§5.4).

### 1.3 Design choices for Kaggle dispatch

| Decision | Option | Pros | Cons | Verdict |
|---|---|---|---|---|
| How a job reaches Kaggle | **A. "Worker-slot" kernels** (`vm-slot-1`, `vm-slot-2`). Each job = a new version whose `run.py` embeds the job JSON. Outputs are fetched by `OWNER/SLUG/VERSION` | 2 slugs = 2 concurrency slots; version history is the audit log; no secrets on Kaggle | needs version bookkeeping | **Recommended** |
| | B. One kernel per job (230+ slugs) | simple mapping | clutter; the title/slug rule adds friction | fallback |
| | C. A long-running Kaggle worker that pulls a queue over the internet | fewer pushes | needs secrets inside Kaggle; still capped at 12 h | reject |
| Resume across the 12 h cap | **A. Laptop relay.** `kernels output` → `datasets version` of `vm-ckpt-<job>` → re-push with it in `dataset_sources` and `RESUME=1` | CLI-only; no tokens on Kaggle | ~0.5 GB of transfer per hop | **Recommended** |
| | B. Push the checkpoint to HF Hub from inside Kaggle | fast | an HF write token must live in Kaggle Secrets, which are set only in the UI **[UNVERIFIED]** | fallback |
| | C. `kernel_sources` chaining (the next kernel mounts the previous output) | no re-upload | self-reference of the same slug is unclear **[UNVERIFIED]** | test once |
| Environment | **A. `docker_image_pinning_type: "original"` + a private "wheels" dataset with exact pins** | reproducible; works without internet | one upload per version | **Recommended** |
| | B. `pip install` from the internet on every run | simple | version drift; needs `enable_internet` | reject for the grid |

### 1.4 End-to-end CLI outline (laptop; Claude can run every line once a token exists)

```bash
# 0. one-time: pip install kaggle==2.2.5 six ; token in $KAGGLE_API_TOKEN or ~/.kaggle/access_token
kaggle quota                                             # read the real weekly GPU quota
kaggle datasets create -p build/code   -r zip            # user/vimath-code   (repo snapshot, versioned)
kaggle datasets create -p build/wheels -r zip            # user/vimath-wheels (pinned wheels)
kaggle datasets create -p build/data   -r zip            # user/vimath-data   (frozen arms; train side only)

# 1. render a job into a slot folder
#    jobs/slot-1/kernel-metadata.json:
#    {"id":"USER/vm-slot-1","title":"vm slot 1","code_file":"run.py","language":"python",
#     "kernel_type":"script","is_private":true,"enable_gpu":true,"enable_internet":true,
#     "machine_shape":"NvidiaTeslaT4","docker_image_pinning_type":"original",
#     "dataset_sources":["USER/vimath-code","USER/vimath-wheels","USER/vimath-data","USER/vm-ckpt-<job>"]}
#    run.py = header that pip-installs from /kaggle/input/vimath-wheels (--no-index), then
#             JOBS=[<job A json>, <job B json>]; starts 2 subprocesses with CUDA_VISIBLE_DEVICES=0 / 1,
#             asserts nvidia-smi shows "Tesla T4", writes /kaggle/working/<job>/{manifest.json,ckpt/,preds.jsonl}
kaggle kernels push -p jobs/slot-1 -t 41400               # 11.5 h, leaves margin for the final save

# 2. poll (adaptive, >=60 s)
while :; do s=$(kaggle kernels status USER/vm-slot-1); case "$s" in
  *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 120; done

# 3. pull outputs; if the job is unfinished -> relay the checkpoint and resume
kaggle kernels output USER/vm-slot-1 -p runs/<job> -o
kaggle datasets version -p runs/<job>/ckpt -m "resume <job> step N" -r zip   # USER/vm-ckpt-<job>
kaggle kernels push -p jobs/slot-1                         # same job with RESUME=1
```

### 1.5 What the user must do once (Claude must not create accounts or handle credentials)

1. Log in to Kaggle and **phone-verify** the account (needed for GPU and internet).
2. Generate a token at `kaggle.com/settings/api`.
3. Save it as `~/.kaggle/access_token` or the `KAGGLE_API_TOKEN` env var. Legacy `kaggle.json` is still
   accepted (auth order: access token → legacy key → OAuth `[SRC authenticate]`).
4. Run `kaggle quota` once to confirm GPU access.

---

## 2. Modal (Starter plan)

### 2.1 Verified facts

| Item | Value | Evidence |
|---|---|---|
| GPU list price | T4 **$0.59/h**, L4 **$0.80/h**, A10 $1.10/h, L40S **$1.95/h**, A100-40 $2.10/h, A100-80 $2.50/h, H100 $3.95/h, H200 $4.54/h, B200 $6.25/h. Page values are per second (e.g. L4 $0.000222/s) | `[DOC pricing]` × 3600, `[COORD modal billing rates]` |
| CPU / RAM | $0.0472/physical core/h (minimum 0.125 core), $0.0080/GiB/h | `[DOC]`, `[COORD]` |
| Starter plan | "$30 / month free credits, 3 workspace seats, 100 containers, 10 GPU concurrency" | `[DOC pricing]` |
| Multipliers | region selection 1.15–1.75×; non-preemptible 3×. **GPU functions cannot be non-preemptible**: "All Modal Functions are subject to preemption by default", and Modal restarts the same input | `[DOC pricing, guide/preemption]` |
| Timeout | default 300 s, maximum **24 h**; each `modal.Retries` attempt gets a new timeout | `[DOC guide/timeouts]` |
| Fan-out | `.map`/`.starmap` run inputs in parallel; `max_containers` caps a Function; hard limit of 4,000 containers per Function; plan limits apply | `[DOC guide/scale]` |
| Async jobs | `fn.spawn(x)` → `call.object_id`; later `FunctionCall.from_id(id).get(timeout=0)`; results kept **7 days** | `[DOC guide/job-queue]` |
| Cold start | "Containers boot in about one second"; ready time "from seconds to minutes"; put weights on disk early (Volume or Image) | `[DOC guide/cold-start]` |
| **Payment method** | "Using a GPU requires having a valid payment method on file." / "you must have a payment method on file in order to use Modal." | `[DOC guide/gpu]`, `[DOC guide/billing]` |
| Spend limit | caps out-of-pocket charges after credits: "When the spend limit is reached, Modal stops workloads that would incur additional out-of-pocket charges." Whether it can be set to $0 is not stated | `[DOC guide/budgets]` |
| Workspace state | $0.06 metered, $0.00 billed this month; the CLI does not show the credit balance | `[COORD]` |

### 2.2 Can the $30 be spent in parallel in one day?

Yes. Nothing limits the spend rate apart from the 10-GPU concurrency.

- At 10 × L40S, with $2.33/h effective per container (L40S $1.95 + 4 cores + 24 GiB = $0.38/h
  `[ESTIMATE]`), **$30 = 12.9 L40S-h, gone in ≈1.3 h of wall-clock.**
- Then nothing more until next month's credit.

**$ per unit of useful compute** (`feasibility_out.txt`):

| GPU | effective $/h | $30 buys | **$ per L40S-eq-h** |
|---|---|---|---|
| T4 | 0.97 | 30.9 h | 6.49 |
| L4 | 1.18 | 25.4 h | 3.53 |
| **L40S** | 2.33 | 12.9 h | **2.33 ← recommended for training** |
| A100-40 | 2.48 | 12.1 h | 2.88 |
| H100 | 4.33 | 6.9 h | 2.38 (MFU on 0.8–4B models is uncertain) |

**Decision D1 (user).** Choose one:

| Option | Effect | Verdict |
|---|---|---|
| (a) Add a card + set **spend limit = $0** (or the minimum Modal allows) | Keeps Modal free with a hard cap. This needs the user to relax the "no payment method" rule | pick if the user agrees |
| (b) No card | If GPU jobs are refused, Modal is out. Kaggle-only becomes **24–33 weeks** | default if no card |
| (c) A second workspace for more credit | Credit farming | reject |

### 2.3 Fan-out pattern (recommended)

| Option | Behaviour | Verdict |
|---|---|---|
| **A. `modal deploy` an app (`train_sft`, `train_grpo`, `evaluate`); the orchestrator calls `Function.from_name(...).spawn(spec)` and stores `object_id` in SQLite** | survives a laptop sleep; results kept 7 days; resumable | **Recommended** |
| B. `modal run --detach` with `.map(specs)` in a local entrypoint | one call for a whole grid | results tied to the client process; harder retries per item |
| C. Sandboxes | flexible | billed at the 3× non-preemptible rate `[DOC pricing]` → reject |

**Function settings:**
- `gpu="L40S"` with no fallback list. A fallback would mix hardware inside a block (§6.4).
- `timeout=6*3600`, `retries=modal.Retries(max_retries=3, backoff_coefficient=2)`.
- `max_containers=10`.
- Volumes: `vm-weights` (HF cache) and `vm-ckpt`. Commit a checkpoint every N steps, because preemption
  restarts the input and the trainer must resume from the latest checkpoint.

### 2.4 Cold-start overhead

- **Image build:** first build only, cached afterwards. Roughly 5–15 min for torch + vLLM **[ESTIMATE]**.
- **Per-container:** about 1 s boot, plus model load from a Volume (2–10 GB, ≈10–60 s) **[ESTIMATE]**.
- 03 used 0.15 h/run; I use **0.10 L40S-h/run** including checkpoint save → 23 L40S-h over 230 runs.
- **Pack several evaluations per container** to amortize it.

### 2.5 What the user does once

1. `pip install modal==1.6.3`, then `modal token new` (browser login).
2. Answer D1.
3. Optionally set a spend limit in Usage & Billing.

---

## 3. Colab and other "free" GPUs

| Source | Programmatic? | Free offering today | Usable? |
|---|---|---|---|
| **Colab free** | **No official execution API.** The FAQ points to Colab Enterprise / GCP / a local runtime for guaranteed resources. Free runtimes are notebook-UI only; at most 12 h "depending on availability" `[DOC research.google.com/colaboratory/faq.html]` | unplannable | ✗ for the orchestrator. Manual debugging only |
| Lightning AI | Studio SDK exists **[UNVERIFIED in this check]** | **one-off** "Up to 30" credits: 5 on sign-up, +25 "if you add a card"; "~50 GPU hours on interruptible machines"; phone number requested; credits expire after 12 months; 2 concurrent GPUs `[DOC lightning.ai/pricing]` | ✗ for the grid (one-off; the card is needed for most of it). At best ≈5 credits ≈ a few T4-h |
| HF Jobs | `hf jobs` CLI | open to accounts "with a positive credit balance"; free accounts get < $0.10/month of Inference credit `[search: HF jobs-pricing docs]` | ✗ |
| HF ZeroGPU | Gradio Spaces only | minutes per day (03 §4) | ✗ (only a public demo) |
| Google Cloud free trial | yes | "you must provide a credit card"; while on the Free Trial you cannot "Add GPUs to your VM instances" `[DOC docs.cloud.google.com/free/docs/free-cloud-features]` | ✗ excluded twice |

**Conclusion:** the plannable free GPU pool is **Kaggle + Modal (subject to D1) + laptop**.

---

## 4. Generation APIs: NVIDIA NIM (+ Gemini)

### 4.1 NIM limits

**Rate limit.** 40 RPM per account/key. The project measured it earlier; third-party pages agree.

**Multiple keys.**
- Whether 40 RPM is per key or per account is **undocumented**.
- A moderator answer, relayed on the NVIDIA forum (2026-07/08), says there is no rate-limit increase on that
  same tier `[DOC forums.developer.nvidia.com/t/…/378450]`.
- **Rule for this plan:** plan with **40 RPM total**.

**Batching and concurrency.** I found no batch endpoint. RPM counts requests, not tokens.

- **Required number of requests in flight** = RPM/60 × latency: latency 10 s → 7, 35 s → 23, 90 s → 60.
- **Throughput levers:**
  - (i) `n>1` samples per request, if the endpoint honours the OpenAI `n` parameter **[UNVERIFIED — test in
    the pilot]**.
  - (ii) Several items per request, e.g. "5 variants as a JSON list". This trades latency and quality risk
    for fewer calls. Measure the pass rate before adopting it.

### 4.2 Items per day

**Ceiling:** 40 × 1,440 = **57,600 calls/day** per 40-RPM quota.

**Call budget for the full data scope** (7 arms, ≤20k items each; `feasibility_out.txt`):

| Stream | Calls |
|---|---|
| T: re-translate 20k (translate + back-check, 0.9 pass) | 44,444 |
| N: 20k checked (generate + 2 independent solves, 0.6 pass) — also yields N-unchecked | 100,000 |
| Y: formalize ~11k seeds | 11,000 |
| Y: 20k checked mutations (rewrite + independent re-formalization + judge, 0.5 pass) — also yields Y-unchecked | 120,000 |
| Pilots + 10 % retries | 29,544 |
| **Total** | **≈305k calls → 5.3 days at 40 RPM** (1.1 days at 200 RPM) |

Pass rates are placeholders. Replace them with the pilot's measured rates.

**Items per day at 40 RPM:**

| Stream | Calls per accepted item | Accepted items/day |
|---|---|---|
| N | 5 | ≈11.5k |
| Y | 6 | ≈9.6k |
| T | 2.2 | ≈26k |

### 4.3 Gemini (the user now has keys)

**Limits.**
- "Rate limits are applied per project, not per API key."
- "Requests per day (RPD) quotas reset at midnight Pacific time."
- The free-tier RPM/RPD table is no longer printed in the docs. It points to the per-project page in AI
  Studio `[DOC aistudio.google.com/docs/rate-limits]`.
- Secondary sources report roughly 5–15 RPM and ~100–1,500 RPD for Flash/Flash-Lite. These conflict and
  are **[UNVERIFIED]**. **Read the real numbers in AI Studio for the project.**

- Because limits are per project, **several keys in one project add nothing**. Count **one project**.

**Throughput value.** ≤~1–1.5k calls/day ≈ **+2–3 %** of NIM capacity.

**Recommendation:**
- Do not use Gemini as a generator. It would add a second generator family, confounding the arms, and
  00 §5.1 already excluded it.
- At most, use it as an **extra judge vote on training-side items**.
- **Never send test or benchmark items** (contamination control).

---

## 5. The 1-day feasibility calculation

### 5.1 Workload

**Grid.** The structure follows 03 §6.6; the students follow 00 §5.2.

| Base | Params | SFT runs |
|---|---|---|
| Qwen3.5-0.8B | 0.87B | 30 |
| Qwen3.5-2B | 2.27B | 92 |
| Qwen3.5-4B | 4.66B | 30 |
| Gemma-4-E2B | ≈2.3B effective compute **[ESTIMATE]** | 78 |
| **Total** | | **230** (sweep 36 + core 84 + data-efficiency 96 + English control 12 + full-FT check 2) |

**Per-run assumptions:**
- Tokens: 10k × 450 tok × 2 epochs = **9.0M tokens**.
- FLOPs: 6N × 1.15 per token (LoRA + gradient checkpointing + attention).
- MFU: 0.30 on L4 and L40S, 0.25 on T4, 0.20 on the laptop.

**Per-run time:**

| Base | FLOP/run | T4 h/run | L40S h/run | Laptop h/run |
|---|---|---|---|---|
| 0.8B | 5.40e16 | 0.92 | 0.14 | 5.0 |
| 2B | 1.41e17 | 2.41 | 0.36 | (OOM) |
| 4B | 2.89e17 | 4.95 | 0.74 | (OOM) |
| G-E2B | 1.43e17 | 2.44 | 0.37 | (OOM) |

**SFT total:** 3.44e19 FLOP = **88 L40S-h = 588 T4-h**.

**Everything else:**
- Overhead: 0.10 L40S-h × 230 = **23**.
- Evaluation: 260 checkpoints × ~6k items × ~350 tokens, vLLM, ≈0.10 L40S-h each = **26**.
- **RLVR:** 30 GRPO runs at 2B scale.
  - 03 §5(b) gives 4.1 L40S-h for 1.7B at 64 prompts × 8 samples × 500 tokens × 300 steps.
  - Scaled ×2.27/1.72, that is 5.4 h/run. At 32 prompts/step it is 2.7 h.
  - 30 runs → **81–162**.
- **Total GPU need: 218–299 L40S-eq-h.** At Modal list prices that is **$509–698**.

### 5.2 Supply

| Source | First 24 h | Per week after that | Hard limits |
|---|---|---|---|
| Modal $30 (if D1 allows GPU) | **12.9** (all of it, in ≈1.3 h) | 3.0 (12.9 per month) | 10 GPUs at once; monthly credit |
| Kaggle (1 account) | **7.2–9.0** (48–60 T4-h) | 9.0 (60 T4-h) | 30 h/week, 12 h/session, concurrency [UNVERIFIED] |
| Laptop RTX 3050 Ti | 0.7 | 2.8 at 60 % duty, **but only the 0.8B block is useful (7.1 total)** | 4 GB VRAM, 7.7 GB RAM |
| Colab | 0 (no API) | 0 | — |
| **Total** | **≈22.5 L40S-eq-h = 8–10 % of the need** | **≈12 + laptop** | |

**Other resources in the first 24 h:**
- NIM: 57,600 of ~305k calls (19 %).
- Humans: ≈18 of 223 person-hours (3 people × 6 h).

### 5.3 Binding constraints and minimum calendar time for the FULL scope

| Constraint | Need | Free rate | Calendar | On the critical path? |
|---|---|---|---|---|
| **GPU (Kaggle + Modal + laptop)** | 218–299 L40S-eq-h | ≈12/week (+ laptop on 0.8B) | **18–24 weeks** (±2× band: 9–49); **24–33 weeks if Modal GPU is unavailable**; Kaggle-only need = 1,410–1,953 T4-h | **Yes** |
| Human annotation | ratings 90 (6 arms × 200 × 3 raters × 1.5 min) + benchmark QA 120 (600 items × 2 readers × 6 min) + pilot reads 13 = **223 person-h** | 18 person-h/day (3 people) | ≈12 days | No (parallel), but the pilot reads gate generation |
| NIM | ≈305k calls | 57.6k/day | 5.3 days | No (S0 / T-off / translated-seed runs start first) |
| Laptop 0.8B block | 30 × 5 h | 24 h/day | ≈6 days at 24/7, ≈10 days at 60 % duty | No |
| Serial chain | sweep → freeze LR/epochs → core/data-efficiency → RLVR (needs SFT checkpoints) → eval | — | adds days, not weeks | yes, inside the GPU span |

**What "1 day" would require** (counterfactual, outside the brief):
- About **$510–700** of GPU.
- More concurrency than Starter's 10 GPUs. At 10 GPUs the GPU work is ≥22–30 h of wall-clock plus the
  serial chain.
- A NIM tier of ≥200 RPM.
- About 12 raters working in parallel.

Even then the pilot gates (generate, then read, then decide) are sequential. **Realistic floor with money:
about 1–2 weeks. Floor at $0: about 4–5.5 months.**

### 5.4 Fastest honest $0 schedule (same scope)

| When | Work | Gate |
|---|---|---|
| Day 0 (user, ~30 min) | Kaggle phone verification + token; `modal token new`; D1; NIM RPM per key/account checked | `kaggle quota` and a Modal T4 smoke job succeed |
| Day 1 (automated) | Smoke-train 50 steps per base on its assigned platform and measure tokens/s → **replace §5.1** | loss finite; tokens/s logged |
| | Probe Kaggle concurrency and `NvidiaL4` | |
| | Gemma-E2B fp16 check on T4 | |
| | Start the NIM pilots and benchmark QA | |
| Days 1–3 | S0 LR × epoch sweep (Modal L40S for 4B; Kaggle for the others); human pilot reads | freeze the recipe |
| Days 3–9 | NIM bulk generation (5.3 days at 40 RPM) | datasheet per arm |
| | GPU: S0, T-off and translated-seed runs | |
| Weeks 2–3 | Human ratings + benchmark QA finished | α reported |
| Weeks 2–~20 | Core + data-efficiency + English control | |
| | RLVR per family as soon as that family's S0/S1/S2/S5 × 3 seeds exist | |
| | All evaluations | |
| +1 week | Analysis | |

**Platform map (determinism rule: one hardware/precision per base and block):**

| Block | Platform | Load |
|---|---|---|
| Qwen3.5-0.8B (30 SFT) | laptop bf16 | ≈150 h; fallback: Kaggle T4, the whole block |
| Qwen3.5-2B (92 SFT + 15 RLVR) | Kaggle T4 fp16 | |
| Gemma-4-E2B (78 SFT) | Kaggle T4 fp16 **if** the fp16 pilot passes, else Modal L40S | |
| Gemma-4-E2B RLVR (15) | Kaggle T4 fp16, same rule | |
| Qwen3.5-4B (30 SFT) | Modal L40S bf16 | ≈25 L40S-h ≈ $59 ≈ 2 monthly credits |
| Full-FT vs LoRA check | Modal L40S, **both arms of the comparison on L40S** | |
| All evaluation | Kaggle T4, vLLM fp16, greedy | one inference platform; validated by a 2-checkpoint cross-check against Modal L4 bf16 (≥99 % identical greedy answers, or accuracy difference inside the CI) |

**Allocation options:**

| Option | GPU calendar | Verdict |
|---|---|---|
| A. Balanced (above) | 18–24 weeks | **Recommended** |
| B. Modal-centric (03 §6.6, corrected prices) | $509–698 → 17–23 months | ✗ |
| C. Kaggle-only (D1 = no) | 24–33 weeks | fallback |

**Levers that do not cut scope:**
- GRPO at 32 prompts/step (already the low end of the range).
- Early-stop RLVR on a dev-set plateau.
- Two jobs per T4×2 session.
- Batch evaluations per session.
- `n>1` / multi-item NIM requests (§4.1).

---

## 6. One orchestrator (laptop CLI) — design

### 6.1 Stack choice

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. `vimath-orch`: ~600 lines of Python, SQLite (WAL), asyncio, platform adapters** | no server; state is auditable; fits Windows; Claude can run and repair it | written by hand | **Recommended** |
| B. Prefect / Airflow | retries and UI built in | server + DB; heavy on 7.7 GB RAM; still needs custom Kaggle/Modal adapters | ✗ |
| C. Makefile / shell loops | trivial | no state, retries or quota awareness | ✗ (only the §1.4 outline) |

### 6.2 Data model

**Job id.** `job_id = sha1(kind, base, arm, n_train, seed, recipe_ver, data_ver, prompt_ver)`, so the same
spec always gets the same id. Submission is idempotent: re-submitting a COMPLETE id is a no-op.

**Job kinds:** `gen | verify | sft | grpo | eval | upload`.

**`jobs` table columns:**
- `id`, `kind`, `spec_json`, `deps[]`, `platform` (derived from the base, never chosen freely), `priority`.
- `status` ∈ {PENDING, READY, SUBMITTED, RUNNING, NEEDS_RESUME, COMPLETE, FAILED, BLOCKED}.
- `attempts`, `external_ref` (Kaggle `slot/version` | Modal `call_id` | PID), `ckpt_uri`.
- `est_cost`, `actual_cost`, `started`, `finished`, `last_error`.

**Other tables:**
- `events` (append-only JSONL mirror).
- `budget` (Modal $ spent, Kaggle hours used from `kaggle quota`, NIM calls per minute and per day).

### 6.3 Scheduler loop (every 60 s)

```text
refresh():  Kaggle -> kernels status per busy slot; Modal -> FunctionCall.from_id(id).get(timeout=0);
            laptop -> poll PID; NIM -> in-process
for job in READY order by (critical_path_rank, longest_first):
    p = PLATFORM[job.base, job.kind]                          # fixed map, §5.4
    if not capacity(p): continue                              # Kaggle slots (measured), Modal ≤10 GPUs, laptop 1
    if p == modal  and spent + est > CAP_USD (=28):  BLOCK until next cycle
    if p == kaggle and quota_remaining < est_h + 0.5: BLOCK until refreshAt
    submit(job)                                               # Kaggle: render slot + push; Modal: spawn
on COMPLETE: fetch artifacts -> verify manifest (GPU name, versions, hashes) -> upload to HF (private) -> unblock deps
on ERROR:    classify -> retry | resume | fail (6.4)
```

**Critical-path ranking:** sweep > S0/S1/S2/S5 core (these feed RLVR) > the other core runs > RLVR >
data-efficiency > English control > evaluation backlog.

### 6.4 Retries, resume, determinism, logging

**Error classes:**

| Class | Examples | Action |
|---|---|---|
| Transient | HTTP 429/5xx, Modal preemption, Kaggle QUEUED > 2 h, network | exponential backoff with jitter, ≤3 attempts |
| Resumable | Kaggle 12 h cap or ERROR after a checkpoint exists, Modal timeout | NEEDS_RESUME → §1.3 relay, or a Modal re-spawn reading `vm-ckpt` |
| Fatal | NaN/inf loss, assertion that the GPU ≠ the expected GPU, OOM at step 0 | FAILED + alert. **No automatic platform switch**: a switch would break the hardware-per-base rule |

**Checkpoints.**
- Save adapter + optimizer + scheduler + RNG states + data-loader position every 30 min (Kaggle) or every
  N steps (Modal Volume `commit()`).
- Trainers are idempotent: at start they look for the newest checkpoint for their `job_id`.

**Determinism rules (enforced, not advisory):**
1. `PLATFORM[base]` is a constant. The run manifest records `nvidia-smi` name, driver, CUDA, torch, vLLM,
   precision, and Kaggle docker pin or Modal image id. The job **fails** if the detected GPU ≠ expected.
   This catches Kaggle's silent fallback from a retired shape `[SRC]`.
2. Fixed seeds (python/numpy/torch, data order = seed). `pip freeze` is saved per run. Wheels are pinned
   (Kaggle wheels dataset; Modal image built from a lock file).
3. One evaluation platform for all checkpoints; greedy decoding; fixed `max_tokens`; vLLM version pinned.
4. Within any compared set (arms × seeds of one base), hardware, precision and library versions are
   identical. The cross-platform evaluation check (§5.4) is a gate.

**Logging.**
- `runs/<job_id>/{spec.json, manifest.json, train_log.jsonl, preds.jsonl, metrics.json}`.
- Every NIM/Gemini/Claude request is stored raw (model id, date, parameters, request, response), as
  00 §7.3 requires.
- The API cache key is `sha1(prompt_ver, input_id, model, params, sample_idx)`, so a re-run never re-pays a
  call.
- `vimath-orch status` prints queue depth, $ spent, quota left and ETA per block.

**API adapters (NIM / Gemini):**
- Token bucket at **38 RPM** per 40-RPM quota (5 % headroom), semaphore of 64 requests in flight.
- On 429, honour `Retry-After`.
- One bucket per quota, never one per key (§4.1).
- Gemini: one bucket per project, using the AI Studio numbers.
- Benchmark and test items are **hard-blocked** from NIM and Gemini (00 §4).

**Secrets.**
- Tokens live only on the laptop (`KAGGLE_API_TOKEN`, `~/.modal.toml`, `NVIDIA_API_KEY*`, `GEMINI_API_KEY`,
  `HF_TOKEN`).
- Kaggle kernels are private and contain no tokens.
- HF uploads happen from the laptop after outputs are fetched.

### 6.5 CLI surface

```text
vimath-orch init            # create SQLite, check tokens (kaggle quota, modal token, NIM ping)
vimath-orch plan grid.yaml  # expand the grid -> jobs (idempotent ids), print GPU-h/$/calendar estimate
vimath-orch run [--only kind=sft base=Q2B] [--dry-run]
vimath-orch status | logs <job> | retry <job> | cancel <job> | budget
vimath-orch replan --measured runs/smoke/*.json   # replace §5 estimates with measured tokens/s
```

---

## 7. Changes vs 03 (§4–§6.6)

| 03 said | Correct value | Effect |
|---|---|---|
| L4 $0.222/h, L40S $0.542/h, … | **L4 $0.80/h, L40S $1.95/h** (the page lists $/s) | $30 = 12.9 L40S-h, not 55. The Modal-only plan costs $509–698, not $167–217 |
| "Free with no payment method" | Modal docs: a payment method is required for GPU and for Modal use | user decision D1 |
| Kaggle T4×2 **or P100** | P100 retired; requests fall back to T4 `[SRC]` | only the T4 path |
| "4–5 Modal cycles" | GPU calendar **18–24 weeks** with Kaggle as the main engine; 24–33 weeks without Modal | Kaggle is now the critical resource, not Modal |
| NIM not quantified | ≈305k calls ≈ 5.3 days at 40 RPM | off the critical path |

## 8. Open items (measure on day 0/1)

- Run `kaggle quota` for the real weekly GPU hours.
- Find the Kaggle concurrent batch-session count (probe with 2 pushes).
- Check whether `NvidiaL4` is available.
- Confirm Gemma-4-E2B trains stably in fp16 on T4.
- Answer D1 (does a Modal GPU job run on this workspace?).
- Measure whether the NIM 40 RPM is per key or per account.
- Read the Gemini free RPM/RPD for the project from AI Studio.
- Check whether NIM honours `n>1`.
- Measure tokens/s per base → `replan`.

## Sources

- **Kaggle:**
  - kaggle 2.2.5 / kagglesdk 0.1.38 source (`kaggle/api/kaggle_api_extended.py`: `_RETIRED_ACCELERATORS`,
    `kernels_push`, `quota_view_cli`, authentication order; `kagglesdk/kernels/types/kernels_enums.py`).
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
  - https://fastero.com/blog/kaggle-notebooks (2026-10-31, quotes the Kaggle GPU-tips text)
  - https://www.kaggle.com/docs/efficient-gpu-usage (did not render)
- **Modal:**
  - https://modal.com/pricing
  - https://modal.com/docs/guide/gpu
  - https://modal.com/docs/guide/billing
  - https://modal.com/docs/guide/budgets
  - https://modal.com/docs/guide/preemption
  - https://modal.com/docs/guide/timeouts
  - https://modal.com/docs/guide/scale
  - https://modal.com/docs/guide/job-queue
  - https://modal.com/docs/guide/cold-start
  - `modal --help`, `modal billing --help` `[CLI]`
- **Colab / other GPUs:**
  - https://research.google.com/colaboratory/faq.html
  - https://lightning.ai/pricing
  - https://docs.cloud.google.com/free/docs/free-cloud-features
  - https://huggingface.co/docs/hub/jobs-pricing (via search)
- **NIM:**
  - https://forums.developer.nvidia.com/t/request-for-nvidia-build-api-rate-limit-increase-40-rpm-200-rpm/378450
- **Gemini:**
  - https://aistudio.google.com/docs/rate-limits
- **RL precision:** https://arxiv.org/abs/2510.26788
- **Arithmetic:** `scratchpad/t6/feasibility.py` → `scratchpad/t6/feasibility_out.txt`
