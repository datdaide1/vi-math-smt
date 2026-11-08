# Datasheet — Vi-ExamMath Core v1.1 (`core_v1.1`)

Contribution C1. Builders: X2 (collection, `vi_math_verified_aug/vi_exam/`, report
[`exec/X2_benchmark.md`](../../docs/research_2026-11/exec/X2_benchmark.md)) and Q1 (key QA with LLM judges,
`vi_math_verified_aug/vi_exam_qa/`, report [`exec/Q1_benchmark_qa.md`](../../docs/research_2026-11/exec/Q1_benchmark_qa.md)).
Contamination flags: X5 ([`exec/X5_decontam.md`](../../docs/research_2026-11/exec/X5_decontam.md) §4.3).
Status: **FROZEN v1.1** (2026-11-08, ~03:10 +07).

## Files and hashes (`data/vi_exam_qa/`, verified with `sha256sum` on 2026-11-08 08:40)

| File | Rows | sha256 |
|---|---|---|
| **`core_v1.1.jsonl`** | **1,545** (VERIFIED 1,367 · ADJUDICATED_LOW_CONFIDENCE 178) | **`ceac95defa5ed8f2b93c31835660551089743d583335c3002d4ad9d5b8c0046c`** (`core_v1.1.sha256`) |
| `invalid_v1.1.jsonl` | 40 | `df7d2f19ca50219c5d7740f6fb38d700198aa3ef92f6f2a27ffa41f7c1473654` |
| `core_v1.1.datacard.md` | — | Q1's own datacard |
| `core_v1.jsonl` (superseded, kept for traceability) | 1,555 (VERIFIED 1,029 · LOW_CONF 526; INVALID 30) | `48bfd1c9287d88d5b7f22556e0914b033048410e7d8232aa420e3e1688c534dc` |
| `decisions_v1.jsonl`, `q1_stats_v1.json`, `audit_results_v1.jsonl`, `batches.jsonl` (62+ events, 0 pending) | — | audit trail |

Upstream collection files (X2): `data/vi_exam/items.jsonl` (public fields), `items_private.jsonl` (third-party
solution text — never release), `summary.json`, `qa_queue.jsonl` (LF-normalised sha256 `0f58ee1c…e36aeaaf`, verified
unchanged by Q1).

## 1. Motivation
A native Vietnamese math benchmark: grade-10 entrance exams ("Toán vào 10"), free-response with gradable final
answers, keys checked by a solver and independent LLM solves, each item tagged "symbolically certifiable or not"
(P3/H4), with a temporal split (exam year / first-seen date) for contamination analysis. Master's-thesis project
(user `datdaide1`); no external funding recorded.

## 2. Composition
| | Value |
|---|---|
| Exams collected | 270 official/school papers, 58 provinces, exam years 2017–2026 (no 2024 papers on the source site) |
| Items (question units) → Core candidates | 4,209 (3,732 figure-free) → **1,585** Core candidates (gradable, figure-free, not proof/MCQ/construction/open, de-duplicated) |
| **core_v1.1** | **1,545**: VERIFIED **1,367**, ADJUDICATED_LOW_CONFIDENCE **178**; INVALID 40 removed (text_defect 15, ill_posed 10, proof_or_open 7, not_gradable 5, other 3) |
| VERIFIED routes | X2 consensus confirmed by the grader 894 · two Claude solves + adjudicator (high) 370 · keys agree after solve #2 88 · independent keys incl. non-Claude 15 |
| Contamination | 37 VERIFIED items carry `contamination.flag` (X5 review rule, mostly 2025 papers reusing textbook problems) → **VERIFIED clean = 1,330** (the set R1 analyses) |
| Post-cutoff (after QA + contamination removal) | 2026: 109 (98 VERIFIED) · 2025+: 313 (272) · first_seen ≥ 2026-06: 84 (75) |
| Certifiable split (H4), contamination-clean | VERIFIED: yes 970 / non-certifiable 360 (maybe 283, no 77) · all Core: yes 1,021 / 485 (392, 93) |
| VERIFIED share by tag (valid items) | yes 95 %, maybe 73 %, no 82 % — report H4 on all valid Core too |

- **Instance:** one sub-question with a single gradable final answer. Fields (X2 §4 + Q1): `id`, problem text, typed
  gold (`answer_canonical`, `answer_type`, locale), topic tags, `symbolically_certifiable`, `qa.status`, keys and
  agreement, `provenance{site, source_url, province, school, exam_year, exam_kind, issuer_type, fetched_at,
  first_seen…}`, `license_basis`, `release_eligible`, `temporal{…}`, `contamination`, `canary` (GUID).
- **Label quality — fresh audit #2** (150 items, uniform over Core, seed 20261108): **2 label errors → 1.3 %
  [0.16 %, 4.7 %]** (Clopper–Pearson; one-sided upper bound 4.1 %; VERIFIED stratum 2/132, upper bound 4.7 %;
  LOW_CONF 0/18, upper bound 15.3 %). If every grader mismatch counted as an error (10/150) the upper bound is 11.0 %.
  Both errors were corrected; both are one defect class (below). Audit #1 (v1): 3/150.
- **Known defect class — sibling sub-questions (X2 segmentation):** an item's text can contain its sibling parts
  ("a) Thực hiện phép tính … b) Rút gọn …") while the key covers one part. Scan of v1.1 for ≥ 2 line-initial
  `a)`/`b)` labels: 6 items, all LOW_CONF, 0 VERIFIED; inline or "2.1"-style labels are not caught. KNOWN_ISSUES
  X2-SEG.
- **Never validity-checked by an LLM:** the 894 X2-consensus and 88 "keys agree" items (audit #2 found its 2 errors in
  the consensus route, 2/82 audited there). No human raters (user decision: LLM judges replace them).
- **Not in public files:** third-party worked-solution text; raw HTML (private bundle).
- **Personal data:** exam papers contain no student data; school names appear for school-set papers. A PII scan has not
  been run (KNOWN_ISSUES).

## 3. Collection process
- Source: loigiaihay.com exam articles (hub + all monthly sitemaps 2017-01…2026-11); official provincial sources
  recorded where found. robots.txt checked per host, fail-closed; ≥ 2 s between requests; generic research user
  agent; every fetch logged (private). 0 URLs blocked by robots.
- Collected 2026-11-07; `first_seen` = min(Wayback first capture, sitemap lastmod, og:image date).
- **Keys and labels:** page solution (third-party), SymPy (`symkey.py`), blind Claude solve #1 (X2: Opus 5.5,
  effort low, Batch, $8.84), Claude solve #2 (Q1: effort high), adjudicator (effort medium), audit re-solves and audit
  judge — all `claude-opus-5-5` via the Message Batches API, prompts `q1_solve.v1`, `q1_adjudicate.v1`,
  `q1_audit_solve.v1`, `q1_audit_judge.v1`. **Benchmark items were sent only to Claude** (`split="benchmark"`
  guard; no Gemini/NIM). Q1 spend **$15.36 of $15.5** (v1 $9.93 + v1.1 $5.43); ledger `logs/api/Q1/claude.jsonl`.
- **Deviations (Q1):** v1 could not afford adjudication of all 631 items (218 adjudicated); v1.1 adjudicated the 382
  two-Claude-only items (reference solution clipped to 2,000 chars); audit #2 re-solved 54 items at effort high and 96
  at effort medium (cap).

## 4. Preprocessing / cleaning / labeling
DOM blocks → question tree + solution alignment (`segment.py`) → tags (`tagging.py`) → provisional key (`answers.py`)
→ SymPy key → typed gold with grader 3.0.2 → Q1 statuses (VERIFIED rule: adjudicator high/medium and ≥ 2 grader-agreeing
keys incl. ≥ 1 non-Claude, or both Claude solves + high-confidence adjudicator; X2 grader-confirmed consensus items
VERIFIED without adjudication). Exclusions before Core: proof 715, MCQ 939, construction 120, open 288, true/false 48,
figure 477, duplicate 23, near-duplicate 5, multi-target 9.

## 5. Uses
- **Evaluation only.** Primary analysis set: **VERIFIED, contamination-clean (1,330)**; sensitivity: all valid Core.
  Drop `contamination.flag` items from post-cutoff claims. In the 11-08 tranche Core is evaluated **greedy-only**
  (deviation from T5's K = 4, cost; T2 §5.2).
- **Grader gaps that affect credit on this set** (KNOWN_ISSUES G-1…G-3): named points `M(0;0)`, compound conditions
  `0 \le x < 4, x \ne 1`, prose with several labelled values.
- **Must not** be trained on (canary GUID in every item); must not be sent to non-Claude APIs before release.

## 6. Distribution and licenses
- Now: internal bundle `data` only (`BUNDLE.md`); third-party text in the private bundle.
- Proposed (06 §4.4–4.5, not legal advice): our annotations/keys/metadata CC BY 4.0; problem statements reproduced for
  non-commercial research, copyright with the issuing authorities; card `license: other` or CC BY-NC-SA 4.0; takedown
  procedure. `license_basis` per item: state-issued papers → reading of Luật SHTT Điều 15(2) + NĐ 17/2023 Điều 8(2)
  (**UNVERIFIED**, confirm with the VNU office); university-school papers (PTNK, KHTN) → research use only, not
  releasable (11 of the 1,585 candidates).
- **Claude provenance:** the keys/adjudications are Claude outputs (`claude-opus-5-5`; logs and prompts listed in
  `README.md`); the problem texts are not.

## 7. Maintenance
`core_v1` → `core_v1.1` kept side by side with hashes; any further key change → `core_v1.2` with an item-level diff
(`id, field, old, new, reason`); ids are stable; removed items go to `invalid_*`. Open: X2 segmentation fix
(sibling sub-questions), PII scan, 2024/HSG/chuyên gap (user decision: not now).
