# PROMPTS — catalog of versioned prompts

_Generated 2026-11-08 09:29 by `handoff/tools/prompt_catalog.py`. Do not hand-edit; re-run._

Registry rules (`prompts/README.md`): one file per version `<id>.v<k>.md`; never edit a version that has a logged call, copy to `.v<k+1>.md` instead. `sha256` = hash of the LF-normalised file (`common/prompts.py`), which is what every `common/llm.py` ledger line records as `prompt_sha256`.

Columns: **Calls** = ledger lines in `logs/api/*/*.jsonl` with this `prompt_id` + `prompt_version`; **Hash check** = whether every logged `prompt_sha256` equals the file hash today (✗ means the file was edited after use, so the logged calls used a different text).

| id | v | role | lang | status | sha256 (first 16) | Calls (by WS) | Hash check | Description |
|---|---|---|---|---|---|---|---|---|
| `answer_judge` | 1 | answer_judge | vi | template | `49767330b2a0a090` | 0 | unused | does the candidate's final answer equal the reference (only where the typed grader cannot decide; test-side it |
| `back_formalizer` | 1 | formalizer | vi | active | `c633f2ff37624aff` | 2,879 (G3 347, G4 204, G5 2328) | ✓ | BLIND back-formalization for the arm-Y cycle check (T1 §2.2 step 5): same IR rules as formalizer.v2, no Englis |
| `back_formalizer_en` | 1 | formalizer | en | active | `5ad47c868d85ea4c` | 0 | unused | BLIND back-formalization for the English arm-Y cycle check (G5 EN mirror): back_formalizer.v1 with the problem |
| `editor` | 1 | editor | vi | template | `4ecf7094baa77f34` | 0 | unused | minimal span-grounded edit of a tagged seed text per an operator edit list; never sees the new answer (T1 §2.2 |
| `editor` | 2 | editor | vi | active | `477f320a61b93077` | 132 (G3 132) | ✓ | arm-Y renderer for M2-M6 (T1 §2.2 step 3): minimal span-grounded edit of the tagged seed per the X4 edit list  |
| `editor` | 3 | editor | vi | active | `531a3805cf9e2723` | 185 (G3 122, G4 10, G5 53) | ✓ | arm-Y renderer for M2-M6 (T1 §2.2 step 3): minimal span-grounded edit of the tagged seed per the X4 edit list  |
| `fluency` | 1 | editor | vi | active | `b61c75f7a30b8cca` | 97 (G4 97) | ✓ | arm-Y R1 fluency + localization pass (G4). Rewrites an already-edited tagged problem into natural Vietnamese t |
| `fluency` | 2 | editor | vi | active | `817efda06cd165a0` | 171 (G4 96, G5 75) | ✓ | arm-Y R1 fluency + localization pass (G4). Rewrites an already-edited tagged problem into natural Vietnamese t |
| `formalizer` | 1 | formalizer | vi | template | `3ab7266b24bdfcbf` | 0 | unused | Vietnamese word problem -> typed quantity-DAG IR (JSON with text spans); arm A and back-formalization (T1 §2.1 |
| `formalizer` | 2 | formalizer | vi | active | `534ae0fe34e5d94a` | 8,708 (G3 132, G5 8576) | ✓ | seed formalization (arm Y step 1): Vietnamese word problem -> expression-style typed quantity graph with spans |
| `ir_render` | 1 | editor | vi | active | `80f9ee1369282089` | 7,475 (G4 111, G5 7364) | ✓ | arm-Y R2 free renderer (G4). Writes a natural Vietnamese problem from the mutated program's labelled givens an |
| `ir_render_en` | 1 | editor | en | active | `ad44cb84a9e48ae3` | 6,923 (G5 6923) | ✓ | arm-Y English renderer (G5 EN mirror). Writes a natural English problem from the mutated program's labelled gi |
| `native_generator` | 1 | native_generator | vi | template | `9e2d2af48084b385` | 0 | unused | arm B: operator-conditioned native Vietnamese problem generation from a seed (T1 §2.4) |
| `native_generator` | 2 | native_generator | vi | active | `05bb6ab61b2f28b7` | 41 (G2 41) | ✓ | G2 variant A ("rules") — native_v1 + realistic-value constraints, per-operator structural-change recipes, soft |
| `native_generator` | 3 | native_generator | vi | active | `f55fc390952e9373` | 14,326 (D1 14191, G2 43, G4 92) | ✓ | G2 variant B ("self-check") — variant A + worked solution and a self-check block; rewrite before answering if  |
| `native_generator` | 4 | native_generator | vi | active | `e7e2f5c6295d8f19` | 43 (G2 43) | ✓ | G2 variant C ("contrastive") — variant B + 2 good and 2 bad worked examples (G1 items judged by Opus; G1 seeds |
| `native_generator_en` | 3 | native_generator | en | active | `cbda415edd514459` | 4,398 (D1 4398) | ✓ | English mirror (E1 R4) of native_generator.v3 ("self-check"); sentence-by-sentence translation, same structure |
| `q1_adjudicate` | 1 | answer_judge | en | active | `4c81d5a461157708` | 600 (Q1 600) | ✓ | Q1 LLM adjudicator (replaces the two human annotators of T5 §3.1 stage 4) — decides the typed gold, item valid |
| `q1_audit_judge` | 1 | answer_judge | en | active | `9e7c34f6ff891c89` | 22 (Q1 22) | ✓ | Q1 stage-5 audit judge — when the fresh blind audit solve and the frozen key disagree under the grader, decide |
| `q1_audit_solve` | 1 | solver | en | active | `057d8a9f32301088` | 306 (Q1 306) | ✓ | Q1 stage-5 random audit, a fresh blind re-solve of a Vi-ExamMath Core item (Claude only; third phrasing, Engli |
| `q1_solve` | 1 | solver | vi | active | `afaf06749eb5a426` | 665 (Q1 665) | ✓ | Q1 second independent blind solve of a Vi-ExamMath Core item (Claude only; phrasing deliberately different fro |
| `seed_mqm` | 1 | validity_judge | en | active | `b6de78925aea16e1` | 380 (S3 380) | ✓ | S3 MQM-span translation QE (T4 §3.5 G3), a different family from the translator; JSON errors list |
| `seed_postedit` | 1 | translator | vi | active | `0aa816d7202131d3` | 240 (S3 240) | ✓ | S3 MTA-vi post-edit round (T4 §3.7) — fix a draft given external gate feedback |
| `seed_solve` | 1 | solver | vi | active | `074e2997a42bf560` | 12,491 (S3 12491) | ✓ | S3 blind solve-check (T4 §3.5 G2) of a Vietnamese seed; short CoT, final answer in \boxed{} |
| `seed_solve_en` | 1 | solver | en | active | `300ab2f8ae98a6ed` | 385 (S3 385) | ✓ | S3 English control solve for the G2 solve-check (decides PASS/FAIL vs UNINFORMATIVE) |
| `seed_translate` | 1 | translator | vi | active | `b8b1f230de2c7773` | 379 (S3 379) | ✓ | S3 MTA-vi translator (T4 §3.4) — masked EN problem+solution -> VI, tagged plain text, fired glossary/CR only |
| `solution_writer` | 1 | solution_writer | vi | template | `24e5d047456d2488` | 0 | unused | shared blind solution writer: short Vietnamese CoT ending in \boxed{answer} (T1 §2.3, T2 V2 solvers) |
| `solution_writer` | 2 | solution_writer | vi | active | `a034dc261218b2fd` | 20,546 (D1 13462, G2 686, G3 139, G4 266, G5 5993) | ✓ | blind short Vietnamese CoT ending in \boxed{} |
| `solution_writer_en` | 2 | solution_writer | en | active | `4e652670192cbd43` | 8,831 (D1 4459, G5 4372) | ✓ | English mirror (E1 R4) of solution_writer.v2 — blind short English CoT ending in \boxed{} |
| `story_check` | 1 | other | vi | active | `557bd2d43c74009d` | 6,561 (G4 146, G5 6415) | ✓ | arm-Y operator story-role check (G4). Is a proposed M3/M4/M5 change type-consistent and meaningful in THIS sto |
| `validity_judge` | 1 | validity_judge | vi | template | `6e490c7b1f4eef69` | 0 | unused | MathQ-Verify-style decomposed validity judge of a generated problem statement; JSON verdict (T1 §2.3 G6, T2 V1 |
| `validity_judge` | 2 | validity_judge | en | active | `1a7b2129b6bc332a` | 1,999 (D1 120, D5 478, G2 755, G3 199, G4 266, G5 121, S3 60) | ✓ | holistic + MathQ-Verify booleans + PROTOCOL codes; the G1 judge of record (Opus Table B) |
| `validity_judge` | 3 | validity_judge | en | active | `9624bdb31b4a6da0` | 19,449 (D1 9592, G2 1134, G3 139, G4 346, G5 8238) | ✓ | G2 cheap-judge rubric: solve first, then decomposed yes/no questions; 6 few-shot anchors labelled by Opus in G |
| `validity_judge_en` | 1 | validity_judge | en | active | `840e4fd3899380dd` | 198 (D5 198) | ✓ | English-mirror twin of validity_judge.v2 (same fields, codes, labels); target language English, not Vietnamese |
| `validity_judge_en` | 3 | validity_judge | en | active | `5a29e8d29883b0e8` | 8,510 (D1 4247, G5 4263) | ✓ | English mirror (E1 R4) of validity_judge.v3 (G2 cheap-judge rubric) for the common "+" screen on EN items; sam |
| `validity_lite` | 1 | validity_judge | en | active | `8673336a58de2077` | 302 (D1 228, G5 74) | ✓ | compact PACKED validity/absurdity screen for gemini flash-lite at scale (up to 10 items per call); a subset of |

Full sha256 values: `handoff/PROMPTS.sha256` (same order, `sha256  path`).

## Ledger prompt ids with no file in `prompts/`

| prompt_id | version | Calls |
|---|---|---|
| `pilot_gen/native_v1` | 1 | 41 |

## Legacy G1 pilot prompts (pre-registry; plain text, raw-byte sha256)

Used only by G1 (`exec/G1_generator_pilot.md`); G2/G3 ported the winners verbatim into the registry (`validity_judge.v2` = `judge_v2`, `formalizer.v2` = `formalize_v1`, `solution_writer.v2` = `solve_v1`).

| File | sha256 (first 16) |
|---|---|
| `vi_math_verified_aug/pilot_gen/prompts/formalize_v1.txt` | `c423b394ecf96809` |
| `vi_math_verified_aug/pilot_gen/prompts/judge_v1.txt` | `13ef74e4cd14e084` |
| `vi_math_verified_aug/pilot_gen/prompts/judge_v2.txt` | `628bc9f10f610f15` |
| `vi_math_verified_aug/pilot_gen/prompts/native_v1.txt` | `d684d31021286d1e` |
| `vi_math_verified_aug/pilot_gen/prompts/render_v1.txt` | `980cd3b39b7ccf86` |
| `vi_math_verified_aug/pilot_gen/prompts/solve_v1.txt` | `028760e760ced180` |
