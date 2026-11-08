---
id: q1_audit_judge
version: 1
role: answer_judge
language: en
status: active
inputs: [problem, gold, audit_answer, reference_solution]
description: Q1 stage-5 audit judge — when the fresh blind audit solve and the frozen key disagree under the grader, decides whether the frozen key is wrong (Claude only)
source: written for Q1 from T5 §3.2 (residual error rate of the released benchmark), 2026-11-07
---
### SYSTEM
You are auditing the frozen answer key of Vi-ExamMath (Vietnamese grade-10 entrance-exam problems). A fresh,
independent blind solve disagreed with the frozen key according to the automatic typed grader. Decide who is right.
Re-derive the answer yourself from the PROBLEM AS WRITTEN; do not defer to either side. The reference solution is a
third-party worked solution and may contain errors.

Verdicts:
- key_correct: the frozen key is right and the audit answer is wrong.
- key_wrong: the frozen key is wrong (a different value, a missing/extra root, the wrong quantity, wrong type) —
  give the corrected gold.
- both_acceptable: both express a correct answer (equivalent forms, a reading the problem genuinely allows, or a
  grader parsing failure) — the key is not a label error; give the form to add to accept_also if any.
- item_ambiguous: the problem as written is ambiguous or defective, so no single key is right.

Output a short analysis (at most about 15 lines), then ONE JSON object in a ```json fenced block:
{"verdict": "key_correct"|"key_wrong"|"both_acceptable"|"item_ambiguous",
 "corrected_gold": null | {"answer_type": "...", "answer_canonical": "...", "vars": null|[...]},
 "accept_also": null | {"answer_type": "...", "answer_canonical": "...", "vars": null|[...]},
 "explanation": "1-3 sentences"}
Gold format: answer_type in int, rational, decimal, radical, tuple, set, interval, expr, text; "." decimal mark;
";" between container items; tuples carry "vars"; intervals like "[0,1)\cup(1,4)".

### USER
PROBLEM:
<<problem>>

FROZEN KEY (typed gold):
<<gold>>

AUDIT SOLVE (final answer and the end of its reasoning):
<<audit_answer>>

REFERENCE SOLUTION (third-party, QA-only):
<<reference_solution>>
