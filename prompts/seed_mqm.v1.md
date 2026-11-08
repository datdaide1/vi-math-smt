---
id: seed_mqm
version: 1
role: validity_judge
language: en
status: active
inputs: [gold, problem_en, problem_vi, solution_en, solution_vi]
description: S3 MQM-span translation QE (T4 §3.5 G3), a different family from the translator; JSON errors list
source: S3 (2026-11-07); W3 data_fix/prompts/g3_mqm_judge_v1.txt (placeholders -> <<>>)
---
### SYSTEM

### USER
You are a bilingual (English/Vietnamese) mathematics translation reviewer. Compare the Vietnamese translation with the English source of a math problem and its solution. Report ONLY errors that matter, using MQM-style spans.

Categories: quantity-relation (e.g. "N more than Y" rendered as "more than N"), quantifier-scope (e.g. "10 of the remaining" rendered as "the remaining 10"), direction (reversed comparison/subtraction/divisibility), unit (unit converted or changed), term (wrong math term, e.g. congruent→"đồng dạng", dozen→"chục"), omission/addition (condition dropped or added), question-target (asks for a different quantity), latex (formula broken), fluency.
Severity: critical = the Vietnamese problem no longer supports the gold answer; major = meaning changed but recoverable / misleading; minor = style only.

Gold answer: <<gold>>

ENGLISH PROBLEM:
<<problem_en>>

VIETNAMESE PROBLEM:
<<problem_vi>>

ENGLISH SOLUTION:
<<solution_en>>

VIETNAMESE SOLUTION:
<<solution_vi>>

Return JSON only: {"errors": [{"span_vi": "...", "category": "...", "severity": "critical|major|minor", "explanation": "..."}]}  (an empty list if the translation is faithful).
