---
id: answer_judge
version: 1
role: answer_judge
language: vi
status: template
inputs: [problem, reference_answer, candidate]
description: does the candidate's final answer equal the reference (only where the typed grader cannot decide; test-side items go to Claude only) (T2 §2.4/§4)
source: pending — fill from the G1 winner (no G1 prompt yet)
---
### SYSTEM

### USER
<<problem>>
<<reference_answer>>
<<candidate>>
