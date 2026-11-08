---
id: validity_judge
version: 1
role: validity_judge
language: vi
status: template
inputs: [seed_problem, operator, problem]
description: MathQ-Verify-style decomposed validity judge of a generated problem statement; JSON verdict (T1 §2.3 G6, T2 V1)
source: pending — fill from the G1 winner (pilot_gen/prompts/judge_v2.txt)
---
### SYSTEM

### USER
<<seed_problem>>
<<operator>>
<<problem>>
