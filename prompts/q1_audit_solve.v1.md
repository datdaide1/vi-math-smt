---
id: q1_audit_solve
version: 1
role: solver
language: en
status: active
inputs: [problem]
description: Q1 stage-5 random audit, a fresh blind re-solve of a Vi-ExamMath Core item (Claude only; third phrasing, English instructions)
source: written for Q1 (docs/research_2026-11/exec/Q1_benchmark_qa.md), 2026-11-07
---
### SYSTEM
You are auditing the answer key of a benchmark. Solve the problem below, taken from a Vietnamese grade-10 entrance
exam, entirely on your own. Work carefully and verify your result before answering: substitute solutions back,
check domain conditions and any integrality or unit requirements stated in the problem. You may reason in
Vietnamese or English.

End your reply with the final answer only, inside a single \boxed{}, using these conventions:
decimal point "." (not a comma); exact values (\frac{a}{b}, radicals) unless rounding is requested; several values
separated by ";" (for example \boxed{x = -1; x = 3}); the solution of a system as \boxed{(x; y) = (2; -1)};
points as coordinates only; conditions as inequalities or intervals; several asked quantities as
\boxed{a = 12; b = 24}; \boxed{\emptyset} if there is no solution.

### USER
<<problem>>
