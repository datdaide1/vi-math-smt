---
id: native_generator
version: 1
role: native_generator
language: vi
status: template
inputs: [seed_problem, operator, localization_card]
description: arm B: operator-conditioned native Vietnamese problem generation from a seed (T1 §2.4)
source: pending — fill from the G1 winner (pilot_gen/prompts/native_v1.txt)
---
### SYSTEM

### USER
<<seed_problem>>
<<operator>>
<<localization_card>>
