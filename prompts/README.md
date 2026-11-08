# prompts/ — versioned prompt registry

One file per prompt version: `<id>.v<version>.md`. Loader: `vi_math_verified_aug/common/prompts.py`.

```markdown
---
id: validity_judge          # = file-name stem before .v<k>
version: 1                  # = the k in .v<k>.md
role: validity_judge        # formalizer | editor | native_generator | solution_writer | validity_judge |
                            # answer_judge | solver | translator | other
language: vi                # vi | en   (English-mirror prompts get their own id, e.g. validity_judge_en)
status: active              # template (empty, refuses to render) | active | retired
inputs: [seed_problem, operator, problem]
description: one line
source: provenance of the text (e.g. G1 winner pilot_gen/prompts/judge_v2.txt)
---
### SYSTEM
...system text...
### USER
...user text with <<placeholders>>...
```

Rules
- **Never edit a version that has been used for a logged call.** Copy to `.v<k+1>.md`, bump `version`.
  The API ledger stores `prompt_id`, `prompt_version` and `prompt_sha256` (LF-normalised file hash).
- Placeholders are `<<name>>` (not `{name}`: LaTeX and JSON examples are full of braces). Every placeholder
  must be listed in `inputs`; rendering fails on a missing or an undeclared value.
- Pin `prompt_version` in run configs; `load_prompt(id)` without a version returns the highest one.

Current files are **empty templates** (`status: template`) for the T1 §2 / T2 §2 roles. Their texts come
from the G1 pilot winners; the declared `inputs` are a draft contract and may change with the G1 port.

| id | role | intended use |
|---|---|---|
| formalizer | formalizer | Vi problem → typed quantity-DAG IR JSON (arm A seed formalization; blind back-formalization) |
| editor | editor | minimal span-grounded edit of a tagged seed per an operator edit list (arm A renderer) |
| native_generator | native_generator | arm B operator-conditioned native Vietnamese generation |
| solution_writer | solution_writer | shared blind solution writer, short Vi CoT ending in `\boxed{}` |
| validity_judge | validity_judge | decomposed (MathQ-Verify-style) problem validity judge, JSON verdict |
| answer_judge | answer_judge | final-answer equivalence where the typed grader cannot decide |
