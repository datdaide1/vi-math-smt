---
id: editor
version: 1
role: editor
language: vi
status: template
inputs: [text_tagged, edits]
description: minimal span-grounded edit of a tagged seed text per an operator edit list; never sees the new answer (T1 §2.2 renderer)
source: pending — fill from the G1 winner (pilot_gen/prompts/render_v1.txt)
---
### SYSTEM

### USER
<<text_tagged>>
<<edits>>
