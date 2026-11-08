---
id: solution_writer
version: 2
role: solution_writer
language: vi
status: active
inputs: [problem]
description: blind short Vietnamese CoT ending in \boxed{}
source: verbatim port of G1 pilot_gen/prompts/solve_v1.txt
---
### SYSTEM


### USER
Giải bài toán sau bằng tiếng Việt, ngắn gọn từng bước, kết thúc bằng \boxed{đáp số} (chỉ ghi số trong \boxed, không kèm đơn vị).

<<problem>>
