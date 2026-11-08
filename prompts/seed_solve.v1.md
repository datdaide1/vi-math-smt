---
id: seed_solve
version: 1
role: solver
language: vi
status: active
inputs: [problem]
description: S3 blind solve-check (T4 §3.5 G2) of a Vietnamese seed; short CoT, final answer in \boxed{}
source: S3 (2026-11-07); W3 g2_solve_v1 wording, extended for MATH-style (non-numeric) answers
---
### SYSTEM

### USER
Giải bài toán sau. Trình bày ngắn gọn, đặt đáp án cuối cùng trong \boxed{} (chỉ ghi giá trị hoặc biểu thức, không kèm đơn vị; nếu đề yêu cầu dạng cụ thể như phân số tối giản hay căn thức thì ghi đúng dạng đó).

<<problem>>
