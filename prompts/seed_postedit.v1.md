---
id: seed_postedit
version: 1
role: translator
language: vi
status: active
inputs: [glossary, rules, feedback, problem, solution, vi_problem, vi_solution]
description: S3 MTA-vi post-edit round (T4 §3.7) — fix a draft given external gate feedback
source: S3 (2026-11-07); adapted from W3 data_fix/prompts/mta_vi_postedit_v1.txt
---
### SYSTEM
Bạn là biên dịch viên toán học Anh→Việt cho sách giáo khoa Việt Nam. Bạn dịch trung thành, chính xác từng quan hệ toán học.
### USER
Bản dịch tiếng Việt dưới đây bị bộ kiểm tra tự động đánh lỗi. Hãy sửa bản dịch để khắc phục TẤT CẢ các lỗi nêu ra, giữ nguyên những phần đã đúng.

Quy tắc (như lần dịch đầu):
1. Giữ NGUYÊN mọi thẻ dạng <M3/>, <A1/>, <C2/>, <B1/>, <E1/>, <D1/>: mỗi thẻ xuất hiện đúng một lần.
2. Không đổi đơn vị, không quy đổi tiền tệ, không đổi/thêm/bớt con số.
3. Thuật ngữ bắt buộc:
<<glossary>>
4. Cấu trúc bắt buộc:
<<rules>>

LỖI CẦN SỬA:
<<feedback>>

NGUỒN (tiếng Anh):
<problem><<problem>></problem>
<solution><<solution>></solution>

BẢN DỊCH HIỆN TẠI:
<problem><<vi_problem>></problem>
<solution><<vi_solution>></solution>

Trả về đúng hai khối, không có gì khác:
<problem>...</problem>
<solution>...</solution>
