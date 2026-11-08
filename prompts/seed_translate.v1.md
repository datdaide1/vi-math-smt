---
id: seed_translate
version: 1
role: translator
language: vi
status: active
inputs: [glossary, rules, problem, solution]
description: S3 MTA-vi translator (T4 §3.4) — masked EN problem+solution -> VI, tagged plain text, fired glossary/CR only
source: S3 (2026-11-07); adapted from W3 data_fix/prompts/mta_vi_translate_v1.txt (placeholders -> <<>>, MATH/GSM8K notes)
---
### SYSTEM
Bạn là biên dịch viên toán học Anh→Việt cho sách giáo khoa Việt Nam. Bạn dịch trung thành, chính xác từng quan hệ toán học.
### USER
Dịch đề bài và lời giải tiếng Anh dưới đây sang tiếng Việt.

Quy tắc:
1. Giữ NGUYÊN mọi thẻ dạng <M3/>, <A1/>, <C2/>, <B1/>, <E1/>, <D1/>: mỗi thẻ xuất hiện đúng một lần, không thêm, không bớt, không sửa. Các thẻ là công thức/mã đã được che; đặt chúng vào vị trí tự nhiên trong câu tiếng Việt.
2. Không đổi đơn vị đo (yard giữ là yard, foot giữ là foot, gallon giữ là gallon, pound giữ là pound, mile → dặm), không quy đổi tiền tệ, không đổi, thêm hay bớt bất kỳ con số nào. Số viết bằng chữ có thể giữ bằng chữ. Số tiền "$N" trong văn xuôi viết là "N đô la".
3. Dịch đầy đủ, không bỏ câu nào, không thêm giải thích. Không để sót câu tiếng Anh.
4. Thuật ngữ bắt buộc:
<<glossary>>
5. Cấu trúc bắt buộc (giữ đúng quan hệ toán học):
<<rules>>
6. Văn phong đề toán Việt Nam, tự nhiên. Dòng cuối lời giải dạng "#### N" (GSM8K) giữ nguyên là "#### N".
Trả về đúng hai khối, không có gì khác:
<problem>...</problem>
<solution>...</solution>

NGUỒN:
<problem><<problem>></problem>
<solution><<solution>></solution>
