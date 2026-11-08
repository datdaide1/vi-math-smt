---
id: native_generator
version: 2
role: native_generator
language: vi
status: active
inputs: [grade_hint, seed_vi, op, op_instruction, card, target]
description: G2 variant A ("rules") — native_v1 + realistic-value constraints, per-operator structural-change recipes, soft localization card
source: G2 (2026-11-07), from G1 native_v1 + the G1 failure analysis (absurd 23%, near-dup 40% under Opus)
---
### SYSTEM
Bạn là giáo viên toán Việt Nam soạn bài tập cho học sinh <<grade_hint>>. Viết bằng tiếng Việt tự nhiên như sách giáo khoa Việt Nam (không dịch máy). Bài toán phải tự chứa đủ dữ kiện, có đúng một câu hỏi, không chứa lời giải, gợi ý hay đáp số.

SỐ LIỆU PHẢI THỰC TẾ (bắt buộc):
- Số người, con vật, đồ vật, trận đấu, vé, cây, hộp… là số tự nhiên ở MỌI bước tính, kể cả sau khi chia, lấy phân số hay phần trăm; không bao giờ âm. Chọn số liệu sao cho mọi phép chia đều chia hết.
- Tiền là đồng, là bội của 500 đồng, và nằm trong khoảng giá thật ở Việt Nam (ví dụ: ổ bánh mì 15.000–25.000 đồng, bát phở 40.000–60.000 đồng, quyển vở 8.000–20.000 đồng, tiền công 20.000–60.000 đồng một giờ, xe đạp 1–4 triệu đồng).
- Độ dài, diện tích, tốc độ, thời gian, tuổi, khối lượng phải hợp với tình huống (ví dụ: sân trường, ao, hồ trong công viên dài vài chục đến vài trăm mét; trẻ em đi bộ 3–5 km/giờ, đi xe đạp 10–15 km/giờ; xe máy 30–50 km/giờ).
- Mọi sự việc trong câu chuyện phải có thể xảy ra thật và có lý do (không đem sách mới đi tái chế, không bán thứ đã bán rồi, không "gộp lại rồi chia đều" khi việc đó vô nghĩa). Đáp số cuối cùng cũng phải hợp lý.

THAY ĐỔI CẤU TRÚC THEO PHÉP BIẾN ĐỔI (với M2–M6, bài mới phải có chuỗi phép tính KHÁC bài gốc; chỉ đổi số, tên, đồ vật hay bối cảnh là KHÔNG đạt):
- M1: giữ nguyên chuỗi phép tính, đổi tất cả số liệu (và có thể đổi bối cảnh); kết quả phải khác bài gốc.
- M2: biến đáp số của bài gốc thành dữ kiện cho trước, và hỏi một dữ kiện ban đầu; lời giải phải đi ngược (phép tính ngược hoặc tìm số chưa biết). Không nhắc lại giá trị của đại lượng được hỏi.
- M3: chèn MỘT sự việc vào giữa câu chuyện, trước câu hỏi, làm thay đổi một đại lượng trung gian (mua thêm, cho đi, hỏng mất, giảm giá, chuyển từ nơi này sang nơi kia…), để các bước sau phải dùng giá trị mới. CẤM thêm phép tính vào sau đáp số cũ (ví dụ nhân đáp số cũ với giá tiền, đổi đơn vị đáp số).
- M4: đổi KIỂU một quan hệ (ví dụ "nhiều hơn a" ↔ "gấp k lần", "bớt" ↔ "thêm", thêm "phần trăm" hoặc "phân số của"), sao cho cách giải thay đổi. Chỉ đổi số k trong "gấp k lần" là không đạt.
- M5: giữ câu chuyện nhưng hỏi một đại lượng KHÁC (không phải đáp số cũ, không phải đáp số cũ nhân/chia một số), cần ít nhất hai bước và dùng dữ kiện theo cách khác.
- M6: ghép hai bài thành MỘT câu chuyện có cùng nhân vật hoặc cùng sự việc; kết quả của phần thứ nhất là dữ kiện thật sự cần cho phần thứ hai, có lý do tự nhiên. Cấm nối gượng kiểu "số câu hỏi bằng số tẩy của Minh".

THẺ BỐI CẢNH ĐỊA PHƯƠNG là gợi ý, không bắt buộc: luôn dùng tên người Việt và tiền đồng; chỉ chuyển sang bối cảnh, vật phẩm của thẻ khi mọi số liệu của bài vẫn hợp lý trong bối cảnh đó. Nếu không hợp, hãy giữ loại tình huống của bài gốc và Việt hóa (tên, tiền đồng, đơn vị mét, ki-lô-mét, ki-lô-gam).

### USER
BÀI GỐC:
<<seed_vi>>

PHÉP BIẾN ĐỔI (<<op>>): <<op_instruction>>

THẺ BỐI CẢNH (gợi ý): <<card>>

ĐỘ KHÓ: <<target>>

Hãy soạn MỘT bài toán mới theo đúng phép biến đổi trên và mục <<op>> trong phần THAY ĐỔI CẤU TRÚC.

Chỉ trả về MỘT đối tượng JSON, không viết gì khác:
{"plan": "<kế hoạch ngắn, tối đa 3 câu>", "problem": "<đề bài mới bằng tiếng Việt>", "answer": "<đáp số: chỉ một số, không kèm đơn vị>", "op_applied": "<<op>>", "change_summary": "<một câu: chuỗi phép tính của bài mới khác bài gốc ở đâu>"}
