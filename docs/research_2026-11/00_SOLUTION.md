# 00 — Solution tổng hợp (re-research 2026-11-07)

**Đầu vào:** 6 báo cáo nghiên cứu (`01`–`06`) + 6 báo cáo soát nội dung dữ liệu (`content_review/R1`–`R6`),
do 12 agent làm song song. Mọi con số dưới đây trỏ về báo cáo gốc (ký hiệu `[01 §5]`, `[R5]`…). Những điểm
then chốt đã được **tự kiểm lại** trong phiên này, ký hiệu `[TK]` (lệnh + kết quả ghi ở §9).

**Ràng buộc (user, 2026-11-07):** mục tiêu **báo Q2, không cắt/thu nhỏ scope**; bàn giao **nội bộ** cho người
tiếp nhận (đủ data + model + code + prompt); không tiền, hạ tầng yếu → ưu tiên miễn phí/nhẹ nhưng **chất lượng
đứng đầu**; có **$100 Claude API**. Mọi kết luận cần evidence, mọi lựa chọn cần so sánh.

---

## 0. TL;DR

1. **Kế hoạch tháng 9 không chạy được nguyên trạng.** Generator đã chọn (`deepseek-v4-pro-0813`) và model yếu
   (`llama-3.1-8b-instruct`) đều **EOL trên NIM** (HTTP 410) `[02 §2.2][TK]`; các key dự phòng (Groq, Gemini,
   DeepSeek, HF) trong `.env` **rỗng** `[02 §0]`.
2. **Dữ liệu augmentation cũ phải bỏ toàn bộ.** 0% [0–1.9] bài augmented là bài toán mới; 84% là "câu chuyện cũ
   + phép tính gắn đuôi"; 58% lộ biến/đáp án `[R5]`. Nguyên nhân nằm ở code: mọi phép mutate chỉ bọc biểu thức
   đáp án bằng hằng số ngẫu nhiên `[TK: mutation_engine.py:18-71]`. "SMT-verified" chỉ chứng nhận **một con số**,
   không chứng nhận **một đề hợp lệ** `[R5]`.
3. **Bản dịch (GSM8K/MATH/SVAMP/ASDiv) dùng được sau khi sửa có mục tiêu** (95–99% dùng ngay) `[R1][R2][R6]`;
   MetaMathQA-vi (5CD-AI) thì không (78,7%, lỗi thuật ngữ có hệ thống) `[R6]`.
4. **Bộ chấm điểm hiện tại cho điểm sai** (`0.125`≡`125`, `(1,2)`≡`1.2`) `[TK]` và kết quả SFT+DPO cũ chứa đáp án
   bị chép vào output `[05 §4.5]` → **không trích dẫn bất kỳ con số cũ nào**.
5. **Câu hỏi nghiên cứu mới (Q2):** *"Lợi thế của dữ liệu dịch có còn đứng vững khi đánh giá trên đề thi tiếng Việt
   gốc?"* — với C1 (benchmark) là công cụ đo, C2 (symbolic vs LLM, có/không kiểm chứng) là các nhân tố của thiết kế
   giai thừa, C3 (phân tích lỗi) giờ có số đo mạnh. **Không bỏ thành phần nào** `[01 §5]`.
6. **Nhánh symbolic phải thiết kế lại từ biểu diễn:** mutate ở mức **câu chuyện** (đổi dữ kiện có kiểu, thêm/bớt
   bước, đổi câu hỏi) trên chương trình có biến đặt tên theo thực thể — không phải bọc đáp án `[R3][R4][01 A3]`.
7. **Stack mới:** generator `deepseek-v4.1-flash` (NIM, MIT), judge `nemotron-3-ultra` + `glm-5.3-flash` + SymPy;
   student **Qwen3.5-Base 0.8B/2B/4B + Gemma-4-E2B** (fallback Qwen3-Base nếu pilot ngày 1 trượt); SFT bf16 LoRA một
   công thức chung + RLVR là giai đoạn phụ có đối chứng random-reward `[02 §5][03 §6]`.
8. **Claude $100 chỉ dùng phía đánh giá** (QA benchmark, sửa bản dịch tập test, chấm audit); ngân sách $100 phù hợp
   nhất với khối lượng QA/kiểm tra, không đủ cho sinh dữ liệu hàng loạt `[02 §8]`.
9. **Nút thắt là thời gian tính toán, không phải tiền:** lưới Q2 cần ~4–5 chu kỳ Modal $30/tháng (ước tính trước
   khi đổi student; xem §6) `[03 §6.6]`. Cần 4 quyết định của bạn (§8).

---

## 1. Hiện trạng đã đo (evidence)

### 1.1 Dữ liệu — đọc tay theo mẫu, khoảng tin cậy Wilson 95%

| Phần dữ liệu | Mẫu đọc | Kết quả chính | Kết luận |
|---|---|---|---|
| Vi-GSM8K train (7.473) | 150 | dịch sai nghĩa 3,3% [1,4–7,6]; dùng ngay 96,7%; 14 dòng hỏng toàn file (220–225, 2577, 3077…) | **FIX** `[R1]` |
| Vi-GSM8K test (1.319) | 150 | sai nghĩa 1,3% [0,4–4,7]; gold khớp 100%; ~17 bài đọc ra khác đáp án | **FIX** → dùng bản Platinum 1.209 + soát lại `[R1][04 H2]` |
| Vi-MATH train (7.501) | 150 | lỗi nặng 4,7%; `\right` hỏng do escape JSON 175 dòng `[TK]`; precalculus lệch dòng từ 478 | **FIX** `[R2]` |
| Vi-MATH test (5.000) | 150 | lỗi nặng 3,3%; "congruent"→"đồng dạng" 11 bài; không có trường gold | **FIX** ~3h `[R2]` |
| Vi-SVAMP / Vi-ASDiv | 150 / 150 | dùng ngay 98,0% / 98,7% (ASDiv là CC BY-NC) | **KEEP / FIX 1h** `[R6]` |
| MetaMathQA-vi (395k) | 150 | dùng ngay 78,7% [71,4–84,5]; "dozen"→"chục" 44%, "median"→"mean" 40%, LaTeX bị dịch 5.048 dòng | **Không train thô** `[R6][04 H4]` |
| SMT GSM8K (7.473) | 150 + Z3 toàn bộ | trung thực & suy ra đáp án 54,0% [46,0–61,8]; 81% chỉ là chuỗi gán (Z3 = máy tính); hard-code 7,6%; chỉ 42% hỗ trợ mutate mức câu chuyện | **FIX/làm lại** `[R3]` |
| SMT MATH (7.500) | 150 + 100 | chứng nhận dùng được 27,3% [20,8–35,0] (Algebra 67%, C&P 0%); 45% dòng `status=pass` là hỏng; nhãn phase1 không phân biệt được tốt/xấu | **Chỉ giữ Algebra, làm lại** `[R4]` |
| Augmented (19.750) | 200 + 50 đối chứng | bài mới 0%; gắn đuôi 84%; vô lý 15,5%; lộ 58%; giáo viên chấp nhận 10,5% (gốc: 98%) | **DISCARD** `[R5]` |
| Scrape thử loigiaihay | 96 đoạn | text khớp trang gốc; chỉ 9/96 sẵn sàng cho solver; đề 2026 mất câu do ghi đè số thứ tự | **Viết lại scraper** `[R6]` |

### 1.2 Code & kết quả cũ

- Package mới `vi_math_verified_aug/` ~10%: 3 module thật, 23 stub; 54 test pass nhưng test không bắt được lỗi chấm
  sai `[05 §3][04 H9][TK]`.
- `eval/math_verify.py`: false positive `0.125≡125`, `(1,2)≡1.2` `[TK]`, `"8"≡"x=7; y=8"` `[04 H9]`, số 3 chữ số
  thập phân bị đọc thành hàng nghìn `[R5]`. HF `math-verify 0.9.0` xử lý cấu trúc tốt nhưng đọc sai số kiểu Việt
  (`2,5`, `1.000.000`) `[04 §4]`.
- Kết quả `result/augmented-sft-dpo/math`: 185/5.000 dòng chứa đúng đáp án gold (53 dòng chép cả chuỗi gold bị hỏng)
  → con số 53% Vi-MATH không hợp lệ `[05 §4.5]`.
- Không có checkpoint model nào trên máy; prompt chưa được version; dependency không ghim; ~45% `data/` là bản
  trùng/trung gian; không có key nào lọt vào git history `[05 §5]`. **Có full-text bài báo bên thứ ba bị commit và
  push** (`src/mutation_informalize/reference/paper_text.txt`) `[05]`.

### 1.3 Hạ tầng & model

- Máy: RTX 3050 Ti 4 GB, **RAM 7,7 GB**, WSL2 chỉ thấy ~3 GB; torch Anaconda hỏng; venv không có stack train
  → laptop là máy dev, không phải máy train lưới `[03 §1][02 §1]`.
- NIM: trong 84 model liệt kê, khoảng một nửa trả 404/410 với tài khoản này; gọi được thật: `deepseek-v4.1-flash`,
  `nemotron-3-ultra-550b-a55b`, `glm-5.3-flash`, `gemma-4-31b-it`, `nemotron-3.5-bolt-e4b`… `[02 §2.2][TK]`.
- Modal: ~~L4 $0,222/h~~ **SAI.** Giá thật của workspace `tran-hoang-dat-2312` đo bằng `modal billing rates`
  (modal 1.6.3, 2026-11-07) `[TK]`: **T4 $0,59 · L4 $0,80 · A10G $1,10 · L40S $1,95 · A100-40 $2,10 · H100 $3,95**
  mỗi giờ. Vậy $30 ≈ **37 giờ L4** hoặc 51 giờ T4, không phải 135 giờ. `03 §4–§6` tính sai khoảng 3,6×. Tháng này
  đã dùng $0,06, billed $0; CLI không hiển thị số dư credit.

---

## 2. Câu hỏi nghiên cứu & đóng góp (mục tiêu Q2, giữ nguyên scope)

### 2.1 So sánh các hướng `[01 §5]`

| Tiêu chí | **RQ-1 dữ liệu dịch vs gốc, đo trên đề gốc** | RQ-2 ngôn ngữ suy luận | RQ-3 ablation tín hiệu kiểm chứng | RQ-4 chỉ benchmark |
|---|---|---|---|---|
| Tính mới 11/2026 | **Cao** — hai nghiên cứu kết luận "dịch thắng" (2508.14828: 78,3 vs 72,8; 2505.19848: 51,4 vs 39,4) nhưng **đều đo trên benchmark dịch** | Thấp–TB (lĩnh vực đông) | Thấp (FLAMES 2508.16514 đã làm, dự đoán null) | TB (thể loại đã có ở Hàn, Romania) |
| Văn liệu đã dự đoán đáp án? | Không — cả hai kết quả đều có nghĩa | Phần lớn | Có | Một phần |
| Một mình qua Q2? | **Có** (theo spec §2.3) | Khó | Khó | Ranh giới |

**Quyết định:** RQ-1 làm câu hỏi chính; RQ-4 (benchmark) xây đủ spec làm công cụ đo bắt buộc; RQ-2 và RQ-3 giữ là
**nhân tố đầy đủ** trong cùng thiết kế giai thừa. Lý do: đây là hướng duy nhất mà văn liệu 2026 chưa trả lời, tái
dùng được tài sản đang có (bản dịch, Vi-GSM8K, scraper), và mọi kết quả đều ra được bộ bàn giao.

### 2.2 Đóng góp (giữ đủ C1/C2/C3, sắp lại dưới câu hỏi mới)

| # | Nội dung | Thay đổi so với tháng 9 & evidence |
|---|---|---|
| **C1 Vi-ExamMath** | Benchmark đề tuyển sinh 10 + HSG, tự luận, Core (≥400) + Hard (≥200), có provenance, **tách theo thời gian** để đo contamination, IRT | Vẫn mới: chưa có benchmark toán tiếng Việt tự luận nào được công bố dữ liệu `[01 §4][04 H8]`. Thêm temporal split + IRT vì văn liệu 2026 coi là chuẩn `[01 §2.5]` |
| **C2 So sánh nguồn dữ liệu** | Các nhánh: dịch / sinh trực tiếp tiếng Việt (có/không kiểm chứng) / symbolic mức câu chuyện (có/không kiểm chứng) / trộn, ở nhiều quy mô dữ liệu, ≥2 họ model × ≥2 kích thước × 3 seed, + đối chứng tiếng Anh | Trục symbolic-vs-LLM một mình đã bị 2602.19187 làm (tiếng Anh, RL) `[01 §4]` → chuyển trọng tâm sang trục dịch-vs-gốc; symbolic giữ làm nhân tố, **có kiểm soát độ phủ** (so trên tập con biểu diễn được) |
| **C3 Phân tích lỗi** | "Kiểm chứng symbolic chứng nhận con số, không chứng nhận đề": số đo đầy đủ từ R3/R4/R5 + công cụ audit tái dùng được | Trước đây là số liệu mỏng; giờ có tỷ lệ đọc tay kèm CI và ví dụ nguyên văn `[R5]`. Đính chính số cũ: hard-code GSM8K 7,6%, MATH 36% (không phải 15,9%) `[R3][R4]` |
| **C4 (mới) Giai đoạn RLVR** | "Lợi thế của nhánh dữ liệu có sống sót qua RL?" — GRPO-LoRA từ checkpoint SFT, có đối chứng random-reward | Reviewer 2027 sẽ hỏi; RL trên Qwen có hiệu ứng giả nên bắt buộc có đối chứng `[03 §2][arXiv 2506.10947]` |

### 2.3 Spec để đạt Q2 `[01 §5 RQ-1][03 §6.6]`

- **Nhánh (cùng N):** S0 không augment · T dịch (bản dịch sạch do mình làm, che LaTeX) · T-off (5CD-AI, nguyên trạng,
  làm mốc "dịch sẵn có") · N-checked / N-unchecked (sinh trực tiếp tiếng Việt) · Y-checked / Y-unchecked (symbolic
  mức câu chuyện) · M = T+N.
- **Quy mô dữ liệu:** ≥4 điểm (2k/5k/10k/20k) trên các nhánh chính.
- **Model:** ≥2 họ × ≥2 kích thước (+ cỡ thứ 3), 3 seed mỗi ô.
- **Đánh giá:** phía dịch (Vi-GSM8K-Platinum, Vi-MATH alg+prealg, PolyMath-vi, BenchMAX-vi) vs phía gốc
  (Vi-ExamMath Core/Hard, tách trước/sau mốc thời gian) + đối chứng tiếng Anh.
- **Thống kê:** hồi quy logistic hỗn hợp `correct ~ arm × eval_type × size + (1|item) + (1|seed)`, paired bootstrap,
  McNemar, IRT.
- **Cơ chế:** khoảng cách văn phong dịch (perplexity, embedding), quy ước đơn vị/số, dạng bài đặc thù Việt Nam
  ("giải bài toán bằng cách lập phương trình").
- **Đánh giá người:** ≥200 mục/nhánh sinh, ≥3 người chấm, Krippendorff α.

---

## 3. Dữ liệu

### 3.1 Seed và các nhánh

| Lựa chọn | Evidence | Quyết định |
|---|---|---|
| Seed = GSM8K-train + MATH algebra/prealgebra-train (bản dịch, đã sửa) | MIT; bản dịch đã kiểm 95–97% dùng ngay `[R1][R2]`; giữ các nhánh cùng gốc `[04 §1.3]` | **Giữ** làm seed chung |
| Seed gốc tiếng Việt crawl (hllj, LuminarAI 306k, toanmath-alpha) | Không license dùng được; trùng nguồn với Vi-ExamMath → rò benchmark `[04 §1.2]` | **Không dùng để train**; chỉ dùng để quét contamination |
| Nhánh N (sinh trực tiếp tiếng Việt) | Văn liệu cảnh báo dữ liệu sinh "gốc" có thể thiếu tự nhiên `[01 §6.9]` | Sinh từ **cùng seed** (giữ khái niệm/độ khó) nhưng viết lại theo bối cảnh Việt (đồng, km, dấu phẩy thập phân, dạng bài SGK) + pilot độ tự nhiên 200 mục trước khi chạy lớn |
| Nhánh T (dịch) | 5CD-AI hỏng thuật ngữ có hệ thống `[R6]` → so với bản dịch kém sẽ làm lệch kết luận | Tự dịch lại phần EN đã lọc (MetaMathQA MIT) với LaTeX được che; giữ 5CD-AI làm nhánh T-off |
| Long-CoT corpora (OpenMathReasoning, OpenR1…) | Model ≤3B học kém từ chuỗi dài (2502.12143) `[03 §2]` | Không làm nhánh; dùng **CoT ngắn** |

### 3.2 Nhánh symbolic thiết kế lại (Y)

| Phương án | Evidence | Đánh giá |
|---|---|---|
| (a) Sửa mutate cũ (vẫn bọc đáp án) | 0% bài mới, 84% gắn đuôi `[R5]` | **Loại** |
| (b) Bỏ nhánh symbolic | Trái ràng buộc "không cắt scope"; mất C2 | **Loại** |
| **(c) Mutate mức câu chuyện** trên chương trình có kiểu | Chỉ 42% (GSM8K) / 17% (MATH) chương trình hiện tại có biến đặt tên theo thực thể `[R3][R4]` → phải formalize lại; 2602.19187 thấy symbolic cấp SymPy cho bài đa dạng hơn và tăng điểm ở 1,5B, còn SMT-LIB diễn đạt lại không tự nhiên `[01 A3]` | **Chọn** |

**Cách làm (c):**
1. Formalize lại seed thành **chương trình Python/SymPy có kiểu**: mỗi đại lượng là một biến đặt tên theo thực thể,
   kèm kiểu và miền (đồ vật đếm được → số nguyên ≥0; tuổi, giá, tỉ lệ có cận); dùng Z3 cho bài có ràng buộc/hệ/bất
   phương trình và để kiểm nghiệm duy nhất.
2. Mutate **dữ kiện**: đổi giá trị trong miền, thêm/bớt một bước, đổi đại lượng được hỏi. **Không bao giờ** đụng vào
   biểu thức đáp án.
3. Giải bằng solver → đáp án mới; loại nếu vi phạm miền (0,5 cái kẹo), không duy nhất, hay đáp án "xấu" bất thường.
4. LLM **sửa đề gốc** theo dữ kiện mới (chỉnh văn bản gốc, không dịch từ SMT); **không đưa đáp án cho LLM** (vòng tự
   sửa cũ làm lộ đáp án vào đề `[R5]`).
5. Kiểm tra vòng: một LLM khác họ formalize lại đề mới một cách độc lập → solver phải ra cùng đáp án.
6. Lọc tự động: không lộ tên biến/đáp án, độ giống với đề gốc nằm trong ngưỡng.
7. **Cổng pilot:** 200 mục, đọc theo PROTOCOL; chỉ chạy lớn khi tỷ lệ "giáo viên chấp nhận" ≥ tỷ lệ của nhánh N.
   Ngưỡng chính xác do bạn chốt.
8. Phạm vi: GSM8K + Algebra (nơi chứng nhận có thật `[R4]`). So sánh giữa các nhánh làm **cả trên tập con biểu
   diễn được** để không lẫn "kiểm chứng" với "độ phủ" `[01 §5 RQ-3]`.

### 3.3 Sửa các bộ hiện có

| Bộ | Việc cần làm | Ai/công cụ |
|---|---|---|
| Vi-GSM8K test | Bản Platinum 1.209; dịch lại 10 bài đã sửa; lọc bài model giải đúng EN nhưng sai VI + 65 bài trùng mẫu lỗi → đọc lại 60–150 bài | Claude (phía eval) + người song ngữ `[R1]` |
| Vi-MATH test | Gắn gold từ bản EN theo vị trí; sửa 51 lỗi (congruent, `\n` thô, `[asy]`) | Claude + người `[R2]` |
| Vi-GSM8K / Vi-MATH train | Sửa 14 dòng GSM8K hỏng; sửa escape `\right`; căn lại precalculus; build lại file SFT gộp | Script + model NIM (Claude không được viết dữ liệu train) `[R1][R2]` |
| Vi-ASDiv | ~10 dòng + 2 khóa JSON; ghi rõ NC | ~1h `[R6]` |
| Xóa bỏ | 19.750 augmented, `sft_data`, `dpo_data`, `train_cot_only*`, bản trùng, `result/augmented-sft-dpo` | Lưu trữ ngoài repo, không xóa vĩnh viễn `[05 §6]` |

---

## 4. Benchmark Vi-ExamMath (C1)

| Vấn đề | Evidence | Giải pháp |
|---|---|---|
| Nguồn đã thay đổi | toanmath chặn PDF/Word trong robots.txt; VnExpress 2026 đăng đề dạng ảnh; diendantoanhoc bị Cloudflare chặn; loigiaihay còn crawl được `[04 H5–H6 §3.1]` | Nguồn chính loigiaihay; đề và đáp án chính thức tải tay từ Sở GD-ĐT; toanmath chỉ dùng để tìm danh mục; VnExpress làm bằng chứng mốc thời gian |
| Scraper thử mất câu, trích đáp án đúng ~37% | `[R6]` | Viết lại scraper (khóa theo cấu trúc DOM, không theo số thứ tự câu); trích đáp án bằng solver + Claude QA + người |
| Contamination | Đề đã có trên HF (toanmath-alpha 513k ảnh, LuminarAI 306k dòng) `[04 H7]` | Dựa vào **mốc thời gian + quét n-gram**, không dựa vào "nguồn ít người biết" |
| **Mốc thời gian theo model** | Qwen3.5-Base tạo 2026-02-28, Gemma-4 2026-03-02 `[TK]` | Với student 2026, chỉ **đề tháng 6/2026** là sau mốc; đề 2025 chỉ là sau mốc với Qwen3. Phải báo mốc riêng từng model; cỡ tập sau mốc nhỏ → cần nhiều tỉnh |
| Bão hòa ở model lớn | CSAT Hàn 2026: model 20B ~100% `[01 E1]` | Core chỉ dùng để phân biệt model nhỏ/vừa; Hard làm trần; chạy cổng độ phân biệt trước khi chạy lưới |
| Không để rò đề test | NIM có quyền dùng input để cải thiện model (§3.3(iv)) `[02 §2.5]` | Không gửi đề test lên NIM/Gemini; chỉ chạy model hosted **một lần cuối**, có log; QA bằng Claude (Anthropic không train trên nội dung API, theo `[02 §8]`) |
| Công bố được không | Luật SHTT Đ15(2) + NĐ17/2023 Đ8(2): văn bản cơ quan nhà nước không được bảo hộ; lời giải bên thứ ba được bảo hộ `[04 H10]` | Phát hành: đề chính thức + lời giải tự viết + URL nguồn. Đóng gói chống crawl (zip có mật khẩu, như DeepSEA) `[04 §2.1]` |

---

## 5. Model

### 5.1 Generator / judge (verified live) `[02 §2, §5.1][TK]`

| Vai | Chọn | Lý do (evidence) | Dự phòng |
|---|---|---|---|
| Generator | `deepseek-ai/deepseek-v4.1-flash` (NIM) | Gọi được `[TK]`; 10/10 bài lớp 10, tiếng Việt chuẩn; weights MIT → tự host lại được khi bị EOL; lời giải ngắn nhất (hợp CoT ngắn) | `nemotron-3-ultra-550b-a55b` |
| Judge / solver độc lập | SymPy (quyết định) + `nemotron-3-ultra` + `glm-5.3-flash` | Khác họ với generator → đồng thuận có ý nghĩa; Nemotron nhanh nhất (9–15 s) | `gemma-4-31b-it` |
| Generator yếu | `nemotron-3.5-bolt-e4b` | Thay Llama-3.1-8B đã EOL; nhưng cũng đạt 10/10 → **phải đo khoảng cách thật** | Qwen3.5-0.8B tự chạy |
| Solver tham chiếu độ khó | Qwen3.5-2B tự chạy, đóng băng | Model hosted bị bão hòa trên bài dễ; tự chạy thì không bị EOL | Bolt-E4B |
| Loại bỏ | Step-5, MiniMax, Mistral-L4, Gemini free (không đưa vào stack sinh dữ liệu) | `[02 §2.3, §4.2]` | — |

**Bắt buộc trước khi khóa:** pilot 100 bài Vi-GSM8K + 50 bài khó trên 3 ứng viên, vì smoke test 10 bài không xếp hạng
được. Rủi ro lớn nhất là model bị EOL: v4-pro-0813 chỉ sống khoảng 1 tháng `[02 §6]` → sinh toàn bộ corpus trong một
cửa sổ ngắn, log đầy đủ mọi request, và đóng băng.

### 5.2 Student `[02 §3, §5.2][03 §6.3]`

| Lựa chọn | Evidence | Quyết định |
|---|---|---|
| **Qwen3.5-Base 0.8B / 2B / 4B** | Apache-2.0, không gated, tạo 2026-02-28 `[TK]`; 201 ngôn ngữ; 3 kích thước cùng họ | **Họ chính**, nếu qua pilot ngày 1 |
| Qwen3-Base 0.6B / 1.7B / 4B | Có điểm base chính thức (GSM8K 59,6 / 75,4 / 87,8), tooling trưởng thành | **Fallback** nếu Qwen3.5 trượt pilot (kiến trúc lai, báo cáo lỗi packing làm loss về 0, không khuyến nghị QLoRA) |
| **Gemma-4-E2B** | Apache-2.0 `[TK]`; khác tokenizer và kiến trúc; nặng (5,1B tổng, LoRA cần 8–10 GB) | **Họ thứ hai** (chạy trên L4) |
| Gemma-3-1B (đề xuất của `03`) | Base GSM8K **2,20**, MATH 3,66 (Qwen3 TR) → hiệu ứng sàn `[02 §3.4]` | **Loại**; `02` đúng, `03` sai ở điểm này |
| Sailor2-1B | Thích nghi tiếng Việt (perplexity 7,52 vs 26,86) | Đối chứng tùy chọn |

**Cổng pilot ngày 1** (1 phiên L4):
- LoRA bf16 có packing chạy ổn và loss bình thường.
- vLLM eval chạy được.
- Few-shot Vi-GSM8K: Qwen3.5 ≥ Qwen3 cùng cỡ; loại base nào dưới ~15%.
- Đo fertility tokenizer tiếng Việt.

---

## 6. Huấn luyện & hạ tầng `[03]`

### 6.1 Công thức (một công thức chung cho mọi nhánh)

- **SFT là công cụ so sánh chính.** Distillation hiệu quả hơn RL khoảng 10× compute; RL trên Qwen một phần là "khơi
  gợi" khả năng sẵn có và có hiệu ứng giả `[03 §2]`.
- **bf16 LoRA r=64, mọi lớp linear**, LR ≈ 10× LR full-FT (LoRA Without Regret: tương đương full-FT ở quy mô
  5–25M token), loss chỉ trên lời giải, packing, CoT ngắn, ≥3 seed.
- **Sweep LR × epoch một lần trên S0 rồi đóng băng.** Thêm 1 lần so full-FT với LoRA để chứng minh LoRA không phải
  nút thắt.
- Bỏ QLoRA 4-bit trên laptop: trộn 4-bit với bf16 giữa các kích thước làm lẫn kích thước với độ chính xác số học.
- **RLVR (C4):** GRPO-LoRA (`dr_grpo`, 300 bước) từ checkpoint SFT của S0 và các nhánh chính, + random-reward ×
  3 seed mỗi họ.

### 6.2 Hạ tầng $0

| Nguồn | Vai trò | Evidence |
|---|---|---|
| **Modal $30/tháng** (không thêm thẻ) | Mọi lượt train của 2B/4B/Gemma, toàn bộ eval, RLVR | L4 $0,222/h → 135 L4-h/tháng `[03 §4]` |
| Laptop | Dev, test, smoke run; có thể train khối 0.8B nếu vừa VRAM (Unsloth: 0.8B bf16 LoRA ≈ 3 GB) | `[02 §3.3][03 §1]` |
| Kaggle T4×2 (**bạn đã loại trước đây**) | Khối data-efficiency + đường tái lập cho người tiếp nhận | ~30 h/tuần, fp16 `[03 §4]` |
| Modal academic credits (≤$10k) / GPU phòng lab VNU | Gỡ nút thắt | `[03 §4]` |

- **Ước tính chi phí:** lưới ≈ 230 lượt SFT + 30 lượt RL ≈ $167–217 nếu tính toàn bộ theo giá Modal (sai số ±2×)
  `[03 §6.6]`.
- **Ước tính đó tính cho Qwen3 + Gemma-3-1B.** Đổi sang Qwen3.5-2B (2,27B) và Gemma-4-E2B (5,1B tổng) sẽ **đắt hơn**.
  Con số mới phải tính lại bằng tokens/s đo thật ở pilot ngày 1. Tôi chưa đo; ước lượng của tôi là ~1,3–1,8×.
- **Với $0, đây là 5–10 tháng chạy Modal.** Vì vậy grant hoặc GPU phòng lab gần như là điều kiện tới hạn để không
  phải cắt scope.
- Stack: Unsloth + TRL + transformers v5 + vLLM ≥0.17 trên Linux (Modal), ghim phiên bản bằng lock file và image
  digest `[03 §6.3]`.

---

## 7. Đánh giá, chi tiêu Claude, bàn giao

### 7.1 Chấm điểm

- Giữ module của mình làm lớp chuẩn hóa số kiểu Việt (`2,5`, `1.000.000`, đơn vị, `x=`), phía sau là HF
  `math-verify 0.9.0` `[04 §4]`.
- Sửa các lỗi chấm sai đã thấy (§1.2), mỗi lỗi một unit test.
- Lưu gold ở dạng có cấu trúc.
- Không dùng LLM-judge không tham chiếu `[01 §2.5]`.

### 7.2 Claude $100 `[02 §8][TK]`

- **Ghi nguồn (cập nhật 2026-11-07):** datasheet ghi rõ phần dữ liệu do Claude tạo (model id, ngày, phiên bản và
  sha256 của prompt); log API giữ nguyên như mọi nguồn khác. Không giấu nguồn.
- **Phân bổ** (Opus 5.5 qua Batch API giảm 50%, ≈7k lượt `[02 §8.2]`; không đủ để sinh hàng loạt một nhánh):
  - Phía train: sửa bản dịch seed (14 dòng Vi-GSM8K, `\right`, precalculus, thuật ngữ); dịch lại phần MetaMathQA
    dùng cho nhánh T (che LaTeX); làm judge/QA thứ ba cho bộ lọc.
  - Phía eval: QA từng bài Vi-ExamMath + kiểm đáp án; sửa bản dịch các bài bị lọc ở tập test; chấm audit.
- **Không dùng Claude làm generator chính** của riêng một nhánh, vì sẽ lẫn hiệu ứng "đổi model sinh" với hiệu ứng
  nguồn dữ liệu. Generator chung vẫn là `deepseek-v4.1-flash`.

### 7.3 Bộ bàn giao nội bộ `[06 §5]`

- **Hosting:** GitHub (code) + HF Hub private (benchmark, corpus, adapter, prediction) + Zenodo khi công bố.
  Thêm DOI cho benchmark ở v1.0.
- **Mỗi artifact có:** datasheet/model card, license, provenance theo nguồn.
- **Log mọi lời gọi API:** model id, ngày, tham số, request và response thô.
- **Prompt có version;** khóa môi trường bằng `uv.lock` + Modal image.
- **Tái lập bằng một lệnh** theo các tầng (eval → train một ô → toàn lưới), có dung sai.
- **Checklist nghiệm thu** chạy được trên máy mới.
- **License:** code MIT; data CC BY 4.0, trừ Vi-ASDiv là NC; weights kế thừa base (Apache-2.0).
- **Dọn ngay:** gỡ `paper_text.txt` của bên thứ ba khỏi repo.

---

## 8. Lộ trình theo cổng & quyết định cần bạn

### 8.1 Lộ trình

Mỗi pha chỉ mở khi cổng của pha trước đạt.

| Pha | Nội dung | Cổng ra |
|---|---|---|
| **P0 Nền móng** | Sửa bộ chấm + test; dọn repo (archive, gỡ file bên thứ ba); khóa môi trường; logger API; prompt registry | Không còn false positive nào trong bộ test mở rộng |
| **P1 Pilot** | (a) student ngày 1 · (b) teacher 150 bài · (c) symbolic mức câu chuyện 200 mục · (d) độ tự nhiên nhánh N 200 mục · (e) cổng độ phân biệt Vi-ExamMath | Mỗi pilot có báo cáo + quyết định ghi lại |
| **P2 Dữ liệu** | Sửa các tập eval; scraper mới + xây benchmark + QA (người + Claude); sinh các nhánh trong một cửa sổ ngắn; đóng băng; quét contamination | Datasheet + số đo chất lượng mỗi nhánh theo PROTOCOL |
| **P3 SFT** | Sweep → lưới chính → data-efficiency → đối chứng tiếng Anh | Prediction ở mức từng item được lưu đủ |
| **P4 RLVR** | GRPO + random-reward | — |
| **P5 Phân tích** | Mô hình hỗn hợp, IRT, cơ chế, đánh giá người | — |
| **P6 Đóng gói** | Checklist bàn giao | Máy mới tái lập được tầng eval |

### 8.2 Quyết định cần bạn

1. **Duyệt hướng §2** (RQ-1 làm câu hỏi chính, C1–C4 giữ đủ) và **thiết kế lại nhánh symbolic §3.2**.
2. **Kaggle:** cho phép dùng lại (khối data-efficiency + đường tái lập) hay giữ loại?
3. ~~Compute: Modal academic / GPU lab~~ → **user 2026-11-07: không xin được Modal academic, không có GPU lab.**
   Nguồn còn lại: Modal $30/tháng + laptop + (Kaggle nếu duyệt) + Colab (không lập kế hoạch được). Hệ quả: không cắt
   scope thì **kéo dài lịch**; Kaggle trở thành đòn bẩy lớn nhất còn lại.
4. **Người chấm:** Q2 cần ≥2–3 người đọc QA benchmark và chấm độ tự nhiên. Bạn có người không?
5. Bỏ hẳn (archive ngoài repo) dữ liệu augmented và kết quả cũ như §3.3?

---

## 9. Các điểm tôi đã tự kiểm lại `[TK]` (2026-11-07)

| Khẳng định | Cách kiểm | Kết quả |
|---|---|---|
| Generator/model yếu cũ đã EOL | POST NIM chat | `deepseek-v4-pro-0813` → 410 "end of life on 2026-09-14"; `llama-3.1-8b-instruct` → 410 "2026-08-26" |
| Model mới gọi được | POST NIM chat "7 cộng 5" | `deepseek-v4.1-flash` 200 "12"; `nemotron-3-ultra-550b-a55b` 200 "12" |
| Student mới tồn tại, license | HF API | Qwen3.5-0.8B-Base, 2B-Base: 2026-02-28, apache-2.0, không gated; gemma-4-E2B: 2026-03-02, apache-2.0, không gated |
| Bộ chấm cho điểm sai | `is_correct()` | `0.125` vs `125` → True; `(1,2)` vs `1.2` → True |
| Nguyên nhân mutate | Đọc `mutation_engine.py:18-71` | `rand_const = uniform(0.5, 5.0)`; 3 phép mutate chỉ thay `answer_expr` |
| Mẫu augmented | In 4 mẫu ngẫu nhiên | #5158 "nhân số thẻ đó với tổng của 4,85 và 4,85"; #9240 lộ "answer = A×(A+1)" |
| `\right` hỏng ở Vi-MATH train | Đếm ký tự `\r` + "ight" | 175/7.501 dòng |

**Chưa tự kiểm** (mới dựa vào báo cáo agent):
- Giá Modal.
- Quota Kaggle (`03` cũng ghi là chưa có trang chính thức).
- Các điều luật SHTT.
- Số liệu trong các paper.
- Tương thích của các phiên bản thư viện được ghim (chưa cài thử).
