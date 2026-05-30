# vi-math-smt

Xây dựng dữ liệu suy luận toán học tiếng Việt có kiểm chứng hình thức (SMT-LIB
+ Z3/CVC5), và pipeline tinh chỉnh mô hình ngôn ngữ lớn (SFT + Online DPO) trên
bộ dữ liệu đó.

> Repo này tập trung trình bày **kiến trúc và phương pháp** của pipeline. Các
> notebook huấn luyện/đánh giá mô hình (`notebooks/`, `src/evaluate/`) là
> triển khai tham khảo theo đúng thiết kế được mô tả — không kèm theo số liệu
> kết quả cụ thể nào trong tài liệu này.

## Kiến trúc tổng quan

```
GSM8K + MATH (tiếng Anh)
        │
        ├──► Hình thức hóa SMT-LIB (LLM sinh, Z3/CVC5 kiểm chứng) ──┐
        │                                                            ├─► Tập dữ liệu cơ sở
        └──► Dịch sang tiếng Việt (Gemini) ─────────────────────────┘         │
                                                                                ▼
                                                          Biến đổi SMT (5 chiến lược)
                                                                                │
                                                                                ▼
                                                          Phi hình thức hóa ngược (LLM)
                                                          + xác minh nhất quán bằng Z3
                                                                                │
                                                                                ▼
                                                              Tập dữ liệu huấn luyện
                                                              tăng cường (SFT + DPO)
```

Quy trình gồm hai nhánh song song hội tụ về một tập dữ liệu cơ sở, sau đó qua
giai đoạn tăng cường dữ liệu:

1. **Hình thức hóa SMT-LIB** — mô hình ngôn ngữ lớn (`openai/gpt-oss-120b` qua
   NVIDIA NIM) sinh mã SMT-LIB 2.6 mô tả cấu trúc toán học của từng bài, được
   kiểm chứng tự động bằng Z3 (và CVC5 cho MATH). Vòng lặp sinh — kiểm chứng —
   sửa lỗi cho phép mô hình tự sửa khi kiểm chứng thất bại.
2. **Dịch thuật** — Gemini 2.5 Flash dịch đề bài và lời giải sang tiếng Việt,
   giữ nguyên công thức toán, LaTeX và các ký hiệu tính toán trung gian.
3. **Hợp nhất tập cơ sở** — ghép mã SMT-LIB đã kiểm chứng với bản dịch tiếng
   Việt theo `index`, chuẩn hóa thành một schema thống nhất.
4. **Tăng cường dữ liệu** (chỉ áp dụng cho MATH):
   - *Biến đổi SMT* — 5 chiến lược biến đổi thuật toán thuần túy trên mã
     SMT-LIB (thay hằng số, biến đổi cấu trúc/biểu thức, kết hợp ràng buộc,
     biến đổi độ khó), mỗi biến thể được kiểm chứng lại bằng Z3.
   - *Phi hình thức hóa ngược* — LLM chuyển mỗi mã SMT-LIB đã biến đổi thành
     bài toán + lời giải tiếng Việt mới, kèm bước xác minh: LLM giải lại bằng
     suy luận từng bước và so khớp với nghiệm Z3.
5. **Xây dựng bộ kiểm thử** — dịch các bộ test gốc (GSM8K, MATH) và hai bộ
   ngoài miền phân phối (SVAMP, ASDiv) sang tiếng Việt theo cùng quy trình.
6. **Tinh chỉnh mô hình** — SFT (LoRA/RSLoRA trên Mistral-7B-Instruct-v0.2,
   qua Unsloth) trên tập cơ sở và tập tăng cường, sau đó Online DPO với một
   rule-based math judge (ưu tiên: đáp án đúng → tuân thủ định dạng `\boxed{}`
   → lời giải ngắn hơn) để tinh chỉnh thêm trên tập tăng cường.

## Cấu trúc thư mục

```
vi-math-smt/
├── notebooks/                 Pipeline SFT + Online DPO (Kaggle T4, numbered 00-06)
├── src/
│   ├── convert/                Tải dataset gốc từ HuggingFace, chuyển định dạng
│   ├── translate/
│   │   ├── gsm8k/                Dịch GSM8K sang tiếng Việt
│   │   ├── math/                 Dịch MATH sang tiếng Việt
│   │   └── test_dataset/         Dịch bộ test OOD (SVAMP, ASDiv)
│   ├── formalize/
│   │   ├── gsm8k/                 Sinh + kiểm chứng SMT-LIB cho GSM8K (ReAct agent)
│   │   └── math/                  Sinh + kiểm chứng SMT-LIB cho MATH (repair loop)
│   ├── mutation_informalize/    Biến đổi SMT + phi hình thức hóa (4 pha)
│   │   ├── tools/                  Script QC/sửa dữ liệu một lần (đứng độc lập)
│   │   ├── analysis/               Thống kê tỷ lệ nhất quán sau tăng cường
│   │   └── reference/              Ghi chú/snippet tham khảo khi viết prompt
│   ├── merge/
│   │   ├── gsm8k/                 Hợp nhất + thống kê phân phối GSM8K
│   │   └── math/                  Hợp nhất + thống kê phân phối MATH
│   ├── analysis/                Tiện ích xử lý/kiểm tra dữ liệu MATH
│   └── evaluate/                Đánh giá bằng Z3 (`z3_evaluate.py`), và so
│       ├── base/                  sánh model theo 3 nhóm: Vi-MathLM-Base,
│       ├── aug/                   Vi-MathLM-Aug (kèm đánh giá OOD), và
│       └── wizardmath/             baseline WizardMATH
├── data/                       (gitignored) toàn bộ dataset/output sinh ra
├── _scratch/                   (gitignored) script debug/patch cũ, giữ làm lịch sử dev
├── .env.example                Danh sách biến môi trường cần thiết
└── requirements.txt
```

Mỗi file trong `src/` và `notebooks/` giả định được chạy **từ chính thư mục
chứa nó** (đường dẫn tương đối trỏ ngược về `data/` ở gốc repo — số lượng
`../` tùy theo độ sâu thư mục). Một số file đi kèm `*.outputs.md` — log kết
quả thực thi thật đã ghi lại trước khi chuyển từ Jupyter notebook sang `.py`,
giữ làm tài liệu tham khảo.

Trong `src/mutation_informalize/`, các file ở thư mục gốc (`config.py`,
`phase1_validate.py` … `phase4_merge.py`, `run_pipeline.py`, v.v.) import lẫn
nhau qua same-directory import (`from config import ...`) nên **phải ở cùng
cấp**, không tách thêm — đây là lý do thư mục này vẫn còn khá nhiều file so
với các thư mục khác.

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env   # rồi điền các API key của bạn
```

Xem `.env.example` để biết đầy đủ biến môi trường cần thiết (HuggingFace,
Gemini, NVIDIA NIM, và một LLM proxy tùy chọn dùng trong một số script cũ).

## Chạy pipeline

```bash
# 1. Tải dữ liệu gốc từ HuggingFace
cd src/convert && python download_datasets.py

# 2. Hình thức hóa SMT-LIB (GSM8K: ReAct agent, MATH: sequential repair loop)
cd ../formalize/gsm8k && python formalize_gsm8k_nvidia_langgraph.py
cd ../math && python formalize_math_nvidia.py

# 3. Dịch sang tiếng Việt
cd ../../translate/gsm8k && python translate_gsm8k.py
cd ../math && python translate_math.py

# 4. Hợp nhất tập cơ sở
cd ../../merge/gsm8k && python gsm8k_complete.py
cd ../math && python math_complete.py
cd .. && python merge_all.py

# 5. Tăng cường dữ liệu (mutation + informalize)
cd ../mutation_informalize && python run_pipeline.py

# 6. Tinh chỉnh mô hình — dán từng file trong notebooks/ vào một Kaggle
#    notebook mới (thứ tự 00 → 06), theo hướng dẫn ghi trong docstring mỗi file
```

Mỗi bước có thể chạy độc lập nếu bạn đã có dữ liệu đầu vào tương ứng (ví dụ
chỉ muốn chạy lại bước dịch thuật trên dữ liệu đã hình thức hóa sẵn).

## Ghi chú

- Bộ dữ liệu tiếng Việt gồm 4 phần: GSM8K, MATH (trong miền phân phối) và
  SVAMP, ASDiv (ngoài miền phân phối, dùng để đánh giá khả năng tổng quát hóa).
- `src/mutation_informalize/config.py` hỗ trợ xoay vòng nhiều API key qua biến
  môi trường `NVIDIA_API_KEYS` (danh sách phân tách bởi dấu phẩy) để tăng
  throughput khi gọi LLM song song.
- `_scratch/` chứa các script vá lỗi/thử nghiệm trong quá trình phát triển,
  không thuộc pipeline chính thức — giữ lại để tham khảo lịch sử phát triển.
