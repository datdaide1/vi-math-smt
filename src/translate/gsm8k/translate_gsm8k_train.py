# %%
import json, time, os, math
from openai import OpenAI
from datetime import datetime

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────
# Set BEEKNOEE_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY    = os.environ.get("BEEKNOEE_API_KEY", "")
BASE_URL   = "http://127.0.0.1:20128/v1" 
MODEL_NAME = "beeknoee/gemini-2.5-flash"

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────
BATCH_SIZE = 50        # chọn 50 là sweet spot giữa tốc độ và ổn định
RPM_LIMIT   = 500      # requests per minute
RPD_LIMIT   = 100      # requests per day

AVG_TOKENS_PER_SAMPLE = 3500  # trung bình mỗi sample ~3500 token
TPM = 250000                  # tokens per minute giới hạn

# Tính số token mỗi batch
tokens_per_batch = BATCH_SIZE * AVG_TOKENS_PER_SAMPLE

# Delay tối thiểu giữa 2 request, tính theo:
# 1. Giới hạn RPM (requests per minute)
# 2. Giới hạn TPM (tokens per minute)
MIN_DELAY = max(60 / RPM_LIMIT + 0.5, tokens_per_batch / TPM * 60)


# Assumes this script is run from its own directory (src/translate/)
INPUT_FILE  = os.path.join("..", "..", "..", "data", "generate", "gsm8k", "train.json")
OUTPUT_FILE = os.path.join("..", "..", "..", "data", "translation", "gsm8k", "gsm8k_train_vietnamese.jsonl")
# ───────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


import json, time

def translate_batch(batch_data, max_retries=5):
    payload = [{"question": x["question"], "answer": x["answer"]} for x in batch_data]

    prompt = f"""Bạn là chuyên gia toán học và dịch thuật. Dịch mảng JSON các bài toán tiếng Anh sang tiếng Việt.

⚠️ QUAN TRỌNG: Chỉ trả về **một mảng JSON**, KHÔNG thêm comment, text, hoặc ký tự nào khác. 
Ví dụ: [{"{"} "question": "...", "answer": "..." {"}"}]

YÊU CẦU:
1. Trả về đúng định dạng mảng JSON gốc với key "question" và "answer".
2. Giữ nguyên biểu thức toán, số, và nội dung trong << >>.
3. Giữ nguyên ký tự ngắt dòng (\\n).

Dữ liệu:
{json.dumps(payload, ensure_ascii=False)}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Always respond with valid JSON array or object."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_content = response.choices[0].message.content.strip()

            # ── CLEAN RAW CONTENT ───────────────────────────────
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()

            # parse JSON
            result = json.loads(raw_content)

            # Nếu AI trả string JSON bên trong
            if isinstance(result, str):
                result = json.loads(result)

            # Lấy mảng JSON chính
            translated_data = None
            if isinstance(result, list):
                translated_data = result
            elif isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, list):
                        translated_data = v
                        break

            # Kiểm tra xem AI có trả về đúng mảng không
            if translated_data is None:
                raise ValueError(f"AI không trả về mảng JSON hợp lệ, kiểu: {type(result)}")

            if len(translated_data) != len(batch_data):
                raise ValueError(f"Số lượng bài không khớp: gửi {len(batch_data)}, nhận {len(translated_data)}")

            return translated_data

        except Exception as e:
            print("\n" + "="*80)
            print(f"⚠️ Lỗi lần {attempt}/{max_retries}: {repr(e)}")
            try:
                print("RAW RESPONSE (2000 ký tự đầu):")
                print(raw_content[:2000])
            except:
                print("Không có raw_content để hiển thị")
            print("="*80 + "\n")

            # Delay tăng dần nếu rate limit
            err_str = str(e).lower()
            wait = 2 ** attempt
            if "429" in err_str or "rate_limit" in err_str:
                wait = (2 ** attempt) * 30
                print(f"⏳ Rate limit, đợi {wait}s trước khi retry...")
            else:
                print(f"⏳ Đợi {wait}s trước khi retry...")
            time.sleep(wait)

    # Nếu vẫn lỗi sau max_retries → dừng pipeline
    print("❌ Hết lượt retry, dừng pipeline để kiểm tra batch này.")
    raise RuntimeError("Batch dịch bị lỗi, cần kiểm tra.")

# ── ĐỌC DATASET & CHECKPOINT ───────────────────────────────────────────────
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

processed = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        processed = sum(1 for line in f if line.strip())
    print(f"✅ Tìm thấy checkpoint: đã dịch {processed}/{len(dataset)} bài.")
else:
    print(f"🚀 Bắt đầu dịch mới.")

# ── VÒNG LẶP DỊCH ─────────────────────────────────────────────────────────
request_count = 0
success_count = 0
start_time    = time.time()

print(f"Bắt đầu dịch lúc {datetime.now().strftime('%H:%M:%S')}\n")

for i in range(processed, len(dataset), BATCH_SIZE):
    if request_count >= RPD_LIMIT:
        print(f"\n🛑 Đạt giới hạn {RPD_LIMIT} requests/ngày. Hẹn ngày mai!")
        break

    batch = dataset[i : i + BATCH_SIZE]
    batch_num = (i - processed) // BATCH_SIZE + 1

    translated_list = translate_batch(batch)
    request_count += 1

    if translated_list and isinstance(translated_list, list) and len(translated_list) == len(batch):
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for j, t in enumerate(translated_list):
                f.write(json.dumps({
                    "question"    : t.get("question", ""),
                    "answer"      : t.get("answer", ""),
                    "ground_truth": batch[j].get("ground_truth", "")
                }, ensure_ascii=False) + '\n')
        
        success_count += 1
        total_done = processed + (success_count * BATCH_SIZE)
        elapsed = (time.time() - start_time) / 60
        print(f"[Batch {batch_num:>4}] ✅ Xong {total_done:,}/{len(dataset):,} bài | "
              f"RPD còn: {RPD_LIMIT - request_count} | {elapsed:.1f} phút")
    else:
        print(f"[Batch {batch_num:>4}] ❌ Lỗi JSON hoặc API. Dừng để kiểm tra.")
        break

    if i + BATCH_SIZE < len(dataset):
        time.sleep(MIN_DELAY)

# ── TỔNG KẾT ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"🏁 Hoàn tất phiên làm việc!")
print(f"   Đã dịch thêm: {success_count * BATCH_SIZE:,} bài")
print(f"   Tổng tiến độ: {processed + (success_count * BATCH_SIZE):,}/{len(dataset):,}")
print(f"{'='*60}")
