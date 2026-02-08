# %%
import json, time, os, math
import google.generativeai as genai
from datetime import datetime

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME  = "gemini-2.5-flash"        # Đổi thành "gemini-2.5-flash-preview-04-17" nếu cần
BATCH_SIZE  = 20                         # 20 bài/request
RPM_LIMIT   = 5                         # Requests Per Minute (free tier)
RPD_LIMIT   = 22                        # Requests Per Day (free tier)
MIN_DELAY   = 60 / RPM_LIMIT + 1        # Delay tối thiểu giữa các request (giây)
# Assumes this script is run from its own directory (src/translate/)
INPUT_FILE  = os.path.join("..", "..", "..", "data", "generate", "gsm8k", "train.json")
OUTPUT_FILE = os.path.join("..", "..", "..", "data", "translation", "gsm8k", "gsm8k_train_vietnamese.jsonl")
# ──────────────────────────────────────────────────────────────────────────────

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

print(f"Model      : {MODEL_NAME}")
print(f"Batch size : {BATCH_SIZE} bài/request")
print(f"Delay      : {MIN_DELAY:.0f}s giữa mỗi request (>= 60/{RPM_LIMIT}+1)")
print(f"RPD limit  : {RPD_LIMIT} requests/ngày → tối đa {RPD_LIMIT * BATCH_SIZE:,} bài/ngày")
print(f"Input      : {INPUT_FILE}")
print(f"Output     : {OUTPUT_FILE}")

# %%
def translate_batch(batch_data, retries=4):
    """Dịch 1 batch với exponential backoff khi gặp lỗi 429."""
    payload = [{"question": x["question"], "answer": x["answer"]} for x in batch_data]

    prompt = f"""Bạn là chuyên gia toán học và dịch thuật. Dịch mảng JSON các bài toán tiếng Anh sang tiếng Việt.

YÊU CẦU:
1. Trả về đúng định dạng mảng JSON gốc với key "question" và "answer".
2. GIỮ NGUYÊN tất cả biểu thức toán, số, và nội dung trong << >> (ví dụ: <<48/2=24>>24).
3. Giữ nguyên ký tự ngắt dòng (\\n).

Dữ liệu:
{json.dumps(payload, ensure_ascii=False)}"""

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            result = json.loads(response.text)
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = (2 ** attempt) * 30  # 30s, 60s, 120s, 240s
                print(f"   ⏳ Rate limit (429). Chờ {wait}s rồi thử lại... (lần {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                print(f"   ❌ Lỗi API: {err[:120]}")
                return None
    print("   ❌ Hết lượt retry.")
    return None

print("✅ Định nghĩa hàm translate_batch() thành công.")

# %%
# Đọc dataset
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Kiểm tra checkpoint
processed = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        processed = sum(1 for line in f if line.strip())
    print(f"Tìm thấy checkpoint: đã dịch {processed}/{len(dataset)} bài. Tiếp tục...")
else:
    print(f"Bắt đầu mới.")

remaining     = len(dataset) - processed
batches_needed = math.ceil(remaining / BATCH_SIZE)
batches_today  = min(batches_needed, RPD_LIMIT)
items_today    = min(remaining, batches_today * BATCH_SIZE)
eta_minutes    = batches_today * MIN_DELAY / 60

print(f"Dataset  : {len(dataset):,} bài tổng | Còn lại: {remaining:,} bài")
print(f"Hôm nay  : ~{items_today:,} bài ({batches_today} batches x {BATCH_SIZE})")
print(f" ETA      : ~{eta_minutes:.0f} phút ({eta_minutes/60:.1f} giờ)")

# %%
# ── VÒNG LẶP DỊCH ─────────────────────────────────────────────────────────────
request_count = 0
success_count = 0
fail_count    = 0
start_time    = time.time()
last_report   = 0   # in summary mỗi 10 batch

print(f"Bắt đầu dịch lúc {datetime.now().strftime('%H:%M:%S')}\n")

for i in range(processed, len(dataset), BATCH_SIZE):
    batch      = dataset[i : i + BATCH_SIZE]
    batch_num  = (i - processed) // BATCH_SIZE + 1

    # Dừng nếu đã dùng hết RPD hôm nay
    if request_count >= RPD_LIMIT:
        print(f"\nĐã đạt {RPD_LIMIT} requests/ngày. Dừng lại, chạy lại vào ngày mai.")
        break

    translated = translate_batch(batch)
    request_count += 1

    if translated and len(translated) == len(batch):
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for j, t in enumerate(translated):
                f.write(json.dumps({
                    "question"    : t["question"],
                    "answer"      : t["answer"],
                    "ground_truth": batch[j]["ground_truth"]
                }, ensure_ascii=False) + '\n')
        success_count += 1
        total_done = processed + success_count * BATCH_SIZE
        pct = total_done / len(dataset) * 100

        # In tóm tắt mỗi 10 batch để tránh truncate
        if batch_num - last_report >= 1 or batch_num == 1:
            elapsed = (time.time() - start_time) / 60
            remaining_req = RPD_LIMIT - request_count
            print(f"[Batch {batch_num:>4}] ✅ {total_done:,}/{len(dataset):,} bài ({pct:.1f}%) "
                  f"| RPD còn: {remaining_req} | Thời gian: {elapsed:.1f} phút")
            last_report = batch_num
    else:
        fail_count += 1
        print(f"[Batch {batch_num:>4}] ❌ Lỗi! Dừng để tránh lãng phí quota.")
        break

    # Delay để giữ đúng RPM
    if i + BATCH_SIZE < len(dataset) and request_count < RPD_LIMIT:
        time.sleep(MIN_DELAY)

# ── KẾT QUẢ ───────────────────────────────────────────────────────────────────
elapsed_total = (time.time() - start_time) / 60
total_done = processed + success_count * BATCH_SIZE
print(f"\n{'='*60}")
print(f"✅ Hoàn tất phiên dịch lúc {datetime.now().strftime('%H:%M:%S')}")
print(f"   Requests dùng  : {request_count} / {RPD_LIMIT}")
print(f"   Bài dịch được  : {success_count * BATCH_SIZE:,} bài mới")
print(f"   Tổng đã có     : {total_done:,} / {len(dataset):,} bài")
print(f"   Thời gian chạy : {elapsed_total:.1f} phút")
if fail_count:
    print(f"   ⚠️  Batch lỗi   : {fail_count}")
print(f"{'='*60}")
