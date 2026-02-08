# %%
import json, time, os, math
import google.generativeai as genai
from datetime import datetime

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME  = "gemini-2.5-flash"

# Assumes this script is run from its own directory (src/translate/)
BASE_DIR = os.path.join("..", "..", "..", "data", "generate", "MATH", "train")

FILE_LIST = [
    os.path.join(BASE_DIR, f) for f in [
        "algebra.json",
        "counting_and_probability.json",
        "geometry.json",
        "intermediate_algebra.json",
        "number_theory.json",
        "prealgebra.json",
        "precalculus.json"
    ]
]

BATCH_SIZE  = 20                         # 20 bài/request
RPM_LIMIT   = 5                         # Requests Per Minute (free tier)
RPD_LIMIT   = 20                        # Requests Per Day (free tier)
MIN_DELAY   = 60 / RPM_LIMIT + 1        # Delay tối thiểu giữa các request (giây)

# ───────────────────────────────────────────────────────────────────────────

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

print(f"Model      : {MODEL_NAME}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Delay      : {MIN_DELAY:.0f}s")
print(f"RPD limit  : {RPD_LIMIT} req/day → {RPD_LIMIT * BATCH_SIZE:,} items/day")


# %%
import json
import re
import time

def translate_batch(batch_data):
    payload = [{"problem": item["problem"], "solution": item["solution"]} for item in batch_data]
        
    prompt = f"""
    Bạn là chuyên gia toán học và dịch thuật. Hãy dịch mảng JSON này sang tiếng Việt.
    
    YÊU CẦU:
    1. Trả về đúng mảng JSON với 2 key: "problem" và "solution".
    2. GIỮ NGUYÊN HOÀN TOÀN các công thức LaTeX và khối code Asymptote [asy]...[/asy]. Không làm biến dạng bất kỳ dấu gạch chéo ngược (\\) nào của công thức gốc.
    
    Dữ liệu:
    {json.dumps(payload, ensure_ascii=False)}
    """
    
    MAX_RETRIES = 3 
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text
            
            try:
                # Trả việc xử lý JSON lại cho Python và Gemini tự lo với nhau
                return json.loads(raw_text)
            except json.JSONDecodeError:
                # TẤM LƯỚI BẢO VỆ: Chỉ kích hoạt khi model vô tình làm hỏng JSON
                cleaned_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', raw_text)
                cleaned_text = cleaned_text.replace(r'\begin', r'\\begin')
                cleaned_text = cleaned_text.replace(r'\bf', r'\\bf')
                cleaned_text = cleaned_text.replace(r'\frac', r'\\frac')
                
                return json.loads(cleaned_text)
                
        except Exception as e:
            print(f"    ⚠️ Lỗi API/JSON ở lần thử {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                return None

print("✅ Ready translate_batch()\n")

# %%
# # ── HÀM DỊCH ────────────────────────────────────────────────────────────────

# import json
# import re
# import time

# def clean_json_string(raw_text):
#     """
#     Hàm dọn dẹp các lỗi escape \ do model sinh ra trong LaTeX
#     trước khi đưa vào json.loads()
#     """
#     # Thay thế các dấu \ đơn (không hợp lệ trong JSON) thành \\
#     # Biểu thức Regex này tìm các dấu \ không đứng trước các ký tự escape hợp lệ của JSON (", \, /, b, f, n, r, t)
#     cleaned_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', raw_text)
    
#     # Xử lý riêng một số case LaTeX hay bị trùng với ký tự JSON hợp lệ
#     # Ví dụ: \begin, \bf, \frac (chữ f và b)
#     cleaned_text = cleaned_text.replace(r'\begin', r'\\begin')
#     cleaned_text = cleaned_text.replace(r'\bf', r'\\bf')
#     cleaned_text = cleaned_text.replace(r'\frac', r'\\frac')
    
#     return cleaned_text

# def translate_batch(batch_data):
#     # Chỉ bóc tách phần cần dịch
#     payload = [{"problem": item["problem"], "solution": item["solution"]} for item in batch_data]
        
#     prompt = f"""
#     Bạn là chuyên gia toán học và dịch thuật. Hãy dịch mảng JSON này sang tiếng Việt.
    
#     YÊU CẦU TỐI QUAN TRỌNG VỀ ĐỊNH DẠNG (FORMATTING):
#     1. Trả về đúng mảng JSON với 2 key: "problem" và "solution".
#     2. GIỮ NGUYÊN HOÀN TOÀN khối code Asymptote [asy]...[/asy].
    
    
#     Dữ liệu:
#     {json.dumps(payload, ensure_ascii=False)}
#     """

# # 3. CHÚ Ý LATEX: Trong kết quả JSON trả về, BẮT BUỘC phải escape tất cả các dấu backslash (\\) của công thức LaTeX thành hai dấu backslash (\\\\). 
# #        - Ví dụ SAI: "\\sqrt{{2}}" hoặc "\\begin{{align*}}"
# #        - Ví dụ ĐÚNG: "\\\\sqrt{{2}}" hoặc "\\\\begin{{align*}}"
# #        - Tuyệt đối không để sót bất kỳ dấu \\ đơn lẻ nào.

#     MAX_RETRIES = 3 # Thử lại tối đa 3 lần nếu parse JSON thất bại
    
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = model.generate_content(prompt)
#             raw_text = response.text
            
#             # Cố gắng parse thẳng xem model có làm chuẩn không
#             try:
#                 return json.loads(raw_text)
#             except json.JSONDecodeError:
#                 # Nếu văng lỗi Invalid \escape, gọi hàm dọn dẹp và thử parse lại
#                 cleaned_text = clean_json_string(raw_text)
#                 return json.loads(cleaned_text)
                
#         except Exception as e:
#             print(f"    ⚠️ Lỗi API/JSON ở lần thử {attempt + 1}/{MAX_RETRIES}: {e}")
#             if attempt < MAX_RETRIES - 1:
#                 print("    -> Đang thử lại trong 3 giây...")
#                 time.sleep(3)
#             else:
#                 print("    ❌ Đã thử 3 lần nhưng vẫn thất bại. Bỏ qua lô này.")
#                 return None


# print("✅ Ready translate_batch()\n")



# %%
# ── MAIN PIPELINE ───────────────────────────────────────────────────────────
OUTPUT_ROOT = os.path.join("..", "..", "..", "data", "translation", "MATH", "train")

total_requests = 0
start_global   = time.time()

print(f"Bắt đầu lúc {datetime.now().strftime('%H:%M:%S')}\n")

for file_name in FILE_LIST:

    print(f"\n{'='*60}")
    print(f"FILE: {file_name}")
    print(f"{'='*60}")

    input_file  = file_name
    output_file = os.path.join(
        OUTPUT_ROOT,
        os.path.basename(file_name).replace(".json", "_vi.jsonl")
    )

    # load data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file → skip")
        continue

    # checkpoint
    processed = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed = sum(1 for _ in f if _.strip())
        print(f"Checkpoint: {processed}/{len(dataset)}")

    remaining = len(dataset) - processed
    batches_needed = math.ceil(remaining / BATCH_SIZE)

    # limit theo ngày
    batches_today = min(batches_needed, RPD_LIMIT - total_requests)
    items_today   = batches_today * BATCH_SIZE

    eta_minutes = batches_today * MIN_DELAY / 60

    print(f"Total     : {len(dataset):,}")
    print(f"Remaining : {remaining:,}")
    print(f"Today     : ~{items_today:,} items ({batches_today} batches)")
    print(f"ETA       : ~{eta_minutes:.1f} phút\n")

    success_count = 0
    fail_count    = 0

    # loop batch
    for i in range(processed, len(dataset), BATCH_SIZE):

        if total_requests >= RPD_LIMIT:
            print("\n🚫 Hết quota ngày → dừng toàn bộ")
            break

        batch = dataset[i:i+BATCH_SIZE]
        batch_num = (i - processed) // BATCH_SIZE + 1

        translated = translate_batch(batch)
        total_requests += 1

        if translated and len(translated) == len(batch):

            with open(output_file, 'a', encoding='utf-8') as f:
                for j, t in enumerate(translated):
                    item = {
                        "problem": t.get("problem", batch[j]["problem"]),
                        "level": batch[j]["level"],
                        "type": batch[j]["type"],
                        "solution": t.get("solution", batch[j]["solution"]),
                        "subject": batch[j]["subject"]
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            success_count += 1

            done = processed + success_count * BATCH_SIZE
            pct  = done / len(dataset) * 100

            # chỉ print mỗi 10 batch hoặc batch cuối
            # if batch_num % 10 == 0 or i + BATCH_SIZE >= len(dataset):
            print(f"[{file_name}][Batch {batch_num:>3}] ✅ "
                f"{done:,}/{len(dataset):,} ({pct:.1f}%) "
                f"| RPD còn: {RPD_LIMIT - total_requests}")

        else:
            fail_count += 1
            print(f"[Batch {batch_num}] ❌ lỗi → dừng file")

            print("\n===== DEBUG INFO =====")
            
            print("➡️ Input batch:")
            for idx, item in enumerate(batch):
                print(f"[{idx}] problem length:", len(item.get("problem", "")))
            
            print("\n➡️ Translated output:")
            print(translated)

            # check chi tiết
            if translated is None:
                print("⚠️ translated = None")
            elif not isinstance(translated, list):
                print(f"⚠️ translated không phải list mà là {type(translated)}")
            else:
                print(f"⚠️ len(translated) = {len(translated)} | len(batch) = {len(batch)}")

            print("======================\n")

            break

        # delay
        if i + BATCH_SIZE < len(dataset) and total_requests < RPD_LIMIT:
            time.sleep(MIN_DELAY)

    print(f"\n✔ Done file {file_name}")
    print(f"   Success batch: {success_count}")
    if fail_count:
        print(f"   ⚠️ Fail batch : {fail_count}")


# ── SUMMARY ──────────────────────────────────────────────────────────────────
elapsed = (time.time() - start_global) / 60

print(f"\n{'='*60}")
print(f"🎯 HOÀN TẤT")
print(f"Requests dùng : {total_requests}/{RPD_LIMIT}")
print(f"Thời gian     : {elapsed:.1f} phút")
print(f"{'='*60}")
