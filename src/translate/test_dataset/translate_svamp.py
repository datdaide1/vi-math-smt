# %%
import json
import time
import google.generativeai as genai
import os

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────────

# Set GEMINI_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME  = "gemini-2.5-flash"           # Phiên bản Gemini sử dụng
BATCH_SIZE  = 20                        # Số lượng bài toán trong một lần gọi API
RPM_LIMIT   = 5                         # Request mỗi phút (Rate limit)
MIN_DELAY   = 60 / RPM_LIMIT + 1        # Độ trễ giữa các request
# Assumes this script is run from its own directory (src/translate/test_dataset/)
# INPUT_FILE comes from src/convert/download_datasets.py's output layout
INPUT_FILE  = os.path.join("..", "..", "..", "data", "raw", "svamp", "train.jsonl")
OUTPUT_FILE = os.path.join("..", "..", "..", "data", "translation_test_output", "svamp", "svamp-train-vi.jsonl")
# ──────────────────────────────────────────────────────────────────────────────

# Khởi tạo API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

print(f"Model      : {MODEL_NAME}")
print(f"Batch size : {BATCH_SIZE} bài/request")
print(f"Delay      : {MIN_DELAY:.0f}s giữa mỗi request")

def translate_batch(batch_data, retries=3):
    """Gửi batch lên Gemini để dịch SVAMP dataset sang tiếng Việt"""
    payload = []
    for item in batch_data:
        payload.append({
            "ID": item.get("ID", ""),
            "Body": item.get("Body", ""), 
            "Question": item.get("Question", ""), 
            "Answer": item.get("Answer", ""),
            "Type": item.get("Type", ""),
            "Equation": item.get("Equation", "")
        })
        
    prompt = f"""
    Bạn là một chuyên gia toán học và dịch thuật. Hãy dịch mảng JSON chứa các bài toán tiếng Anh này sang tiếng Việt.
    
    YÊU CẦU:
    1. Dịch chuẩn xác ngữ cảnh toán học trong các trường "Body" và "Question".
    2. Dịch phần đơn vị/danh từ trong trường "Answer" (ví dụ nếu Answer là "27" hoặc "27 (birds)" thì dịch lại đơn vị).
    3. GIỮ NGUYÊN tuyệt đối:
       - Trường "Type" (loại phép toán: Subtraction, Addition, v.v.)
       - Trường "Equation" (công thức toán học)
       - Trường "ID"
       - Tất cả các con số trong bài
    4. Trả về MỘT MẢNG JSON chứa đầy đủ các bài toán đã dịch, giữ đúng các key: ID, Body, Question, Answer, Type, Equation.
    
    Dữ liệu cần dịch:
    {json.dumps(payload, ensure_ascii=False)}
    """
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            translated_data = json.loads(response.text)
            
            if isinstance(translated_data, list) and len(translated_data) == len(batch_data):
                return translated_data
            else:
                print(f"⚠️ Cảnh báo: Batch trả về {len(translated_data)} bài, mong đợi {len(batch_data)}. Thử lại...")
                time.sleep(2)
                
        except Exception as e:
            print(f"⚠️ Lỗi ở lần thử {attempt + 1}/{retries}: {e}")
            time.sleep(MIN_DELAY)
            
    return None

# ── THỰC THI & TRACKING ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Đọc toàn bộ dữ liệu đầu vào
    full_dataset = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    full_dataset.append(json.loads(line.strip()))
    else:
        print(f"❌ Không tìm thấy file: {INPUT_FILE}")
        exit()

    total_items = len(full_dataset)
    print(f"Tổng số bài toán trong file gốc: {total_items}")

    # 2. Kiểm tra tracking (số bài đã dịch)
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed_count += 1
    
    print(f"Đã dịch thành công trước đó   : {processed_count} bài")

    # 3. Lọc bỏ các bài đã dịch
    if processed_count >= total_items:
        print("✅ Toàn bộ dữ liệu đã được dịch xong. Không cần chạy tiếp!")
        exit()
        
    dataset_to_process = full_dataset[processed_count:]
    print(f"Số bài toán CẦN DỊCH TIẾP     : {len(dataset_to_process)}")
    print("-" * 60)

    start_time = time.time()
    success_count = 0

    # 4. Mở file ở chế độ 'a' (Append) để ghi tiếp
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for i in range(0, len(dataset_to_process), BATCH_SIZE):
            batch = dataset_to_process[i:i+BATCH_SIZE]
            
            # Tính toán lại Index/Batch thực tế so với tổng file gốc để dễ theo dõi
            current_start_index = processed_count + i
            current_end_index = current_start_index + len(batch) - 1
            batch_num = (current_start_index // BATCH_SIZE) + 1
            
            print(f"Đang xử lý Batch {batch_num} (Index {current_start_index} -> {current_end_index})...")
            
            translated_batch = translate_batch(batch)
            
            if translated_batch:
                for item in translated_batch:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                # Force flush để đảm bảo dữ liệu ghi xuống ổ cứng ngay lập tức
                f_out.flush() 
                os.fsync(f_out.fileno())
                
                success_count += len(translated_batch)
                print(f"   => Xong Batch {batch_num}!")
            else:
                print(f"❌ Batch {batch_num} thất bại hoàn toàn. Dừng tiến trình để kiểm tra lỗi!")
                break # Dừng lại nếu lỗi liên tục để tránh lãng phí request
            
            # Quản lý Rate Limit
            if i + BATCH_SIZE < len(dataset_to_process):
                time.sleep(MIN_DELAY)

    # Tổng kết
    elapsed = (time.time() - start_time) / 60
    total_done_now = processed_count + success_count
    print("-" * 60)
    print(f"✅ Hoàn tất phiên chạy!")
    print(f"Thời gian chạy phiên này: {elapsed:.1f} phút")
    print(f"Bài dịch mới trong phiên: {success_count} bài")
    print(f"Tiến độ tổng cộng       : {total_done_now} / {total_items} bài ({(total_done_now/total_items)*100:.1f}%)")

# %%
# import os
# import json

# # ── KIỂM TRA TIẾN ĐỘ DỊCH SVAMP ───────────────────────────────────────────────
# INPUT_CHECK  = os.path.join("..", "..", "..", "data", "raw", "svamp", "test.jsonl")
# OUTPUT_CHECK = os.path.join("..", "..", "..", "data", "translation_test_output", "svamp", "svamp-test-vi.jsonl")

# check_dataset = []
# if os.path.exists(INPUT_CHECK):
#     with open(INPUT_CHECK, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 check_dataset.append(json.loads(line.strip()))
# else:
#     print(f"❌ Không tìm thấy file: {INPUT_CHECK}")
#     exit()

# total = len(check_dataset)
# print(f"📊 KIỂM TRA DỊCH SVAMP DATASET")
# print(f"{'='*60}")
# print(f"Tổng số bài toán trong file gốc: {total}")

# # 2. Kiểm tra tracking (số bài đã dịch)
# check_count = 0
# if os.path.exists(OUTPUT_CHECK):
#     with open(OUTPUT_CHECK, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 check_count += 1

# print(f"Đã dịch thành công trước đó   : {check_count} bài")

# # 3. Lọc bỏ các bài đã dịch
# if check_count >= total:
#     print("✅ Toàn bộ dữ liệu đã được dịch xong. Không cần chạy tiếp!")
#     print(f"{'='*60}")
#     exit()
    
# checking = check_dataset[check_count:]
# print(f"Số bài toán CẦN DỊCH TIẾP     : {len(checking)}")
# print(f"Tiến độ                       : {check_count}/{total} ({(check_count/total)*100:.1f}%)")
# print(f"{'='*60}")

# # Hiển thị vài ví dụ cần dịch
# print(f"\n📝 Ví dụ bài toán cần dịch:")
# for idx, item in enumerate(checking[:2], 1):
#     print(f"\n  Bài {idx}:")
#     print(f"    ID: {item.get('ID', 'N/A')}")
#     print(f"    Body: {item.get('Body', 'N/A')[:80]}...")
#     print(f"    Question: {item.get('Question', 'N/A')[:80]}...")
