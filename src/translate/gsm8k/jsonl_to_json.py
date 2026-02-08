# %%
import json
import os

# Assumes this script is run from its own directory (src/translate/)
INPUT_JSONL  = os.path.join("..", "..", "..", "data", "translation", "gsm8k", "gsm8k_test_vietnamese.jsonl")   # <-- đổi thành file JSONL của bạn
OUTPUT_JSON  = os.path.join("..", "..", "..", "data", "translation", "gsm8k", "gsm8k_test_vietnamese.json")    # <-- tên file JSON đầu ra

print(f"Input : {INPUT_JSONL}")
print(f"Output: {OUTPUT_JSON}")

# %%
# Đọc từng dòng JSONL và gộp thành list
records = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # bỏ qua dòng trống
            records.append(json.loads(line))

print(f"Đọc được {len(records)} bản ghi từ {INPUT_JSONL}")

# Ghi ra file JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Đã lưu thành công: {OUTPUT_JSON}")

# %%
# Kiểm tra nhanh kết quả
with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Tổng số bản ghi: {len(data)}")
print(f"Keys của bản ghi đầu tiên: {list(data[0].keys())}")
print()
print("--- Bản ghi đầu tiên ---")
print(json.dumps(data[0], ensure_ascii=False, indent=2))
