# %% [markdown]
# # JSON / JSONL Converter & Index Tagger
# 
# Notebook này gồm 2 phần:
# 1. **Convert** file JSON array → JSONL (mỗi dòng 1 JSON object)
# 2. **Gắn index** vào file JSONL có sẵn (thêm trường `index` vào đầu mỗi record)

# %%
import json
import os
import shutil

# %% [markdown]
# ---
# ## Phần 1 – Convert JSON array → JSONL (có index)

# %%
# === Config Part 1 === (assumes this script is run from its own directory, src/convert/)
INPUT_JSON   = os.path.join("..", "..", "data", "generate", "gsm8k", "train.json")
OUTPUT_JSONL = os.path.join("..", "..", "data", "generate", "gsm8k", "train.jsonl")
INDEX_FIELD  = "index"
START_INDEX  = 0        # 0 = 0-based, 1 = 1-based

# %%
# Load JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} records from: {INPUT_JSON}")

# Ghi JSONL
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for i, record in enumerate(data):
        indexed_record = {INDEX_FIELD: i + START_INDEX, **record}
        f.write(json.dumps(indexed_record, ensure_ascii=False) + "\n")

print(f"[INFO] Written {len(data)} records → {OUTPUT_JSONL}")
print(f"[INFO] Index range: {START_INDEX} → {START_INDEX + len(data) - 1}")

# %% [markdown]
# ---
# ## Phần 2 – Gắn index vào file JSONL có sẵn
# 
# > Dùng cho file JSONL chưa có trường `index` (ví dụ: `gsm8k_train_vietnamese.jsonl`).
# > File gốc sẽ được backup trước khi ghi đè.

# %%
# === Config Part 2 ===
JSONL_FILE   = os.path.join("..", "..", "data", "translation", "gsm8k", "gsm8k_train_vietnamese.jsonl")
INDEX_FIELD  = "index"
START_INDEX  = 0        # 0 = 0-based, 1 = 1-based
BACKUP       = True     # True = tạo file backup .bak trước khi ghi đè

# %%
# Backup file gốc (nếu bật)
if BACKUP:
    backup_path = JSONL_FILE + ".bak"
    shutil.copy2(JSONL_FILE, backup_path)
    print(f"[INFO] Backup saved → {backup_path}")

# Đọc tất cả dòng
records = []
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"[INFO] Loaded {len(records)} records from: {JSONL_FILE}")

# Gắn index vào đầu mỗi record và ghi đè
with open(JSONL_FILE, "w", encoding="utf-8") as f:
    for i, record in enumerate(records):
        # Bỏ qua nếu đã có index rồi
        record.pop(INDEX_FIELD, None)
        indexed_record = {INDEX_FIELD: i + START_INDEX, **record}
        f.write(json.dumps(indexed_record, ensure_ascii=False) + "\n")

print(f"[INFO] Done! Indexed {len(records)} records → {JSONL_FILE}")
print(f"[INFO] Index range: {START_INDEX} → {START_INDEX + len(records) - 1}")

# %%
# Preview 3 dòng đầu
print("\n=== Preview (3 records dau) ===")
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for _ in range(3):
        line = f.readline().strip()
        if line:
            obj = json.loads(line)
            # Chỉ hiện index + 50 ký tự đầu của question
            print(f"  index={obj.get('index')}  question={str(obj.get('question',''))[:60]}...")
