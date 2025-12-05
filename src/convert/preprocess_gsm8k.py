# %% [markdown]
# # Preprocess GSM8K
# Trích xuất đáp án sau `####` từ trường `answer` và lưu vào trường `ground_truth`.
# Run after src/convert/download_datasets.py's download_gsm8k(), or against
# data/generate/gsm8k/{train,test}.json if you're using that intermediate stage.

# %%
import json
import re
from pathlib import Path

# Assumes this script is run from its own directory (src/convert/)
BASE_DIR = Path("..") / ".." / "data" / "generate" / "gsm8k"

def extract_ground_truth(answer: str) -> str:
    """Lấy phần số sau '####' trong trường answer."""
    match = re.search(r'####\s*(.+)', answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""

for split in ["train", "test"]:
    path = BASE_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["ground_truth"] = extract_ground_truth(item.get("answer", ""))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ {split}.json — {len(data)} mẫu đã cập nhật")

# %%
# Kiểm tra kết quả
with open(BASE_DIR / "train.json", "r", encoding="utf-8") as f:
    train = json.load(f)

print(f"Tổng train: {len(train)} mẫu\n")
print("=" * 60)
for item in train[:5]:
    print(f"Q : {item['question'][:80]}...")
    print(f"GT: {item['ground_truth']}")
    print("-" * 60)

# %%
# Thống kê: bao nhiêu mẫu có ground_truth rỗng
for split in ["train", "test"]:
    with open(BASE_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    empty = sum(1 for d in data if not d.get("ground_truth"))
    print(f"{split}: {len(data)} mẫu | ground_truth rỗng: {empty}")
