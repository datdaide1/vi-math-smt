# %%
import os
import json
from pathlib import Path

# One-time migration script — DATA_DIR pointed at an external MATH mirror
# outside this repo. Prefer src/convert/download_datasets.py for a fresh
# download instead; this is kept for reference only.
DATA_DIR   = Path("path/to/external/MATH/train")  # <-- set to your own MATH source
OUTPUT_DIR = Path(os.path.join("..", "..", "data", "generate", "MATH", "train"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lấy danh sách 7 folder dạng toán
folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
print(f"Tìm thấy {len(folders)} folder dạng toán:")
for f in folders:
    print(f"  - {f.name}")

# %%
# ============================================================
# HÀM ĐỌC TẤT CẢ FILE JSON TRONG 1 FOLDER
# ============================================================
def load_json_files(folder: Path) -> list[dict]:
    """Đọc tất cả file .json trong folder, thêm trường 'source_file' và 'subject'."""
    records = []
    json_files = sorted(folder.glob("*.json"), key=lambda p: p.stem)
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Bổ sung metadata
            data["subject"]     = folder.name          # tên dạng toán
            # data["source_file"] = fp.name              # tên file gốc (vd: 1.json)
            records.append(data)
        except Exception as e:
            print(f"  [WARN] Lỗi đọc {fp}: {e}")
    return records

# %%
# ============================================================
# XUẤT TỪNG FOLDER → 1 FILE JSON RIÊNG
# ============================================================
summary = {}

for folder in folders:
    records = load_json_files(folder)

    # Lưu file riêng cho từng dạng toán
    out_path = OUTPUT_DIR / f"{folder.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary[folder.name] = len(records)
    print(f"✅  {folder.name:35s} → {len(records):5d} bài  →  {out_path.name}")

total = sum(summary.values())
print(f"\nTổng cộng: {total} bài toán từ {len(folders)} dạng toán.")


# %%
# ============================================================
# THỐNG KÊ TỔNG HỢP
# ============================================================
print("=" * 55)
print("THỐNG KÊ DỮ LIỆU MATH TEST")
print("=" * 55)
print(f"\n{'Dạng toán':<35s} {'Số bài':>8}")
print("-" * 45)
for subject, count in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"{subject:<35s} {count:>8,d}")
print("-" * 45)
print(f"{'TỔNG':<35s} {sum(summary.values()):>8,d}")
