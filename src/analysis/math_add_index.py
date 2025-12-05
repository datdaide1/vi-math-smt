# %% [markdown]
# ## MATH – Gắn `index` cho tất cả file VN & ENG
# 
# Script này:
# 1. Thêm trường **`index`** (số nguyên, bắt đầu từ **0**) vào **đầu** mỗi record
# 2. Ghi trực tiếp vào **file gốc** (VN: JSONL, ENG: JSON)
# 3. In bảng tổng hợp kết quả và preview 2 record đầu mỗi file

# %%
import json
import os

VN_DIR  = os.path.join("..", "..", "data", "translation", "MATH", "train")
ENG_DIR = os.path.join("..", "..", "data", "generate", "MATH", "train")

SUBJECTS = [
    ("algebra",                 "algebra_vi.jsonl",                 "algebra.json"),
    ("counting_and_probability","counting_and_probability_vi.jsonl","counting_and_probability.json"),
    ("geometry",                "geometry_vi.jsonl",                "geometry.json"),
    ("intermediate_algebra",    "intermediate_algebra_vi.jsonl",    "intermediate_algebra.json"),
    ("number_theory",           "number_theory_vi.jsonl",           "number_theory.json"),
    ("prealgebra",              "prealgebra_vi.jsonl",              "prealgebra.json"),
    ("precalculus",             "precalculus_vi.jsonl",             "precalculus.json"),
]


def add_index_jsonl(filepath):
    """Đọc JSONL, chèn 'index' vào đầu mỗi record, ghi lại file gốc."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with open(filepath, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(records):
            # Chèn 'index' vào đầu dict
            new_item = {"index": idx, **item}
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    return len(records)


def add_index_json(filepath):
    """Đọc JSON list, chèn 'index' vào đầu mỗi phần tử, ghi lại file gốc."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = [{"index": idx, **item} for idx, item in enumerate(data)]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    return len(new_data)


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"{'Subject':<32} {'VN records':>12} {'ENG records':>12}")
print("-" * 60)

for subject, vn_file, eng_file in SUBJECTS:
    vn_path  = os.path.join(VN_DIR,  vn_file)
    eng_path = os.path.join(ENG_DIR, eng_file)

    vn_total  = add_index_jsonl(vn_path)
    eng_total = add_index_json(eng_path)

    print(f"{subject:<32} {vn_total:>12,} {eng_total:>12,}")

print("\n✅ Đã gắn 'index' vào tất cả các file gốc.")

# %% [markdown]
# ## Cell 2 – Preview: kiểm tra record đầu mỗi file

# %%
import json
import os

VN_DIR  = os.path.join("..", "..", "data", "translation", "MATH", "train")
ENG_DIR = os.path.join("..", "..", "data", "generate", "MATH", "train")

SUBJECTS = [
    ("algebra",                 "algebra_vi.jsonl",                 "algebra.json"),
    ("counting_and_probability","counting_and_probability_vi.jsonl","counting_and_probability.json"),
    ("geometry",                "geometry_vi.jsonl",                "geometry.json"),
    ("intermediate_algebra",    "intermediate_algebra_vi.jsonl",    "intermediate_algebra.json"),
    ("number_theory",           "number_theory_vi.jsonl",           "number_theory.json"),
    ("prealgebra",              "prealgebra_vi.jsonl",              "prealgebra.json"),
    ("precalculus",             "precalculus_vi.jsonl",             "precalculus.json"),
]

for subject, vn_file, eng_file in SUBJECTS:
    print(f"\n{'='*60}")
    print(f"  {subject.upper()}")
    print(f"{'='*60}")

    # VN: đọc record đầu
    vn_path = os.path.join(VN_DIR, vn_file)
    with open(vn_path, 'r', encoding='utf-8') as f:
        vn_first = json.loads(f.readline().strip())
    print(f"\n[VN] index={vn_first['index']} | keys={list(vn_first.keys())}")
    print(f"     ground_truth: {vn_first.get('ground_truth', 'N/A')}")

    # ENG: đọc record đầu
    eng_path = os.path.join(ENG_DIR, eng_file)
    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_first = json.load(f)[0]
    print(f"\n[ENG] index={eng_first['index']} | keys={list(eng_first.keys())}")
    print(f"      ground_truth: {eng_first.get('ground_truth', 'N/A')}")

    # Kiểm tra index tương ứng
    matched = vn_first['index'] == eng_first['index'] == 0
    print(f"\n  → Index đầu khớp (cùng = 0): {'✅' if matched else '❌'}")

print("\n\n✅ Preview hoàn tất.")
