# %%
def extract_math_ground_truth(solution_text):
    """
    Trích xuất nội dung bên trong thẻ \\boxed{...} cuối cùng.
    Hỗ trợ dấu ngoặc nhọn lồng nhau (phân số, căn thức, v.v.).
    """
    start_idx = solution_text.rfind("\\boxed{")
    if start_idx == -1:
        return "NOT_FOUND"

    content_start = start_idx + len("\\boxed{")
    open_braces = 1

    for i in range(content_start, len(solution_text)):
        if solution_text[i] == '{':
            open_braces += 1
        elif solution_text[i] == '}':
            open_braces -= 1
        if open_braces == 0:
            return solution_text[content_start:i].strip()

    return "NOT_FOUND"

# %% [markdown]
# ## MATH – Extract Ground Truth from `\\boxed{}`
# 
# Script này:
# 1. Đọc 7 file JSONL (tiếng Việt) và 7 file JSON (tiếng Anh)
# 2. Extract `ground_truth` từ `solution` – nội dung trong `\\boxed{...}` cuối cùng
# 3. Ghi trường `ground_truth` trực tiếp vào **file gốc**
# 4. So sánh ground truth của 2 bộ để kiểm tra độ khớp

# %%
def extract_math_ground_truth(solution_text):
    """
    Trích xuất nội dung bên trong thẻ \\boxed{...} cuối cùng.
    Hỗ trợ dấu ngoặc nhọn lồng nhau (phân số, căn thức, v.v.).
    """
    start_idx = solution_text.rfind("\\boxed{")
    if start_idx == -1:
        return "NOT_FOUND"

    content_start = start_idx + len("\\boxed{")
    open_braces = 1

    for i in range(content_start, len(solution_text)):
        if solution_text[i] == '{':
            open_braces += 1
        elif solution_text[i] == '}':
            open_braces -= 1
        if open_braces == 0:
            return solution_text[content_start:i].strip()

    return "NOT_FOUND"

# %% [markdown]
# ## Cell 2 – Thêm `ground_truth` vào TẤT CẢ file tiếng Việt (JSONL) và tiếng Anh (JSON)

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


def parse_line_objects(line):
    """Parse 1 hoặc nhiều JSON object bị ghép dính trên cùng 1 dòng."""
    decoder = json.JSONDecoder()
    items = []
    idx = 0
    while idx < len(line):
        while idx < len(line) and line[idx] in ' \t\r\n':
            idx += 1
        if idx >= len(line):
            break
        try:
            obj, end = decoder.raw_decode(line, idx)
            items.append(obj)
            idx = end
        except json.JSONDecodeError:
            break
    return items


def process_jsonl(filepath):
    """Đọc JSONL, thêm ground_truth vào mỗi dòng, ghi lại file gốc."""
    records = []
    not_found = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for item in parse_line_objects(line):
                gt = extract_math_ground_truth(item.get("solution", ""))
                if gt == "NOT_FOUND":
                    not_found += 1
                item["ground_truth"] = gt
                records.append(item)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(records), not_found


def process_json(filepath):
    """Đọc JSON (list), thêm ground_truth vào mỗi phần tử, ghi lại file gốc."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    not_found = 0
    for item in data:
        gt = extract_math_ground_truth(item.get("solution", ""))
        if gt == "NOT_FOUND":
            not_found += 1
        item["ground_truth"] = gt
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return len(data), not_found


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"{'Subject':<32} {'VN records':>11} {'VN NOT_FOUND':>13} {'ENG records':>12} {'ENG NOT_FOUND':>14}")
print("-" * 85)

for subject, vn_file, eng_file in SUBJECTS:
    vn_path  = os.path.join(VN_DIR,  vn_file)
    eng_path = os.path.join(ENG_DIR, eng_file)
    vn_total,  vn_nf  = process_jsonl(vn_path)
    eng_total, eng_nf = process_json(eng_path)
    print(f"{subject:<32} {vn_total:>11,} {vn_nf:>13,} {eng_total:>12,} {eng_nf:>14,}")

print("\n✅ Đã ghi ground_truth vào tất cả các file gốc.")

# %% [markdown]
# ## Cell 3 – So sánh ground truth tiếng Việt vs tiếng Anh

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

all_match      = True
grand_total    = 0
grand_mismatch = 0

print(f"{'Subject':<32} {'Records':>8} {'Match':>8} {'Mismatch':>9} {'Match %':>9}")
print("-" * 72)

for subject, vn_file, eng_file in SUBJECTS:
    vn_path  = os.path.join(VN_DIR,  vn_file)
    eng_path = os.path.join(ENG_DIR, eng_file)

    # Load VN ground truth
    vn_gts = []
    with open(vn_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                vn_gts.append(json.loads(line).get("ground_truth", "NOT_FOUND"))

    # Load ENG ground truth
    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_data = json.load(f)
    eng_gts = [item.get("ground_truth", "NOT_FOUND") for item in eng_data]

    n = min(len(vn_gts), len(eng_gts))
    mismatches = [(i, vn_gts[i], eng_gts[i]) for i in range(n) if vn_gts[i] != eng_gts[i]]

    len_diff       = len(vn_gts) - len(eng_gts)
    mismatches_count = len(mismatches) + abs(len_diff)
    matched        = n - len(mismatches)
    pct            = matched / max(n, 1) * 100
    grand_total    += n
    grand_mismatch += mismatches_count

    status = "✅" if mismatches_count == 0 else "❌"
    if mismatches_count > 0:
        all_match = False

    note = f"  ⚠️  VN={len(vn_gts)}, ENG={len(eng_gts)}" if len_diff != 0 else ""
    print(f"{subject:<32} {n:>8,} {matched:>8,} {mismatches_count:>9,} {pct:>8.2f}%  {status}{note}")

    # In tối đa 5 mismatch đầu để debug
    for idx, vgt, egt in mismatches[:5]:
        print(f"    [idx={idx}] VN='{vgt}'")
        print(f"             ENG='{egt}'")

print("-" * 72)
grand_matched = grand_total - grand_mismatch
grand_pct     = grand_matched / max(grand_total, 1) * 100
print(f"{'TỔNG CỘNG':<32} {grand_total:>8,} {grand_matched:>8,} {grand_mismatch:>9,} {grand_pct:>8.2f}%")
print()

if all_match:
    print("🎉 Ground truth của tất cả các file VN và ENG KHỚP 100%!")
else:
    print("⚠️  Có sự khác biệt giữa ground truth VN và ENG. Phân tích nguyên nhân:")
    print("   - Dịch trong \\boxed{}: các từ/cụm lặng xuất hiện trong đáp án bằng tiếng Việt")
    print("   - Lỗi \\frac không có khoảng cách: '\\\\frac' vs '\\frac' (escape thừa)")
    print("   - Số record lệch: precalculus VN=747, ENG=746 (1 dòng ghép bị tách)")
