# %% [markdown]
# ## MATH – Fix Issues in VN Train Files
# 
# **Vấn đề 1**: Nhiều sample chưa được dịch hoặc dịch thiếu (đề bài tiếng Việt nhưng lời giải tiếng Anh).  
# **Vấn đề 2**: Lỗi ký tự `\x0c` (form feed) và `\x08` (backspace) làm hỏng LaTeX.
# 
# **Giải pháp**:
# - Cell 2: Fix `\x0c` -> `f` và `\x08` -> `b` trong tất cả file.
# - Cell 3: Re-translate các bài chưa dịch bằng cách DỊCH TỪNG TRƯỜNG VĂN BẢN (KHÔNG DÙNG JSON) để tránh lỗi format.
# - Cell 4: Kiểm tra lại toàn bộ.

# %%
import json, os, re, time
from datetime import datetime

# Assumes this script is run from its own directory (src/translate/)
VN_DIR  = os.path.join("..", "..", "..", "data", "translation", "MATH", "train")
ENG_DIR = os.path.join("..", "..", "..", "data", "generate", "MATH", "train")

SUBJECTS = [
    ("algebra",                 "algebra_vi.jsonl",                 "algebra.json"),
    ("counting_and_probability","counting_and_probability_vi.jsonl","counting_and_probability.json"),
    ("geometry",                "geometry_vi.jsonl",                "geometry.json"),
    ("intermediate_algebra",    "intermediate_algebra_vi.jsonl",    "intermediate_algebra.json"),
    ("number_theory",           "number_theory_vi.jsonl",           "number_theory.json"),
    ("prealgebra",              "prealgebra_vi.jsonl",              "prealgebra.json"),
    ("precalculus",             "precalculus_vi.jsonl",             "precalculus.json"),
]

VI_RE = re.compile(
    r"[àáảãạăắặẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẶẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)
# Mở rộng ENG_WORDS_RE để bắt được cả các lệnh toán học phổ biến
ENG_WORDS_RE = re.compile(
    r"\b(the|is|are|if|find|let|what|how|given|show|prove|compute|evaluate|determine|expand|simplify|solve|calculate|express|write)\b",
    re.IGNORECASE
)

def extract_math_ground_truth(solution_text):
    if not solution_text or not isinstance(solution_text, str):
        return "NOT_FOUND"
    start_idx = solution_text.rfind("\\boxed{")
    if start_idx != -1:
        content_start = start_idx + 7
        open_braces = 1
        for i in range(content_start, len(solution_text)):
            if solution_text[i] == '{':
                open_braces += 1
            elif solution_text[i] == '}':
                open_braces -= 1
            if open_braces == 0:
                return solution_text[content_start:i].strip()
        return solution_text[content_start:].strip().rstrip('.$')
    match = re.search(r"\\boxed\s+([^{}\s\$\.\,]+)", solution_text[solution_text.rfind("\\boxed"):] if "\\boxed" in solution_text else "")
    if match: return match.group(1).strip()
    return "NOT_FOUND"

print("✅ Setup hoàn tất với logic ENG_WORDS_RE mạnh mẽ, bỏ các đếm ký tự dễ nhầm lẫn.")

# %% [markdown]
# ## Cell 2 – Vệ sinh dữ liệu (Đã chạy xong, có thể bỏ qua)
# 
# Dùng để sửa \x0c and \x08.

# %%
def sanitize_string(s):
    if not isinstance(s, str): return s
    return s.replace("\x0c", "f").replace("\x08", "b")

# print("✅ Đã vệ sinh dữ liệu.")

# %% [markdown]
# ## Cell 3 – Re-translate (KHÔNG DÙNG JSON ĐỂ TRÁNH LỖI)
# 
# Quét những câu còn lại và dịch thẳng từng dòng chữ bằng text thuần tuý.

# %%
from openai import OpenAI

# Set BEEKNOEE_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY    = os.environ.get("BEEKNOEE_API_KEY", "")
BASE_URL   = "http://127.0.0.1:20128/v1"
MODEL_NAME = "beeknoee/gemini-2.5-flash-lite"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
MIN_DELAY   = 1.0

def translate_single_text(text):
    if not text: return text
    prompt = f"""Bạn là chuyên gia toán học. Dịch đoạn văn bản tiếng Anh sau sang tiếng Việt.
⚠️ CHÚ Ý QUAN TRỌNG:
- Giữ nguyên các thẻ định dạng, mã như [asy]...[/asy].
- Giữ nguyên CÁC CÔNG THỨC TOÁN HỌC bằng LaTeX (bắt đầu bằng $ hoặc \\). KHÔNG sửa hoặc làm mất bất kỳ dấu gạch chéo ngược (\\) nào!
- CHỈ trả về đoạn văn bản đã dịch, không thêm chú thích.

Văn bản cần dịch:
{text}"""
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=10000
            )
            res = response.choices[0].message.content.strip()
            if res: return res
        except: time.sleep(2)
    return text

total_fixed = 0

for subject, vn_file, eng_file in SUBJECTS:
    vn_path = os.path.join(VN_DIR, vn_file)
    eng_path = os.path.join(ENG_DIR, eng_file)
    vn_records = {}
    with open(vn_path, 'r', encoding='utf-8') as f:
        for line in f: 
            item = json.loads(line)
            vn_records[item['index']] = item
    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_data = json.load(f)
    eng_by_idx = {item['index']: item for item in eng_data}
    
    to_translate = []
    for idx, item in sorted(vn_records.items()):
        prob, sol = item.get("problem", ""), item.get("solution", "")
        is_p_un = not VI_RE.search(prob) and ENG_WORDS_RE.search(prob)
        is_s_un = not VI_RE.search(sol) and ENG_WORDS_RE.search(sol)
        if is_p_un or is_s_un:
            to_translate.append((idx, is_p_un, is_s_un))
            
    if not to_translate: print(f"{subject:<32}: OK ✅"); continue
    print(f"Processing {subject}: {len(to_translate)} items")
    
    for idx, is_p_un, is_s_un in to_translate:
        orig_eng = eng_by_idx[idx]
        if is_p_un:
            print(f"  -> Dịch Problem cho Index {idx}")
            vn_records[idx]['problem'] = translate_single_text(orig_eng['problem'])
            time.sleep(MIN_DELAY)
        if is_s_un:
            print(f"  -> Dịch Solution cho Index {idx}")
            vn_records[idx]['solution'] = translate_single_text(orig_eng['solution'])
            time.sleep(MIN_DELAY)
            
        vn_records[idx]['ground_truth'] = extract_math_ground_truth(vn_records[idx]['solution'])
        total_fixed += 1
        
    with open(vn_path, 'w', encoding='utf-8') as f:
        for idx in sorted(vn_records.keys()): f.write(json.dumps(vn_records[idx], ensure_ascii=False) + '\n')
print(f"\n🎯 Xong! Đã sửa triệt để {total_fixed} câu.")

# %% [markdown]
# ## Cell 4 – Kiểm tra cuối cùng

# %%
print(f"{'Subject':<32} {'Total':>7} {'Untrans':>8} {'NOT_FOUND':>11}")
print("-" * 60)
for subject, vn_file, _ in SUBJECTS:
    path = os.path.join(VN_DIR, vn_file)
    total, untrans, nf = 0, 0, 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total += 1
            p, s = item.get('problem',''), item.get('solution','')
            is_p_un = not VI_RE.search(p) and ENG_WORDS_RE.search(p)
            is_s_un = not VI_RE.search(s) and ENG_WORDS_RE.search(s)
            if is_p_un or is_s_un: untrans += 1
            if item.get('ground_truth') == 'NOT_FOUND': nf += 1
    print(f"{subject:<32} {total:>7,} {untrans:>8,} {nf:>11,}")
