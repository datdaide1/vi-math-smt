# %%
import json, os, time
from datetime import datetime
from openai import OpenAI
import re

# ── CẤU HÌNH ──────────────────────────────────────────────────────────────
# Set BEEKNOEE_API_KEY in your environment (or a .env file loaded before this cell)
API_KEY    = os.environ.get("BEEKNOEE_API_KEY", "")
BASE_URL   = "http://127.0.0.1:20128/v1"
MODEL_NAME = "beeknoee/gemini-2.5-flash-lite"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Assumes this script is run from its own directory (src/translate/)
BASE_DIR = os.path.join("..", "..", "..", "data", "generate", "MATH", "train")
OUTPUT_ROOT = os.path.join("..", "..", "..", "data", "translation", "MATH", "train")

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

# ── RATE LIMIT & BATCH ─────────────────────────────────────────────────────
BATCH_SIZE = 8       # số bài dịch mỗi batch
RPM_LIMIT  = 500      # requests per minute
RPD_LIMIT  = 10000      # requests per day
AVG_TOKENS_PER_SAMPLE = 2500
TPM = 1000000
MAX_RETRIES = 10

# tokens_per_batch = BATCH_SIZE * AVG_TOKENS_PER_SAMPLE
# MIN_DELAY = max(60 / RPM_LIMIT + 0.5, tokens_per_batch / TPM * 60)
MIN_DELAY = 0.3

# ── TRANSLATE BATCH ─────────────────────────────────────────────────────────
def translate_batch(batch_data, max_retries=MAX_RETRIES):
    payload = [{"problem": x["problem"], "solution": x["solution"]} for x in batch_data]

    prompt = f"""Bạn là chuyên gia toán học và dịch thuật. Dịch mảng JSON này sang tiếng Việt.

⚠️ QUAN TRỌNG: Chỉ trả về **một mảng JSON**, KHÔNG thêm comment, text, hoặc ký tự nào khác.
Ví dụ: [{"{"} "problem": "...", "solution": "..." {"}"}]

YÊU CẦU:
1. Trả về đúng định dạng mảng JSON gốc với key "problem" và "solution".
2. GIỮ NGUYÊN HOÀN TOÀN các công thức LaTeX và khối code Asymptote [asy]...[/asy]. Không làm biến dạng bất kỳ dấu gạch chéo ngược (\\) nào của công thức gốc.
3. Giữ nguyên ký tự ngắt dòng (\\n).

Dữ liệu:
{json.dumps(payload, ensure_ascii=False)}
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Always respond with valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=20000
            )

            raw_content = response.choices[0].message.content.strip()

            # ── CLEAN RAW CONTENT (GIỮ NGUYÊN LOGIC CỦA BẠN) ─────────────────
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()

            try:
                return json.loads(raw_content)

            except json.JSONDecodeError:
                # 🔥 CLEAN BACKSLASH (PHẦN QUAN TRỌNG NHẤT)
                cleaned_text = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', r'\\\\', raw_content)
                cleaned_text = cleaned_text.replace(r'\begin', r'\\begin')
                cleaned_text = cleaned_text.replace(r'\bf', r'\\bf')
                cleaned_text = cleaned_text.replace(r'\frac', r'\\frac')

                return json.loads(cleaned_text)

        except Exception as e:
            print("\n" + "="*80)
            print(f"⚠️ Lỗi lần {attempt}/{max_retries}: {repr(e)}")
            try:
                print("RAW RESPONSE (2000 ký tự đầu):")
                print(raw_content[:2000])
            except:
                print("Không có raw_content")
            print("="*80 + "\n")

            wait = 2 ** attempt
            err_str = str(e).lower()

            if "429" in err_str or "rate_limit" in err_str:
                wait = (2 ** attempt) * 30
                print(f"⏳ Rate limit, đợi {wait}s...")
            else:
                print(f"⏳ Đợi {wait}s...")

            time.sleep(wait)

    print("❌ Hết lượt retry, dừng pipeline.")
    raise RuntimeError("Batch dịch bị lỗi")

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────
total_requests = 0
start_global = time.time()

print(f"Bắt đầu dịch lúc {datetime.now().strftime('%H:%M:%S')}\n")

for file_name in FILE_LIST:
    print(f"\n{'='*60}")
    print(f"FILE: {file_name}")
    print(f"{'='*60}")

    output_file = os.path.join(OUTPUT_ROOT, os.path.basename(file_name).replace(".json", "_vi.jsonl"))

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file → skip")
        continue

    processed = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed = sum(1 for _ in f if _.strip())
        print(f"✅ Checkpoint: đã dịch {processed}/{len(dataset)} bài")
    else:
        print("🚀 Bắt đầu dịch mới")

    success_count = 0
    fail_count = 0

    for i in range(processed, len(dataset), BATCH_SIZE):
        if total_requests >= RPD_LIMIT:
            print("\n🛑 Hết quota ngày → dừng toàn bộ")
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
                        "level": batch[j].get("level", ""),
                        "type": batch[j].get("type", ""),
                        "solution": t.get("solution", batch[j]["solution"]),
                        "subject": batch[j].get("subject", "")
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            success_count += 1
            total_done = processed + success_count * BATCH_SIZE
            pct = total_done / len(dataset) * 100
            print(f"[{file_name}][Batch {batch_num:>3}] ✅ {total_done:,}/{len(dataset):,} ({pct:.1f}%) "
                  f"| RPD còn: {RPD_LIMIT - total_requests}")

        else:
            fail_count += 1
            print(f"[Batch {batch_num}] ❌ lỗi → dừng file")
            break

        if i + BATCH_SIZE < len(dataset) and total_requests < RPD_LIMIT:
            time.sleep(MIN_DELAY)

    print(f"\n✔ Done file {file_name}")
    print(f"   Success batch: {success_count}")
    if fail_count:
        print(f"   ⚠️ Fail batch : {fail_count}")

elapsed = (time.time() - start_global) / 60
print(f"\n{'='*60}")
print(f"🎯 HOÀN TẤT")
print(f"Requests dùng : {total_requests}/{RPD_LIMIT}")
print(f"Thời gian     : {elapsed:.1f} phút")
print(f"{'='*60}")
