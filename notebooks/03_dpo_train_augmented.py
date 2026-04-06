"""
KAGGLE NOTEBOOK 3: Online DPO Training — Vi-MathLM-Aug
========================================================
Runtime: T4 GPU
Input: SFT-on-augmented adapter (from 02_sft_train_augmented.py) + sft_data.jsonl
       (prompts + ground truths are extracted directly from this file)
Output: Vi-MathLM-Aug (final DPO adapter)

Implements Online DPO (Bảng 3.2): at each step the model samples two
completions for a prompt, and a rule-based math judge picks the preferred
one — (1) mathematically correct answer wins, (2) if tied, \\boxed{} format
compliance wins, (3) if still tied, the shorter completion wins (anti
repetition-collapse).

Instructions:
1. Upload the adapter produced by 02_sft_train_augmented.py as a Kaggle
   Dataset (e.g. "model-aug-sft")
2. Upload sft_data.jsonl (same augmented set used for SFT) as a Kaggle Dataset
3. Enable GPU (T4), Internet ON
4. Paste this entire script and run
"""

# ====================== CELL 1: Install ======================
# !pip uninstall unsloth unsloth-zoo transformers trl -y
# !pip install --no-cache-dir "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-cache-dir --no-deps trl==0.12.1 peft accelerate bitsandbytes
# !pip install --no-cache-dir transformers==4.51.3

# ====================== CELL 2: Config =======================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, re
import matplotlib.pyplot as plt

from unsloth import FastLanguageModel
from trl.experimental.online_dpo import OnlineDPOTrainer, OnlineDPOConfig
from trl.experimental.judges import BasePairwiseJudge
from datasets import load_dataset, Dataset

# -- Paths (adjust Dataset names) --
SFT_ADAPTER_PATH = "/kaggle/input/model-aug-sft/model_aug_sft"
SFT_DATA_PATH = "/kaggle/input/datasets/datuni/aug-data-new/sft_data.jsonl"
OUTPUT_DIR = "/kaggle/working/model_aug_dpo"

# -- Hyperparameters (Bảng 3.2) --
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
MAX_NEW_TOKENS = 512
LEARNING_RATE = 5e-7
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 16
KL_BETA = 0.1

# ====================== CELL 3: Load Model ===================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    random_state=42,
)

# ====================== CELL 4: Rule-Based Math Judge =========
def check_format_compliance(text):
    if not isinstance(text, str):
        return False
    if "\\boxed" in text or "\\fbox" in text:
        return True
    indicators = [
        "đáp án", "đáp số", "kết quả là", "vậy ta có", "vậy ta được",
        "the answer", "final answer",
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


def clean_latex(text: str) -> str:
    if not text:
        return ""
    for token in ["\\text{", "\\dfrac{", "\\tfrac{", "\\frac{", "{", "}", "\\"]:
        text = text.replace(token, "")
    return text


def extract_numbers(s: str):
    temp = ""
    for char in s:
        if char.isdigit() or char in ".-/":
            temp += char
        else:
            temp += " "
    return [w for w in temp.split() if w.strip()]


def extract_answer(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    nums = extract_numbers(text)
    if nums:
        return nums[-1]
    return None


def verify_correctness(model_ans: str, ground_truth: str) -> bool:
    if not model_ans:
        return False

    m_clean = clean_latex(str(model_ans)).strip().lower()
    g_clean = clean_latex(str(ground_truth)).strip().lower()

    for char in " $,.()[]_":
        m_clean = m_clean.replace(char, "")
        g_clean = g_clean.replace(char, "")

    if m_clean == g_clean:
        return True

    m_nums = extract_numbers(m_clean)
    g_nums = extract_numbers(g_clean)

    if m_nums and g_nums:
        try:
            if "/" in m_nums[0] and "/" in g_nums[0]:
                m_parts = m_nums[0].split("/")
                g_parts = g_nums[0].split("/")
                if float(m_parts[0]) / float(m_parts[1]) == float(g_parts[0]) / float(g_parts[1]):
                    return True
            else:
                if float(m_nums[0]) == float(g_nums[0]):
                    return True
        except Exception:
            pass

    return False


class MathOnlineJudge(BasePairwiseJudge):
    """Ưu tiên: (1) đáp án đúng, (2) tuân thủ định dạng \\boxed{}, (3) lời giải ngắn hơn."""

    def __init__(self, problem_to_gt):
        super().__init__()
        self.problem_to_gt = problem_to_gt

    def normalize_key(self, text):
        return "".join(text.split()).lower()

    def judge(self, prompts, completions, shuffle_order=False):
        results = []
        for prompt, comps in zip(prompts, completions):
            clean_p = self.normalize_key(prompt)
            gt = self.problem_to_gt.get(clean_p, "")

            c0, c1 = comps
            ans0 = extract_answer(c0)
            ans1 = extract_answer(c1)

            ok0 = verify_correctness(ans0, gt)
            ok1 = verify_correctness(ans1, gt)

            if ok0 and not ok1:
                results.append(0)
            elif ok1 and not ok0:
                results.append(1)
            else:
                fmt0 = check_format_compliance(c0)
                fmt1 = check_format_compliance(c1)
                if fmt0 and not fmt1:
                    results.append(0)
                elif fmt1 and not fmt0:
                    results.append(1)
                else:
                    results.append(0 if len(c0) <= len(c1) else 1)
        return results

# ====================== CELL 5: Load Data ====================
print("Loading SFT dataset to extract prompts and ground truths dynamically...")
sft_dataset = load_dataset("json", data_files=SFT_DATA_PATH, split="train")

problem_to_gt = {}
train_records = []

for row in sft_dataset:
    text = row.get("text", "")
    if "[/INST]" not in text:
        continue
    parts = text.split("[/INST]")
    prompt = parts[0] + "[/INST]"
    solution = parts[1]

    gt = extract_answer(solution)
    clean_p = "".join(prompt.split()).lower()
    problem_to_gt[clean_p] = str(gt or "")

    train_records.append({"prompt": prompt})

train_dataset = Dataset.from_list(train_records)
train_dataset = train_dataset.shuffle(seed=42)

print(f"Loaded {len(train_dataset)} prompts for Online DPO")

# ====================== CELL 6: Train ==========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

online_judge = MathOnlineJudge(problem_to_gt)

training_args = OnlineDPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    beta=KL_BETA,
    max_new_tokens=MAX_NEW_TOKENS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_length=MAX_SEQ_LENGTH,
)

dpo_trainer = OnlineDPOTrainer(
    model=model,
    judge=online_judge,
    args=training_args,
    tokenizer=tokenizer,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

print("Starting Online DPO training...")
dpo_trainer.train()

# ====================== CELL 7: Save & Plot ==================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

log_history = dpo_trainer.state.log_history
steps = [x["step"] for x in log_history if "loss" in x]
losses = [x["loss"] for x in log_history if "loss" in x]

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Online DPO Loss", color="orange")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Online DPO Training Curve — Vi-MathLM-Aug")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "dpo_metrics_aug.png"), dpi=300)
plt.show()

print(f"\nVi-MathLM-Aug adapter saved to {OUTPUT_DIR}")
print("Next: run 05_inference_augmented.py for evaluation")
