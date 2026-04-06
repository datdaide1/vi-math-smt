"""
KAGGLE NOTEBOOK 1: SFT Training — Model A (CoT only)
=====================================================
Runtime: T4 GPU, ~5 hours
Input: Upload train_cot_only_clean.jsonl as Kaggle Dataset
Output: Model A adapter (save as Kaggle Output)

Instructions:
1. Upload data/finetune/processed/train_cot_only_clean.jsonl (from your local
   repo checkout) to a Kaggle Dataset (e.g. "smt-train-data")
2. Create new notebook, add that dataset
3. Enable GPU (T4), Internet ON
4. Paste this entire script and run
"""

# ====================== CELL 1: Install ======================
# !pip uninstall unsloth unsloth-zoo -y
# !pip install --upgrade --no-cache-dir unsloth unsloth-zoo
# !pip install --no-deps trl peft accelerate bitsandbytes

# ====================== CELL 2: Config =======================
import os, json

# -- Paths (adjust if your Kaggle Dataset name differs) --
DATA_PATH = "/kaggle/input/datasets/datuni/smt-train-data-new/train_cot_only_clean.jsonl"
OUTPUT_DIR = "/kaggle/working/model_a_sft"

# -- Hyperparameters (OPTIMIZED for T4) --
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
# Set HF_TOKEN via Kaggle Add-ons > Secrets (bind to env var HF_TOKEN), or set the
# environment variable manually before running this cell.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_SEQ_LENGTH = 2048
LORA_R = 32          # 64 → 32 (more stable on T4, still sufficient)
LORA_ALPHA = 32      # Match r for balanced scaling
LEARNING_RATE = 2e-5  # 1e-4 → 2e-5 (CRITICAL: preserve base reasoning)
NUM_EPOCHS = 2        # 3 → 2 (avoid overfitting)
BATCH_SIZE = 1        # Increased with gradient checkpointing
GRAD_ACCUM = 8        # effective batch = 8
WARMUP_RATIO = 0.1    # 0.05 → 0.1 (gentler warmup)

# ====================== CELL 3: Load Model ===================
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
print(f"Loaded: {BASE_MODEL}")

# ====================== CELL 4: Apply LoRA ===================
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    random_state=42,
)
print("LoRA applied")

# ====================== CELL 5: Load Data ====================
from datasets import load_dataset

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.02)
print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")
print(f"Sample:\n{dataset['train'][0]['text'][:300]}")

# ====================== CELL 6: Train ========================
from trl import SFTTrainer
from transformers import TrainingArguments

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.05,            # 0.01 → 0.05 (better regularization)
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,  # Auto-select best checkpoint
        metric_for_best_model="eval_loss",
        seed=42,
        report_to="none",
        average_tokens_across_devices=False,
    ),
)

print("Starting SFT training for Model A...")
stats = trainer.train()
print(f"Done! Runtime: {stats.metrics['train_runtime']:.0f}s, Loss: {stats.metrics['train_loss']:.4f}")

# ====================== CELL 7: Save =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(stats.metrics, f, indent=2)

print(f"Model A saved to {OUTPUT_DIR}")
print("Download from Kaggle Output and upload as new Dataset for inference later.")

# ====================== CELL 8: Plot Loss ====================
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
    json.dump(log_history, f, indent=2)

train_steps = []
train_loss = []
eval_steps_list = []
eval_loss = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    elif "eval_loss" in log and "step" in log:
        eval_steps_list.append(log["step"])
        eval_loss.append(log["eval_loss"])

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label="Loss Huấn luyện", color="blue", alpha=0.6)
if eval_loss:
    plt.plot(eval_steps_list, eval_loss, label="Loss Đánh giá", color="red", marker="o", linewidth=2)

plt.xlabel("Số bước (Steps)")
plt.ylabel("Mức độ lỗi (Loss)")
plt.title("Biểu đồ Loss Huấn luyện và Đánh giá (Model A)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
plt.show()

print(f"Loss curve saved to {OUTPUT_DIR}/loss_curve.png")
