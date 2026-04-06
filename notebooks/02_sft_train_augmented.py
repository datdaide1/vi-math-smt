"""
KAGGLE NOTEBOOK 2: SFT Training — Augmented Data (pre-DPO)
===========================================================
Runtime: T4 GPU, ~8 hours
Input: Upload the augmented training set (train_augmented_final.jsonl,
       formatted into the SFT "text" field — see notebooks/00_prepare_data.py)
Output: interim SFT adapter, later refined by 03_dpo_train_augmented.py into
        Vi-MathLM-Aug

Same hyperparameters as 01_sft_train_base.py (Kịch bản I) — this stage differs
only in the training set: the 34,620-sample augmented set instead of the
14,870-sample base set.

Instructions:
1. Upload sft_data.jsonl (augmented, formatted) to a Kaggle Dataset
   (e.g. "aug-data-new")
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
DATA_PATH = "/kaggle/input/datasets/datuni/aug-data-new/sft_data.jsonl"
OUTPUT_DIR = "/kaggle/working/model_aug_sft"

# -- Hyperparameters (same as Kịch bản I — Bảng 3.1) --
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
# Set HF_TOKEN via Kaggle Add-ons > Secrets (bind to env var HF_TOKEN), or set the
# environment variable manually before running this cell.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_SEQ_LENGTH = 2048
LORA_R = 32
LORA_ALPHA = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.1

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
        weight_decay=0.05,
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        report_to="none",
        average_tokens_across_devices=False,
    ),
)

print("Starting SFT training on augmented data...")
checkpoints_dir = OUTPUT_DIR
import glob
checkpoints = glob.glob(os.path.join(checkpoints_dir, "checkpoint-*"))
if checkpoints:
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    print(f">>> Resuming from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    trainer.train()

# ====================== CELL 7: Save & Plot ==================
import matplotlib.pyplot as plt

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

log_history = trainer.state.log_history
train_steps = [x["step"] for x in log_history if "loss" in x]
train_loss = [x["loss"] for x in log_history if "loss" in x]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.title("SFT (Augmented Data) Training Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.show()

print(f"\nSFT-on-augmented adapter saved to {OUTPUT_DIR}")
print("Next: feed this adapter into 03_dpo_train_augmented.py")
