"""
KAGGLE NOTEBOOK 6: Final Evaluation & Comparison
=================================================
Runtime: CPU / T4 GPU, ~10 mins
Input: Predictions from Vi-MathLM-Base (04_inference_base.py) and
       Vi-MathLM-Aug (05_inference_augmented.py)
Output: Comparison plots + Accuracy + Format Compliance reports

"Model A" below refers to Vi-MathLM-Base, "Model C" to Vi-MathLM-Aug —
naming kept consistent with the predictions/ folders produced by 04 and 05.

Instructions:
1. Upload predictions from 04_inference_base.py and 05_inference_augmented.py
   as Kaggle Datasets
2. Adjust paths below
3. Paste and run
"""

# ====================== CELL 1: Config =======================
import os, json, re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Paths - Adjust to your Kaggle Input paths
MODEL_A_DIR = "/kaggle/input/predictions-model-a/model_a_sft"
MODEL_C_DIR = "/kaggle/input/predictions-model-c/model_c_dpo"
OUTPUT_DIR = "/kaggle/working/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== CELL 2: Utils ========================
def normalize_answer(ans):
    if ans is None: return ""
    ans = str(ans).strip()
    ans = re.sub(r'^\$\$?|\$\$?$', '', ans)
    ans = re.sub(r',', '', ans).strip()
    try:
        v = float(ans)
        return str(int(v)) if v == int(v) else str(v)
    except: pass
    return ans.lower().strip()

def check_answer(pred, gt):
    p, g = normalize_answer(pred), normalize_answer(str(gt))
    if not p or not g: return False
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return False

def load_preds(file_path):
    results = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

# ====================== CELL 3: Load Data ====================
print("Loading Model A Predictions...")
a_gsm8k = load_preds(os.path.join(MODEL_A_DIR, "gsm8k_predictions.jsonl"))
a_math = load_preds(os.path.join(MODEL_A_DIR, "math_predictions.jsonl"))

print("Loading Model C Predictions...")
c_gsm8k = load_preds(os.path.join(MODEL_C_DIR, "gsm8k_predictions.jsonl"))
c_math = load_preds(os.path.join(MODEL_C_DIR, "math_predictions.jsonl"))

# ====================== CELL 4: Calculate Metrics ============
def get_metrics(preds, name):
    if not preds: return {"name": name, "acc": 0, "total": 0, "correct": 0, "format_ok": 0}
    correct = sum(1 for p in preds if check_answer(p["model_answer"], p["ground_truth"]))
    fmt_ok = sum(1 for p in preds if p.get("format_ok", False))
    return {
        "name": name,
        "acc": correct / len(preds) * 100,
        "total": len(preds),
        "correct": correct,
        "format_pct": fmt_ok / len(preds) * 100 if preds else 0,
    }

stats = [
    get_metrics(a_gsm8k, "Model A (GSM8K)"),
    get_metrics(c_gsm8k, "Model C (GSM8K)"),
    get_metrics(a_math, "Model A (MATH)"),
    get_metrics(c_math, "Model C (MATH)"),
]

df_stats = pd.DataFrame(stats)
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(df_stats.to_string(index=False))

# ====================== CELL 5: Subject Analysis =============
def get_subject_stats(preds):
    subjects = {}
    for p in preds:
        sub = p.get("subject", "unknown")
        if sub not in subjects: subjects[sub] = {"correct": 0, "total": 0}
        subjects[sub]["total"] += 1
        if check_answer(p["model_answer"], p["ground_truth"]):
            subjects[sub]["correct"] += 1
    return {s: (v["correct"]/v["total"]*100) for s, v in subjects.items()}

a_math_subs = get_subject_stats(a_math)
c_math_subs = get_subject_stats(c_math)

# Merge for comparison
all_subs = sorted(list(set(a_math_subs.keys()) | set(c_math_subs.keys())))
comp_data = []
for s in all_subs:
    comp_data.append({"Subject": s, "Model A (CoT)": a_math_subs.get(s, 0), "Model C (CoT+SMT+DPO)": c_math_subs.get(s, 0)})

df_comp = pd.DataFrame(comp_data)
print("\nMATH Per-Subject Comparison:")
print(df_comp.to_string(index=False))

# ====================== CELL 6: Difficulty Analysis ===========
def get_level_stats(preds):
    levels = {}
    for p in preds:
        lvl = str(p.get("level", "?"))
        if lvl not in levels: levels[lvl] = {"correct": 0, "total": 0}
        levels[lvl]["total"] += 1
        if check_answer(p["model_answer"], p["ground_truth"]):
            levels[lvl]["correct"] += 1
    return {l: (v["correct"]/v["total"]*100) for l, v in levels.items() if v["total"] > 0}

a_math_levels = get_level_stats(a_math)
c_math_levels = get_level_stats(c_math)

level_data = []
for lvl in sorted(set(a_math_levels.keys()) | set(c_math_levels.keys())):
    level_data.append({"Level": lvl, "Model A": a_math_levels.get(lvl, 0), "Model C": c_math_levels.get(lvl, 0)})

df_levels = pd.DataFrame(level_data)
print("\nMATH Per-Level Comparison:")
print(df_levels.to_string(index=False))

# ====================== CELL 7: Visualization ================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Overall comparison
models = ["Model A\n(CoT)", "Model C\n(CoT+SMT+DPO)"]
gsm_accs = [stats[0]["acc"], stats[1]["acc"]]
math_accs = [stats[2]["acc"], stats[3]["acc"]]

x = range(len(models))
w = 0.35
axes[0].bar([i - w/2 for i in x], gsm_accs, w, label="GSM8K", color="#4ECDC4")
axes[0].bar([i + w/2 for i in x], math_accs, w, label="MATH", color="#FF6B6B")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Overall Accuracy")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Subject comparison
if comp_data:
    df_plot = df_comp.set_index("Subject")
    df_plot.plot(kind="barh", ax=axes[1], color=["#4ECDC4", "#FF6B6B"])
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("MATH Accuracy by Subject")
    axes[1].grid(axis='x', alpha=0.3)

# Plot 3: Difficulty comparison
if level_data:
    df_lp = df_levels.set_index("Level")
    df_lp.plot(kind="bar", ax=axes[2], color=["#4ECDC4", "#FF6B6B"])
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("MATH Accuracy by Difficulty Level")
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_plots.png"), dpi=300, bbox_inches='tight')
plt.show()

# ====================== CELL 8: Conclusion ===================
print("\n" + "="*60)
print("FINAL CONCLUSION")
print("="*60)

a_combined = (stats[0]["correct"] + stats[2]["correct"]) / max(stats[0]["total"] + stats[2]["total"], 1) * 100
c_combined = (stats[1]["correct"] + stats[3]["correct"]) / max(stats[1]["total"] + stats[3]["total"], 1) * 100
improvement = c_combined - a_combined

print(f"Model A Combined: {a_combined:.1f}%")
print(f"Model C Combined: {c_combined:.1f}%")
print(f"Improvement:      {improvement:+.1f}%")

if improvement > 0:
    print("\n✅ Model C (SMT-Augmented + DPO) outperformed Model A!")
elif improvement == 0:
    print("\n🟡 Models performed equally. Check per-subject differences.")
else:
    print("\n❌ Model C did not show overall improvement. Investigate per-subject/level.")

# Save full report
report = {
    "model_a_gsm8k": stats[0],
    "model_c_gsm8k": stats[1],
    "model_a_math": stats[2],
    "model_c_math": stats[3],
    "model_a_combined": a_combined,
    "model_c_combined": c_combined,
    "improvement": improvement,
    "subject_comparison": comp_data,
    "level_comparison": level_data,
}
with open(os.path.join(OUTPUT_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nFull report saved to {OUTPUT_DIR}/evaluation_report.json")
print(f"Plots saved to {OUTPUT_DIR}/comparison_plots.png")
