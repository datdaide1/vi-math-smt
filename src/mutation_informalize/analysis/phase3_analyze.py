# %%
"""
Phase 3 Clean Data Analysis - Thống kê toàn diện + Tỷ lệ nhất quán
Chạy: python analyze_phase3_clean.py
Output: thư mục analysis_plots/
"""
import json, os, re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

# Assumes this script is run from its own directory (src/mutation_informalize/)
CLEAN_FILE = os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized_clean.jsonl")
PHASE2_FILE = os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase2_mutated_smt.jsonl")
SAVE_DIR = os.path.join("..", "..", "..", "data", "finetune", "augmented", "analysis_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Load Data ───
print("Loading data...", flush=True)
data = [json.loads(l) for l in open(CLEAN_FILE, 'r', encoding='utf-8')]
phase2 = [json.loads(l) for l in open(PHASE2_FILE, 'r', encoding='utf-8')]

ok = [d for d in data if d['status'] == 'ok']
failed = [d for d in data if d['status'] == 'failed']
rescued = [d for d in ok if 'rescued:' in d.get('feedback', '')]

print(f"Total: {len(data)}, OK: {len(ok)}, Failed: {len(failed)}, Rescued: {len(rescued)}")

# ─── 1. TỔNG QUAN ───
print("\n" + "="*70)
print("1. TỔNG QUAN")
print("="*70)
print(f"  Tổng entries sau cleanup: {len(data)}")
print(f"  OK:      {len(ok)} ({len(ok)/len(data)*100:.1f}%)")
print(f"  Failed:  {len(failed)} ({len(failed)/len(data)*100:.1f}%)")
print(f"  Rescued: {len(rescued)}")

# Rescue breakdown
rescue_types = Counter(d.get('feedback','').replace('rescued:','') for d in rescued)
print(f"\n  Rescue breakdown:")
for k, v in rescue_types.most_common():
    print(f"    {k}: {v}")

# ─── 2. TỶ LỆ NHẤT QUÁN (Consistency Rate) ───
print("\n" + "="*70)
print("2. TỶ LỆ NHẤT QUÁN (Consistency Rate)")
print("="*70)

# Overall
consistency_rate = len(ok) / len(data) * 100
print(f"  Overall: {consistency_rate:.1f}% ({len(ok)}/{len(data)})")

# By subject
subjects = sorted(set(d['subject'] for d in data))
print(f"\n  By Subject:")
subj_rates = {}
for subj in subjects:
    s_all = [d for d in data if d['subject'] == subj]
    s_ok = [d for d in s_all if d['status'] == 'ok']
    rate = len(s_ok) / len(s_all) * 100 if s_all else 0
    subj_rates[subj] = rate
    print(f"    {subj:30s}: {rate:5.1f}% ({len(s_ok)}/{len(s_all)})")

# By level
print(f"\n  By Level:")
level_rates = {}
for lvl in sorted(set(d['level'] for d in data)):
    l_all = [d for d in data if d['level'] == lvl]
    l_ok = [d for d in l_all if d['status'] == 'ok']
    rate = len(l_ok) / len(l_all) * 100 if l_all else 0
    level_rates[lvl] = rate
    print(f"    Level {lvl}: {rate:5.1f}% ({len(l_ok)}/{len(l_all)})")

# By mutation strategy
print(f"\n  By Mutation Strategy:")
mut_rates = {}
for mut in sorted(set(d.get('mutation_strategy', 0) for d in data)):
    m_all = [d for d in data if d.get('mutation_strategy', 0) == mut]
    m_ok = [d for d in m_all if d['status'] == 'ok']
    rate = len(m_ok) / len(m_all) * 100 if m_all else 0
    mut_rates[mut] = rate
    print(f"    Mutation {mut}: {rate:5.1f}% ({len(m_ok)}/{len(m_all)})")

# ─── 3. SOURCE COVERAGE ───
print("\n" + "="*70)
print("3. SOURCE COVERAGE (Độ phủ bài gốc)")
print("="*70)
ok_indices = set(d['index'] for d in ok)
sources_ok = set()
for idx in ok_indices:
    if idx < len(phase2):
        sources_ok.add(phase2[idx].get('source_index'))
print(f"  Bài gốc có ít nhất 1 OK mutation: {len(sources_ok)} / 6817 ({len(sources_ok)/6817*100:.1f}%)")

# Mutations per source
from collections import defaultdict
source_mut_count = defaultdict(int)
for d in ok:
    idx = d['index']
    if idx < len(phase2):
        src = phase2[idx].get('source_index')
        source_mut_count[src] += 1
mut_dist = Counter(source_mut_count.values())
print(f"\n  Phân bố số mutations OK/bài gốc:")
for k in sorted(mut_dist.keys()):
    print(f"    {k} mutations: {mut_dist[k]} bài gốc")

# ─── 4. PHÂN BỐ ĐỘ DÀI ───
print("\n" + "="*70)
print("4. PHÂN BỐ ĐỘ DÀI")
print("="*70)
prob_lens = [len(d.get('problem', '')) for d in ok]
sol_lens = [len(d.get('solution', '')) for d in ok]
print(f"  Problem: mean={np.mean(prob_lens):.0f}, median={np.median(prob_lens):.0f}, min={min(prob_lens)}, max={max(prob_lens)}")
print(f"  Solution: mean={np.mean(sol_lens):.0f}, median={np.median(sol_lens):.0f}, min={min(sol_lens)}, max={max(sol_lens)}")

# ═══════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════
print("\nGenerating plots...", flush=True)

# Plot 1: OK vs Failed Pie
fig, ax = plt.subplots(figsize=(6, 5))
sizes = [len(ok), len(failed)]
labels = [f'OK\n{len(ok)} ({len(ok)/len(data)*100:.1f}%)', f'Failed\n{len(failed)} ({len(failed)/len(data)*100:.1f}%)']
colors = ['#2ecc71', '#e74c3c']
ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
ax.set_title('Phase 3 Clean: OK vs Failed', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '01_ok_vs_failed_pie.png'), bbox_inches='tight')
plt.close()

# Plot 2: Consistency Rate by Subject
fig, ax = plt.subplots(figsize=(10, 5))
subj_names = [s.replace('_', ' ').title() for s in subjects]
rates = [subj_rates[s] for s in subjects]
colors_subj = plt.cm.Set2(np.linspace(0, 1, len(subjects)))
bars = ax.barh(subj_names, rates, color=colors_subj, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Consistency Rate (%)')
ax.set_title('Tỷ lệ nhất quán theo chủ đề', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{rate:.1f}%', va='center', fontsize=10)
ax.axvline(x=consistency_rate, color='red', linestyle='--', alpha=0.7, label=f'Overall: {consistency_rate:.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '02_consistency_by_subject.png'), bbox_inches='tight')
plt.close()

# Plot 3: Consistency Rate by Level
fig, ax = plt.subplots(figsize=(8, 5))
lvls = sorted(level_rates.keys())
rates_l = [level_rates[l] for l in lvls]
colors_l = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
bars = ax.bar([f'Level {l}' for l in lvls], rates_l, color=colors_l, edgecolor='white', width=0.6)
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('Tỷ lệ nhất quán theo mức độ khó', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
for bar, rate in zip(bars, rates_l):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=consistency_rate, color='red', linestyle='--', alpha=0.7, label=f'Overall: {consistency_rate:.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '03_consistency_by_level.png'), bbox_inches='tight')
plt.close()

# Plot 4: Consistency Rate by Mutation
fig, ax = plt.subplots(figsize=(8, 5))
muts = sorted(mut_rates.keys())
rates_m = [mut_rates[m] for m in muts]
colors_m = plt.cm.Pastel1(np.linspace(0, 0.6, len(muts)))
bars = ax.bar([f'Mut {m}' for m in muts], rates_m, color=colors_m, edgecolor='gray', width=0.6)
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('Tỷ lệ nhất quán theo chiến lược đột biến', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
for bar, rate in zip(bars, rates_m):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=consistency_rate, color='red', linestyle='--', alpha=0.7, label=f'Overall: {consistency_rate:.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '04_consistency_by_mutation.png'), bbox_inches='tight')
plt.close()

# Plot 5: Subject Distribution (OK entries)
fig, ax = plt.subplots(figsize=(10, 5))
subj_ok_counts = Counter(d['subject'] for d in ok)
names = [s.replace('_', ' ').title() for s in subjects]
counts = [subj_ok_counts.get(s, 0) for s in subjects]
ax.barh(names, counts, color=colors_subj, edgecolor='white')
ax.set_xlabel('Number of OK entries')
ax.set_title('Phân bố OK entries theo chủ đề', fontsize=14, fontweight='bold')
for i, c in enumerate(counts):
    ax.text(c + 30, i, str(c), va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '05_subject_distribution.png'), bbox_inches='tight')
plt.close()

# Plot 6: Solution Length Distribution (Histogram)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sol_lens, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(x=np.mean(sol_lens), color='red', linestyle='--', label=f'Mean: {np.mean(sol_lens):.0f}')
ax.axvline(x=np.median(sol_lens), color='orange', linestyle='--', label=f'Median: {np.median(sol_lens):.0f}')
ax.set_xlabel('Solution Length (chars)')
ax.set_ylabel('Count')
ax.set_title('Phân bố độ dài lời giải (OK entries)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '06_solution_length_dist.png'), bbox_inches='tight')
plt.close()

# Plot 7: Problem Length Distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(prob_lens, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
ax.axvline(x=np.mean(prob_lens), color='red', linestyle='--', label=f'Mean: {np.mean(prob_lens):.0f}')
ax.axvline(x=np.median(prob_lens), color='orange', linestyle='--', label=f'Median: {np.median(prob_lens):.0f}')
ax.set_xlabel('Problem Length (chars)')
ax.set_ylabel('Count')
ax.set_title('Phân bố độ dài đề bài (OK entries)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '07_problem_length_dist.png'), bbox_inches='tight')
plt.close()

# Plot 8: Heatmap Subject x Level
fig, ax = plt.subplots(figsize=(10, 6))
heatmap_data = np.zeros((len(subjects), 5))
for d in ok:
    si = subjects.index(d['subject'])
    li = d['level'] - 1
    heatmap_data[si][li] += 1
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(5))
ax.set_xticklabels([f'Level {i+1}' for i in range(5)])
ax.set_yticks(range(len(subjects)))
ax.set_yticklabels([s.replace('_', ' ').title() for s in subjects])
for i in range(len(subjects)):
    for j in range(5):
        ax.text(j, i, f'{int(heatmap_data[i][j])}', ha='center', va='center', fontsize=9,
                color='white' if heatmap_data[i][j] > heatmap_data.max()*0.6 else 'black')
ax.set_title('Heatmap: Subject × Level (OK entries)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '08_subject_level_heatmap.png'), bbox_inches='tight')
plt.close()

# Plot 9: Consistency Rate Heatmap (Subject x Level)
fig, ax = plt.subplots(figsize=(10, 6))
rate_heatmap = np.zeros((len(subjects), 5))
for si, subj in enumerate(subjects):
    for li in range(5):
        lvl = li + 1
        all_sl = [d for d in data if d['subject'] == subj and d['level'] == lvl]
        ok_sl = [d for d in all_sl if d['status'] == 'ok']
        rate_heatmap[si][li] = len(ok_sl) / len(all_sl) * 100 if all_sl else 0
im = ax.imshow(rate_heatmap, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
ax.set_xticks(range(5))
ax.set_xticklabels([f'Level {i+1}' for i in range(5)])
ax.set_yticks(range(len(subjects)))
ax.set_yticklabels([s.replace('_', ' ').title() for s in subjects])
for i in range(len(subjects)):
    for j in range(5):
        ax.text(j, i, f'{rate_heatmap[i][j]:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if rate_heatmap[i][j] < 55 else 'black')
ax.set_title('Tỷ lệ nhất quán: Subject × Level', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Consistency Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '09_consistency_heatmap.png'), bbox_inches='tight')
plt.close()

print(f"\nAll plots saved to: {SAVE_DIR}")
print("DONE!", flush=True)


# %%
