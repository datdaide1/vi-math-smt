# %%
import json
import os

# Define paths — assumes this script is run from its own directory (src/merge/)
smt_file_path = os.path.join("..", "..", "..", "data", "formalize_output", "gsm8k", "gsm8k-smt-nvidia-FULL.jsonl")
vi_file_path = os.path.join("..", "..", "..", "data", "translation", "gsm8k", "gsm8k_train_vietnamese.jsonl")
output_dir = os.path.join("..", "..", "..", "data", "finetune", "gsm8k")
output_file_path = os.path.join(output_dir, 'gsm8k_train_vietnamese_smt.jsonl')

# Create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

print(f"Reading SMT data from: {smt_file_path}")
# Read SMT data into a dictionary
smt_data = {}
with open(smt_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        idx = item['index']
        # Use the formalization result
        smt_data[idx] = item.get('smt_lib', '')

print(f"Reading Vietnamese data from: {vi_file_path}")
# Read Vietnamese data and merge
merged_data = []
missing_indices = []

with open(vi_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        idx = item['index']
        if idx in smt_data:
            item['smt'] = smt_data[idx]
        else:
            item['smt'] = None
            missing_indices.append(idx)
        merged_data.append(item)

print(f"Total items in Vietnamese data: {len(merged_data)}")
print(f"Total items in SMT data: {len(smt_data)}")
if missing_indices:
    print(f"Warning: {len(missing_indices)} items missing SMT data.")
else:
    print("Success: All items merged successfully.")

# Sort by index for consistency
merged_data.sort(key=lambda x: x['index'])

# Save to output file
print(f"Saving merged data to: {output_file_path}")
with open(output_file_path, 'w', encoding='utf-8') as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Step 1 Done.")

# %%
import json
import os
import re
from collections import OrderedDict

# Define paths for the second step
input_file = os.path.join("..", "..", "..", "data", "finetune", "gsm8k", "gsm8k_train_vietnamese_smt.jsonl")
out_dir = os.path.join("..", "..", "..", "data", "finetune", "gsm8k")
out_json = os.path.join(out_dir, 'gsm8k_train_vietnamese_final.json')
out_jsonl = os.path.join(out_dir, 'gsm8k_train_vietnamese_final.jsonl')

def calculate_level(solution):
    # Count reasoning steps marked by <<...>>
    steps = len(re.findall(r'<<.*?>>', solution))
    if steps <= 2:
        return 1
    elif steps <= 4:
        return 2
    else:
        return 3

print(f"Processing final transformation from: {input_file}")
final_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        item = json.loads(line)
        
        # Transform and reorder based on mapping:
        # index -> problem -> smt -> level -> type -> solution -> subject -> ground_truth
        transformed = OrderedDict([
            ('index', item['index']),
            ('problem', item['question']),       # Renamed from question
            ('smt', item.get('smt', '')),
            ('level', calculate_level(item['answer'])),
            ('type', 'Arithmetic'),              # New field
            ('solution', item['answer']),        # Renamed from answer
            ('subject', 'arithmetic'),           # New field
            ('ground_truth', item['ground_truth'])
        ])
        final_data.append(transformed)

print(f"Total items processed: {len(final_data)}")

# Export to both JSON and JSONL
print(f"Exporting to JSON: {out_json}")
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"Exporting to JSONL: {out_jsonl}")
with open(out_jsonl, 'w', encoding='utf-8') as f:
    for item in final_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Final step completed successfully.")
