# %%
import json
import os
from pathlib import Path

# Define paths
input_dir = r'output\svamp'
output_dir = r'output\svamp'

train_file = os.path.join(input_dir, 'svamp-train-vi.jsonl')
test_file = os.path.join(input_dir, 'svamp-test-vi.jsonl')
output_file = os.path.join(output_dir, 'svamp-vi.jsonl')

print(f"Train file: {train_file}")
print(f"Test file: {test_file}")
print(f"Output file: {output_file}")

# %%
# Load both files
data = []

# Load train file
print("Loading train file...")
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data)} records from train file")

# Load test file
print("Loading test file...")
initial_count = len(data)
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
print(f"Loaded {len(data) - initial_count} records from test file")
print(f"Total records: {len(data)}")

# %%
# Check the structure of data
if data:
    print("Sample record:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False)[:500])

# %%
# Sort by ID - detecting the ID field
# Check common ID field names
id_field = None
if data:
    first_record = data[0]
    if 'id' in first_record:
        id_field = 'id'
    elif 'ID' in first_record:
        id_field = 'ID'
    elif 'problem_id' in first_record:
        id_field = 'problem_id'
    else:
        # Show available keys
        print("Available keys:", first_record.keys())
        id_field = 'id'  # Default assumption

print(f"Using '{id_field}' as ID field")

# %%
# Sort the data by ID
def extract_id(record):
    """Extract and convert ID to sortable value"""
    if id_field in record:
        id_val = record[id_field]
        # Try to convert to int if possible for proper sorting
        try:
            return int(id_val) if isinstance(id_val, str) else id_val
        except (ValueError, TypeError):
            return id_val
    return 0

print("Sorting data by ID...")
data_sorted = sorted(data, key=extract_id)
print("Sorting completed!")

# %%
# Show first and last few records after sorting
print("First 3 records after sorting:")
for i, record in enumerate(data_sorted[:3]):
    print(f"  Record {i}: ID = {record.get(id_field, 'N/A')}")

print("\nLast 3 records after sorting:")
for i, record in enumerate(data_sorted[-3:]):
    print(f"  Record {len(data_sorted)-3+i}: ID = {record.get(id_field, 'N/A')}")

# %%
# Save the merged and sorted data
print(f"Saving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for record in data_sorted:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
print(f"Successfully saved {len(data_sorted)} records to {output_file}")

# %%
# Verify the output file
if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"✓ Output file created successfully")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Records: {len(data_sorted)}")
else:
    print("✗ Error: Output file was not created")
