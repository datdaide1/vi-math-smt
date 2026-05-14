# Outputs captured from `merge_and_sort.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: 722ea638)

```
Train file: output\svamp\svamp-train-vi.jsonl
Test file: output\svamp\svamp-test-vi.jsonl
Output file: output\svamp\svamp-vi.jsonl
```

## Cell 1 (id: 1006943e)

```
Loading train file...
Loaded 700 records from train file
Loading test file...
Loaded 300 records from test file
Total records: 1000
```

## Cell 2 (id: f58c434c)

```
Sample record:
{
  "ID": "chal-777",
  "Body": "Trong bộ sưu tập của Philip có 87 quả cam và 290 quả chuối. Nếu chuối được chia thành 2 nhóm và cam được chia thành 93 nhóm",
  "Question": "Mỗi nhóm chuối có bao nhiêu quả?",
  "Answer": "145 quả chuối",
  "Type": "Common-Division",
  "Equation": "( 290.0 / 2.0 )"
}
```

## Cell 3 (id: b3f7a1c5)

```
Using 'ID' as ID field
```

## Cell 4 (id: d5a32199)

```
Sorting data by ID...
Sorting completed!
```

## Cell 5 (id: 8cb2b1bf)

```
First 3 records after sorting:
  Record 0: ID = chal-1
  Record 1: ID = chal-10
  Record 2: ID = chal-100

Last 3 records after sorting:
  Record 997: ID = chal-997
  Record 998: ID = chal-998
  Record 999: ID = chal-999
```

## Cell 6 (id: ddc8325d)

```
Saving to output\svamp\svamp-vi.jsonl...
Successfully saved 1000 records to output\svamp\svamp-vi.jsonl
```

## Cell 7 (id: 597f0932)

```
✓ Output file created successfully
  File size: 341,919 bytes
  Records: 1000
```
