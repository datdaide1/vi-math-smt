# Outputs captured from `merge-all.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: c1)

```
GSM8K: 7473 entries
MATH:  7397 entries
Total: 14870 entries
```

## Cell 1 (id: c2)

```
Merged dataset: 14870 entries (shuffled, re-indexed)
Columns: ['index', 'problem', 'smt', 'level', 'type', 'solution', 'subject', 'ground_truth', 'source']
```

## Cell 2 (id: c3)

```
=== Source distribution ===
source
gsm8k    7473
math     7397
Name: count, dtype: int64

=== Level distribution ===
level
1    3237
2    4908
3    2823
4    1667
5    2235
Name: count, dtype: int64

=== Subject distribution ===
subject
arithmetic                  7473
algebra                     1742
intermediate_algebra        1269
prealgebra                  1203
number_theory                855
geometry                     832
counting_and_probability     768
precalculus                  728
Name: count, dtype: int64

=== SMT coverage ===
With SMT: 14870 (100.0%)
Without:  0 (0.0%)
```

## Cell 3 (id: c4)

```
<Figure size 1600x1200 with 4 Axes>

Saved: <repo>\src\data-complete\img\merged_distribution.png
```

## Cell 4 (id: c5)

```
Exporting JSON: <repo>\data\finetune\mine\train_vietnamese_final.json
Exporting JSONL: <repo>\data\finetune\mine\train_vietnamese_final.jsonl

✅ Done! 14870 entries exported.
   JSON:  <repo>\data\finetune\mine\train_vietnamese_final.json
   JSONL: <repo>\data\finetune\mine\train_vietnamese_final.jsonl
```
