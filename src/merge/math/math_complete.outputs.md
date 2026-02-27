# Outputs captured from `math-complete.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: b22583ea)

```
Output directory: <repo>\data\finetune\math
```

## Cell 2 (id: step1)

```

=== SMT Data Statistics ===
Total records:               7500
Kept (pass):                 7131
Kept (cannot_parse_gt):      266
Dropped (bad quality):       103
Total available for merge:   7397
```

## Cell 4 (id: step2)

```
  algebra                         kept=1742   dropped=2
  counting_and_probability        kept=768    dropped=3
  geometry                        kept=832    dropped=38
  intermediate_algebra            kept=1269   dropped=26
  number_theory                   kept=855    dropped=14
  prealgebra                      kept=1203   dropped=2
  precalculus                     kept=728    dropped=19

=== Merge Results ===
Total entries:   7397
Dropped:         104
```

## Cell 6 (id: step3)

```
Exporting JSON:  <repo>\data\finetune\math\math_train_vietnamese_final.json
Exporting JSONL: <repo>\data\finetune\math\math_train_vietnamese_final.jsonl

✅ Done! 7397 entries exported.
   JSON:  <repo>\data\finetune\math\math_train_vietnamese_final.json
   JSONL: <repo>\data\finetune\math\math_train_vietnamese_final.jsonl
```

## Cell 8 (id: step4)

```
Total entries:  7397
Columns:        ['index', 'problem', 'smt', 'level', 'type', 'solution', 'subject', 'ground_truth']
Null SMT:       0
Empty SMT:      0

=== Per subject ===
subject
algebra                     1742
counting_and_probability     768
geometry                     832
intermediate_algebra        1269
number_theory                855
prealgebra                  1203
precalculus                  728

=== Sample entry ===
  index: 0
  problem: Cho \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ nếu }x>2, \\
x-5 &\text{ nếu } -2 \le x \le 2, ...
  smt: (set-logic ALL)
(declare-const a Real)
(declare-const b Real)
(declare-const answer Real)

; Continu...
  level: Level 5
  type: Algebra
  solution: Để hàm số từng phần liên tục, các trường hợp phải "gặp nhau" tại $2$ và $-2$. Ví dụ, $ax+3$ và $x-5$...
  subject: algebra
  ground_truth: 0
```
