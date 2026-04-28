# Outputs captured from `z3_evaluate.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 2 (id: cell_imports)

```
Import OK
```

## Cell 3 (id: cell_load_data)

```
Tổng số bài: 3
Bài có SMT code: 0
Bài không có SMT code: 3
```

## Cell 4 (id: cell_solver_fn)

```
Hàm parse_and_solve_smtlib đã sẵn sàng
```

## Cell 5 (id: cell_compare_fn)

```
Hàm normalize & compare đã sẵn sàng
```

## Cell 6 (id: cell_run_eval)

```
 Idx  ✓/✗  Status      Z3 Answer       Ground Truth    Problem
-------------------------------------------------------------------------------------
   0  ✓   sat         2               2               
   1  ✓   sat         10              10              
   2  ✓   sat         9/7             9/7             
-------------------------------------------------------------------------------------
Đã xử lý xong!
```

## Cell 7 (id: cell_summary)

```
=======================================================
TỔNG KẾT
=======================================================
  Tổng số bài           : 3
  Có SMT code → SAT     : 3
  Không có SMT code     : 0
  Lỗi parse / runtime   : 0
  UNSAT / Unknown       : 0
-------------------------------------------------------
  Đúng / Tổng SAT       : 3 / 3  (100.0%)
  Sai (trong số SAT)    : 0
-------------------------------------------------------
  Accuracy tổng thể     : 3 / 3  (100.0%)
=======================================================
```

## Cell 8 (id: cell_wrong_list)

```
Bài giải SAI (0 bài):
---------------------------------------------------------------------------
```

## Cell 9 (id: cell_error_list)

```
Bài lỗi parse (0 bài):
---------------------------------------------------------------------------
```

## Cell 10 (id: cell_save)

```
Đã lưu kết quả vào: <repo>\z3_evaluation_results.json
```
