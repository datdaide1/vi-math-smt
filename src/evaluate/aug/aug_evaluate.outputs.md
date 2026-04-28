# Outputs captured from `data\result\augmented-sft-dpo\augmented-sft-dpo_evaluation.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: cell-0)

```
Đã khai báo toàn bộ thư viện và cấu hình đồ họa tiếng Việt thành công!
```

## Cell 1 (id: cell-1)

```
Đã cài đặt các hàm chuẩn hóa LaTeX và kiểm tra bằng SymPy thành công!
```

## Cell 2 (id: cell-2)

```
Đã cài đặt các hàm tính FCR (Inclusive) và Reasoning Similarity (RSS) thành công!
```

## Cell 3 (id: cell-3)

```
Đang nạp các tập dữ liệu đối chứng (Ground-truth)... 
Nạp thành công 1319 lời giải mẫu GSM8K và 4999 lời giải mẫu MATH!
Đang xử lý kết quả dự đoán GSM8K...
Đang xử lý kết quả dự đoán MATH...

<string>:1: SyntaxWarning: 'set' object is not callable; perhaps you missed a comma?


--- HOÀN THÀNH SO KHỚP VÀ XỬ LÝ DỮ LIỆU CẢ 2 TẬP ---
```

## Cell 4 (id: cell-4)

```
1. BỘ TẬP DỮ LIỆU GSM8K:
  - Số lượng mẫu thử: 1319
  - Độ chính xác acc (Accuracy): 63.76%
  - Tỷ lệ tuân thủ định dạng (FCR): 94.62%
  - Độ tương đồng lý luận (RSS): 56.70%
------------------------------------------------------------
2. BỘ TẬP DỮ LIỆU MATH:
  - Số lượng mẫu thử: 5000
  - Độ chính xác acc (Accuracy): 39.58%
  - Tỷ lệ tuân thủ định dạng (FCR): 82.30%
  - Độ tương đồng lý luận (RSS): 54.02%
============================================================

============================================================
  PHÂN TÍCH ĐỘ CHÍNH XÁC acc & TƯƠNG ĐỒNG THEO CHỦ ĐỀ (MATH)
============================================================
                          Số lượng  Độ chính xác acc (%)  Tuân thủ định dạng (%)  Tương đồng lý luận (%)
subject                                                                                                 
Algebra                       1187                 46.76                   89.72                   56.34
Counting And Probability       474                 49.79                   82.91                   59.28
Geometry                       479                 24.63                   77.24                   46.67
Intermediate Algebra           903                 28.57                   72.87                   48.38
Number Theory                  540                 44.81                   82.04                   54.50
Prealgebra                     871                 48.22                   88.86                   57.96
Precalculus                    546                 27.47                   75.46                   53.46

============================================================
  PHÂN TÍCH ĐỘ CHÍNH XÁC acc & TƯƠNG ĐỒNG THEO ĐỘ KHÓ (MATH)
============================================================
         Số lượng  Độ chính xác acc (%)  Tuân thủ định dạng (%)  Tương đồng lý luận (%)
level                                                                                  
Level 1       437                 54.23                   94.05                   61.37
Level 2       894                 48.77                   91.50                   59.93
Level 3      1131                 44.21                   86.38                   57.27
Level 4      1214                 36.90                   80.40                   52.81
Level 5      1324                 27.04                   70.47                   45.95
```

## Cell 5 (id: cell-5)

```
<Figure size 1000x600 with 1 Axes>

<Figure size 1200x600 with 1 Axes>

<Figure size 1000x600 with 1 Axes>

<Figure size 1200x500 with 2 Axes>

Đã tạo và lưu thành công toàn bộ 4 biểu đồ báo cáo tiếng Việt vào thư mục 'plots'!
```
