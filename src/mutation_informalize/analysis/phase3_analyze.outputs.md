# Outputs captured from `phase3_analyze.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: 59f2b5ff)

```
Loading data...
Total: 27024, OK: 19750, Failed: 7274, Rescued: 255

======================================================================
1. TỔNG QUAN
======================================================================
  Tổng entries sau cleanup: 27024
  OK:      19750 (73.1%)
  Failed:  7274 (26.9%)
  Rescued: 255

  Rescue breakdown:
    numeric_match: 148
    string_match: 83
    sympy_match: 24

======================================================================
2. TỶ LỆ NHẤT QUÁN (Consistency Rate)
======================================================================
  Overall: 73.1% (19750/27024)

  By Subject:
    algebra                       :  67.9% (4184/6162)
    counting_and_probability      :  63.3% (1882/2974)
    geometry                      :  73.9% (2128/2879)
    intermediate_algebra          :  76.6% (3605/4709)
    number_theory                 :  74.2% (2242/3021)
    prealgebra                    :  79.5% (3606/4533)
    precalculus                   :  76.6% (2103/2746)

  By Level:
    Level 1:  70.3% (1500/2134)
    Level 2:  72.5% (3589/4951)
    Level 3:  73.2% (4273/5841)
    Level 4:  74.5% (4551/6106)
    Level 5:  73.0% (5837/7992)

  By Mutation Strategy:
    Mutation 0:  73.0% (4860/6655)
    Mutation 1:  72.6% (4807/6622)
    Mutation 2:  73.0% (3714/5085)
    Mutation 3:  73.2% (3170/4333)
    Mutation 4:  73.9% (3199/4329)

======================================================================
3. SOURCE COVERAGE (Độ phủ bài gốc)
======================================================================
  Bài gốc có ít nhất 1 OK mutation: 5670 / 6817 (83.2%)

  Phân bố số mutations OK/bài gốc:
    1 mutations: 593 bài gốc
    2 mutations: 1306 bài gốc
    3 mutations: 795 bài gốc
    4 mutations: 720 bài gốc
    5 mutations: 2256 bài gốc

======================================================================
4. PHÂN BỐ ĐỘ DÀI
======================================================================
  Problem: mean=343, median=291, min=21, max=3143
  Solution: mean=904, median=808, min=156, max=6005

Generating plots...

All plots saved to: <repo>\data\finetune\augmented\analysis_plots
DONE!
```
