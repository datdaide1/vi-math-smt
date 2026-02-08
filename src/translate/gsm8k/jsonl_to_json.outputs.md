# Outputs captured from `jsonl_to_json.ipynb` before converting to a plain .py script.

# Preserved for reference — these are real recorded execution results, not regenerated.


## Cell 0 (id: cell-config)

```
Input : <repo>/data_translation/data/vn/gsm8k/gsm8k_test_vietnamese.jsonl
Output: <repo>/data_translation/data/vn/gsm8k/gsm8k_test_vietnamese.json
```

## Cell 1 (id: cell-convert)

```
Đọc được 1319 bản ghi từ <repo>/data_translation/data/vn/gsm8k/gsm8k_test_vietnamese.jsonl
Đã lưu thành công: <repo>/data_translation/data/vn/gsm8k/gsm8k_test_vietnamese.json
```

## Cell 2 (id: cell-verify)

```
Tổng số bản ghi: 1319
Keys của bản ghi đầu tiên: ['question', 'answer', 'ground_truth']

--- Bản ghi đầu tiên ---
{
  "question": "Vịt của Janet đẻ 16 quả trứng mỗi ngày. Cô ấy ăn ba quả vào bữa sáng mỗi sáng và làm bánh nướng xốp cho bạn bè mỗi ngày với bốn quả. Cô ấy bán số trứng còn lại tại chợ nông sản hàng ngày với giá 2 đô la cho mỗi quả trứng vịt tươi. Hỏi mỗi ngày cô ấy kiếm được bao nhiêu đô la tại chợ nông sản?",
  "answer": "Janet bán 16 - 3 - 4 = <<16-3-4=9>>9 quả trứng vịt mỗi ngày.\nMỗi ngày cô ấy kiếm được 9 * 2 = $<<9*2=18>>18 tại chợ nông sản.\n#### 18",
  "ground_truth": "18"
}
```
