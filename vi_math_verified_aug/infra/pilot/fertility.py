"""Vietnamese tokenizer fertility (02_models §3.2; W2 pilot task c). CPU only.

fertility = tokens / whitespace words (Vietnamese words here are syllables) on FLORES-200 vie_Latn devtest
(all 1,012 sentences), plus eng_Latn devtest for the vi/en ratio, plus mean tokens per Vi-GSM8K train item in
the T3 template (used to re-estimate the grid's tokens per run).

Needs: tokenizers, huggingface_hub (installed in scratchpad/w2/tokenv, NOT the project venv).
Usage: python -I fertility.py FLORES_DIR TRAIN_JSONL OUT_JSON
"""
import json
import re
import statistics
import sys

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODELS = ["Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base",
          "google/gemma-4-E2B", "sail/Sailor2-1B"]


def clean_solution(sol, gt):
    s = re.sub(r"<<[^<>]*>>", "", sol).rsplit("####", 1)[0].rstrip()
    return f"{s}\nĐáp số: \\boxed{{{gt}}}"


def main(flores_dir, train_jsonl, out):
    vi = [l.rstrip("\n") for l in open(f"{flores_dir}/devtest/vie_Latn.devtest", encoding="utf-8")]
    en = [l.rstrip("\n") for l in open(f"{flores_dir}/devtest/eng_Latn.devtest", encoding="utf-8")]
    train = [json.loads(l) for l in open(train_jsonl, encoding="utf-8") if l.strip()]
    items = [f"Bài toán:\n{r['problem']}\n\nLời giải:\n{clean_solution(r['solution'], r['ground_truth'])}"
             for r in train]
    comp = [f"{clean_solution(r['solution'], r['ground_truth'])}" for r in train]
    vi_words = sum(len(s.split()) for s in vi)
    en_words = sum(len(s.split()) for s in en)
    vi_chars = sum(len(s) for s in vi)
    res = {"flores": {"n_sent": len(vi), "vi_words": vi_words, "en_words": en_words, "vi_chars": vi_chars}}
    for m in MODELS:
        try:
            tk = Tokenizer.from_file(hf_hub_download(m, "tokenizer.json"))
        except Exception as e:  # noqa
            res[m] = {"error": repr(e)[:300]}
            continue
        nv = sum(len(e.ids) for e in tk.encode_batch(vi, add_special_tokens=False))
        ne = sum(len(e.ids) for e in tk.encode_batch(en, add_special_tokens=False))
        lens = [len(e.ids) for e in tk.encode_batch(items, add_special_tokens=False)]
        clens = [len(e.ids) for e in tk.encode_batch(comp, add_special_tokens=False)]
        lens_sorted = sorted(lens)
        res[m] = {"vocab": tk.get_vocab_size(), "vi_tokens": nv, "vi_fertility_tok_per_word": round(nv / vi_words, 3),
                  "vi_chars_per_token": round(vi_chars / nv, 3), "en_fertility_tok_per_word": round(ne / en_words, 3),
                  "vi_over_en_token_ratio": round(nv / ne, 3),
                  "train_item_tokens_mean": round(statistics.mean(lens), 1),
                  "train_item_tokens_p99": lens_sorted[int(0.99 * len(lens_sorted)) - 1],
                  "train_completion_tokens_mean": round(statistics.mean(clens), 1),
                  "train_total_tokens": sum(lens)}
        print(m, res[m], flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
