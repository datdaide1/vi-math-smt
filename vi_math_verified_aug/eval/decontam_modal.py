"""X5 embedding stage on Modal (L4, bf16). Companion of ``decontam.py``; reuses its Encoder / EmbeddingIndex /
calibrate_pairs (the package source is shipped into the image).

Run from the repo root (inputs prepared locally into data/contam/embed_inputs/inputs.json, see X5 report):

    modal run -m vi_math_verified_aug.eval.decontam_modal                       # bge-m3 scan + calibration
    modal volume get x5-embed / data/contam/embed_modal/                        # pull incremental outputs

Budget guard (EXEC_BRIEF2 "Modal budget": X5 cap $0.50 on L4 at $0.80/h): hard function timeout 1,800 s
(= $0.40 GPU), per-stage time budgets, outputs written to the ``x5-embed`` volume after every stage so a timeout
loses nothing already done.

Corpora are streamed/downloaded inside the container from the HF hub (never via this laptop):
vinhnq29 (4 splits), hllj_mcq, hllj_crawl, LuminarAI, 5CD-AI MetaMathQA-vi (``query_vi`` column only).
toanmath.com-alpha is NOT embedded: its ``label`` column is only the document slug (see X5 report).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("x5-embed")
vol = modal.Volume.from_name("x5-embed", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.22.2", "pyarrow==26.0.1", "numpy==2.4.6",
                 "huggingface_hub==2.4.0", "sentencepiece==0.2.3")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("vi_math_verified_aug")
)

HF = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{path}"
CORPORA = [
    # name, repo, parquet path(s), loader
    ("vinhnq29", "vinhnq29/vietnamese-elementary-math",
     ["default/vigsm8k_test/0000.parquet", "default/vigsm8k_train/0000.parquet",
      "default/vimathqa_train_v1/0000.parquet", "default/vimathqa_test_v1/0000.parquet"]),
    ("hllj_mcq", "hllj/vi_grade_school_math_mcq", ["default/train/0000.parquet"]),
    ("hllj_crawl", "hllj/vi_math_problem_crawl", ["default/train/0000.parquet"]),
    ("metamathqa_vi", "5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated", ["default/train/0000.parquet"]),
    # English side of the same file: the 15k GSM8K/MATH-train seeds (cross-lingual Vi↔EN check, τ = 0.80)
    ("metamathqa_en", "5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated", ["default/train/0000.parquet"]),
    ("luminarai", "LuminarAI/vietnamese-ms-hs-textbook-math-300K", ["default/train/0000.parquet"]),
]
# Embedding units (why not whole rows):
#  - metamathqa_vi: the 15,013 unique ``original_question_vi`` (the GSM8K/MATH-train seeds every one of the 395k
#    rephrased/SV/FOBAR rows derives from); a test item close to a rephrasing is close to its seed.
#  - luminarai: rows are whole montoan.com.vn pages (median 9.2 kB; a 256-token window would embed only the
#    marketing intro), so we embed the problem-statement segments from ``decontam.problem_segments``.


def _rows(name: str, repo: str, paths: list[str]):
    """Yield (doc_id, embed_text) with the same id scheme as decontam.py's lexical scan (metamath: HF row order)."""
    import urllib.request

    import pyarrow.parquet as pq
    for path in paths:
        local = Path("/tmp") / (repo.replace("/", "__") + "__" + path.replace("/", "_"))
        if not local.exists():
            urllib.request.urlretrieve(HF.format(repo=repo, path=path), local)
        pf = pq.ParquetFile(local)
        split = path.split("/")[1]
        i = 0
        cols = {"metamathqa_vi": ["original_question_vi"], "metamathqa_en": ["original_question_en"], "luminarai": ["question"], "vinhnq29": ["question"],
                "hllj_crawl": ["contents", "id", "title"], "hllj_mcq": ["problems", "id"]}[name]
        for rb in pf.iter_batches(batch_size=4096, columns=cols):
            for r in rb.to_pylist():
                if name == "vinhnq29":
                    yield f"vinhnq29/{split}/{i}", r["question"] or ""
                elif name == "luminarai":
                    from vi_math_verified_aug.eval.decontam import problem_segments
                    for k, seg in enumerate(problem_segments(r["question"] or "")):
                        yield f"luminarai/{i}#s{k}", seg
                elif name == "metamathqa_vi":
                    yield f"metamathqa_vi/orig/{i}", r["original_question_vi"] or ""
                elif name == "metamathqa_en":
                    yield f"metamathqa_en/orig/{i}", r["original_question_en"] or ""
                elif name == "hllj_crawl":
                    c = r.get("contents") or ""
                    c = "\n".join(map(str, c)) if isinstance(c, (list, tuple)) else str(c)
                    yield f"hllj_crawl/{r.get('id', i)}", (c[:1500] or r.get("title") or "")
                elif name == "hllj_mcq":
                    for k, pr in enumerate(r.get("problems") or []):
                        q = pr.get("question") if isinstance(pr, dict) else str(pr)
                        yield f"hllj_mcq/{r.get('id', i)}/{k}", str(q or "")
                i += 1


@app.function(gpu=["L4", "A10G"], image=image, timeout=840, retries=0, volumes={"/out": vol}, memory=8192, cpu=2)
def run(inputs_json: bytes, model: str = "BAAI/bge-m3", max_length: int = 256, batch: int = 256,
        deadline_s: int = 1650, corpora: list[str] | None = None, calib_small: bool = True,
        do_calib: bool = True, tag: str = "run") -> dict:
    import subprocess

    import numpy as np
    import torch

    from vi_math_verified_aug.eval import decontam as dc

    t_start = time.time()
    out = Path("/out") / tag
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a", encoding="utf-8")

    def say(*a):
        msg = f"[{time.time() - t_start:7.1f}s] " + " ".join(map(str, a))
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    import transformers
    say(f"ENVCHECK gpu={smi} torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"transformers={transformers.__version__} numpy={np.__version__} model={model}")
    inp = json.loads(inputs_json)
    targets = inp["targets"]
    summary: dict = {"model": model, "max_length": max_length, "batch": batch, "env": smi, "stages": {}}

    def save_summary():
        (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        vol.commit()

    # ---- stage 0: encoder load
    enc = dc.Encoder(model, batch_size=batch, max_length=max_length, dtype="bf16")
    say("encoder loaded")

    # ---- stage 1: calibration (budget 240 s)
    t0 = time.time()
    calib = {} if not do_calib else {k: dc.calibrate_pairs(enc, a, b) for k, (a, b) in inp["calib"].items()}
    summary["calibration"] = {model: calib}
    summary["stages"]["calibration_s"] = round(time.time() - t0, 1)
    say("calibration", json.dumps({k: (v["tau_fpr_random"], v["recall_at_tau_random"], v["top1_acc"])
                                   for k, v in calib.items()}))
    save_summary()
    if calib_small and do_calib:
        t0 = time.time()
        small = dc.Encoder("intfloat/multilingual-e5-small", batch_size=batch, max_length=max_length, dtype="bf16")
        summary["calibration"]["intfloat/multilingual-e5-small"] = {
            k: dc.calibrate_pairs(small, a, b) for k, (a, b) in inp["calib"].items()}
        del small
        torch.cuda.empty_cache()
        summary["stages"]["calibration_e5small_s"] = round(time.time() - t0, 1)
        save_summary()

    # ---- stage 2: target embeddings
    t0 = time.time()
    T = enc.encode([t["text"] for t in targets])
    np.save(out / "target_emb.npy", T)
    (out / "target_ids.json").write_text(json.dumps([[t["set"], t["id"]] for t in targets]), encoding="utf-8")
    summary["stages"]["targets_s"] = round(time.time() - t0, 1)
    say(f"targets embedded: {T.shape}")
    save_summary()

    # ---- stage 3: corpora, small → large; each saved when done
    for name, repo, paths in CORPORA:
        if corpora and name not in corpora:
            continue
        if time.time() - t_start > deadline_s:
            say(f"deadline reached before {name}")
            summary["stages"][f"{name}_skipped"] = True
            break
        t0 = time.time()
        ix = dc.EmbeddingIndex(T, k=3)
        ids, texts, keep_text = [], [], {}
        n = 0
        hist = np.zeros(101, dtype=np.int64)            # histogram of corpus-row max cosine (0.00..1.00)
        seen = set()
        partial = False

        def flush():
            nonlocal ids, texts
            if not texts:
                return
            E = enc.encode(texts)
            pos = np.arange(n - len(texts), n)
            rmax = ix.update(E, pos)
            np.add.at(hist, np.clip((rmax * 100).round().astype(int), 0, 100), 1)
            top = set(ix.best_id.ravel().tolist())
            for p_, i_, t_ in zip(pos.tolist(), ids, texts):
                if p_ in top:
                    keep_text[p_] = (i_, t_[:600])
            ids, texts = [], []

        for doc_id, text in _rows(name, repo, paths):
            if name in ("metamathqa_vi", "metamathqa_en", "luminarai"):    # embed each distinct text once (first id kept)
                h = hash(text)
                if h in seen:
                    continue
                seen.add(h)
            ids.append(doc_id)
            texts.append(text)
            n += 1
            if len(texts) >= 16384:
                flush()
                if time.time() - t_start > deadline_s:
                    partial = True
                    say(f"{name}: deadline hit after {n} docs (partial)")
                    break
        flush()
        res = []
        for i, t in enumerate(targets):
            res.append({"set": t["set"], "id": t["id"],
                        "cos": [round(float(x), 4) for x in ix.best_val[i]],
                        "doc": [keep_text.get(int(p), (None,))[0] for p in ix.best_id[i]]})
        with open(out / f"embed_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"snippets_{name}.jsonl", "w", encoding="utf-8") as f:
            for p_, (i_, t_) in keep_text.items():
                f.write(json.dumps({"id": i_, "text": t_}, ensure_ascii=False) + "\n")
        summary["stages"][name] = {"n_docs_embedded": n, "partial": partial, "seconds": round(time.time() - t0, 1),
                                   "docs_per_s": round(n / max(time.time() - t0, 1e-6), 1),
                                   "corpus_rowmax_hist": hist.tolist()}
        say(f"{name}: {n} docs in {time.time() - t0:.0f}s")
        save_summary()
    summary["total_s"] = round(time.time() - t_start, 1)
    summary["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    save_summary()
    return summary


@app.local_entrypoint()
def main(inputs: str = "data/contam/embed_inputs/inputs.json", model: str = "BAAI/bge-m3", batch: int = 256,
         max_length: int = 256, deadline_s: int = 1650, corpora: str = "", calib_small: bool = True,
         do_calib: bool = True, tag: str = "run"):
    data = Path(inputs).read_bytes()
    s = run.remote(data, model=model, batch=batch, max_length=max_length, deadline_s=deadline_s,
                   corpora=[c for c in corpora.split(",") if c] or None, calib_small=calib_small,
                   do_calib=do_calib, tag=tag)
    os.makedirs("data/contam/embed_modal", exist_ok=True)
    Path(f"data/contam/embed_modal/summary_return_{tag}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "calibration"}, indent=1)[:4000])
