"""QA-T: decode real SFT labels with the REAL tokenizers on a Modal CPU container (laptop has no tokenizers).

    E:/venv_tools/Scripts/modal.exe run vi_math_verified_aug/qa/modal_tokcheck.py

Uses train/data.encode + train/guards.preflight_items exactly as sft.py does (torch-free), for Qwen3-0.6B-Base and
LFM2.5-2.6B-Base, on S0-VI, S0-EN, dev300 and the N-arm pilot traces.  Reports special tokens, BOS/EOS placement,
the decoded trained text, the label fraction, and per-source token-length quantiles (drop-rate proxy at 1024).
CPU only, no GPU, retries=0; cost ~ cents.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("transformers==5.20.2", "tokenizers", "jinja2", "pyyaml", "huggingface_hub")
         .add_local_dir(str(PKG), "/root/vi_math_verified_aug", ignore=["**/__pycache__/**"]))
app = modal.App("qa-t-tokcheck", image=image)


def _rows(path: Path, n: int | None = None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if n and len(out) >= n:
                    break
    return out


@app.function(cpu=2, memory=4096, timeout=1200, retries=0)
def check(payload: dict) -> dict:
    import sys
    sys.path.insert(0, "/root")
    from transformers import AutoTokenizer
    from vi_math_verified_aug.train import data as D, guards as G, template as T
    from vi_math_verified_aug.train.modeling import bos_prefix
    res = {}
    for model in ("Qwen/Qwen3-0.6B-Base", "LiquidAI/LFM2.5-2.6B-Base"):
        try:
            tok = AutoTokenizer.from_pretrained(model)
        except Exception as e:  # noqa: BLE001
            res[model] = {"error": repr(e)[:300]}
            continue
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        try:
            from transformers import GenerationConfig
            gc = GenerationConfig.from_pretrained(model)
            gen_eos = gc.eos_token_id
        except Exception as e:  # noqa: BLE001
            gen_eos = repr(e)[:100]
        r = {"bos_token": tok.bos_token, "bos_id": tok.bos_token_id, "eos_token": tok.eos_token,
             "eos_id": tok.eos_token_id, "pad_token": tok.pad_token, "pad_id": tok.pad_token_id,
             "generation_config_eos": gen_eos, "bos_prefix": bos_prefix(tok),
             "add_special_tokens_ids_of_x": tok("x")["input_ids"]}
        dec = lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)  # noqa
        for name, (rows, lang) in payload.items():
            fields = {"solution": ("solution", "trace")} if name.startswith("N_") else None
            items, stats = D.prepare_pool(rows, tok, 1024, fields, lang)
            for it in items:  # keep the source fields for the preflight, as SFTRun does
                pass
            g = dict(G.DEFAULTS, n_decode=2)
            try:
                pf = G.preflight_items(items, dec, prompt_fn=lambda q: T.prompt(q, lang),
                                       completion_fn=lambda s, a: T.completion(s, a, lang), eos_id=tok.eos_token_id,
                                       pad_id=tok.pad_token_id, g=g)
                pf_ok = True
            except G.GuardAbort as e:
                pf, pf_ok = {"abort": str(e)[:600]}, False
            it = items[0]
            first = next(i for i, x in enumerate(it["labels"]) if x != -100)
            # boundary check: joint tokenisation of prompt+completion vs the separate one used in training
            p = T.prompt(it["q"], lang)
            c = T.completion(it["sol"], it["ans"], lang)
            joint = tok(p + c, add_special_tokens=False)["input_ids"]
            sep = tok(p, add_special_tokens=False)["input_ids"] + tok(c, add_special_tokens=False)["input_ids"]
            comp = sorted(x["n_completion"] for x in items)
            tot = sorted(len(x["ids"]) for x in items)

            def q(xs, f):
                return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
            r[name] = {"stats": stats, "preflight_ok": pf_ok, "preflight": pf,
                       "first_ids": it["ids"][:4], "last_ids": it["ids"][-3:],
                       "first_trained_tok": dec([it["ids"][first]]), "prompt_tail": dec(it["ids"][max(0, first - 6):first]),
                       "boundary_joint_equals_separate": joint == sep,
                       "n_joint": len(joint), "n_sep": len(sep),
                       "completion_tokens_p50_p90_p99_max": [q(comp, .5), q(comp, .9), q(comp, .99), comp[-1]],
                       "sequence_tokens_p50_p90_p99_max": [q(tot, .5), q(tot, .9), q(tot, .99), tot[-1]],
                       "trained_text_example": dec([x for x in it["labels"] if x != -100])[-300:]}
        res[model] = r
    return res


@app.local_entrypoint()
def main():
    d = REPO / "data"
    n_rows = []
    probs = {r["key"]: r.get("problem") for r in _rows(d / "train/N/_work/vi/gen.jsonl")}
    import re
    box = re.compile(r"\\boxed\{([^{}]*)\}")
    for r in _rows(d / "train/N/_work/vi/solve.jsonl"):
        if r.get("trace") and r.get("boxed") and probs.get(r["key"]):
            # same as scale_gen.checks.clean_trace (BOX.sub -> group 1), inlined to avoid importing its deps
            n_rows.append({"id": r["key"], "question": probs[r["key"]], "trace": box.sub(lambda m: m.group(1),
                                                                                          r["trace"]).strip(),
                           "answer": r["boxed"]})
    payload = {"S0_vi": (_rows(d / "t1r/s0_train.jsonl", 1500), "vi"),
               "S0_en": (_rows(d / "t1r/s0_train_en.jsonl", 1500), "en"),
               "dev300_vi": (_rows(d / "t1r/dev300.jsonl"), "vi"),
               "N_pilot_vi": (n_rows, "vi")}
    out = check.remote(payload)
    p = REPO / "jobs" / "outputs" / "QA_T" / "tokcheck.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)
