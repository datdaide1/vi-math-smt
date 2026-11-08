"""D1 decontamination gate for generated N items (X5 G4 rule, T1 §2.3), applied to N+ and N- alike.

Targets: S3's ``seeds.decontam.targets()`` = every ``data/eval/*.jsonl`` item (v1.1, VI and EN) + Vi-ExamMath
(``data/vi_exam/items.jsonl``). Nothing is sent to an LLM; embeddings run on Modal.

Reject an item if any of (X5 G4 thresholds):
* lexical: >= 1 shared 13-gram with a target item (exact or diacritic-stripped variant), or >= 0.5 coverage of a
  target item's number-templated 13-grams (a value-perturbed copy);
* bge-m3 cosine >= 0.85 vs a same-language target, or >= 0.80 vs a cross-language target (VI<->EN).
Within-arm near-duplicates (X5 G5, MinHash 5-gram J >= 0.8 vs an earlier item in rank order) are rejected too.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import ROOT, WORK, Store, read_jsonl

COS_SAME = 0.85
COS_X = 0.80
MODAL = Path(r"E:\venv_tools\Scripts\modal.exe")


def targets() -> list[dict]:
    from ..seeds.decontam import targets as s3_targets
    return s3_targets()


def lexical(lang: str, items: list[tuple[str, str]], T: list[dict]) -> dict[str, dict]:
    """items = [(key, text)] in rank order. Returns {key: {g4_ngram, g5_dup}}."""
    from ..eval.decontam import MinHasher, MinHashLSH, NgramIndex, token_hashes, variant_tokens
    ix = {v: NgramIndex(13) for v in ("exact", "nodiac", "template")}
    for i, t in enumerate(T):
        vt = variant_tokens(t["text"])
        for v in ix:
            ix[v].add(i, token_hashes(vt[v]))
    for v in ix.values():
        v.finalize()
    mh, lsh = MinHasher(), MinHashLSH(bands=16)
    out = {}
    for n, (key, text) in enumerate(items):
        vt = variant_tokens(text)
        hit = None
        for v, x in ix.items():
            res = x.query(token_hashes(vt[v]))
            if v == "template":
                res = {i: c for i, c in res.items() if c / max(x.n_grams[i], 1) >= 0.5}
            if res:
                i = max(res, key=res.get)
                hit = {"variant": v, "tid": T[i]["tid"], "shared": int(res[i]),
                       "cov": round(res[i] / max(x.n_grams[i], 1), 3)}
                break
        dup = None
        sig = mh.signature(token_hashes(vt["exact"]))
        if sig is not None:
            q = lsh.query(sig, 0.8)
            if q:
                dup = {"dup_of": items[q[0][0]][0], "j": round(float(q[0][1]), 3)}
            else:
                lsh.insert(n, sig)
        out[key] = {"g4_ngram": hit, "g5_dup": dup}
    return out


def write_inputs(path: Path, items: list[dict], T: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items, "targets": [{"tid": t["tid"], "lang": t["lang"], "text": t["text"]}
                                                            for t in T]}, ensure_ascii=False), encoding="utf-8")
    return path


def run_modal(inputs: Path, out: Path) -> dict:
    cmd = [str(MODAL), "run", "-m", "vi_math_verified_aug.scale_gen.embed_modal", "--inputs", str(inputs),
           "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = WORK / "logs" / f"modal_{out.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""), encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"modal run failed ({p.returncode}); see {log}")
    return json.loads(out.read_text(encoding="utf-8"))


def apply(lang: str, items: list[tuple[str, str]], tag: str, *, embed: bool = True) -> Store:
    """Lexical + embedding gate for ``items`` (rank order) -> ``<WORK>/<lang>/decontam.jsonl`` rows."""
    st = Store(WORK / lang / "decontam.jsonl")
    T = targets()
    lex = lexical(lang, items, T)
    cos = {}
    if embed:
        todo = [{"id": k, "lang": lang, "text": t} for k, t in items]
        inp = write_inputs(WORK / "embed" / f"inputs_{tag}.json", todo, T)
        res = run_modal(inp, WORK / "embed" / f"result_{tag}.json")
        for i, k in enumerate(res["ids"]):
            cos[k] = {"vi": (res["top1"]["vi"]["cos"][i], res["top1"]["vi"]["tid"][i]),
                      "en": (res["top1"]["en"]["cos"][i], res["top1"]["en"]["tid"][i])}
    other = "en" if lang == "vi" else "vi"
    for k, _ in items:
        lx = lex[k]
        c = cos.get(k)
        hit_cos = None
        if c:
            if c[lang][0] >= COS_SAME:
                hit_cos = {"pair": f"{lang}>{lang}", "cos": c[lang][0], "tid": c[lang][1]}
            elif c[other][0] >= COS_X:
                hit_cos = {"pair": f"{lang}>{other}", "cos": c[other][0], "tid": c[other][1]}
        st.put(k, {"g4_ngram": lx["g4_ngram"], "g5_dup": lx["g5_dup"], "g4_cos": hit_cos,
                   "cos_same": c[lang][0] if c else None, "cos_x": c[other][0] if c else None,
                   "embedded": bool(c),
                   "pass": not (lx["g4_ngram"] or lx["g5_dup"] or hit_cos) and (bool(c) or not embed)})
    return st


def load_items(path: Path) -> list[dict]:
    return read_jsonl(path)
