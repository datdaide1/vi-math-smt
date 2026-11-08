"""Build the out-of-git delivery bundle(s): data/ (git-ignored), trained adapters, predictions, API ledgers.

    # 1) plan only (default): list every file that would go in, with bytes; no hashing, no zipping
    .venv/Scripts/python.exe handoff/tools/make_bundle.py
    # 2) plan + sha256 (streams files, constant memory; slow on ~1 GB)
    .venv/Scripts/python.exe handoff/tools/make_bundle.py --hash [--only data]
    # 3) build zips into OUT (hashes, writes MANIFEST.sha256 inside each zip, re-reads every member to verify)
    .venv/Scripts/python.exe handoff/tools/make_bundle.py --build --out D:/vimath_bundle [--only data,runs]

Bundles (BUNDLE.md explains each and who may receive it):
  data     frozen datasets: eval sets, Vi-ExamMath public items + QA, seed pool, training pools, contamination flags
  runs     run manifests/metrics/configs, final LoRA adapters, item-level predictions (no optimizer checkpoints)
  private  NEVER public: third-party solution text, raw scraped pages, API ledgers (prompts+responses incl. test items)

Writes handoff/BUNDLE_MANIFEST.tsv (bundle, path, bytes, sha256-or-"-"). Missing paths are reported, not fatal:
pools that are not frozen yet simply do not appear. Pure stdlib.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (bundle, root-relative directory or file, include globs, exclude globs). Globs match the path relative to ROOT.
SPEC = [
    # ---- data ----------------------------------------------------------------------------------------------------
    ("data", "data/eval", ["*"], ["data/eval/_work/*"]),                       # v1.1 frozen sets + _sources manifest
    ("data", "data/vi_exam", ["data/vi_exam/items.jsonl", "data/vi_exam/exams.jsonl", "data/vi_exam/summary.json",
                              "data/vi_exam/coverage.csv", "data/vi_exam/exam_index.jsonl",
                              "data/vi_exam/first_seen.jsonl", "data/vi_exam/official_sources.jsonl",
                              "data/vi_exam/qa_queue*.json*", "data/vi_exam/claude_keys.jsonl",
                              "data/vi_exam/core_v1*", "data/vi_exam/SHA256SUMS", "data/vi_exam/*.datacard.json"], []),
    ("data", "data/vi_exam_qa", ["*"], ["*.log"]),                             # Q1 adjudication -> core_v1
    ("data", "data/seeds", ["*"], ["data/seeds/_work/embed/*", "*.log"]),       # S3 pool_v1 (+ _work audit trail)
    ("data", "data/train", ["*"], []),                                          # Y/N x vi/en pools (when frozen)
    ("data", "data/contam", ["data/contam/vi_exam_flags.jsonl", "data/contam/scans/*", "data/contam/logs/*"],
     ["*corpus_snippets*"]),                                                    # snippets = third-party text -> private
    ("data", "data/pilot", ["*"], []),                                          # G1-G4 pilot items + judge labels
    ("data", "data/t1r", ["*"], []),                                            # S0 train/dev used by first runs
    ("data", "data/audit", ["*"], []),                                          # D5 gate audit, R1 tables, diversity
    ("data", "data/train/final", ["*"], []),                                    # explicit (also under data/train)
    # ---- runs ----------------------------------------------------------------------------------------------------
    # Everything under jobs/outputs (manifests, configs, train/dev logs, pre-flight, pack gates, post-run checks,
    # item-level predictions incl. jobs/outputs/t2-eval/ev-*.jsonl, GRPO smoke, G6 sweep) and the 25 tranche adapters
    # once pulled from the Modal Volume into jobs/outputs/t2-modal/<run_id>/adapters/epoch3 (BUNDLE.md §Adapters).
    ("runs", "jobs/outputs", ["*"],
     ["*/ckpt/*", "*optimizer.pt", "*/adapter.pt",                            # final adapters only, no optimizer state
      "jobs/outputs/t1r-smoke-*/*/adapters/*",                                 # smoke-test adapters: manifests only
      "jobs/outputs/t1r-runs-v1/*/adapters/*"]),                               # Kaggle PILOT adapters (not comparison rows)
    ("runs", "jobs/queue", ["*"], []),                                          # queue + launcher state
    ("runs", "bench/r_gpu/results", ["*"], []),                                 # R-GPU raw results + per-item preds
    # ---- private -------------------------------------------------------------------------------------------------
    ("private", "data/vi_exam", ["data/vi_exam/items_private.jsonl", "data/vi_exam/raw/*",
                                 "data/vi_exam/fetch_log.jsonl", "data/vi_exam/listings.jsonl"], []),
    ("private", "logs/api", ["*"], ["*.lock"]),
    ("private", "data/contam", ["*corpus_snippets*"], []),                     # third-party corpus excerpts
]


def _match(rel: str, pats) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, f"*/{p}") for p in pats)


def plan(only: set[str] | None):
    files, missing = [], []
    for bundle, base, inc, exc in SPEC:
        if only and bundle not in only:
            continue
        p = ROOT / base
        if not p.exists():
            missing.append((bundle, base))
            continue
        it = [p] if p.is_file() else (x for x in p.rglob("*") if x.is_file())
        for f in it:
            rel = f.relative_to(ROOT).as_posix()
            if (inc == ["*"] or _match(rel, inc)) and not _match(rel, exc):
                files.append((bundle, rel, f.stat().st_size))
    seen, uniq = set(), []
    for t in files:
        if (t[0], t[1]) not in seen:
            seen.add((t[0], t[1]))
            uniq.append(t)
    return sorted(uniq), missing


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(files, out: Path, tag: str):
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    for b, rel, size, digest in files:
        by.setdefault(b, []).append((rel, size, digest))
    for b, items in by.items():
        zp = out / f"vimath-{b}-{tag}.zip"
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
            for rel, _, _ in items:
                z.write(ROOT / rel, arcname=rel)
            z.writestr("MANIFEST.sha256", "".join(f"{d}  {rel}\n" for rel, _, d in items))
        with zipfile.ZipFile(zp) as z:  # verify by re-reading every member
            for rel, _, d in items:
                h = hashlib.sha256()
                with z.open(rel) as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != d:
                    raise SystemExit(f"verify failed: {zp.name}:{rel}")
        with open(out / f"{zp.name}.sha256", "w", encoding="utf-8", newline="\n") as fh:  # LF so `sha256sum -c` works
            fh.write(f"{sha256(zp)}  {zp.name}\n")
    with open(out / "MANIFEST.sha256", "w", encoding="utf-8", newline="\n") as fh:   # every zip in --out
        for z in sorted(out.glob("vimath-*.zip")):
            fh.write(f"{sha256(z)}  {z.name}\n")
        print(f"{zp}  {len(items)} files  {zp.stat().st_size / 2**20:.1f} MiB  verified")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma list of bundles: data,runs,private")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", help="output dir for zips (required with --build; keep it OUTSIDE the repo)")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d"))
    ap.add_argument("--manifest", default=str(ROOT / "handoff" / "BUNDLE_MANIFEST.tsv"))
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    files, missing = plan(only)
    do_hash = a.hash or a.build
    rows = [(b, rel, size, sha256(ROOT / rel) if do_hash else "-") for b, rel, size in files]
    with open(a.manifest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("bundle\tpath\tbytes\tsha256\n")
        for b, rel, size, d in rows:
            fh.write(f"{b}\t{rel}\t{size}\t{d}\n")
    tot = {}
    for b, _, size, _ in rows:
        n, s = tot.get(b, (0, 0))
        tot[b] = (n + 1, s + size)
    for b, (n, s) in sorted(tot.items()):
        print(f"{b:8s} {n:6d} files {s / 2**20:9.1f} MiB")
    for b, base in missing:
        print(f"{b:8s} MISSING {base} (not produced/frozen yet)")
    if a.build:
        if not a.out:
            ap.error("--build needs --out")
        out = Path(a.out).resolve()
        if ROOT in out.parents or out == ROOT:
            ap.error("--out must be outside the repository (zips must never be committed)")
        build(rows, out, a.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
