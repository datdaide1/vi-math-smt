"""Build handoff/PROMPTS.md: every prompt file with id, version, role, language, status, sha256, and how often the
API ledgers used it (and whether the logged prompt_sha256 matches the file today).

    .venv/Scripts/python.exe handoff/tools/prompt_catalog.py

Hashes for `prompts/*.md` come from `vi_math_verified_aug.common.prompts.load_file` (LF-normalised text, the same
hash the ledger records). Legacy G1 prompts (`pilot_gen/prompts/*.txt`) are hashed as raw bytes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def ledger_usage(root: Path):
    """(prompt_id, version) -> Counter(sha256 -> calls), and per-WS counts. Streams every apilog-format ledger."""
    use = defaultdict(Counter)
    ws_use = defaultdict(Counter)
    for f in sorted((root / "logs" / "api").glob("*/*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if '"prompt_id"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = r.get("prompt_id")
                if not pid:
                    continue
                key = (pid, r.get("prompt_version"))
                use[key][r.get("prompt_sha256")] += 1
                ws_use[key][f.parent.name] += 1
    return use, ws_use


def main() -> None:
    from vi_math_verified_aug.common.prompts import load_file

    use, ws_use = ledger_usage(ROOT)
    rows = []
    for p in sorted((ROOT / "prompts").glob("*.v*.md")):
        pr = load_file(p)
        rows.append(pr)
    L = ["# PROMPTS — catalog of versioned prompts", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `handoff/tools/prompt_catalog.py`. Do not hand-edit; re-run._", "",
         "Registry rules (`prompts/README.md`): one file per version `<id>.v<k>.md`; never edit a version that has a "
         "logged call, copy to `.v<k+1>.md` instead. `sha256` = hash of the LF-normalised file "
         "(`common/prompts.py`), which is what every `common/llm.py` ledger line records as `prompt_sha256`.", "",
         "Columns: **Calls** = ledger lines in `logs/api/*/*.jsonl` with this `prompt_id` + `prompt_version`; "
         "**Hash check** = whether every logged `prompt_sha256` equals the file hash today "
         "(✗ means the file was edited after use, so the logged calls used a different text).", "",
         "| id | v | role | lang | status | sha256 (first 16) | Calls (by WS) | Hash check | Description |",
         "|---|---|---|---|---|---|---|---|---|"]
    drift = []
    for pr in rows:
        key = (pr.id, pr.version)
        shas = use.get(key, Counter())
        n = sum(shas.values())
        if not n:
            chk = "unused"
        elif set(shas) == {pr.sha256}:
            chk = "✓"
        else:
            bad = sum(c for s, c in shas.items() if s != pr.sha256)
            chk = f"✗ {bad}/{n} differ"
            drift.append((pr.id, pr.version, bad, n))
        by_ws = ", ".join(f"{w} {c}" for w, c in sorted(ws_use.get(key, {}).items()))
        desc = str(pr.meta.get("description", "") if hasattr(pr, "meta") else "").replace("|", "/")[:110]
        L.append(f"| `{pr.id}` | {pr.version} | {pr.role} | {pr.language} | {getattr(pr, 'status', '')} | "
                 f"`{pr.sha256[:16]}` | {n:,}{' (' + by_ws + ')' if by_ws else ''} | {chk} | {desc} |")
    known = {(pr.id, pr.version) for pr in rows}
    orphans = [(k, sum(v.values())) for k, v in use.items() if k not in known]
    L += ["", "Full sha256 values: `handoff/PROMPTS.sha256` (same order, `sha256  path`).", ""]
    if orphans:
        L += ["## Ledger prompt ids with no file in `prompts/`", "",
              "| prompt_id | version | Calls |", "|---|---|---|"]
        L += [f"| `{k[0]}` | {k[1]} | {n:,} |" for k, n in sorted(orphans, key=str)]
        L.append("")
    if drift:
        L += ["## Hash drift (prompt edited after it was used)", ""]
        L += [f"- `{i}` v{v}: {b} of {n} logged calls carry a different sha256 than the file." for i, v, b, n in drift]
        L.append("")
    legacy = sorted((ROOT / "vi_math_verified_aug" / "pilot_gen" / "prompts").glob("*.txt"))
    L += ["## Legacy G1 pilot prompts (pre-registry; plain text, raw-byte sha256)", "",
          "Used only by G1 (`exec/G1_generator_pilot.md`); G2/G3 ported the winners verbatim into the registry "
          "(`validity_judge.v2` = `judge_v2`, `formalizer.v2` = `formalize_v1`, `solution_writer.v2` = `solve_v1`).", "",
          "| File | sha256 (first 16) |", "|---|---|"]
    sums = []
    for f in legacy:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        L.append(f"| `{f.relative_to(ROOT).as_posix()}` | `{h[:16]}` |")
        sums.append(f"{h}  {f.relative_to(ROOT).as_posix()}")
    (ROOT / "handoff" / "PROMPTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    full = [f"{pr.sha256}  prompts/{Path(pr.path).name}" if hasattr(pr, "path") else f"{pr.sha256}  {pr.id}.v{pr.version}"
            for pr in rows] + sums
    (ROOT / "handoff" / "PROMPTS.sha256").write_text("\n".join(full) + "\n", encoding="utf-8")
    print(f"{len(rows)} registry prompts, {len(legacy)} legacy, {len(orphans)} orphan ids, {len(drift)} drifted")


if __name__ == "__main__":
    main()
