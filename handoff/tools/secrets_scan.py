"""Scan the working tree for anything that looks like an API key or token. Prints PATHS ONLY (plus the rule name,
match count and whether git tracks the file) -- never the matched value.

    .venv/Scripts/python.exe handoff/tools/secrets_scan.py [--out handoff/SECRETS_SCAN.md]

Two kinds of rule:
  * shape rules: provider key formats (Anthropic, Google, NVIDIA, HF, GitHub, OpenAI, AWS, Kaggle, Modal, PEM keys,
    generic `api_key = "<long literal>"` assignments);
  * exact rule: every secret-looking value in `.env` / `.env.bak` (names *KEY*, *TOKEN*, *SECRET*, *PASSWORD*, *_ID
    for the workspace id) is searched verbatim. Values are held in memory only and never printed or written.
Skipped: .git/, .venv/, venvs, `.env*` themselves, binaries (*.pt, *.safetensors, *.bin, *.sqlite*, *.zip, images),
files > 300 MB. Lines are streamed (constant memory).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".uv-cache"}
SKIP_EXT = {".pt", ".safetensors", ".bin", ".sqlite", ".sqlite-wal", ".sqlite-shm", ".zip", ".png", ".jpg", ".jpeg",
            ".gif", ".pdf", ".parquet", ".npy", ".npz", ".pyc", ".whl", ".gz", ".zst", ".tar", ".ico", ".webp"}
RULES = {
    "anthropic_key": re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "google_api_key": re.compile(rb"AIza[0-9A-Za-z_\-]{35}"),
    "nvidia_key": re.compile(rb"nvapi-[A-Za-z0-9_\-]{30,}"),
    "hf_token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(rb"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "kaggle_token": re.compile(rb"\bKGAT_[A-Za-z0-9]{20,}|\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "modal_token": re.compile(rb"\b(?:ak|as)-[A-Za-z0-9]{20,}\b"),
    "private_key_pem": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----"),
    "generic_assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|secret|token|password)\b[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_\-\.]{24,}[\"']"),
}


def env_values() -> list[bytes]:
    vals = []
    for name in (".env", ".env.bak", ".env.local"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip().strip("\"'")
            if re.search(r"KEY|TOKEN|SECRET|PASSWORD|WORKSPACE_ID", k, re.I):
                for part in re.split(r"[,\s;]+", v):  # pools like GEMINI_API_KEYS=a,b,c
                    if len(part) >= 16:
                        vals.append(part.encode())
    return vals


def tracked() -> set[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
        return set(out.decode("utf-8", "replace").split("\0"))
    except Exception:
        return set()


def walk():
    for dp, dns, fns in os.walk(ROOT):
        rel = Path(dp).relative_to(ROOT)
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith("venv")]
        for f in fns:
            if f.startswith(".env"):
                continue
            p = Path(dp) / f
            yield p, (rel / f).as_posix()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "handoff" / "SECRETS_SCAN.md"))
    a = ap.parse_args()
    exact = env_values()
    git = tracked()
    hits, skipped, scanned = {}, 0, 0
    for p, rel in walk():
        if p.suffix.lower() in SKIP_EXT:
            skipped += 1
            continue
        try:
            if p.stat().st_size > 300 * 2**20:
                skipped += 1
                continue
            scanned += 1
            with open(p, "rb") as fh:
                for line in fh:
                    for name, rx in RULES.items():
                        if rx.search(line):
                            hits.setdefault((rel, name), 0)
                            hits[(rel, name)] += 1
                    for v in exact:
                        if v in line:
                            hits.setdefault((rel, "EXACT_.env_value"), 0)
                            hits[(rel, "EXACT_.env_value")] += 1
        except OSError:
            skipped += 1
    L = ["# Secrets scan (paths only, never values)", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `handoff/tools/secrets_scan.py`. Scanned {scanned:,} files; "
         f"skipped {skipped:,} binary/oversize files; {len(exact)} `.env` secret values searched verbatim._", "",
         "Excluded by design: `.env*`, `.venv/`, `E:\\venv_*`, `.git/`. `EXACT_.env_value` = a real secret from `.env` "
         "appears verbatim in the file (must be fixed before the file leaves the laptop); shape rules can be dummy test "
         "values (check by eye, without printing them).", "",
         "| Path | Rule | Lines | Tracked by git |", "|---|---|---|---|"]
    # Collapse big groups (e.g. hundreds of cached HTML pages carrying the same site-embedded browser key) into one
    # row per (directory, rule); exact .env hits and git-tracked files are always listed one by one.
    groups = {}
    for (rel, name), n in hits.items():
        d = rel.rsplit("/", 1)[0] if "/" in rel else "."
        groups.setdefault((d, name), []).append((rel, n))
    for (d, name), items in sorted(groups.items(), key=lambda x: (x[0][1] != "EXACT_.env_value", x[0])):
        if len(items) > 10 and name != "EXACT_.env_value" and not any(r in git for r, _ in items):
            L.append(f"| `{d}/` ({len(items)} files) | {name} | {sum(n for _, n in items)} | no |")
            continue
        for rel, n in sorted(items):
            L.append(f"| `{rel}` | {name} | {n} | {'YES' if rel in git else 'no'} |")
    if not hits:
        L.append("| – | no hits | – | – |")
    n_git = sum(1 for (r, _) in hits if r in git)
    L += ["", f"**Summary:** {len(hits)} (path, rule) hits; {n_git} in git-tracked files; "
          f"{sum(1 for (_, n) in hits if n == 'EXACT_.env_value')} verbatim `.env` secrets."]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    ex = sum(1 for (r, n) in hits if n == "EXACT_.env_value")
    print(f"scanned {scanned} files, {len(hits)} (path, rule) hits, {ex} exact .env-value hits -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
