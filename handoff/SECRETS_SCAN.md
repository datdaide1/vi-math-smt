# Secrets scan (paths only, never values)

_Generated 2026-11-08 09:35 by `handoff/tools/secrets_scan.py`. Scanned 8,217 files; skipped 70 binary/oversize files; 16 `.env` secret values searched verbatim._

Excluded by design: `.env*`, `.venv/`, `E:\venv_*`, `.git/`. `EXACT_.env_value` = a real secret from `.env` appears verbatim in the file (must be fixed before the file leaves the laptop); shape rules can be dummy test values (check by eye, without printing them).

| Path | Rule | Lines | Tracked by git |
|---|---|---|---|
| `data/vi_exam/raw/loigiaihay.com/` (341 files) | generic_assignment | 341 | no |
| `data/vi_exam/raw/loigiaihay.com/` (341 files) | google_api_key | 341 | no |
| `logs/api/S3/claude.jsonl` | google_api_key | 1 | no |

**Summary:** 683 (path, rule) hits; 0 in git-tracked files; 0 verbatim `.env` secrets.
