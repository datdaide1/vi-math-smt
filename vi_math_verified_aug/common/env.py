"""Read secrets from ``.env`` + the process environment without ever printing them.

``read_env()`` returns a plain dict (process env wins over the file). ``secret_values()`` lists the values of
secret-looking variables so the API logger can scrub them from anything it writes (defence in depth: the
client never puts keys into log records in the first place).
"""
from __future__ import annotations

import os
from pathlib import Path

from .paths import REPO_ROOT

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def parse_dotenv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, _, v = line.partition("=")
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:  # inline comment on an unquoted value
            v = v.split(" #", 1)[0].rstrip()
        out[k.strip()] = v
    return out


def read_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    p = Path(path) if path else REPO_ROOT / ".env"
    env: dict[str, str] = {}
    if p.is_file():
        env.update(parse_dotenv(p.read_text(encoding="utf-8")))
    env.update({k: v for k, v in os.environ.items()})
    return env


def split_keys(value: str | None) -> list[str]:
    """``"k1, k2,,k3"`` -> ``["k1", "k2", "k3"]`` (order kept, duplicates dropped)."""
    seen, out = set(), []
    for k in (value or "").split(","):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def secret_values(env: dict[str, str]) -> list[str]:
    vals: list[str] = []
    for k, v in env.items():
        if any(m in k.upper() for m in SECRET_MARKERS) and v:
            vals.extend(x for x in split_keys(v) if len(x) >= 8)
    return sorted(set(vals), key=len, reverse=True)
