"""Versioned prompt files: '### SYSTEM' / '### USER' sections, str.format placeholders."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def load_prompt(rel: str) -> tuple[str, str]:
    txt = (PKG / rel).read_text(encoding="utf-8")
    sys_part, user_part = txt.split("### USER", 1)
    sys_part = sys_part.replace("### SYSTEM", "", 1).strip()
    return sys_part, user_part.strip()


def build_messages(rel: str, **kw) -> list:
    s, u = load_prompt(rel)
    return [{"role": "system", "content": s.format(**kw)}, {"role": "user", "content": u.format(**kw)}]
