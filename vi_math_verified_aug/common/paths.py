"""Default on-disk locations. Every path can be overridden by an env var (useful in tests / Kaggle)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    return Path(v) if v else default


def api_log_dir(ws: str) -> Path:
    """``logs/api/<WS>/`` — the API ledger of one workstream (git-ignored, released on HF later)."""
    return _p("VIMATH_API_LOG_DIR", REPO_ROOT / "logs" / "api") / ws


def cache_dir() -> Path:
    """``.cache/vimath/`` — response cache + spend accounting (git-ignored)."""
    return _p("VIMATH_CACHE_DIR", REPO_ROOT / ".cache" / "vimath")


def state_dir() -> Path:
    """``state/`` — orchestrator state such as the job ledger (git-ignored)."""
    return _p("VIMATH_STATE_DIR", REPO_ROOT / "state")


def prompts_dir() -> Path:
    """Top-level ``prompts/`` registry."""
    return _p("VIMATH_PROMPTS_DIR", REPO_ROOT / "prompts")
