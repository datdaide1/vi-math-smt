"""Typed run configs and deterministic ``run_id`` / ``job_id``.

``RunConfig`` holds the identity of a run (kind, arm, lang, model, size, seed, versions) plus non-identity
settings (budget, notes). Only identity fields enter the hash, so changing a budget cap does not create a
new run.

* ``job_id``  = sha1(canonical_json([kind, base, arm, n_train, seed, recipe_ver, data_ver, prompt_ver])),
  exactly the T6 §6.2 tuple (``base`` = ``model``, ``n_train`` = ``size``). When ``id_extra`` is non-empty
  it is appended as a 9th element (for kinds whose identity needs more, e.g. the benchmark of an eval job).
* ``run_id``  = ``<kind>-<arm>-<lang>-<model slug>-n<size>-s<seed>-<job_id[:10]>`` — readable and unique.

Configs load from ``.toml`` (stdlib ``tomllib``) or ``.json``; ``.yaml`` works if PyYAML happens to be
installed. ``write_resolved`` stores the fully resolved config (with ids) into a run directory.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hashing import canonical_json, stable_hash

JOB_KINDS = ("gen", "verify", "sft", "grpo", "eval", "upload")
LANGS = ("vi", "en")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Budget:
    usd: float | None = None          # hard cap for paid API spend (None = no cap)
    calls: int | None = None          # max API calls
    gpu_hours: float | None = None

    def __post_init__(self):
        for k in ("usd", "calls", "gpu_hours"):
            v = getattr(self, k)
            if v is not None and v < 0:
                raise ConfigError(f"budget.{k} must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    arm: str                           # e.g. S0, T, N+, Y-, M, A, B-V2 ... (free token, validated syntactically)
    lang: str                          # vi | en
    model: str                         # base/student model (sft/eval) or generator model (gen)
    seed: int
    size: int | None = None            # n_train items (None where it does not apply)
    kind: str = "sft"
    recipe_ver: str = "v1"
    data_ver: str = "v1"
    prompt_ver: str = "v1"
    id_extra: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    notes: str = ""

    def __post_init__(self):
        if self.kind not in JOB_KINDS:
            raise ConfigError(f"kind {self.kind!r} not in {JOB_KINDS}")
        if self.lang not in LANGS:
            raise ConfigError(f"lang {self.lang!r} not in {LANGS}")
        if not _TOKEN.match(self.arm or ""):
            raise ConfigError(f"bad arm token {self.arm!r}")
        if not self.model or any(c.isspace() for c in self.model):
            raise ConfigError(f"bad model {self.model!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError("seed must be an int")
        if self.size is not None and (isinstance(self.size, bool) or not isinstance(self.size, int)
                                      or self.size < 0):
            raise ConfigError("size must be a non-negative int or None")
        canonical_json(self.id_extra)  # raises on NaN / non-JSON values

    # ---------------------------------------------------------------- identity
    def identity(self) -> list[Any]:
        t = [self.kind, self.model, self.arm, self.size, self.seed, self.recipe_ver, self.data_ver,
             self.prompt_ver]
        if self.id_extra:
            t.append(self.id_extra)
        return t

    def job_id(self) -> str:
        return stable_hash(self.identity(), "sha1")

    def run_id(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", self.model.split("/")[-1]).strip("-").lower()
        arm = re.sub(r"-$", "m", self.arm.replace("+", "p"))          # N+ -> Np, N- -> Nm (readability only;
        arm = re.sub(r"[^A-Za-z0-9]+", "_", arm)                      # uniqueness comes from the hash)
        n = "na" if self.size is None else str(self.size)
        return f"{self.kind}-{arm}-{self.lang}-{slug}-n{n}-s{self.seed}-{self.job_id()[:10]}"

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_spec(self) -> dict[str, Any]:
        """The ``spec_json`` shape the job ledger expects (T6 names: base, n_train)."""
        d = self.to_dict()
        d["base"] = d.pop("model")
        d["n_train"] = d.pop("size")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = dict(d)
        d.pop("run_id", None)
        d.pop("job_id", None)
        if "base" in d and "model" not in d:
            d["model"] = d.pop("base")
        if "n_train" in d and "size" not in d:
            d["size"] = d.pop("n_train")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ConfigError(f"unknown config keys {sorted(unknown)}")
        b = d.get("budget")
        if isinstance(b, dict):
            d["budget"] = Budget(**b)
        elif b is None:
            d.pop("budget", None)
        return cls(**d)

    def resolved(self) -> dict[str, Any]:
        return {**self.to_dict(), "job_id": self.job_id(), "run_id": self.run_id()}


def load_config(path: str | os.PathLike) -> RunConfig:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".toml":
        import tomllib
        d = tomllib.loads(p.read_text(encoding="utf-8"))
    elif suf == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
    elif suf in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ConfigError("YAML configs need PyYAML; use .toml or .json") from e
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        raise ConfigError(f"unsupported config type {p.suffix}")
    return RunConfig.from_dict(d)


def write_resolved(cfg: RunConfig, run_dir: str | os.PathLike) -> Path:
    out = Path(run_dir) / "config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.resolved(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out
