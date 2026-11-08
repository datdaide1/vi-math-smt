"""Canonical JSON and stable hashes.

``canonical_json`` is the single serialisation used for every identity hash in the project (request hash,
run_id, job_id): sorted keys, no whitespace, UTF-8 kept as is, floats via ``repr`` (Python's shortest
round-trip form, stable across platforms), tuples as lists. NaN/inf are rejected so a hash never depends
on a non-standard JSON token.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any


def _norm(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _norm(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(_norm(v) for v in o)
    return o


def canonical_json(obj: Any) -> str:
    return json.dumps(_norm(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(obj: Any, algo: str = "sha256") -> str:
    return hashlib.new(algo, canonical_json(obj).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
