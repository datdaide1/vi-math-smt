"""Versioned prompt registry: top-level ``prompts/<id>.v<version>.md``.

File format (the section markers and the ``<<name>>`` placeholders are the ones G1's pilot prompts already use,
so G1 winners port by adding front matter)::

    ---
    id: validity_judge
    version: 1
    role: validity_judge
    language: vi
    status: active            # template | active | retired
    inputs: [seed, operator, problem]
    description: one line
    source: where the text came from (e.g. "G1 winner pilot_gen/prompts/judge_v2.txt")
    ---
    ### SYSTEM
    ...
    ### USER
    ... <<problem>> ...

Rules: never edit a released version in place — copy to ``.v<k+1>.md`` and bump ``version``. ``sha256`` is
taken over the file text with line endings normalised to LF, so a Windows checkout (CRLF) and a Linux checkout
hash the same; the API ledger records it next to ``prompt_id`` / ``prompt_version``.
Rendering is strict: every declared input must be supplied, nothing undeclared may be supplied, and every
``<<placeholder>>`` must be declared. ``status: template`` prompts refuse to render (contents pending).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import prompts_dir

ROLES = ("formalizer", "editor", "native_generator", "solution_writer", "validity_judge", "answer_judge",
         "solver", "translator", "other")
LANGUAGES = ("vi", "en")
STATUSES = ("template", "active", "retired")
REQUIRED = ("id", "version", "role", "language", "status")

_FNAME = re.compile(r"^(?P<id>[a-z0-9_]+)\.v(?P<v>\d+)\.md$")
_PH = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)>>")
_SECTION = re.compile(r"^###\s*(SYSTEM|USER)\s*$", re.I | re.M)


class PromptError(ValueError):
    pass


class PromptNotReady(PromptError):
    pass


def _scalar(v: str) -> Any:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x) for x in v[1:-1].split(",") if x.strip()]
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none", "~", ""):
        return None
    return v


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Minimal YAML subset: ``key: value`` lines, inline ``[a, b]`` lists, quoted strings, ints, bools."""
    if not text.startswith("---\n"):
        raise PromptError("missing front matter (file must start with '---')")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise PromptError("unterminated front matter")
    meta: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise PromptError(f"bad front-matter line: {line!r}")
        k, _, v = line.partition(":")
        if " #" in v and not v.strip().startswith(("'", '"')):
            v = v.split(" #", 1)[0]
        meta[k.strip()] = _scalar(v)
    return meta, text[end + 5:]


def split_sections(body: str) -> tuple[str, str]:
    marks = list(_SECTION.finditer(body))
    if [m.group(1).upper() for m in marks] != ["SYSTEM", "USER"]:
        raise PromptError("body must contain exactly '### SYSTEM' then '### USER'")
    system = body[marks[0].end():marks[1].start()].strip("\n")
    user = body[marks[1].end():].strip("\n")
    return system.strip(), user.strip()


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    version: int
    sha256: str
    role: str
    language: str
    system: str
    user: str


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    role: str
    language: str
    status: str
    inputs: tuple[str, ...]
    system: str
    user: str
    sha256: str
    path: str
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def placeholders(self) -> set[str]:
        return set(_PH.findall(self.system)) | set(_PH.findall(self.user))

    def render(self, *, allow_template: bool = False, **values: Any) -> RenderedPrompt:
        if self.status == "template" and not allow_template:
            raise PromptNotReady(f"{self.id}.v{self.version} is an empty template (status: template)")
        if self.status == "retired":
            raise PromptNotReady(f"{self.id}.v{self.version} is retired")
        need, got = set(self.inputs), set(values)
        if need - got:
            raise PromptError(f"{self.id}.v{self.version}: missing inputs {sorted(need - got)}")
        if got - need:
            raise PromptError(f"{self.id}.v{self.version}: undeclared inputs {sorted(got - need)}")

        def sub(t: str) -> str:
            return _PH.sub(lambda m: str(values[m.group(1)]), t)

        return RenderedPrompt(self.id, self.version, self.sha256, self.role, self.language, sub(self.system),
                              sub(self.user))


def load_file(path: str | os.PathLike) -> Prompt:
    p = Path(path)
    raw = p.read_bytes().decode("utf-8")
    text = raw.replace("\r\n", "\n").lstrip("﻿")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta, body = parse_front_matter(text)
    missing = [k for k in REQUIRED if meta.get(k) in (None, "")]
    if missing:
        raise PromptError(f"{p.name}: missing front-matter fields {missing}")
    m = _FNAME.match(p.name)
    if not m or m.group("id") != meta["id"] or int(m.group("v")) != meta["version"]:
        raise PromptError(f"{p.name}: file name must be '<id>.v<version>.md' matching the front matter")
    if meta["role"] not in ROLES:
        raise PromptError(f"{p.name}: role {meta['role']!r} not in {ROLES}")
    if meta["language"] not in LANGUAGES:
        raise PromptError(f"{p.name}: language {meta['language']!r} not in {LANGUAGES}")
    if meta["status"] not in STATUSES:
        raise PromptError(f"{p.name}: status {meta['status']!r} not in {STATUSES}")
    inputs = meta.get("inputs") or []
    if not isinstance(inputs, list):
        raise PromptError(f"{p.name}: inputs must be a [list]")
    system, user = split_sections(body)
    pr = Prompt(meta["id"], int(meta["version"]), meta["role"], meta["language"], meta["status"],
                tuple(str(x) for x in inputs), system, user, sha, str(p), meta)
    undeclared = pr.placeholders - set(pr.inputs)
    if undeclared:
        raise PromptError(f"{p.name}: placeholders not declared in inputs: {sorted(undeclared)}")
    if pr.status == "active":
        unused = set(pr.inputs) - pr.placeholders
        if unused:
            raise PromptError(f"{p.name}: declared inputs never used: {sorted(unused)}")
        if not pr.user:
            raise PromptError(f"{p.name}: active prompt with an empty USER section")
    return pr


class PromptRegistry:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else prompts_dir()
        self._by_key: dict[tuple[str, int], Prompt] = {}
        for f in sorted(self.root.glob("*.md")):
            if f.name.lower() == "readme.md":
                continue
            pr = load_file(f)
            k = (pr.id, pr.version)
            if k in self._by_key:
                raise PromptError(f"duplicate prompt {pr.id}.v{pr.version}")
            self._by_key[k] = pr

    def get(self, prompt_id: str, version: int | None = None) -> Prompt:
        """``version=None`` = highest version (convenient while iterating; pin it in run configs)."""
        if version is not None:
            try:
                return self._by_key[(prompt_id, int(version))]
            except KeyError:
                raise PromptError(f"no prompt {prompt_id}.v{version} in {self.root}") from None
        vs = [v for (i, v) in self._by_key if i == prompt_id]
        if not vs:
            raise PromptError(f"no prompt {prompt_id!r} in {self.root}")
        return self._by_key[(prompt_id, max(vs))]

    def all(self) -> list[Prompt]:
        return [self._by_key[k] for k in sorted(self._by_key)]

    def by_role(self, role: str, language: str | None = None) -> list[Prompt]:
        return [p for p in self.all() if p.role == role and (language is None or p.language == language)]

    def index(self) -> list[dict[str, Any]]:
        return [{"id": p.id, "version": p.version, "role": p.role, "language": p.language, "status": p.status,
                 "sha256": p.sha256} for p in self.all()]


_default: PromptRegistry | None = None


def load_prompt(prompt_id: str, version: int | None = None) -> Prompt:
    global _default
    if _default is None:
        _default = PromptRegistry()
    return _default.get(prompt_id, version)
