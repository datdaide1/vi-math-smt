"""common/prompts.py (registry + loader), common/config.py (typed run configs, run_id), common/hashing.py."""
from __future__ import annotations

import json
import math

import pytest

from vi_math_verified_aug.common.config import Budget, ConfigError, RunConfig, load_config, write_resolved
from vi_math_verified_aug.common.hashing import canonical_json, stable_hash
from vi_math_verified_aug.common.llm import LLMRequest
from vi_math_verified_aug.common.prompts import (PromptError, PromptNotReady, PromptRegistry, load_file)

ROLES_REQUIRED = {"formalizer", "editor", "native_generator", "solution_writer", "validity_judge", "answer_judge"}

ACTIVE = """---
id: toy_judge
version: 2
role: validity_judge
language: vi
status: active
inputs: [problem, seed]
description: toy
---
### SYSTEM
Bạn là giám khảo. JSON: {"verdict": "valid"} — giữ nguyên ngoặc {} và \\frac{a}{b}.
### USER
Bài gốc: <<seed>>
Bài mới: <<problem>>
"""


# ------------------------------------------------------------------------------------------------- prompts
def test_repo_registry_has_all_role_templates():
    reg = PromptRegistry()
    roles = {p.role for p in reg.all()}
    assert ROLES_REQUIRED <= roles
    for p in reg.all():
        assert p.language == "vi" and p.version >= 1 and len(p.sha256) == 64
        if p.status == "template":
            with pytest.raises(PromptNotReady):
                p.render(**{k: "x" for k in p.inputs})
            r = p.render(allow_template=True, **{k: "x" for k in p.inputs})
            assert r.prompt_id == p.id


def test_render_active_prompt_strict(tmp_path):
    f = tmp_path / "toy_judge.v2.md"
    f.write_text(ACTIVE, encoding="utf-8")
    p = PromptRegistry(tmp_path).get("toy_judge")
    r = p.render(problem="P mới", seed="S cũ")
    assert "Bài gốc: S cũ" in r.user and "Bài mới: P mới" in r.user
    assert "{\"verdict\": \"valid\"}" in r.system and "\\frac{a}{b}" in r.system     # braces untouched
    with pytest.raises(PromptError):
        p.render(problem="x")                                  # missing input
    with pytest.raises(PromptError):
        p.render(problem="x", seed="y", extra="z")             # undeclared input
    req = LLMRequest.from_prompt(r, "claude-haiku-5-5", max_tokens=10)
    assert (req.prompt_id, req.prompt_version, req.prompt_sha256) == ("toy_judge", 2, p.sha256)
    assert req.system == r.system and req.messages == [{"role": "user", "content": r.user}]


def test_sha_is_line_ending_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "toy_judge.v2.md").write_bytes(ACTIVE.encode("utf-8"))
    (b / "toy_judge.v2.md").write_bytes(ACTIVE.replace("\n", "\r\n").encode("utf-8"))
    assert load_file(a / "toy_judge.v2.md").sha256 == load_file(b / "toy_judge.v2.md").sha256


@pytest.mark.parametrize("name,text", [
    ("toy_judge.v3.md", ACTIVE),                                               # version != file name
    ("toy_judge.v2.md", ACTIVE.replace("inputs: [problem, seed]", "inputs: [problem]")),   # undeclared <<seed>>
    ("toy_judge.v2.md", ACTIVE.replace("role: validity_judge", "role: poet")),
    ("toy_judge.v2.md", ACTIVE.replace("### USER\n", "")),
    ("toy_judge.v2.md", ACTIVE.replace("inputs: [problem, seed]", "inputs: [problem, seed, unused]")),
])
def test_bad_prompt_files_rejected(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    with pytest.raises(PromptError):
        load_file(tmp_path / name)


def test_latest_and_pinned_versions(tmp_path):
    (tmp_path / "toy_judge.v2.md").write_text(ACTIVE, encoding="utf-8")
    (tmp_path / "toy_judge.v3.md").write_text(ACTIVE.replace("version: 2", "version: 3"), encoding="utf-8")
    reg = PromptRegistry(tmp_path)
    assert reg.get("toy_judge").version == 3 and reg.get("toy_judge", 2).version == 2
    with pytest.raises(PromptError):
        reg.get("toy_judge", 9)


# ------------------------------------------------------------------------------------------------- hashing
def test_canonical_json_stable():
    assert canonical_json({"b": 1, "a": [1, (2, 3)]}) == canonical_json({"a": [1, [2, 3]], "b": 1})
    assert stable_hash({"x": "Việt"}) == stable_hash({"x": "Việt"})
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


# ------------------------------------------------------------------------------------------------- config
def _cfg(**kw):
    base = dict(arm="N+", lang="vi", model="Qwen/Qwen3-1.7B", seed=1, size=10000, kind="sft")
    base.update(kw)
    return RunConfig(**base)


def test_run_id_deterministic_and_budget_independent():
    a, b = _cfg(), _cfg(budget=Budget(usd=5.0), notes="x")
    assert a.run_id() == b.run_id() and a.job_id() == b.job_id()
    assert a.run_id().startswith("sft-Np-vi-qwen3-1.7b-n10000-s1-")
    assert a.run_id() != _cfg(seed=2).run_id() != _cfg(arm="N-").run_id()
    assert _cfg(id_extra={"bench": "vi-exammath"}).job_id() != a.job_id()
    assert _cfg().job_id() == "{}".format(a.job_id())                        # pure function of the fields


@pytest.mark.parametrize("bad", [dict(lang="fr"), dict(kind="train"), dict(arm="a b"), dict(seed="1"),
                                 dict(size=-1), dict(model=""), dict(budget_usd=-1)])
def test_config_validation(bad):
    with pytest.raises(ConfigError):
        _cfg(**bad) if "budget_usd" not in bad else Budget(usd=bad["budget_usd"])


def test_load_toml_json_and_resolved(tmp_path):
    t = tmp_path / "c.toml"
    t.write_text('arm = "Y-"\nlang = "en"\nmodel = "gemma-4-e2b"\nseed = 3\nsize = 500\nkind = "eval"\n'
                 '[budget]\nusd = 2.5\n', encoding="utf-8")
    c = load_config(t)
    assert c.budget.usd == 2.5 and c.arm == "Y-"
    j = tmp_path / "c.json"
    j.write_text(json.dumps({**c.to_spec(), "run_id": "ignored"}), encoding="utf-8")
    assert load_config(j) == c                                                # spec names (base/n_train) accepted
    out = json.loads(write_resolved(c, tmp_path / "run").read_text(encoding="utf-8"))
    assert out["run_id"] == c.run_id() and out["job_id"] == c.job_id()
    j.write_text(json.dumps({**c.to_dict(), "typo": 1}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(j)
