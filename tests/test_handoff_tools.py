"""D4 hand-off tools: model-card filler, spend ledger, bundle manifest. Tiny fixtures, stdlib only."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "handoff" / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fmc = _load("fill_model_cards")
ledger = _load("ledger")


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_cards_attach_evals_and_fill_every_field(tmp_path, monkeypatch):
    monkeypatch.setattr(fmc, "ROOT", tmp_path)
    out = tmp_path / "jobs" / "outputs"
    _w(out / "k1" / "sft-N+-0.6B-s1-aa" / "manifest.json", {
        "run_id": "sft-N+-0.6B-s1-aa", "kind": "sft", "arm": "N+", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B",
        "seed": 1, "status": "COMPLETE", "n_params": 596049920, "lora": {"suffixes": ["q_proj"]},
        "data_stats": {"n_items": 10000, "train_path": "/vol/data/train/N/vi/train_10k_plus.jsonl"},
        "summary": {"exec_steps": 625, "skipped_steps": 0, "dev_loss_final": 0.5}})
    _w(out / "ev" / "ev-1" / "manifest.json", {
        "run_id": "ev-1", "kind": "eval", "arm": "N+", "eval_of": "sft-N+-0.6B-s1-aa", "grader": "3.0.2",
        "decoding": {"k_samples": 4, "temperature": 0.6}})
    _w(out / "ev" / "ev-1" / "metrics.json", {"per_dataset": {"gsm8k_platinum": {"n_items": 600, "mean_pass1": 0.4,
                                                                                "maj@4": 0.45, "greedy_acc": 0.41}}})
    _w(out / "ev" / "ev-base" / "manifest.json", {"run_id": "ev-base", "kind": "eval", "arm": "base",
                                                   "eval_of": "base-qwen3-0.6b", "model": "Qwen/Qwen3-0.6B-Base"})
    cards = tmp_path / "cards"
    tpl = ROOT / "handoff" / "MODEL_CARD_TEMPLATE.md"
    assert fmc.main(["--outputs", str(out), "--out", str(cards), "--template", str(tpl)]) == 0
    card = (cards / "sft-N+-0.6B-s1-aa.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert "base_model_relation: adapter" in card and "library_name: peft" in card
    assert "| `ev-1` | gsm8k_platinum | 600 | 0.4 | 0.45 | 0.41 |" in card
    assert "written by Anthropic Claude models" in card  # Claude provenance is never dropped for N/Y arms
    assert (cards / "base-qwen3-0.6b.md").exists()
    idx = (cards / "INDEX.md").read_text(encoding="utf-8")
    assert "sft-N+-0.6B-s1-aa" in idx and "base-qwen3-0.6b" in idx


def test_cards_never_invent_missing_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(fmc, "ROOT", tmp_path)
    out = tmp_path / "o"
    _w(out / "r" / "manifest.json", {"run_id": "sft-Y-0.6B-s2-bb", "kind": "sft", "arm": "Y-", "model": "m"})
    cards = tmp_path / "c"
    fmc.main(["--outputs", str(out), "--out", str(cards), "--template", str(ROOT / "handoff" / "MODEL_CARD_TEMPLATE.md")])
    card = (cards / "sft-Y-0.6B-s2-bb.md").read_text(encoding="utf-8")
    assert "_No evaluation attached yet" in card and "Status of the run: **n/a**" in card


def test_cards_t2_mode_reads_state_metrics_and_volume_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(fmc, "ROOT", tmp_path)
    out = tmp_path / "jobs" / "outputs"
    rid = "t2-sft-Np-vi-0p6B-s0"
    _w(out / "t2-modal" / rid / "manifest.json", {
        "run_id": rid, "kind": "sft", "arm": "N+", "model": "Qwen/Qwen3-0.6B-Base", "size": "0.6B", "seed": 0,
        "status": "COMPLETE", "platform": "modal_l40s", "precision": "bf16",
        "data_stats": {"n_items": 1808, "train_path": "/vol/train_data/train/final/N/vi/train_plus.jsonl"}})
    state = tmp_path / "jobs" / "queue" / "t2_state.json"
    _w(state, {"runs": {
        rid: {"status": "TRAINED", "adapter_vol": "{vol}/train/" + rid + "/adapters/epoch3",
              "eval_metrics": {"ev-" + rid + "-primary": {"gsm8k_platinum": {
                  "n": 600, "pass1": 0.488, "pass1_ci": [0.45, 0.52], "greedy": 0.5, "greedy_ci": [0.46, 0.54]}}}},
        "t2-sft-Np-en-0p6B-s1": {"status": None}},   # HOLD row without a manifest: no card
        "base_rows": {"Qwen/Qwen3-0.6B-Base": ["ev-b"]},
        "base_metrics": {"ev-b": {"vi_exam_core_verified": {"n": 1367, "pass1": None, "greedy": 0.3424,
                                                            "greedy_ci": [0.3168, 0.3672]}}}})
    cards = tmp_path / "cards"
    fmc.main(["--outputs", str(out), "--out", str(cards), "--state", str(state),
              "--template", str(ROOT / "handoff" / "MODEL_CARD_TEMPLATE.md")])
    card = (cards / f"{rid}.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert "| gsm8k_platinum | 600 | 48.8 [45.0, 52.0] | 50.0 [46.0, 54.0] |" in card
    assert "modal volume get t1r-eval /train/" + rid + "/adapters/epoch3" in card
    assert "data/train/final/N/vi/train_plus.jsonl" in card
    assert not (cards / "t2-sft-Np-en-0p6B-s1.md").exists()
    base = (cards / "base-qwen3-0.6b-4shot.md").read_text(encoding="utf-8")
    assert "| vi_exam_core_verified | 1367 | n/a | 34.2 [31.7, 36.7] |" in base


def test_cards_t2_mode_adds_zero_shot_base_card(tmp_path, monkeypatch):
    monkeypatch.setattr(fmc, "ROOT", tmp_path)
    out = tmp_path / "jobs" / "outputs"
    state = tmp_path / "jobs" / "queue" / "t2_state.json"
    _w(state, {"runs": {}, "base_rows": {}, "base_metrics": {}})
    _w(tmp_path / "jobs" / "t2" / "ev-base-qwen3-0.6b-0shot-vi.metrics.json", {
        "ev-base-qwen3-0.6b-0shot-vi-exam-greedy": {"vi_exam_core_verified": {
            "n": 1367, "pass1": None, "greedy": 0.1807, "greedy_ci": [0.1602, 0.2004]}}})
    cards = tmp_path / "cards"
    fmc.main(["--outputs", str(out), "--out", str(cards), "--state", str(state),
              "--template", str(ROOT / "handoff" / "MODEL_CARD_TEMPLATE.md")])
    card = (cards / "base-qwen3-0.6b-0shot.md").read_text(encoding="utf-8")
    assert "0-shot in the arms template" in card
    assert "| vi_exam_core_verified | 1367 | n/a | 18.1 [16.0, 20.0] |" in card


def test_ledger_uses_logged_cost_and_recomputes_only_batch_results(tmp_path):
    api = tmp_path / "logs" / "api"
    (api / "G9").mkdir(parents=True)
    (api / "G9" / "claude.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"provider": "claude", "model_requested": "claude-opus-5-5", "usage": {"input_tokens": 10**6,
         "output_tokens": 0}, "cost_usd": 2.0, "ok": True, "status": 200},  # Batch-discounted cost logged: keep 2.0
        {"event": "batch_submit", "provider": "claude", "reserved_usd": 99.0},
        {"provider": "claude", "model_requested": "claude-haiku-5-5", "cost_usd": 0.0, "cache_hit": True},
    ]) + "\n", encoding="utf-8")
    (api / "X3").mkdir()
    (api / "X3" / "a_results.jsonl").write_text(json.dumps(
        {"model": "claude-opus-5-5", "usage": {"input_tokens": 10**6, "output_tokens": 10**6}}) + "\n", encoding="utf-8")
    (api / "X3" / "a_requests.jsonl").write_text(json.dumps({"model": "claude-opus-5-5"}) + "\n", encoding="utf-8")
    (api / "X2").mkdir()
    (api / "X2" / "cost_ledger.jsonl").write_text(json.dumps({"succeeded": 3, "cost_usd": 1.25}) + "\n", encoding="utf-8")
    (api / "X2" / "x_results.jsonl").write_text(json.dumps({"custom_id": "a", "usage": {"in": 9, "out": 9}}) + "\n",
                                                 encoding="utf-8")
    agg, _ = ledger.scan_api(tmp_path)
    by = {k: v["usd"] for k, v in agg.items()}
    assert by[("G9", "claude", "claude-opus-5-5")] == pytest.approx(2.0)
    assert by[("G9", "claude", "claude-haiku-5-5")] == 0.0
    assert by[("X3", "claude", "claude-opus-5-5")] == pytest.approx((4.0 + 20.0) * 0.5)  # batch rate
    assert by[("X2", "claude", "claude-opus-5-5")] == pytest.approx(1.25)
    assert sum(v["calls"] for k, v in agg.items() if k[0] == "G9") == 2  # batch_submit skipped
