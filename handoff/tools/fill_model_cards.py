"""Fill handoff/MODEL_CARD_TEMPLATE.md: one card per trained adapter (and one for the evaluated base model).

    .venv/Scripts/python.exe handoff/tools/fill_model_cards.py                       # T2 mode (default when the state exists)
    .venv/Scripts/python.exe handoff/tools/fill_model_cards.py --state none --outputs jobs/outputs   # generic mode

T2 mode (the 2026-11-08 tranche): runs = `jobs/queue/t2_state.json["runs"]` whose training manifest exists at
`jobs/outputs/t2-modal/<run_id>/manifest.json`; per-dataset eval metrics come from the state's `eval_metrics` /
`base_metrics` (item-level preds: `jobs/outputs/t2-eval/<eval_run_id>.jsonl`); adapters live on the Modal Volume
`t1r-eval` at `/train/<run_id>/adapters/epoch3`.
Generic mode: every `manifest.json` under --outputs; `kind: eval` manifests are attached to the run they evaluated
(`eval_of`), with a sibling `metrics.json` taking precedence.
Missing fields render as `n/a`; nothing is invented. Writes `<out>/<run_id>.md` and `<out>/INDEX.md`. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "handoff" / "MODEL_CARD_TEMPLATE.md"
VOLUME = "t1r-eval"

ARMS = {
    "base": "no fine-tuning (base model; 4-shot rows, and 0-shot rows in the arms' own template)",
    "S0": "seed problems only: the translated seed set with its upstream solutions (VI: Vi-GSM8K train; EN: GSM8K train "
          "originals), n-matched subsample; the anchor arm",
    "Y+": "symbolic arm: typed-IR mutation + render, shared solution writer, verified (trace = program answer, common "
          "Haiku-v3 screen, blind Qwen3-32B-AWQ cycle check, decontamination)",
    "Y-": "symbolic arm, unverified (wave-1 first item per seed, format/leak/decontam gates only)",
    "N+": "LLM-native arm: Claude Haiku 5.5 `native_generator.v3`, shared solution writer, verified (trace = generator "
          "answer, common Haiku-v3 screen, decontamination)",
    "N-": "LLM-native arm, unverified (first slot per seed, parse/format/decontam gates only)",
}
DATASHEETS = {"S0": "`handoff/DATASHEETS/seed_pool_v1.md` (S0 section)", "base": "n/a (no training)"}
for _a in ("Y+", "Y-", "N+", "N-"):
    DATASHEETS[_a] = "`handoff/DATASHEETS/train_pools.md`"
_CLAUDE = ("problems and solutions were written by Anthropic Claude models (`claude-haiku-5-5` generator/renderer/"
           "solution writer/screen; ids per stage in the datasheet; every call logged under `logs/api/<WS>/claude.jsonl`). "
           "Model ids, prompt hashes and API-log locations per stage: `handoff/DATASHEETS/README.md`.")
PROVENANCE = {
    "base": "n/a",
    "S0": "no LLM-generated text from this project: VI = the legacy Vietnamese translation of GSM8K train "
          "(`data/translation/gsm8k/gsm8k_train_vietnamese.jsonl`; translator Gemini 2.5 Flash per 06 §4.3, UNVERIFIED), "
          "EN = the GSM8K originals; built by `train/build_s0.py`.",
    **{a: _CLAUDE for a in ("Y+", "Y-", "N+", "N-")},
}
T2_DECODING = ("primary sets: K = 4 samples (T 0.6, top-p 0.95, top-k 20, one seed per request) + greedy, cap 1,024 "
               "tokens; Vi-ExamMath Core VERIFIED: greedy only (deviation from T5, cost); vLLM 0.33 bf16 on Modal L40S, "
               "`VLLM_BATCH_INVARIANT=1`; 0-shot in the training template (base rows: 4-shot)")


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d or d[p] is None:
            return default
        d = d[p]
    return d


def _na(v):
    return "n/a" if v is None or v == "" else v


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def _ci(c):
    return "" if not c else f" [{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def _rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ----------------------------------------------------------------------------------------------- generic mode loader
def load_runs(outputs: Path):
    runs, evals = {}, []
    for mf in sorted(outputs.rglob("manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(m, dict) or "run_id" not in m:
            continue
        m["_path"] = mf
        met = mf.with_name("metrics.json")
        if met.exists():
            try:
                mm = json.loads(met.read_text(encoding="utf-8"))
                m["_metrics"] = mm.get("per_dataset") or mm.get("metrics") or mm
                m["_metrics_path"] = met
            except json.JSONDecodeError:
                pass
        if m.get("kind") == "eval":
            evals.append(m)
        else:
            runs[m["run_id"]] = m
    for e in evals:  # base-model evals become their own card
        if e.get("arm") == "base":
            key = e.get("eval_of") or e["run_id"]
            runs.setdefault(key, {"run_id": key, "kind": "base", "arm": "base", "model": e.get("model"),
                                  "size": e.get("size"), "seed": e.get("seed"), "status": e.get("status"),
                                  "_path": e["_path"], "_synthetic": True})
    attach = {}
    for e in evals:
        target = e.get("eval_of")
        if target not in runs:
            ap = str(e.get("adapter_path") or "")
            target = next((rid for rid in runs if rid and rid in ap), target)
        attach.setdefault(target, []).append(e)
    return runs, attach


# --------------------------------------------------------------------------------------------------- T2 mode loader
def load_t2(state_path: Path, outputs: Path):
    st = json.loads(state_path.read_text(encoding="utf-8"))
    runs, attach = {}, {}
    for rid, r in (st.get("runs") or {}).items():
        mf = outputs / "t2-modal" / rid / "manifest.json"
        if not mf.exists():
            continue  # not trained (e.g. HOLD rows)
        m = json.loads(mf.read_text(encoding="utf-8"))
        m["_path"] = mf
        m["_state"] = r
        runs[rid] = m
        attach[rid] = [{"run_id": evid, "_state_metrics": met, "_preds": outputs / "t2-eval" / f"{evid}.jsonl"}
                       for evid, met in sorted((r.get("eval_metrics") or {}).items())]
    base = st.get("base_metrics") or {}
    for model, evids in (st.get("base_rows") or {}).items():
        rid = "base-" + model.split("/")[-1].lower().replace("-base", "") + "-4shot"
        runs[rid] = {"run_id": rid, "kind": "base", "arm": "base", "model": model, "size": "0.6B", "seed": 0,
                     "status": "COMPLETE" if all(e in base for e in evids) else "PARTIAL", "_path": state_path,
                     "_synthetic": True}
        attach[rid] = [{"run_id": e, "_state_metrics": base[e], "_preds": outputs / "t2-eval" / f"{e}.jsonl"}
                       for e in evids if e in base]
    # base rows in the arms' own 0-shot template (jobs/t2/ev-base-*-0shot-*.metrics.json, T2 §5.5)
    for mf in sorted((state_path.parent.parent / "t2").glob("ev-base-*-0shot-*.metrics.json")):
        met = json.loads(mf.read_text(encoding="utf-8"))
        stem = mf.name[: -len(".metrics.json")]                      # ev-base-qwen3-0.6b-0shot-vi
        rid = stem[len("ev-"):].rsplit("-", 1)[0]                   # base-qwen3-0.6b-0shot
        model = "Qwen/Qwen3-0.6B-Base" if "qwen3-0.6b" in stem else stem
        runs[rid] = {"run_id": rid, "kind": "base", "arm": "base", "model": model, "size": "0.6B", "seed": 0,
                     "status": "COMPLETE", "_path": mf, "_synthetic": True, "_shots": 0}
        attach.setdefault(rid, []).extend(
            {"run_id": e, "_state_metrics": m, "_preds": outputs / "t2-eval" / f"{e}.jsonl"} for e, m in sorted(met.items()))
    return runs, attach


def eval_table(evs):
    if not evs:
        return "_No evaluation attached yet (see KNOWN_ISSUES: evaluation pending)._", "n/a", "n/a"
    if any("_state_metrics" in e for e in evs):
        rows = ["| Eval run | Dataset | n | pass@1 over K=4 [95 % CI] | greedy [95 % CI] | item-level preds |",
                "|---|---|---|---|---|---|"]
        for e in evs:
            for ds, v in sorted(e["_state_metrics"].items()):
                rows.append(f"| `{e['run_id']}` | {ds} | {_na(v.get('n'))} | {_pct(v.get('pass1'))}{_ci(v.get('pass1_ci'))} | "
                            f"{_pct(v.get('greedy'))}{_ci(v.get('greedy_ci'))} | `{_rel(e['_preds'])}` |")
        return "\n".join(rows), "3.0.2 (math-verify 0.9.0 backend)", T2_DECODING
    rows = ["| Eval run | Dataset | n | mean pass@1 (K) | maj@K | greedy acc | trunc. (greedy) | no-answer |",
            "|---|---|---|---|---|---|---|---|"]
    grader = decoding = None
    for e in evs:
        grader = grader or e.get("grader")
        d = e.get("decoding") or {}
        decoding = decoding or (f"T={d.get('temperature')}, top-p {d.get('top_p')}, top-k {d.get('top_k')}, "
                                f"K={d.get('k_samples')} + greedy, max_new_tokens {d.get('max_new_tokens')}, "
                                f"{d.get('shots', 0)}-shot; backend {e.get('backend')} {e.get('precision', '')} "
                                f"on {_get(e, 'info', 'gpu', default='?')}")
        met = e.get("_metrics") or e.get("metrics") or {}
        k = d.get("k_samples", 4)
        for ds, v in sorted(met.items()):
            if not isinstance(v, dict):
                continue
            rows.append(f"| `{e['run_id']}` | {ds} | {_na(v.get('n_items'))} | {_na(v.get('mean_pass1'))} | "
                        f"{_na(v.get(f'maj@{k}'))} | {_na(v.get('greedy_acc'))} | "
                        f"{_na(v.get('truncation_rate_greedy'))} | {_na(v.get('no_answer_rate'))} |")
    return "\n".join(rows), _na(grader), _na(decoding)


def _adapter_location(run: dict) -> str:
    av = _get(run, "_state", "adapter_vol") or _get(run, "summary", "final_adapter")
    if not av:
        return "n/a"
    if "{vol}" in str(av) or str(av).startswith("/vol/"):
        p = str(av).replace("{vol}", "").replace("/vol", "", 1)
        local = ROOT / "jobs" / "outputs" / "t2-modal" / run["run_id"] / "adapters" / "epoch3" / "adapter_model.safetensors"
        here = (f"local copy `{_rel(local.parent)}/` (in the `runs` bundle; sha256 in `handoff/ADAPTERS.sha256`); "
                if local.exists() else "")
        return (f"{here}Modal Volume `{VOLUME}`, path `{p}` (pull: `modal volume get {VOLUME} {p} "
                f"jobs/outputs/t2-modal/{run['run_id']}/adapters/epoch3`; see `handoff/BUNDLE.md` §Adapters)")
    return f"`{av}`"


def _lang(run: dict) -> str:
    if run.get("kind") == "base":
        return "vi, en"
    rid = run["run_id"]
    if "-en-" in rid or rid.endswith("-en") or "/en/" in str(_get(run, "data_stats", "train_path", default="")) \
            or "_en." in str(_get(run, "data_stats", "train_path", default="")):
        return "en"
    return "vi"


def render(run: dict, evs: list, template: str) -> str:
    ds = run.get("data_stats") or {}
    sm = run.get("summary") or {}
    env = run.get("env") or {}
    arm = run.get("arm") or "?"
    table, grader, decoding = eval_table(evs)
    plan = run.get("plan")
    lang = _lang(run)
    rid = run["run_id"]
    sft = run.get("kind") in ("sft", "grpo")
    tp = ds.get("train_path")
    if tp and str(tp).startswith("/vol/train_data/"):
        tp = "data/" + str(tp)[len("/vol/train_data/"):]
    lora = run.get("lora") or {}
    vals = {
        "license": "apache-2.0",
        "language": lang,
        "library_name": "peft" if sft else "transformers",
        "base_model": _na(run.get("model")),
        "base_model_relation": "adapter" if sft else "n/a (base model)",
        "datasets": "" if arm == "base" else f"vi-math-smt/{arm}-{lang}",
        "arm_tag": f"arm-{arm}",
        "title": (f"{run.get('model', '?')} — base model, {'0-shot in the arms template' if run.get('_shots') == 0 else '4-shot'} ({rid})"
                  if arm == "base" else f"{run.get('model', '?')} — arm {arm}, {lang.upper()}, seed {run.get('seed')} ({rid})"),
        "manifest_path": _rel(run["_path"]),
        "metrics_note": "" if not evs else (" plus `jobs/queue/t2_state.json` (eval metrics)" if any(
            "_state_metrics" in e for e in evs) else " plus " + ", ".join(
            f"`{_rel(e.get('_metrics_path', e['_path']))}`" for e in evs)),
        "status": _na(run.get("status")),
        "run_id": rid, "kind": _na(run.get("kind")), "config_hash": _na(run.get("config_hash")),
        "size": _na(run.get("size")), "n_params": f"{run['n_params']:,}" if run.get("n_params") else "n/a",
        "arm": arm, "arm_description": ARMS.get(arm, "see handoff/README.md (arm table)"),
        "seed": _na(run.get("seed")),
        "lora_modules": ", ".join(lora.get("suffixes") or lora.get("modules") or []) or "n/a",
        "trainable_params": f"{run['trainable_params']:,}" if run.get("trainable_params") else "n/a",
        "adapter_location": _adapter_location(run),
        "train_path": _na(tp), "datasheet": DATASHEETS.get(arm, "TODO"),
        "n_items": _na(ds.get("n_items")), "n_pool": _na(ds.get("n_pool")),
        "n_dropped": _na(ds.get("n_dropped_too_long")), "max_length": _na(ds.get("max_length")),
        "fingerprint": _na(ds.get("fingerprint")),
        "provenance": PROVENANCE.get(arm, "TODO: record generator models and API log location"),
        "engine": _na(run.get("engine")), "packed": _na(run.get("packed")),
        "precision": _na(run.get("precision")), "platform": _na(run.get("platform")),
        "micro_batch": _na(run.get("micro_batch")), "accum": _na(run.get("accum")),
        "grad_ckpt": _na(run.get("grad_ckpt")),
        "plan": "n/a" if not plan else (f"{plan.get('total')} steps ({plan.get('steps_per_epoch')}/epoch × "
                                        f"{plan.get('epochs')} epochs), warm-up {plan.get('warmup')}, "
                                        f"decay {plan.get('decay')} (WSD); hedge branches at epochs "
                                        f"{[b.get('epoch') for b in plan.get('branches') or []]} (stored, not reported)"),
        "steps": _na(sm.get("exec_steps")), "skipped": _na(sm.get("skipped_steps")),
        "skipped_frac": _na(sm.get("skipped_frac")), "nonfinite": _na(sm.get("nonfinite_losses")),
        "dev_loss": _na(sm.get("dev_loss_final")), "tok_s": _na(sm.get("real_tokens_per_s")),
        "peak_mem": _na(sm.get("peak_mem_gb")),
        "env": ", ".join(f"{k} {v}" for k, v in env.items() if k in
                         ("python", "torch", "transformers", "peft", "trl", "vllm", "gpu", "cuda") and v) or "n/a",
        "code": (f"git `{_get(run, 'code', 'git_sha', default='?')[:12]}` "
                 f"(dirty: {_get(run, 'code', 'dirty', default='?')})") if run.get("code") else "n/a",
        "started": _na(run.get("started_utc")), "updated": _na(run.get("updated_utc")),
        "grader": grader, "decoding": decoding, "eval_table": table,
        "reproduce": (f"# config: {_rel(Path(run['_path']).with_name('config.yaml'))}\n"
                      "python -m vi_math_verified_aug.train.launcher ...   # cell runner; see handoff/RUNBOOK.md") if sft else
                     "python -m vi_math_verified_aug.train.launcher ...   # base rows; see handoff/RUNBOOK.md",
    }
    out = template
    for k, v in vals.items():
        out = out.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{(\w+)\}\}", out)
    if left:
        raise KeyError(f"unfilled template fields: {sorted(set(left))}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "jobs" / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "handoff" / "MODEL_CARDS"))
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--state", default=None, help="t2_state.json (default: jobs/queue/t2_state.json if present; "
                                                  "'none' = generic mode)")
    ap.add_argument("--include-smoke", action="store_true", help="generic mode: also card smoke runs")
    a = ap.parse_args(argv)
    template = Path(a.template).read_text(encoding="utf-8")
    state = Path(a.state) if a.state and a.state != "none" else (ROOT / "jobs" / "queue" / "t2_state.json")
    if a.state != "none" and state.exists():
        runs, attach = load_t2(state, Path(a.outputs))
        mode = f"T2 mode from `{_rel(state)}`"
    else:
        runs, attach = load_runs(Path(a.outputs))
        mode = "generic mode (all manifests)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()  # cards are generated; stale cards must not survive a regeneration
    idx = ["# Model cards index", "",
           f"_Generated by `handoff/tools/fill_model_cards.py` ({mode}); re-run after new runs land._", "",
           "| Run | Arm | Lang | Base | Seed | Status | Evals | Card |", "|---|---|---|---|---|---|---|---|"]
    n = 0
    for rid, run in sorted(runs.items()):
        if rid.startswith("smoke") and not a.include_smoke:
            continue
        evs = attach.get(rid, [])
        (out / f"{rid}.md").write_text(render(run, evs, template), encoding="utf-8")
        idx.append(f"| `{rid}` | {run.get('arm')} | {_lang(run)} | {run.get('model')} | {run.get('seed')} | "
                   f"{run.get('status')} | {len(evs)} | [{rid}.md]({rid}.md) |")
        n += 1
    (out / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"{n} cards ({mode}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
