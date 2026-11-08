"""One YAML file per run.  Every run carries its identity (kind, arm, model, size, seed) and its budget.

Example (``train/configs/sft_example.yaml``)::

    kind: sft                       # sft | grpo | eval
    arm: S0
    model: Qwen/Qwen3-1.7B-Base     # HF id, a local dir, or "tiny-random" (offline smoke model)
    size: 1.7B
    seed: 1                         # the ONE seed: LoRA init, data order and the training subsample
    budget: {n_train: 10000, epochs: 3, wall_s: 39600}
    data: {train: /kaggle/input/vimath-data/arms/S0.jsonl}
    sft: {lr: 2.0e-4, max_length: 1024, branch_epochs: [1, 2]}

Unknown keys are an error (a typo must not silently fall back to a default).  ``run_id`` defaults to
``<kind>-<arm>-<size>-s<seed>-<8 hex of the canonical config>`` so the same config always gets the same id.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

KINDS = ("sft", "grpo", "eval")

# Defaults = T3 §3.1 (SFT), T3 §4.1 (GRPO), T5 §4.1 (eval).  Changing one is a recorded design decision.
DEFAULTS: dict[str, Any] = {
    "kind": None,
    "run_id": None,
    "config_hash": None,               # filled by build(); ignored on input
    "arm": None,
    "model": None,
    "size": None,
    "seed": None,
    "lang": "vi",
    "precision": "fp16",               # fp16 (T4) | bf16 (L4/laptop parity) | fp32 (CPU smoke)
    "device": "auto",                  # auto | cuda | cpu
    "code": None,                      # {git_sha, dirty, paths}: stamped by the launcher (train/launcher.code_stamp)
    "platform": None,                  # kaggle_t4 | modal_l40s | modal_l4 ...: recorded per run (GLMM check covariate)
    "attn_implementation": "sdpa",
    "engine": "hf",                    # hf (plain transformers+peft) | unsloth (FastLanguageModel, T1r/O1; needs its gate)
    "out_dir": "runs/{run_id}",
    "resume": "auto",                  # auto | never | <checkpoint dir>
    "resume_dirs": [],                 # extra places to look for checkpoints (Kaggle relay dataset mount)
    "notes": "",
    "budget": {
        "n_train": None,               # items drawn (seeded) from the eligible pool; None = all eligible
        "epochs": 1,
        "max_steps": None,             # GRPO: optimiser steps; SFT: optional hard cap (smoke tests)
        "wall_s": None,                # per-task wall-clock budget; on expiry: checkpoint + exit 75
        "reserve_s": 300,              # stop this long before wall_s to leave time for the final save
    },
    "data": {
        "train": None,                 # JSONL path (SFT pool / GRPO prompt pool)
        "fields": {},                  # column remapping, e.g. {question: vi_question}
        "lang_filter": None,
    },
    "template": {
        "name": "vi_plain_v1",         # see template.py
        "fewshot_path": None,          # eval of base rows only (4-shot, same template)
    },
    "lora": {
        "r": 32,
        "alpha": 32,
        "dropout": 0.0,
        "targets": "all-linear",       # all Linear layers of the text decoder; never lm_head / vision / audio
        "init_adapter": None,          # GRPO: start adapter (if the SFT model is not merged)
    },
    "sft": {
        "lr": 2.0e-4,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        "seq_per_step": 32,
        "micro_batch": 8,              # halved automatically on OOM at the probe
        "max_length": 1024,            # T3: p99 of the pooled lengths; longer items are DROPPED, never truncated
        "warmup_frac": 0.03,
        "decay_frac": 0.10,            # WSD: linear decay to 0 over the last 10% of a (branch) run's steps
        "branch_epochs": [],           # e.g. [1, 2] in a 3-epoch sweep run -> epoch-1 and epoch-2 models
        "grad_clip": 1.0,
        "init_scale": 65536.0,
        "max_skip_frac": 0.01,         # abort if more than 1% of optimiser steps are skipped (fp16 overflow)
        "skip_check_after": 50,        # ... evaluated from this many steps on, and always at the end
        "grad_ckpt": "auto",           # auto (on for >= 1.5B params) | true | false
        "ckpt_every_min": 15,          # user rule: <= 15 min of work lost on any interruption (+ every epoch)
        "ckpt_every_steps": None,
        "keep_ckpts": 2,
        "length_grouping": "within_step",  # within_step: sort the step's 32 items by length, cut micro-batches
        "packing": False,              # True only with engine=unsloth: rows concatenated, position_ids reset,
                                       # packed_seq_lengths, use_cache=False, start-of-job logits gate (below)
        "pack_gate_max_dnll": 0.02,    # packed-vs-separate per-item NLL gate (O1 §3.1); abort the run on failure
        "pack_gate_min_argmax": 0.97,  # upper bound; the effective threshold is min(this, measured floor - margin)
        "pack_gate_argmax_margin": 0.02,  # floor = argmax agreement of separate vs padded rows, same precision
        "dev_path": None,              # held-out JSONL: completion-NLL (token mean) logged as dev_loss
        "dev_every_steps": None,       # also at step 0 and at the end of the main run
        "dev_max_items": 300,
        "act_monitor_every": 0,        # >0: max |Linear output| hooked on every Nth step (fp16 headroom; S2 §1)
        "log_every": 1,
    },
    "grpo": {
        "sft_model": None,             # merged fp16 SFT checkpoint (dir or HF id); defaults to `model`
        "reward": "math_verify",       # math_verify | random (Bernoulli(0.5) control)
        "random_reward_p": 0.5,
        "loss_type": "dr_grpo",
        "beta": 0.0,
        "scale_rewards": False,
        "num_generations": 8,
        "prompts_per_step": 32,
        "micro_batch": 16,             # completions per forward/backward; accumulation fills 32 x 8 = 256
        "max_steps": 200,
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "lr": 2.0e-5,
        "warmup_steps": 10,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "num_iterations": 1,
        "mask_truncated_completions": True,
        "lora_alpha": 64,              # T3 §4.1: fresh LoRA r=32, alpha=64
        "use_vllm": "auto",            # auto | true | false
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.35,
        "vllm_max_model_length": 1024,
        # precision: GRPO runs fp16 on every GPU (T3 §4.1, 2510.26788: bf16 LoRA-GRPO collapses); merge + vLLM fp16
        "save_steps": 25,
        "logging_steps": 1,
        "dev_eval_every": 25,
    },
    "guards": None,                    # filled from train/guards.DEFAULTS in build(); see guards.py
    "eval": {
        "source_run_id": None,         # the TRAINING run being evaluated (GLMM `run` effect); default: run_id
        "model_path": None,            # merged model dir / HF id; defaults to `model`
        "adapter_path": None,          # LoRA adapter on top of model_path (vLLM enable_lora / PEFT)
        "sets": [],                    # [{name, path, lang, fields}]
        "backend": "auto",             # auto (vllm if importable + CUDA) | vllm | hf
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "k_samples": 4,
        "greedy": True,
        "max_new_tokens": 1024,        # O1 lever 2 (2048 doubled the waste of runaway samples; T1r spot-check)
        "shots": 0,                    # 0 after SFT; 4 for base rows (same template)
        "batch_size": 64,              # HF backend prompts per generate() call
        "chunk_items": 256,            # items per write: predictions are appended to preds.jsonl per chunk
        "save_text": True,             # keep raw responses: re-score, never re-generate
        "max_items": None,             # smoke tests only
        "vllm_gpu_memory_utilization": 0.85,
        "vllm_max_model_len": 4096,
        "vllm_max_lora_rank": 32,
        "vllm_enable_prefix_caching": True,
        "dataset_from_row": "dataset",  # Y2 schema: language-neutral `dataset` taken from this row field if present
        "model_tag": None,             # Y2 `model` factor (e.g. qwen3-0.6b); default: derived from `model`
    },
}

ID_FIELDS = ("kind", "arm", "model", "size", "seed")


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ConfigError(f"unknown config key {path + k!r}")
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: dict) -> str:
    """8 hex of the canonical JSON of the config without run_id/out_dir/resume (pure run identity + recipe)."""
    ident = {k: v for k, v in cfg.items() if k not in ("run_id", "config_hash", "out_dir", "resume", "resume_dirs", "notes",
                                                       "platform", "code")}
    # the wall-clock budget differs per Kaggle session of the same run; it must not invalidate checkpoints
    ident["budget"] = {k: v for k, v in (ident.get("budget") or {}).items() if k not in ("wall_s", "reserve_s")}
    blob = json.dumps(ident, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def validate(cfg: dict) -> dict:
    if cfg["kind"] not in KINDS:
        raise ConfigError(f"kind must be one of {KINDS}, got {cfg['kind']!r}")
    for k in ID_FIELDS:
        if cfg.get(k) in (None, ""):
            raise ConfigError(f"missing required field {k!r} (every run carries kind/arm/model/size/seed)")
    if not isinstance(cfg["seed"], int):
        raise ConfigError("seed must be an int")
    if cfg["precision"] not in ("fp16", "bf16", "fp32"):
        raise ConfigError("precision must be fp16 | bf16 | fp32")
    if cfg["engine"] not in ("hf", "unsloth"):
        raise ConfigError("engine must be hf | unsloth")
    s = cfg["sft"]
    if s["packing"] and cfg["engine"] != "unsloth":
        raise ConfigError("sft.packing=true needs engine: unsloth.  Plain-HF packing is exact only with "
                          "use_cache=False (the W2 v1 leak was the DynamicCache, S2 §3 / O1 §0.3) but it is SLOWER "
                          "than padded length-grouped batches on T4 (937-2,167 vs 1,071-2,352 tok/s); the bf16 "
                          "flex-attention varlen path (O1 §3.1) is not implemented.")
    if s["length_grouping"] not in ("within_step", "none"):
        raise ConfigError("sft.length_grouping must be within_step | none")
    if s["seq_per_step"] < 1 or s["micro_batch"] < 1:
        raise ConfigError("seq_per_step and micro_batch must be >= 1")
    if not (0 <= s["warmup_frac"] < 1 and 0 < s["decay_frac"] <= 1):
        raise ConfigError("warmup_frac in [0,1), decay_frac in (0,1]")
    ep = cfg["budget"]["epochs"]
    if cfg["kind"] == "sft":
        if not cfg["data"]["train"]:
            raise ConfigError("sft needs data.train")
        if not isinstance(ep, int) or ep < 1:
            raise ConfigError("budget.epochs must be an int >= 1 for SFT")
        bad = [e for e in s["branch_epochs"] if not (isinstance(e, int) and 1 <= e < ep)]
        if bad:
            raise ConfigError(f"sft.branch_epochs must be ints in [1, epochs-1]; bad: {bad}")
    g = cfg["grpo"]
    if cfg["kind"] == "grpo" and cfg["precision"] != "fp16":
        raise ConfigError("GRPO runs fp16 only (T3 §4.1; bf16 LoRA-GRPO collapses, 2510.26788)")
    if cfg["kind"] == "grpo":
        if not cfg["data"]["train"]:
            raise ConfigError("grpo needs data.train (the fixed prompt pool)")
        if g["reward"] not in ("math_verify", "random"):
            raise ConfigError("grpo.reward must be math_verify | random")
        if (g["prompts_per_step"] * g["num_generations"]) % g["micro_batch"]:
            raise ConfigError("grpo.micro_batch must divide prompts_per_step * num_generations")
    e = cfg["eval"]
    if cfg["kind"] == "eval":
        if not e["sets"]:
            raise ConfigError("eval needs eval.sets")
        for st in e["sets"]:
            if not {"name", "path"} <= set(st):
                raise ConfigError(f"each eval set needs name and path: {st}")
        if e["backend"] not in ("auto", "vllm", "hf"):
            raise ConfigError("eval.backend must be auto | vllm | hf")
    return cfg


def build(raw: dict, overrides: dict | None = None) -> dict:
    from .guards import DEFAULTS as GUARD_DEFAULTS
    base = copy.deepcopy(DEFAULTS)
    base["guards"] = copy.deepcopy(GUARD_DEFAULTS)
    cfg = _merge(base, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    validate(cfg)
    if not cfg["run_id"]:
        size = str(cfg["size"]).replace(".", "p")
        cfg["run_id"] = f"{cfg['kind']}-{cfg['arm']}-{size}-s{cfg['seed']}-{config_hash(cfg)}"
    cfg["out_dir"] = str(cfg["out_dir"]).format(run_id=cfg["run_id"])
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def load(path: str | Path, overrides: dict | None = None) -> dict:
    import yaml  # PyYAML is in the Kaggle image and in requirements-kaggle.txt

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build(raw, overrides)


def dump(cfg: dict, path: str | Path) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def identity(cfg: dict) -> dict:
    """The fields every result row / manifest carries."""
    return {"run_id": cfg["run_id"], "kind": cfg["kind"], "arm": cfg["arm"], "model": cfg["model"],
            "size": cfg["size"], "seed": cfg["seed"], "config_hash": cfg.get("config_hash")}
