"""GRPO stage (T3 §4.1) on TRL ``GRPOTrainer``.

* Start: the arm's SFT model **merged to fp16** (``grpo.sft_model``), plus a fresh LoRA r=32, alpha=64, all-linear
  (LoRA init seeded by ``seed``).
* Loss: ``loss_type="dr_grpo"``, ``beta=0`` (no KL, no reference model), ``scale_rewards="none"``, eps 0.2 /
  eps_high 0.28, ``num_iterations=1``, ``mask_truncated_completions=True``.
* Shape: G=8 completions per prompt, 32 prompts per optimiser step (256 rollouts = micro x accumulation),
  200 steps, completions <= 512 tokens at T=1.0 / top-p 1.0, LR 2e-5 constant after 10 warm-up steps.
* fp16 trainer (and fp16 vLLM when ``use_vllm``), LoRA weights fp32.
* Reward: binary ``math_verify`` (no format reward) or the Bernoulli(0.5) ``random`` control.
* Prompts: the fixed pool in ``data.train``; prompts longer than ``max_prompt_length`` tokens are dropped (TRL 1.x
  has no prompt truncation option, and truncating a problem statement would change it).
* Resumable through the HF Trainer checkpoints (``save_steps``); the wall-clock budget callback saves and stops,
  the CLI then exits 75.

TRL field names were checked against trl 1.17.1 (``GRPOConfig`` dataclass); unknown names are refused loudly.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

from . import EXIT_NEEDS_RESUME, EXIT_OK
from . import config as C
from . import data as D
from . import modeling as M
from . import reward as R
from . import template as T
from .sft import _write_json, env_info, set_seed


def build_prompt_rows(cfg: dict, tok) -> tuple[list[dict], dict]:
    g, fields = cfg["grpo"], cfg["data"]["fields"]
    rows = D.read_jsonl(cfg["data"]["train"])
    out, n_long, n_bad = [], 0, 0
    for i, r in enumerate(rows):
        q = D.get_field(r, "question", fields)
        gold = D.get_field(r, "gold", fields)
        if gold is None:
            gold = D.item_answer(r, fields)
        if not q or gold is None:
            n_bad += 1
            continue
        p = T.prompt(q, cfg.get("lang", "vi"))
        if len(tok(p, add_special_tokens=False)["input_ids"]) > g["max_prompt_length"]:
            n_long += 1
            continue
        out.append({"prompt": p, "gold": json.dumps(gold, ensure_ascii=False) if isinstance(gold, dict) else str(gold),
                    "item_id": str(D.get_field(r, "id", fields) or f"row{i}")})
    stats = {"n_pool": len(rows), "n_malformed": n_bad, "n_dropped_prompt_too_long": n_long, "n_prompts": len(out)}
    return out, stats


def grpo_args(cfg: dict, out_dir: Path, device: str) -> dict:
    g = cfg["grpo"]
    rollouts = g["prompts_per_step"] * g["num_generations"]
    use_vllm = g["use_vllm"]
    if use_vllm == "auto":
        try:
            import vllm  # noqa: F401
            use_vllm = device.startswith("cuda")
        except ImportError:
            use_vllm = False
    fp16 = cfg["precision"] == "fp16"
    args = dict(
        output_dir=str(out_dir / "trainer"), seed=cfg["seed"], data_seed=cfg["seed"],
        learning_rate=g["lr"], lr_scheduler_type="constant_with_warmup", warmup_steps=g["warmup_steps"],
        max_steps=cfg["budget"]["max_steps"] or g["max_steps"], weight_decay=0.0, max_grad_norm=1.0,
        optim="adamw_torch",
        per_device_train_batch_size=g["micro_batch"], gradient_accumulation_steps=rollouts // g["micro_batch"],
        num_generations=g["num_generations"],
        loss_type=g["loss_type"], beta=g["beta"], scale_rewards="none" if g["scale_rewards"] is False
        else g["scale_rewards"], epsilon=g["epsilon"], epsilon_high=g["epsilon_high"],
        num_iterations=g["num_iterations"], mask_truncated_completions=g["mask_truncated_completions"],
        max_completion_length=g["max_completion_length"], temperature=g["temperature"], top_p=g["top_p"],
        fp16=fp16, bf16=cfg["precision"] == "bf16", use_cpu=not device.startswith("cuda"),
        use_vllm=bool(use_vllm), logging_steps=g["logging_steps"], save_steps=g["save_steps"],
        save_total_limit=2, report_to="none", remove_unused_columns=False, log_completions=False,
        dataloader_num_workers=0,
    )
    if use_vllm:
        args.update(vllm_mode=g["vllm_mode"], vllm_gpu_memory_utilization=g["vllm_gpu_memory_utilization"],
                    vllm_max_model_length=g["vllm_max_model_length"])  # prompt 512 + completion 512: colocate fits
    return args


def make_config(args: dict):
    from trl import GRPOConfig
    names = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(args) - names)
    if unknown:
        raise RuntimeError(f"GRPOConfig in this TRL version has no fields {unknown}; update grpo.py")
    return GRPOConfig(**args)


class _Budget:
    """TrainerCallback factory: save + stop when the wall-clock budget is about to run out."""

    def __new__(cls, deadline: float):
        from transformers import TrainerCallback

        class BudgetCallback(TrainerCallback):
            hit = False

            def on_step_end(self, args, state, control, **kw):
                if time.time() >= deadline:
                    BudgetCallback.hit = True
                    control.should_save = True
                    control.should_training_stop = True
                return control

        return BudgetCallback()


def last_checkpoint(d: str | Path) -> str | None:
    """Newest ``checkpoint-<step>`` dir under d (HF Trainer layout), else None."""
    best = None
    for p in Path(d).glob("checkpoint-*"):
        if p.is_dir() and p.name.split("-")[-1].isdigit():
            if best is None or int(p.name.split("-")[-1]) > int(best.name.split("-")[-1]):
                best = p
    return str(best) if best else None


def run(cfg: dict, t_start: float | None = None) -> int:
    t_start = time.time() if t_start is None else t_start
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    C.dump(cfg, out / "config.yaml")
    manifest = {**C.identity(cfg), "status": "STARTING", "env": env_info(), "reward": cfg["grpo"]["reward"],
                "grader": R.grader_version()}
    _write_json(out / "manifest.json", manifest)
    if (out / "DONE").exists() and cfg["resume"] != "never":
        return EXIT_OK
    try:
        from datasets import Dataset
        from trl import GRPOTrainer
        set_seed(cfg["seed"])
        g = cfg["grpo"]
        device = M.pick_device(cfg["device"])
        if device == "cpu" and cfg["precision"] == "fp16":
            raise C.ConfigError("precision fp16 needs CUDA; use fp32 for CPU smoke runs")
        model_path = M.resolve_model(g["sft_model"] or cfg["model"], cache_root=out.parent)
        tok = M.load_tokenizer(model_path)
        tok.padding_side = "left"
        rows, pstats = build_prompt_rows(cfg, tok)
        manifest.update(model_path=model_path, prompt_stats=pstats)
        base = M.load_model(model_path, cfg["precision"], device, cfg["attn_implementation"])
        set_seed(cfg["seed"])
        model = M.add_lora(base, cfg["lora"]["r"], g["lora_alpha"], cfg["lora"]["dropout"], cfg["lora"]["targets"])
        if g["reward"] == "random":
            reward_fn = R.RandomReward(cfg["seed"], g["random_reward_p"], locale=None)
        else:
            reward_fn = R.MathVerifyReward(locale=None)
        args = grpo_args(cfg, out, device)
        manifest["grpo_args"] = args
        _write_json(out / "manifest.json", manifest)
        b = cfg["budget"]
        deadline = (t_start + b["wall_s"] - b["reserve_s"]) if b["wall_s"] else float("inf")
        budget_cb = _Budget(deadline)
        trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=make_config(args),
                              train_dataset=Dataset.from_list(rows), processing_class=tok, callbacks=[budget_cb])
        ckpt = None
        if cfg["resume"] != "never":
            for d in [args["output_dir"]] + list(cfg["resume_dirs"]):
                if os.path.isdir(d):
                    ckpt = last_checkpoint(d) or ckpt
                    if ckpt:
                        break
        manifest.update(status="RUNNING", resumed_from=ckpt)
        _write_json(out / "manifest.json", manifest)
        trainer.train(resume_from_checkpoint=ckpt)
        with open(out / "train_log.jsonl", "w", encoding="utf-8") as f:
            for row in trainer.state.log_history:
                f.write(json.dumps(row) + "\n")
        if type(budget_cb).hit and trainer.state.global_step < args["max_steps"]:
            manifest.update(status="NEEDS_RESUME", global_step=trainer.state.global_step)
            _write_json(out / "manifest.json", manifest)
            return EXIT_NEEDS_RESUME
        M.save_adapter(trainer.model, out / "adapters" / "final", tok, meta=C.identity(cfg))
        manifest.update(status="COMPLETE", global_step=trainer.state.global_step,
                        final_adapter=str(out / "adapters" / "final"))
        _write_json(out / "manifest.json", manifest)
        (out / "DONE").write_text("ok", encoding="utf-8")
        return EXIT_OK
    except Exception:
        manifest.update(status="FAILED", error=traceback.format_exc()[-6000:])
        _write_json(out / "manifest.json", manifest)
        raise
