"""Model, tokenizer and LoRA helpers shared by SFT, GRPO, merge and eval.

``model: tiny-random`` materialises an offline 2-layer Qwen3 with a character tokenizer covering Vietnamese, so the
whole pipeline can be smoke-tested on a laptop CPU without any download.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

TINY = "tiny-random"
SKIP_RE = re.compile(r"(vision|visual|audio|multi_modal|embed_vision|embed_audio|lm_head|mtp|patch_embed|merger)",
                     re.I)


# ----------------------------------------------------------------------------------------------- tiny model
def _tiny_charset() -> list[str]:
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x323), (0x1E00, 0x1EFF), (0x2010, 0x2030), (0x2190, 0x2265)]
    chars = ["\n", "\t"] + [chr(c) for a, b in ranges for c in range(a, b + 1)]
    return list(dict.fromkeys(chars))


def materialize_tiny(out_dir: str | Path, *, hidden: int = 64, layers: int = 2, seed: int = 0) -> str:
    """Write a tiny random Qwen3 + char tokenizer to ``out_dir`` (idempotent) and return the path."""
    out = Path(out_dir)
    if (out / "config.json").exists() and (out / "tokenizer.json").exists():
        return str(out)
    import torch
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials + _tiny_charset())}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tk.normalizer = normalizers.NFC()
    tk.decoder = decoders.Fuse()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                     unk_token="<unk>")
    cfg = Qwen3Config(vocab_size=len(vocab), hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=hidden // 4, max_position_embeddings=4096, tie_word_embeddings=False,
                      pad_token_id=0, eos_token_id=1, bos_token_id=2)
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(cfg)
    m.generation_config.eos_token_id = 1
    m.generation_config.pad_token_id = 0
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    hf_tok.save_pretrained(out)
    return str(out)


def resolve_model(name: str, cache_root: str | Path = "runs") -> str:
    if name == TINY:
        return materialize_tiny(Path(cache_root) / "_tiny_random")
    return name


# ----------------------------------------------------------------------------------------------- loading
def torch_dtype(precision: str):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def bos_prefix(tok) -> list[int]:
    """The ids the tokenizer itself prepends with ``add_special_tokens=True`` (LFM2: ``<|startoftext|>``; Qwen3:
    none).  Training prepends exactly these, so SFT matches vLLM, which tokenises prompts with special tokens."""
    try:
        with_sp = tok("x", add_special_tokens=True)["input_ids"]
        without = tok("x", add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001  (tiny char tokenizers etc.)
        return []
    n = len(with_sp) - len(without)
    return list(with_sp[:n]) if n > 0 and with_sp[n:] == without else []


# ----------------------------------------------------------------------------------------------- families
# Per-family facts used by the loaders (T1r).  `unsloth` = Unsloth's fast path verified for this family.
FAMILIES = {
    "qwen3": {"match": ("qwen3",), "lora_suffixes": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                     "up_proj", "down_proj"], "unsloth": True},
    # LFM2 / LFM2.5 (Liquid; `Lfm2ForCausalLM` in transformers >= 4.55): hybrid of gated short-conv blocks
    # (conv.in_proj / conv.out_proj Linear + depthwise Conv1d) and GQA blocks (q/k/v/out_proj), SwiGLU MLP
    # (feed_forward.w1/w2/w3).  The tokenizer prepends <|startoftext|>.  Causal short-conv => right padding is
    # exact for training; never left-pad for HF generate.  Unsloth support unverified -> engine hf.
    "lfm2": {"match": ("lfm2",), "lora_suffixes": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1",
                                                   "w2", "w3"], "unsloth": False},
}


def family_of(name_or_path: str, config=None) -> str | None:
    key = (getattr(config, "model_type", None) or "").lower() + "|" + str(name_or_path).lower()
    for fam, f in FAMILIES.items():
        if any(m in key for m in f["match"]):
            return fam
    return None


def load_model(path: str, precision: str, device: str, attn: str = "sdpa"):
    """Plain-HF loader (Qwen3, LFM2/LFM2.5, SmolLM3, ...).  ``use_cache`` is switched off for training: a
    DynamicCache disables transformers' packed-sequence detection and is never needed in a training forward."""
    from transformers import AutoModelForCausalLM
    dtype = torch_dtype(precision)
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation=attn)
    except (ValueError, ImportError):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager")
    m.config.use_cache = False
    return m.to(device)


def load_unsloth_base(path: str, precision: str, *, max_seq_length: int):
    """The Unsloth base model + tokenizer WITHOUT LoRA (shared across the arms of a cell, ``sft.run_cell``)."""
    from unsloth import FastLanguageModel
    kw = dict(model_name=path, max_seq_length=max_seq_length, dtype=torch_dtype(precision), load_in_4bit=False,
              load_in_8bit=False, full_finetuning=False)
    try:  # never silently swap to an unsloth/* mirror repo
        return FastLanguageModel.from_pretrained(use_exact_model_name=True, **kw)
    except TypeError:
        return FastLanguageModel.from_pretrained(**kw)


def lora_init_sha(model) -> str:
    """sha256 over every trainable (LoRA) tensor, in name order -- the adapter-init fingerprint (parity tests)."""
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(((n, p) for n, p in model.named_parameters() if p.requires_grad), key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def load_unsloth(path: str, precision: str, *, max_seq_length: int, r: int, alpha: int, dropout: float, seed: int,
                 grad_ckpt: bool, targets=None, base=None):
    """Unsloth ``FastLanguageModel`` + LoRA (O1 §3.1).  ``import unsloth`` must already have happened before
    transformers was imported (``train.cli`` does it for engine=unsloth).  Returns (peft model, unsloth tokenizer).
    Trainable (LoRA) weights are forced to fp32 as in the HF path (fp16 base + GradScaler needs fp32 grads)."""
    import torch
    from unsloth import FastLanguageModel
    if base is None:
        m, tok = load_unsloth_base(path, precision, max_seq_length=max_seq_length)
    else:   # shared base of a cell: LoRA-free, re-wrapped with a FRESH, seeded adapter
        m, tok = base
    torch.manual_seed(seed)
    fam = family_of(path, getattr(m, "config", None))
    tm = targets if isinstance(targets, list) else FAMILIES.get(fam or "qwen3")["lora_suffixes"]
    m = FastLanguageModel.get_peft_model(m, r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                                         target_modules=tm, random_state=seed,
                                         use_gradient_checkpointing="unsloth" if grad_ckpt else False,
                                         max_seq_length=max_seq_length)
    for p in m.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    try:
        m.config.use_cache = False
    except Exception:  # noqa: BLE001
        pass
    return m, tok


def text_linear_targets(model) -> list[str]:
    """Full module names of every nn.Linear in the text decoder (``all-linear``; no lm_head/vision/audio)."""
    import torch.nn as nn
    return [n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and not SKIP_RE.search(n)]


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def add_lora(model, r: int, alpha: int, dropout: float, targets="all-linear"):
    """Attach a LoRA adapter; trainable weights are forced to fp32 (fp16 base, fp32 adapters, T3 §3.1)."""
    import torch
    from peft import LoraConfig, get_peft_model
    names = text_linear_targets(model) if targets == "all-linear" else targets
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=names, bias="none",
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)  # autocast_adapter_dtype=True (default) already keeps adapters fp32
    for p in pm.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return pm


def lora_targets_summary(model) -> dict:
    names = text_linear_targets(model)
    return {"n_modules": len(names), "suffixes": sorted({n.split(".")[-1] for n in names})}


# ----------------------------------------------------------------------------------------------- loss
def _lm_parts(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    dec = getattr(base, "model", None)
    head = base.get_output_embeddings()
    cap = getattr(base.config, "final_logit_softcapping", None)
    if cap is None and hasattr(base.config, "get_text_config"):
        cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
    return dec, head, cap


def completion_nll_sum(model, input_ids, attention_mask, labels, position_ids=None):
    """Sum of next-token NLL over completion positions only (labels != -100), CE in fp32.

    Projects only the selected hidden states through lm_head (no [B, L, V] logits tensor), which matters for
    151k-token vocabularies on a 16 GB T4.  Applies final-logit soft-capping when the config has it.
    """
    import torch
    import torch.nn.functional as F
    dec, head, cap = _lm_parts(model)
    tgt = labels[:, 1:]
    mask = tgt != -100
    if dec is not None:
        kw = {"position_ids": position_ids} if position_ids is not None else {}
        h = dec(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kw).last_hidden_state
        logits = head(h[:, :-1][mask]).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
    else:  # unknown layout: full logits
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1][mask].float()
    return F.cross_entropy(logits, tgt[mask], reduction="sum")


def unsloth_nll_sum(model, batch: dict):
    """Sum of completion NLL through Unsloth's own (fused-CE) loss: ``loss`` is the mean over the valid shifted
    targets, so sum = loss x count(labels[:, 1:] != -100).  Packed rows pass ``position_ids`` +
    ``packed_seq_lengths`` and no attention mask; padded rows pass the attention mask.  The equality of this sum
    with the per-item logits path is asserted by the start-of-job gate (``sft.SFTRun.pack_gate``)."""
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "position_ids", "packed_seq_lengths", "labels")
          if batch.get(k) is not None}
    n = (batch["labels"][:, 1:] != -100).sum()
    out = model(use_cache=False, **kw)
    return out.loss.float() * n


def item_nll_from_logits(logits, labels) -> tuple[float, int]:
    """(sum NLL, n targets) of one row's completion from full logits [L, V] and labels [L] (fp32 CE)."""
    import torch.nn.functional as F
    tgt = labels[1:]
    m = tgt != -100
    return float(F.cross_entropy(logits[:-1][m].float(), tgt[m], reduction="sum")), int(m.sum())


# ----------------------------------------------------------------------------------------------- adapters
def save_adapter(model, out_dir: str | Path, tok=None, meta: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if tok is not None:
        tok.save_pretrained(str(out))
    if meta:
        (out / "y1_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def merge_adapter(base: str, adapter_dir: str, out_dir: str, precision: str = "fp16") -> str:
    """Fold a LoRA adapter into the base (fp32 math on CPU) and save it in ``precision`` (fp16 for GRPO/vLLM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    m = PeftModel.from_pretrained(m, adapter_dir).merge_and_unload()
    m = m.to(torch_dtype(precision))
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir)
    load_tokenizer(adapter_dir if Path(adapter_dir, "tokenizer_config.json").exists() else base).save_pretrained(
        out_dir)
    return out_dir
