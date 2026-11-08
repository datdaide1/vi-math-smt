"""Verify the pulled tranche adapters against their run manifests and record their sha256.

    .venv/Scripts/python.exe handoff/tools/verify_adapters.py      # writes handoff/ADAPTERS.md + handoff/ADAPTERS.sha256

For every run in jobs/queue/t2_state.json with status TRAINED, checks
jobs/outputs/t2-modal/<run_id>/adapters/epoch3/{adapter_model.safetensors, adapter_config.json}:
  * both files exist; the safetensors header parses (stdlib: 8-byte length + JSON) and the file size equals
    header + data end offset (no truncated download);
  * adapter_config: base_model_name_or_path == manifest model, r / lora_alpha == 32, target modules == manifest
    lora.modules (when recorded), peft_type LORA;
  * a training-time adapter hash, if the manifest records one (`adapter_sha256` / `final_adapter_sha256`), must match.
The manifests of the 11-08 tranche record no final-adapter hash (only `lora_init_sha`), so the sha256 computed here
becomes the reference hash from now on. Stdlib only; streams the files.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_safetensors(p: Path):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    end = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    dtypes = sorted({v["dtype"] for v in tensors.values()})
    ok = p.stat().st_size == 8 + n + end
    return ok, len(tensors), dtypes


def main(argv=None) -> int:
    state = json.loads((ROOT / "jobs" / "queue" / "t2_state.json").read_text(encoding="utf-8"))
    rows, sums, bad = [], [], 0
    for rid, r in sorted(state["runs"].items()):
        if r.get("status") != "TRAINED":
            continue
        d = ROOT / "jobs" / "outputs" / "t2-modal" / rid
        ad = d / "adapters" / "epoch3"
        st, cfgp = ad / "adapter_model.safetensors", ad / "adapter_config.json"
        problems = []
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
        if not st.exists() or not cfgp.exists():
            problems.append("missing file")
            rows.append((rid, "–", "–", "–", "; ".join(problems)))
            bad += 1
            continue
        ok, n_t, dtypes = check_safetensors(st)
        if not ok:
            problems.append("size != header+data (truncated?)")
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        if cfg.get("base_model_name_or_path") not in (man.get("model"), None):
            problems.append(f"base model {cfg.get('base_model_name_or_path')} != manifest {man.get('model')}")
        if cfg.get("r") != 32 or cfg.get("lora_alpha") != 32:
            problems.append(f"r/alpha {cfg.get('r')}/{cfg.get('lora_alpha')}")
        if str(cfg.get("peft_type", "LORA")).upper() != "LORA":
            problems.append(f"peft_type {cfg.get('peft_type')}")
        mods = (man.get("lora") or {}).get("modules") or (man.get("lora") or {}).get("suffixes")
        tm = cfg.get("target_modules")
        if mods and isinstance(tm, list) and sorted(tm) != sorted(mods):
            problems.append(f"target_modules {sorted(tm)} != manifest {sorted(mods)}")
        h = sha256(st)
        rec = man.get("adapter_sha256") or (man.get("summary") or {}).get("final_adapter_sha256")
        if rec and rec != h:
            problems.append("sha256 != manifest")
        sums.append(f"{h}  jobs/outputs/t2-modal/{rid}/adapters/epoch3/adapter_model.safetensors")
        sums.append(f"{sha256(cfgp)}  jobs/outputs/t2-modal/{rid}/adapters/epoch3/adapter_config.json")
        verdict = ("OK" if not problems else "; ".join(problems)) + ("" if rec else " (no training-time hash recorded)")
        bad += bool(problems)
        rows.append((rid, f"{st.stat().st_size / 2**20:.1f}", f"{n_t} {'/'.join(dtypes)}", f"`{h[:16]}`", verdict))
    L = ["# Tranche adapters — verification", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `handoff/tools/verify_adapters.py`. Full sha256 of both files "
         "per adapter: `handoff/ADAPTERS.sha256`._", "",
         "Location: `jobs/outputs/t2-modal/<run_id>/adapters/epoch3/` (pulled from the Modal Volume `t1r-eval`, "
         "`/train/<run_id>/adapters/epoch3`; git-ignored; shipped in the `runs` bundle). Checks: safetensors header parses "
         "and the file size matches it; `adapter_config.json` base model = run manifest model, r = α = 32, LoRA, target "
         "modules = manifest. The 11-08 manifests record no final-adapter hash, so these sha256 are the reference "
         "from now on.", "",
         f"**{len(rows) - bad}/{len(rows)} adapters pass.**", "",
         "| Run | MiB | tensors / dtype | sha256 (first 16) | Verdict |", "|---|---|---|---|---|"]
    L += [f"| `{a}` | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    (ROOT / "handoff" / "ADAPTERS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "handoff" / "ADAPTERS.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows) - bad}/{len(rows)} adapters OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
