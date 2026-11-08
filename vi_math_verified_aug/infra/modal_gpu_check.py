"""Smallest possible Modal GPU probe (W2, decision D1 in T6 §2.2).

Run:  modal run vi_math_verified_aug/infra/modal_gpu_check.py
Cost: one T4 container, timeout 90 s, no extra packages -> worst case ~$0.02.
Purpose: does a GPU function run on this workspace WITHOUT a payment method on file?
Do NOT add a payment method to make this pass.
"""
import modal

app = modal.App("w2-gpu-check")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(gpu="T4", image=image, timeout=90, retries=0)
def gpu_probe() -> str:
    import subprocess
    import time

    t = time.time()
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return f"nvidia-smi rc={out.returncode} out={out.stdout.strip()} err={out.stderr.strip()} t={time.time()-t:.2f}s"


@app.local_entrypoint()
def main():
    print("RESULT:", gpu_probe.remote())
