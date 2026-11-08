"""R1: Y2's lme4 GLMM (models A/B, pre-registered fallback order) on a PRIVATE Kaggle CPU R kernel (Modal is out of
credit; R is not installed on the laptop).

    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.c3_r1_glmm_kaggle export --pred "jobs/outputs/t2-eval/*.jsonl"
    E:\\venv_tools\\Scripts\\kaggle.exe datasets create -p data/audit/r1/kaggle_glmm      (or: datasets version -m ...)
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.c3_r1_glmm_kaggle kernel
    E:\\venv_tools\\Scripts\\kaggle.exe kernels push -p jobs/kernels/d5-r1-lme4
    E:\\venv_tools\\Scripts\\kaggle.exe kernels output datuni/d5-r1-lme4 -p data/audit/r1/glmm_out
The export uses R1's loader (Core VERIFIED & clean, greedy-only Core recoded as K = 1), so the GLMM sees exactly the
data of the bootstrap tables. Results (out_{A,B}_contrasts.csv) feed ``c3_r1_results --r-results`` (decision rule).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data" / "audit" / "r1" / "kaggle_glmm"
KDIR = ROOT / "jobs" / "kernels" / "d5-r1-lme4"
OWNER = "datuni"

KERNEL_R = r'''
inp <- list.files("/kaggle/input", pattern = "glmm_lme4.R$", recursive = TRUE, full.names = TRUE)[1]
d <- dirname(inp)
cat("R", R.version.string, "lme4 available:", requireNamespace("lme4", quietly = TRUE), "\n")
for (w in c("A", "B")) {
  f_data <- file.path(d, paste0("glmm_data_", w, ".csv"))
  if (!file.exists(f_data)) next
  t0 <- Sys.time()
  args_w <- c(f_data, file.path(d, paste0("contrasts_", w, ".csv")), w, file.path("/kaggle/working", paste0("out_", w)))
  commandArgs <- function(trailingOnly = TRUE) args_w
  tryCatch(source(inp, local = new.env()), error = function(e) cat("ERROR", w, conditionMessage(e), "\n"))
  cat("model", w, "done in", format(Sys.time() - t0), "\n")
}
'''


def export(pred: list[str], binary: str = "yes_vs_rest"):
    from . import contrasts, glmm, schema
    from .c3_r1_results import load_rows
    rows, item_meta, stats = load_rows(pred, binary)
    d = schema.build(rows, item_meta=item_meta, strict=False)
    C, notes = contrasts.library(d, include_exploratory=False)
    DS.mkdir(parents=True, exist_ok=True)
    w = glmm.export_for_r(d, C, str(DS))
    shutil.copy2(Path(__file__).with_name("glmm_lme4.R"), DS / "glmm_lme4.R")
    (DS / "dataset-metadata.json").write_text(json.dumps({"title": "d5-r1-glmm", "id": f"{OWNER}/d5-r1-glmm",
                                                          "licenses": [{"name": "other"}]}), encoding="utf-8")
    (DS / "export_info.json").write_text(json.dumps({"written": w, "inputs": stats, "notes": notes, "binary": binary},
                                                    indent=1, default=str), encoding="utf-8")
    print(json.dumps(w, indent=1))


KERNEL_PY = r'''
# Kaggle no longer offers R kernels: install R + lme4 from Ubuntu packages, then run Y2's glmm_lme4.R
import glob, os, subprocess, time
t0 = time.time()
def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode
if subprocess.run("which Rscript", shell=True).returncode != 0:
    sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq r-base-core r-cran-lme4 > /dev/null")
sh("Rscript -e 'library(lme4); cat(R.version.string, as.character(packageVersion(\"lme4\")), \"\n\")'")
inp = glob.glob("/kaggle/input/**/glmm_lme4.R", recursive=True)[0]
d = os.path.dirname(inp)
for w in ("A", "B"):
    f = os.path.join(d, f"glmm_data_{w}.csv")
    if os.path.exists(f):
        t = time.time()
        sh(f"Rscript {inp} {f} {os.path.join(d, f'contrasts_{w}.csv')} {w} /kaggle/working/out_{w}")
        print("model", w, "seconds", round(time.time() - t), flush=True)
print("total seconds", round(time.time() - t0))
'''


def kernel():
    KDIR.mkdir(parents=True, exist_ok=True)
    (KDIR / "glmm.py").write_text(KERNEL_PY, encoding="utf-8")
    meta = {"id": f"{OWNER}/d5-r1-lme4", "title": "d5-r1-lme4", "code_file": "glmm.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{OWNER}/d5-r1-glmm"], "kernel_sources": [],
            "model_sources": [], "competition_sources": []}
    (KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(KDIR)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        pred = sys.argv[sys.argv.index("--pred") + 1:]
        export(pred)
    elif cmd == "kernel":
        kernel()
