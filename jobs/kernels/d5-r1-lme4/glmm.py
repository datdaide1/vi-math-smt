
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
