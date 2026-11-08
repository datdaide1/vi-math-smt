#!/usr/bin/env Rscript
# Primary binomial GLMM for the Vi-math study (T5 §2.2, E1 §5.5).  Designed for a free Kaggle / Colab CPU
# R kernel (R is not installed on the laptop).  Inputs are written by
#   python -m vi_math_verified_aug.analysis.run export --pred preds.jsonl --out glmm_export/
#
# Usage:  Rscript glmm_lme4.R <glmm_data_X.csv> <contrasts_X.csv> <A|B> <out_prefix>
#
#   glmm_data_X.csv  one row per (run, item): run, lang, arm, model, size, seed, seed_key, item, dataset,
#                    evaltype, cell, correct, K
#   contrasts_X.csv  column `cell` (cell-mean levels) + one weight column per hypothesis
#
# Fixed effects: cell means (0 + cell), cell = lang x arm x model x slice  (== full factorial).
# Random effects and the PRE-REGISTERED convergence-fallback order:
#   A (VI grid):  full  (1|item)+(1|item:arm)+(1|item:model)+(1|run)+(1|run:evaltype)
#                 1. drop (1|item:model)
#                 2. drop (1|run:evaltype)  -> its variance is reported from separate per-model fits
#                 3. give up -> status "python_vb" (run analysis.glmm.fit_vb locally)
#   B (mirror):   full  (1|item)+(1|item:lang)+(1|item:arm)+(1|item:arm:lang)+(1|run)+(1|seed_key)
#                 1. drop (1|item:arm:lang)   2. drop (1|seed_key)   3. python_vb
# Each candidate: nAGQ = 0, then nAGQ = 1 (Laplace) started from the nAGQ = 0 estimates, optimizer bobyqa,
# calc.derivs = FALSE.  GLMM_NAGQ0_ONLY=1 stops after nAGQ = 0 (pre-registered time fallback, see Y2 report).
# "Converged" = bobyqa return code 0 and no convergence warning on the final fit.  Singular fits are recorded,
# not refit: a zero variance component does not invalidate the fixed-effect contrasts.
#
# Contrasts: logit-scale Wald z = w'b / sqrt(w'Vw); pp scale by the delta method on plogis(b) (identical to
# emmeans regrid = "response").  Holm over the confirmatory family, BH over the rest (contrast_meta.csv if
# present next to the weights file; otherwise names starting with "H" are confirmatory).
#
# Outputs: <out_prefix>_contrasts.csv, <out_prefix>_varcomp.csv, <out_prefix>_status.json,
#          <out_prefix>_fixef.csv

suppressPackageStartupMessages({
  if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4", repos = "https://cloud.r-project.org")
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) stop("usage: Rscript glmm_lme4.R data.csv contrasts.csv A|B out_prefix")
f_data <- args[1]; f_w <- args[2]; which <- args[3]; out <- args[4]

d <- read.csv(f_data, stringsAsFactors = FALSE)
W <- read.csv(f_w, stringsAsFactors = FALSE, check.names = FALSE)
d$cell <- factor(d$cell, levels = W$cell)
if (any(is.na(d$cell))) stop("data contains cells missing from the weights file")
for (v in c("item", "arm", "model", "run", "evaltype", "lang", "seed_key"))
  if (v %in% names(d)) d[[v]] <- factor(d[[v]])

RE <- list(
  A = list(full = c("item", "item:arm", "item:model", "run", "run:evaltype"),
           drop_item_model = c("item", "item:arm", "run", "run:evaltype"),
           drop_run_evaltype = c("item", "item:arm", "run")),
  B = list(full = c("item", "item:lang", "item:arm", "item:arm:lang", "run", "seed_key"),
           drop_item_arm_lang = c("item", "item:lang", "item:arm", "run", "seed_key"),
           drop_seed_key = c("item", "item:lang", "item:arm", "run")))[[which]]

# a term is only estimable if it has more levels than its parent (e.g. item:model with one model == item)
usable <- function(term, data) {
  vars <- strsplit(term, ":")[[1]]
  all(vars %in% names(data)) &&
    (length(vars) == 1 || nlevels(interaction(data[vars], drop = TRUE)) >
       nlevels(interaction(data[vars[-length(vars)]], drop = TRUE)))
}

# calc.derivs = FALSE: the finite-difference Hessian over (theta, all cell means) dominated run time (model A,
# 9k rows, 56 cells: > 9 min on Modal).  Convergence is judged from the optimizer return code instead, and
# vcov() uses the standard RX-based fixed-effect covariance.
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5), calc.derivs = FALSE)
nagq1 <- !identical(Sys.getenv("GLMM_NAGQ0_ONLY"), "1")   # set GLMM_NAGQ0_ONLY=1 to skip the Laplace refit

fit_structure <- function(terms, data) {
  terms <- Filter(function(t) usable(t, data), terms)
  form <- as.formula(paste("cbind(correct, K - correct) ~ 0 + cell +",
                           paste0("(1|", terms, ")", collapse = " + ")))
  warns <- character(0)
  catch <- function(expr) withCallingHandlers(expr, warning = function(w) {
    warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  t0 <- Sys.time()
  m0 <- tryCatch(catch(glmer(form, data = data, family = binomial, nAGQ = 0, control = ctrl)),
                 error = function(e) e)
  if (inherits(m0, "error")) return(list(ok = FALSE, err = conditionMessage(m0), terms = terms))
  warns0 <- warns; warns <- character(0)
  t_nagq0 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  m1 <- m0
  if (nagq1) {
    m1 <- tryCatch(catch(update(m0, nAGQ = 1, control = ctrl,
                                start = list(theta = getME(m0, "theta"), fixef = fixef(m0)))),
                   error = function(e) e)
    if (inherits(m1, "error")) return(list(ok = FALSE, err = conditionMessage(m1), terms = terms, warn0 = warns0))
  }
  # a singular (boundary) fit is NOT a convergence failure: a zero variance component leaves the fixed-effect
  # contrasts valid.  It is recorded and the structure is kept (bug fix after Modal run 3, where the singular
  # warning was counted as non-convergence and triggered fallback step 1).
  is_sing <- function(m) grepl("singular", m, ignore.case = TRUE)
  conv_msgs <- c(warns, unlist(m1@optinfo$conv$lme4$messages))
  conv_msgs <- conv_msgs[nzchar(conv_msgs) & !is_sing(conv_msgs)]
  ok <- length(conv_msgs) == 0 && m1@optinfo$conv$opt == 0
  list(ok = ok, model = m1, warnings = conv_msgs, warn0 = warns0, terms = terms, singular = isSingular(m1),
       nAGQ = if (nagq1) 1 else 0, seconds_nagq0 = t_nagq0,
       seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
}

status <- list(model = which, attempts = list())
chosen <- NULL
for (nm in names(RE)) {
  res <- fit_structure(RE[[nm]], d)
  status$attempts[[nm]] <- list(ok = res$ok, terms = res$terms, warnings = res$warnings, error = res$err,
                                singular = res$singular, nAGQ = res$nAGQ, seconds_nagq0 = res$seconds_nagq0,
                                seconds = res$seconds)
  cat(sprintf("[%s] structure %s: %s\n", which, nm, if (isTRUE(res$ok)) "converged" else "FAILED"))
  if (isTRUE(res$ok)) { chosen <- nm; fit <- res$model; break }
}

if (is.null(chosen)) {
  status$status <- "python_vb"
  writeLines(paste0('{"model":"', which, '","status":"python_vb"}'), paste0(out, "_status.json"))
  quit(status = 0)
}
status$status <- "lme4"; status$structure <- chosen

# fallback step 2 for A: report the run:evaltype variance from per-model fits
if (which == "A" && chosen == "drop_run_evaltype") {
  per <- list()
  for (m in levels(droplevels(d$model))) {
    sub <- droplevels(d[d$model == m, ])
    r <- fit_structure(c("item", "item:arm", "run", "run:evaltype"), sub)
    if (isTRUE(r$ok)) {
      vc <- as.data.frame(VarCorr(r$model))
      per[[m]] <- vc$sdcor[vc$grp == "run:evaltype"]
    }
  }
  status$run_evaltype_sd_per_model <- per
}

b <- fixef(fit)
V <- as.matrix(vcov(fit))
lv <- levels(d$cell)
names_b <- sub("^cell", "", names(b))
idx <- match(lv, names_b)
meta_f <- file.path(dirname(f_w), "contrast_meta.csv")
meta <- if (file.exists(meta_f)) read.csv(meta_f, stringsAsFactors = FALSE) else NULL

rows <- list()
for (h in setdiff(names(W), "cell")) {
  w <- W[[h]]
  if (any(is.na(idx[w != 0]))) { warning(paste("contrast", h, "uses an inestimable cell")); next }
  wb <- numeric(length(b)); wb[idx[!is.na(idx)]] <- w[!is.na(idx)]
  est <- sum(wb * b); se <- sqrt(as.numeric(t(wb) %*% V %*% wb))
  p <- plogis(b); g <- wb * p * (1 - p)
  est_pp <- sum(wb * p); se_pp <- sqrt(as.numeric(t(g) %*% V %*% g))
  fam <- if (!is.null(meta) && h %in% meta$name) meta$family[meta$name == h] else
    if (grepl("^H[0-9]", h)) "confirmatory" else "exploratory"
  rows[[h]] <- data.frame(name = h, family = fam, estimate = est, se = se, z = est / se,
                          p = 2 * pnorm(-abs(est / se)), ci_lo = est - 1.959964 * se, ci_hi = est + 1.959964 * se,
                          est_pp = est_pp, se_pp = se_pp, ci_lo_pp = est_pp - 1.959964 * se_pp,
                          ci_hi_pp = est_pp + 1.959964 * se_pp, fit = paste0("lme4:", chosen))
}
res <- do.call(rbind, rows)
res$p_adj <- NA_real_
for (fam in unique(res$family)) {
  k <- res$family == fam
  res$p_adj[k] <- p.adjust(res$p[k], method = if (fam == "confirmatory") "holm" else "BH")
}
write.csv(res, paste0(out, "_contrasts.csv"), row.names = FALSE)
vc <- as.data.frame(VarCorr(fit))
write.csv(vc[, c("grp", "vcov", "sdcor")], paste0(out, "_varcomp.csv"), row.names = FALSE)
write.csv(data.frame(cell = names_b, estimate = b, se = sqrt(diag(V))), paste0(out, "_fixef.csv"),
          row.names = FALSE)

# minimal JSON writer (avoids a jsonlite dependency)
to_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    nm <- names(x)
    inner <- vapply(seq_along(x), function(i) to_json(x[[i]]), "")
    if (is.null(nm)) return(paste0("[", paste(inner, collapse = ","), "]"))
    return(paste0("{", paste0('"', nm, '":', inner, collapse = ","), "}"))
  }
  if (is.character(x)) {
    s <- paste0('"', gsub('"', "'", x), '"')
    return(if (length(x) == 1) s else paste0("[", paste(s, collapse = ","), "]"))
  }
  if (is.logical(x)) { s <- tolower(as.character(x)); return(if (length(x) == 1) s else paste0("[", paste(s, collapse = ","), "]")) }
  s <- format(x, digits = 8)
  if (length(x) == 1) s else paste0("[", paste(s, collapse = ","), "]")
}
writeLines(to_json(status), paste0(out, "_status.json"))
cat("done:", which, chosen, "\n")
print(res[, c("name", "family", "estimate", "se", "p", "p_adj", "est_pp", "se_pp")])
print(vc[, c("grp", "sdcor")])
