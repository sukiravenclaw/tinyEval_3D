suppressPackageStartupMessages({
  library(psych); library(lavaan); library(mirt)
})
options(warn = 1)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
ROOT <- if (length(script_path) > 0) normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE) else normalizePath("..")
RES     <- file.path(ROOT, "results")
FIGS    <- file.path(RES, "figures")
R_DATA  <- file.path(RES, "r_data")
dir.create(FIGS,   recursive = TRUE, showWarnings = FALSE)

CRITERIA <- c("geometric_consistency", "structural_consistency",
              "semantic_consistency", "aesthetics", "text_3d_alignment")

cat("E1: Loading wide_final.csv ...\n")
df <- read.csv(file.path(R_DATA, "wide_final.csv"))
cat("  Rows:", nrow(df), "| Cols:", paste(names(df), collapse=", "), "\n")

# Keep only criterion columns, drop rows with NA
dat <- df[, CRITERIA]
dat <- dat[complete.cases(dat), ]
cat("  Complete cases:", nrow(dat), "\n")

if (nrow(dat) < 20) stop("Too few complete cases (", nrow(dat), "). Check data prep.")

# ── 1. Horn's parallel analysis ────────────────────────────────────────────
cat("Running parallel analysis ...\n")
png(file.path(FIGS, "F1_parallel_analysis.png"), width=800, height=600, res=100)
tryCatch({
  pa <- fa.parallel(dat, fa = "fa", n.iter = 100, plot = TRUE,
                    main = "Parallel Analysis Scree Plot — Eval3D Criteria")
  cat("  Suggested factors:", pa$nfact, "\n")
}, error = function(e) {
  plot.new(); text(0.5, 0.5, paste("Parallel analysis failed:", e$message))
  cat("  Parallel analysis error:", e$message, "\n")
})
dev.off()

# ── 2. EFA k=1,2,3 ────────────────────────────────────────────────────────
cat("Running EFA (k=1,2,3) ...\n")
efa_results <- list()
efa_fit_rows <- list()
for (k in 1:3) {
  tryCatch({
    fit <- fa(dat, nfactors = k, rotate = "oblimin", fm = "ml")
    efa_results[[k]] <- fit
    efa_fit_rows[[k]] <- data.frame(
      k = k, RMSR = round(fit$rms, 4), RMSEA = round(fit$RMSEA[1], 4),
      TLI = round(fit$TLI, 4), BIC = round(fit$BIC, 2)
    )
    cat("  k=", k, "RMSR=", round(fit$rms,4), "RMSEA=", round(fit$RMSEA[1],4),
        "TLI=", round(fit$TLI,4), "\n")
  }, error = function(e) cat("  EFA k=", k, "failed:", e$message, "\n"))
}
efa_fit_df <- do.call(rbind, Filter(Negate(is.null), efa_fit_rows))

# Save factor loadings (best solution by TLI)
best_k <- if (!is.null(efa_fit_df) && nrow(efa_fit_df) > 0)
  efa_fit_df$k[which.max(efa_fit_df$TLI)] else 1
if (!is.null(efa_results[[best_k]])) {
  loads <- as.data.frame(round(loadings(efa_results[[best_k]])[,], 3))
  loads$criterion <- rownames(loads)
  write.csv(loads, file.path(RES, "T1_factor_loadings.csv"), row.names = FALSE)
  cat("  Saved T1_factor_loadings.csv (best k=", best_k, ")\n")
}
if (!is.null(efa_fit_df)) {
  write.csv(efa_fit_df, file.path(RES, "T1_efa_fit_indices.csv"), row.names = FALSE)
}

# ── 3. CFA on EFA-suggested structure ─────────────────────────────────────
cat("Running CFA ...\n")
# 2-factor model: geometry (gc, sc) vs semantic-aesthetic (sem, aes, t3d)
cfa_model_2f <- '
  geometry =~ geometric_consistency + structural_consistency
  semantic  =~ semantic_consistency + aesthetics + text_3d_alignment
'
# 1-factor model
cfa_model_1f <- 'quality =~ geometric_consistency + structural_consistency +
                             semantic_consistency + aesthetics + text_3d_alignment'
cfa_fit_rows <- list()
for (nm in c("1F", "2F")) {
  mstr <- if (nm == "1F") cfa_model_1f else cfa_model_2f
  tryCatch({
    fit <- cfa(mstr, data = dat, estimator = "MLR")
    idx <- fitMeasures(fit, c("cfi","tli","rmsea","srmr","bic"))
    cfa_fit_rows[[nm]] <- data.frame(
      model = nm, CFI = round(idx["cfi"],3), TLI = round(idx["tli"],3),
      RMSEA = round(idx["rmsea"],3), SRMR = round(idx["srmr"],3),
      BIC = round(idx["bic"],1)
    )
    cat("  CFA", nm, "— CFI=", round(idx["cfi"],3), "RMSEA=", round(idx["rmsea"],3), "\n")
  }, error = function(e) cat("  CFA", nm, "failed:", e$message, "\n"))
}
cfa_fit_df <- do.call(rbind, Filter(Negate(is.null), cfa_fit_rows))
if (!is.null(cfa_fit_df)) {
  write.csv(cfa_fit_df, file.path(RES, "T1_cfa_fit_indices.csv"), row.names = FALSE)
}

# ── 4. Graded-response MIRT k=1,2,3 ───────────────────────────────────────
# mirt graded model requires 0-based consecutive integer categories.
# Scores vary by criterion (semantic_consistency 0-1, geometric_consistency 2-7, etc.),
# so shift each column to start at 0, then cap at 4 to keep at most 5 categories.
dat_mirt <- as.data.frame(lapply(dat, function(x) {
  pmin(4L, as.integer(x) - as.integer(min(x, na.rm = TRUE)))
}))
cat("Running graded-response MIRT (k=1,2,3) on 0-4 recoded data ...\n")
cat("  Recoded ranges:\n")
for (col in names(dat_mirt)) {
  cat("   ", col, ":", min(dat_mirt[[col]]), "-", max(dat_mirt[[col]]),
      " (", length(unique(dat_mirt[[col]])), "categories)\n")
}
mirt_bic <- c()
for (k in 1:3) {
  tryCatch({
    fit <- mirt(dat_mirt, model = k, itemtype = "graded", verbose = FALSE,
                technical = list(NCYCLES = 2000))
    bic_val2 <- tryCatch({
      val <- extract.mirt(fit, "BIC")
      if (is.null(val)) NA_real_ else as.numeric(val)
    }, error = function(e) NA_real_)
    mirt_bic[k] <- bic_val2
    cat("  MIRT k=", k, "BIC=", round(bic_val2, 1), "\n")
  }, error = function(e) {
    mirt_bic[k] <<- NA_real_
    cat("  MIRT k=", k, "failed:", e$message, "\n")
  })
}
mirt_df <- data.frame(k = 1:3, BIC = mirt_bic)
write.csv(mirt_df, file.path(RES, "T1_mirt_bic.csv"), row.names = FALSE)

# ── Summary table ──────────────────────────────────────────────────────────
all_fit <- list()
if (!is.null(efa_fit_df) && nrow(efa_fit_df) > 0) {
  efa_row <- efa_fit_df[which.max(efa_fit_df$TLI), ]
  all_fit[["EFA (best)"]] <- data.frame(Method="EFA", k=efa_row$k,
    RMSR=efa_row$RMSR, RMSEA=efa_row$RMSEA, TLI=efa_row$TLI, BIC=efa_row$BIC)
}
write.csv(mirt_df, file.path(RES, "T1_fit_indices.csv"), row.names = FALSE)
cat("Done: e1_factor_structure.R\n")
