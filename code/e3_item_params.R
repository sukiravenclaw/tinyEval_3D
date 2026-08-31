suppressPackageStartupMessages({ library(mirt) })
options(warn = 1)

args <- commandArgs(trailingOnly = FALSE)
sp   <- sub("--file=", "", args[grep("--file=", args)])
ROOT <- if (length(sp) > 0 && nchar(sp) > 0) normalizePath(file.path(dirname(sp), ".."), mustWork = FALSE) else normalizePath("..")
RES    <- file.path(ROOT, "results")
FIGS   <- file.path(RES, "figures")
# Override to point at the automated-score stream (results/r_data_dev) instead
# of the default Delphi stream, e.g.:
# E3_R_DATA_DIR=r_data_dev E3_OUT_SUFFIX=_automated Rscript e3_item_params.R
R_DATA_DIR_NAME <- Sys.getenv("E3_R_DATA_DIR", "r_data")
OUT_SUFFIX      <- Sys.getenv("E3_OUT_SUFFIX", "")
R_DATA <- file.path(RES, R_DATA_DIR_NAME)
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

CRITERIA <- c("geometric_consistency", "structural_consistency",
              "semantic_consistency", "aesthetics", "text_3d_alignment")

# ── Load item matrix (rows = models, cols = prompts) ─────────────────────────
cat("E3: Loading item_matrix.csv ...\n")
raw <- read.csv(file.path(R_DATA, "item_matrix.csv"), row.names = 1,
                check.names = FALSE)
item_mat_raw <- raw[, sapply(raw, is.numeric), drop = FALSE]
cat("  Dimensions (models x prompts):", nrow(item_mat_raw), "x", ncol(item_mat_raw), "\n")

# ── Classical prompt-level stats (kept for E5 item selection) ────────────────
# p_value = proportion of models that pass each prompt
# point_biserial = how well passing prompt p discriminates between models
pvals <- colMeans(item_mat_raw, na.rm = TRUE)         # length = N_prompts
total <- rowSums(item_mat_raw, na.rm = TRUE)           # total score per model
pbis  <- sapply(item_mat_raw, function(x) {
  tryCatch(cor(x, total, use = "complete.obs", method = "pearson"),
           warning = function(w) NA_real_,
           error   = function(e) NA_real_)
})
classical_df <- data.frame(
  item           = names(pvals),
  p_value        = round(pvals, 3),
  point_biserial = round(pbis, 3)
)
write.csv(classical_df, file.path(RES, paste0("T3_classical_stats", OUT_SUFFIX, ".csv")), row.names = FALSE)
cat("  Saved T3_classical_stats.csv (", nrow(classical_df), " prompts)\n", sep = "")

# ── Classical model-level stats ───────────────────────────────────────────────
# Pass rate per model = proportion of prompts each model passes
model_pass_rate <- rowMeans(item_mat_raw, na.rm = TRUE)
model_classical_df <- data.frame(
  model          = rownames(item_mat_raw),
  pass_rate      = round(model_pass_rate, 3),
  n_pass         = rowSums(item_mat_raw, na.rm = TRUE),
  n_items        = ncol(item_mat_raw)
)
model_classical_df <- model_classical_df[order(model_classical_df$pass_rate, decreasing = TRUE), ]
cat("  Model pass rates:\n"); print(model_classical_df)

# ── Filter zero-variance prompts (all models pass or all fail) ────────────────
item_var    <- apply(item_mat_raw, 2, var)
nonzero_var <- item_var > 0
cat("  Zero-variance prompts removed:", sum(!nonzero_var),
    "(all-pass or all-fail across all models)\n")
cat("  Prompts available for IRT:", sum(nonzero_var), "\n")
item_mat_irt <- item_mat_raw[, nonzero_var, drop = FALSE]

# ── FLIPPED ORIENTATION: prompts as persons, models as items ──────────────────
# Transpose: rows = 386 prompts (persons), cols = 6 models (items)
# With N_persons = 386 (>> 200 threshold), 2PL parameters for models are stable.
item_mat_flipped <- t(item_mat_irt)   # 386 prompts x 6 models
cat("\nFlipped IRT orientation:\n")
cat("  Persons (prompts):", nrow(item_mat_flipped), "\n")
cat("  Items   (models):", ncol(item_mat_flipped), "\n")
cat("Fitting 2PL IRT (N_persons=", nrow(item_mat_flipped),
    ", N_items=", ncol(item_mat_flipped), ") ...\n", sep = "")

fit2pl <- tryCatch(
  mirt(item_mat_flipped, model = 1, itemtype = "2PL", verbose = FALSE,
       technical = list(NCYCLES = 3000)),
  error = function(e) { cat("  2PL failed:", e$message, "\n"); NULL }
)

model_params_df <- NULL
prompt_theta_df <- NULL

if (!is.null(fit2pl)) {
  # ── Model parameters (the "items" in this orientation) ─────────────────────
  params    <- coef(fit2pl, IRTpars = TRUE, simplify = TRUE)$items
  model_params_df <- as.data.frame(params)
  model_params_df$model <- rownames(model_params_df)

  if ("a" %in% names(model_params_df)) {
    model_params_df$discrimination <- round(model_params_df$a, 3)
    model_params_df$difficulty      <- round(model_params_df$b, 3)
  }

  # Pass rate and classical rank for each model
  model_params_df$pass_rate <- round(model_pass_rate[model_params_df$model], 3)

  # Model item information peak (theta at which this model is most informative)
  theta_grid <- matrix(seq(-3, 3, by = 0.1))
  info_mat   <- tryCatch(iteminfo(fit2pl, Theta = theta_grid), error = function(e) NULL)
  if (!is.null(info_mat)) {
    model_params_df$info_peak_theta <- round(theta_grid[apply(info_mat, 2, which.max)], 2)
    model_params_df$info_peak_value <- round(apply(info_mat, 2, max), 4)
  }

  model_params_df <- model_params_df[order(model_params_df$difficulty), ]
  write.csv(model_params_df, file.path(RES, paste0("T3_model_parameters", OUT_SUFFIX, ".csv")), row.names = FALSE)
  cat("  Saved T3_model_parameters.csv\n")
  cat("  Model difficulty (b) — lower = easier model (more prompts pass):\n")
  print(model_params_df[, c("model", "difficulty", "discrimination", "pass_rate")])

  # Keep T3_item_parameters.csv pointing to model parameters for backward compat
  write.csv(model_params_df, file.path(RES, paste0("T3_item_parameters", OUT_SUFFIX, ".csv")), row.names = FALSE)

  # ── Prompt ability estimates (the "persons" in this orientation) ────────────
  thetas <- tryCatch(fscores(fit2pl, method = "EAP"), error = function(e) NULL)
  if (!is.null(thetas)) {
    prompt_theta_df <- data.frame(
      prompt    = rownames(item_mat_flipped),
      theta     = round(thetas[, 1], 3),
      p_value   = round(pvals[rownames(item_mat_flipped)], 3)
    )
    prompt_theta_df <- prompt_theta_df[order(prompt_theta_df$theta), ]
    write.csv(prompt_theta_df, file.path(RES, paste0("T3_prompt_theta", OUT_SUFFIX, ".csv")), row.names = FALSE)
    cat("  Saved T3_prompt_theta.csv (", nrow(prompt_theta_df), " prompts)\n", sep = "")
    cat("  Prompt theta range: [",
        round(min(prompt_theta_df$theta), 3), ",",
        round(max(prompt_theta_df$theta), 3), "]\n")
    cat("  Saturated prompts (p>=0.8):", sum(pvals >= 0.8, na.rm = TRUE), "\n")
  }

  # ── Model information curve plot ───────────────────────────────────────────
  if (!is.null(info_mat)) {
    png(file.path(FIGS, paste0("F3_item_information", OUT_SUFFIX, ".png")), width = 900, height = 600, res = 100)
    matplot(theta_grid, info_mat, type = "l", lwd = 2,
            col = rainbow(ncol(info_mat)), lty = 1,
            main = "Model Information Functions\n(Flipped IRT: prompts as persons, models as items)",
            xlab = expression(theta[p] ~ "(prompt generatability)"),
            ylab = "Information")
    legend("topright", legend = colnames(item_mat_flipped),
           col = rainbow(ncol(info_mat)), lty = 1, lwd = 2, cex = 0.85)
    dev.off()
    cat("  Saved F3_item_information.png\n")
  }

  # ── Prompt ability distribution plot ──────────────────────────────────────
  if (!is.null(prompt_theta_df)) {
    png(file.path(FIGS, paste0("F3_prompt_ability", OUT_SUFFIX, ".png")), width = 900, height = 600, res = 100)
    hist(prompt_theta_df$theta, breaks = 25, col = "steelblue", border = "white",
         main = "Prompt Generatability Distribution\n(IRT theta_p from flipped 2PL)",
         xlab = expression(hat(theta)[p] ~ "(estimated prompt generatability)"),
         ylab = "Number of prompts")
    abline(v = mean(prompt_theta_df$theta), lty = 2, col = "firebrick", lwd = 2)
    legend("topright", "Mean theta", lty = 2, col = "firebrick", cex = 0.9)
    dev.off()
    cat("  Saved F3_prompt_ability.png\n")
  }

} else {
  cat("  2PL failed. Falling back to classical model pass rates.\n")
  model_params_df <- model_classical_df
  model_params_df$discrimination <- NA
  model_params_df$difficulty     <- 1 - model_classical_df$pass_rate
  write.csv(model_params_df, file.path(RES, paste0("T3_model_parameters", OUT_SUFFIX, ".csv")), row.names = FALSE)
  write.csv(model_params_df, file.path(RES, paste0("T3_item_parameters", OUT_SUFFIX, ".csv")), row.names = FALSE)

  # Fallback prompt theta = classical p_value (for E5)
  prompt_theta_df <- data.frame(
    prompt  = names(pvals[nonzero_var]),
    theta   = round(1 - pvals[nonzero_var], 3),  # harder prompts have lower p → higher difficulty
    p_value = round(pvals[nonzero_var], 3)
  )
  write.csv(prompt_theta_df, file.path(RES, paste0("T3_prompt_theta", OUT_SUFFIX, ".csv")), row.names = FALSE)

  png(file.path(FIGS, paste0("F3_item_information", OUT_SUFFIX, ".png")), width = 900, height = 600, res = 100)
  barplot(model_classical_df$pass_rate, names.arg = model_classical_df$model,
          col = "steelblue", las = 2,
          main = "Model Pass Rates (Classical)", ylab = "Proportion of prompts passed")
  dev.off()
}

# ── Per-criterion discrimination distribution (kept for supplementary) ────────
png(file.path(FIGS, paste0("F3_discrimination_by_criterion", OUT_SUFFIX, ".png")),
    width = 900, height = 600, res = 100)
par(mfrow = c(2, 3))
for (crit in CRITERIA) {
  raw_c <- tryCatch(
    read.csv(file.path(R_DATA, paste0("item_matrix_", crit, ".csv")),
             row.names = 1, check.names = FALSE),
    error = function(e) NULL)
  if (is.null(raw_c)) { plot.new(); title(crit); next }

  mat_c   <- raw_c[, sapply(raw_c, is.numeric), drop = FALSE]
  # Per-model pass rate per criterion
  pass_rates_c <- rowMeans(mat_c, na.rm = TRUE)
  barplot(sort(pass_rates_c, decreasing = TRUE),
          col = "steelblue", las = 2, main = crit, ylim = c(0, 1),
          ylab = "Pass rate", cex.names = 0.8)
}
par(mfrow = c(1, 1))
dev.off()
cat("  Saved F3_discrimination_by_criterion.png\n")

cat("Done: e3_item_params.R\n")
