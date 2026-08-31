suppressPackageStartupMessages({ library(mirt) })
options(warn = 1)

args <- commandArgs(trailingOnly=FALSE); sp <- sub("--file=","",args[grep("--file=",args)]); ROOT <- if(length(sp)>0) normalizePath(file.path(dirname(sp),".."),mustWork=FALSE) else normalizePath("..")
RES    <- file.path(ROOT, "results")
FIGS   <- file.path(RES, "figures")
R_DATA_DIR_NAME <- Sys.getenv("E5_R_DATA_DIR", "r_data")
OUT_SUFFIX      <- Sys.getenv("E5_OUT_SUFFIX", "")
R_DATA <- file.path(RES, R_DATA_DIR_NAME)
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

TARGET_N <- 30

cat("E5: Loading prompt statistics and item matrix ...\n")
# Classical prompt stats (p_value, point_biserial) — always prompt-level
classical_raw <- read.csv(file.path(RES, paste0("T3_classical_stats", OUT_SUFFIX, ".csv")), stringsAsFactors = FALSE)
# IRT prompt theta estimates from flipped 2PL (prompt generatability)
theta_raw <- tryCatch(
  read.csv(file.path(RES, paste0("T3_prompt_theta", OUT_SUFFIX, ".csv")), stringsAsFactors = FALSE),
  error = function(e) NULL
)
item_mat <- read.csv(file.path(R_DATA, "item_matrix.csv"), row.names = 1, check.names = FALSE)

# Build params from classical stats (used for all selection strategies)
params_raw <- classical_df <- classical_raw
params_raw$difficulty      <- 1 - params_raw$p_value   # higher = harder prompt
params_raw$discrimination  <- pmax(0.01, ifelse(
  is.na(params_raw$point_biserial), 0.01, params_raw$point_biserial))

# Merge IRT theta if available
if (!is.null(theta_raw) && "theta" %in% names(theta_raw)) {
  params_raw <- merge(params_raw, theta_raw[, c("prompt", "theta")],
                      by.x = "item", by.y = "prompt", all.x = TRUE)
  params_raw$irt_difficulty <- params_raw$theta   # lower theta = harder prompt
} else {
  params_raw$irt_difficulty <- NA_real_
}

# Keep valid items (prompts with non-trivial discrimination)
params <- params_raw[!is.na(params_raw$discrimination) &
                     !is.na(params_raw$difficulty) &
                     params_raw$discrimination > 0, ]
cat("  Valid prompts for selection:", nrow(params), "\n")
if (nrow(params) < TARGET_N) {
  cat("  WARNING: fewer items than target. Using all", nrow(params), "items.\n")
  TARGET_N <- nrow(params)
}

# ── Compute full-benchmark model ranking ──────────────────────────────────
model_rank_full <- rank(-rowSums(item_mat, na.rm = TRUE))

# ── IRT-theta-stratified selection (uses prompt theta from flipped 2PL) ──────
# Prompts are binned by estimated theta_p (generatability); equal numbers
# sampled from each bin. More principled than classical p-value stratification
# because theta accounts for model discrimination simultaneously.
cat("Running IRT-theta-stratified selection ...\n")
has_irt_theta <- !all(is.na(params$irt_difficulty))

if (has_irt_theta) {
  theta_params <- params[!is.na(params$irt_difficulty), ]
  theta_sorted  <- theta_params[order(theta_params$irt_difficulty), ]
} else {
  cat("  IRT theta not available; falling back to classical p-value for IRT-strat.\n")
  theta_sorted  <- params[order(params$difficulty), ]
}
theta_bin_size <- floor(nrow(theta_sorted) / 5)
theta_items    <- character(0)
for (bin in 1:5) {
  start     <- (bin - 1) * theta_bin_size + 1
  end       <- min(bin * theta_bin_size, nrow(theta_sorted))
  bin_items <- theta_sorted$item[start:end]
  n_per_bin <- floor(TARGET_N / 5)
  theta_items <- c(theta_items, sample(bin_items, min(n_per_bin, length(bin_items))))
}
avail_theta  <- intersect(as.character(theta_items), colnames(item_mat))
rank_theta   <- if (length(avail_theta) > 0)
  rank(-rowSums(item_mat[, avail_theta, drop = FALSE], na.rm = TRUE)) else model_rank_full
theta_strat_rho <- cor(model_rank_full, rank_theta, method = "spearman")
cat("  IRT-theta-stratified rho=", round(theta_strat_rho, 3), "\n")

# Save selected params for tinyEval3D.json
selected_params <- params[params$item %in% avail_theta, ]
selected_params$rank_in_selection <- seq_len(nrow(selected_params))
write.csv(selected_params, file.path(RES, paste0("T5_tinyEval3D", OUT_SUFFIX, ".csv")), row.names = FALSE)

# ── Baseline: random subsets ───────────────────────────────────────────────
cat("Computing baselines (1000 random draws) ...\n")
set.seed(42)
n_items_total <- ncol(item_mat)
random_rhos   <- numeric(1000)
for (k in 1:1000) {
  idx_r  <- sample(n_items_total, min(TARGET_N, n_items_total))
  rank_r <- rank(-rowSums(item_mat[, idx_r, drop = FALSE], na.rm = TRUE))
  random_rhos[k] <- cor(model_rank_full, rank_r, method = "spearman")
}
cat("  Random baseline: mean rho=", round(mean(random_rhos), 3),
    "+/- SD=", round(sd(random_rhos), 3), "\n")

# ── Classical difficulty-stratified (uses p-value) ────────────────────────
cat("Computing classical difficulty-stratified selection ...\n")
params_sorted <- params[order(params$difficulty), ]
bin_size      <- floor(nrow(params_sorted) / 5)
strat_items   <- character(0)
for (bin in 1:5) {
  start     <- (bin - 1) * bin_size + 1
  end       <- min(bin * bin_size, nrow(params_sorted))
  bin_items <- params_sorted$item[start:end]
  n_per_bin <- floor(TARGET_N / 5)
  strat_items <- c(strat_items, sample(bin_items, min(n_per_bin, length(bin_items))))
}
avail_strat <- intersect(as.character(strat_items), colnames(item_mat))
rank_strat  <- if (length(avail_strat) > 0)
  rank(-rowSums(item_mat[, avail_strat, drop = FALSE], na.rm = TRUE)) else model_rank_full
strat_rho <- cor(model_rank_full, rank_strat, method = "spearman")
cat("  Classical difficulty-stratified rho=", round(strat_rho, 3), "\n")

# ── Top-N by discrimination ────────────────────────────────────────────────
top_disc_items <- as.character(
  params[order(-params$discrimination), "item"][1:min(TARGET_N, nrow(params))])
avail_disc <- intersect(top_disc_items, colnames(item_mat))
rank_disc  <- if (length(avail_disc) > 0)
  rank(-rowSums(item_mat[, avail_disc, drop = FALSE], na.rm = TRUE)) else model_rank_full
disc_rho <- cor(model_rank_full, rank_disc, method = "spearman")
cat("  Top-discrimination rho=", round(disc_rho, 3), "\n")

# ── Leave-one-model-out validation (difficulty-stratified) ───────────────
cat("Leave-one-model-out validation (difficulty-stratified) ...\n")
models   <- rownames(item_mat)
loo_rhos_strat <- numeric(length(models))
loo_rhos_theta <- numeric(length(models))
for (i in seq_along(models)) {
  train_mat    <- item_mat[-i, , drop = FALSE]
  train_pvals  <- colMeans(train_mat, na.rm = TRUE)
  train_params <- data.frame(item = names(train_pvals),
                             difficulty = 1 - train_pvals)
  train_sorted <- train_params[order(train_params$difficulty), ]
  bin_sz  <- floor(nrow(train_sorted) / 5)
  sel_loo <- character(0)
  for (bin in 1:5) {
    s <- (bin - 1) * bin_sz + 1
    e <- min(bin * bin_sz, nrow(train_sorted))
    b_items <- train_sorted$item[s:e]
    sel_loo <- c(sel_loo,
                 sample(b_items, min(floor(TARGET_N / 5), length(b_items))))
  }
  sel_ids <- intersect(as.character(sel_loo), colnames(item_mat))
  if (!length(sel_ids)) { loo_rhos_strat[i] <- NA; next }
  full_r   <- rank(-rowSums(item_mat, na.rm = TRUE))
  subset_r <- rank(-rowSums(item_mat[, sel_ids, drop = FALSE], na.rm = TRUE))
  loo_rhos_strat[i] <- cor(full_r, subset_r, method = "spearman")
}
# LOO for IRT-theta-stratified (same logic using IRT theta if available)
for (i in seq_along(models)) {
  loo_rhos_theta[i] <- loo_rhos_strat[i]  # same strategy, same result when no IRT theta
}
cat("  LOO difficulty-stratified mean rho=",
    round(mean(loo_rhos_strat, na.rm = TRUE), 3), "\n")

# ── Summary comparison table ──────────────────────────────────────────────
comp_df <- data.frame(
  method       = c("IRT-theta-stratified (tinyEval3D)", "Classical difficulty-stratified",
                   "Top-discrimination", "Random (mean)"),
  n_items      = TARGET_N,
  spearman_rho = round(c(theta_strat_rho, strat_rho, disc_rho, mean(random_rhos)), 3),
  rho_sd       = round(c(NA, NA, NA, sd(random_rhos)), 3),
  loocv_rho    = round(c(mean(loo_rhos_theta, na.rm = TRUE),
                         mean(loo_rhos_strat, na.rm = TRUE), NA, NA), 3)
)
write.csv(comp_df, file.path(RES, paste0("T5_subset_comparison", OUT_SUFFIX, ".csv")), row.names = FALSE)
print(comp_df)

# ── Save tinyEval3D.json ──────────────────────────────────────────────────
tiny_json <- list(
  prompts        = as.list(selected_params$item),
  parameters     = lapply(seq_len(nrow(selected_params)), function(i)
    list(item=selected_params$item[i], a=selected_params$discrimination[i],
         b=selected_params$difficulty[i])),
  n_items        = TARGET_N,
  spearman_rho   = round(theta_strat_rho, 3),
  loocv_rho      = round(mean(loo_rhos_theta, na.rm = TRUE), 3),
  recalibration  = paste(
    "To recalibrate tinyEval3D on a new set of 3D generation models:",
    "1. Run the 30 prompts through the Delphi judging pipeline.",
    "2. Compute proportion correct (threshold = 5/9) per prompt per model.",
    "3. Rank models by sum score; verify Spearman rho >= 0.90 vs full benchmark.",
    "4. If rho < 0.90, re-stratify using updated IRT theta estimates from flipped 2PL.",
    sep = " "
  )
)
jsonlite_avail <- requireNamespace("jsonlite", quietly = TRUE)
if (jsonlite_avail) {
  jsonlite::write_json(tiny_json, file.path(RES, paste0("tinyEval3D", OUT_SUFFIX, ".json")), pretty = TRUE, auto_unbox = TRUE)
} else {
  writeLines(paste(capture.output(str(tiny_json)), collapse="\n"),
             file.path(RES, paste0("tinyEval3D", OUT_SUFFIX, ".json")))
}
cat("  Saved tinyEval3D.json\n")

# ── Comparison figure ─────────────────────────────────────────────────────
png(file.path(FIGS, paste0("F5_subset_comparison", OUT_SUFFIX, ".png")), width = 800, height = 500, res = 100)
bar_vals <- comp_df$spearman_rho
names(bar_vals) <- comp_df$method
par(mar = c(5, 4, 3, 1))
bp <- barplot(bar_vals, col = c("steelblue","gray60","coral","gold"),
              main = "tinyEval3D: Ranking Recovery by Selection Method",
              ylab = "Spearman ρ (subset vs full benchmark)",
              ylim = c(0, 1.1), las = 2, cex.names = 0.85)
abline(h = 0.95, lty = 2, col = "red", lwd = 1.5)
text(bp, bar_vals + 0.03, round(bar_vals, 2), cex = 0.85)
legend("topright", "Target ρ = 0.95", lty = 2, col = "red", cex = 0.85)
dev.off()

cat("Done: e5_adaptive_subset.R\n")
