suppressPackageStartupMessages({ library(difR) })
options(warn = 1)

args <- commandArgs(trailingOnly = FALSE)
sp   <- sub("--file=", "", args[grep("--file=", args)])
ROOT <- if (length(sp) > 0 && nchar(sp) > 0) normalizePath(file.path(dirname(sp), ".."), mustWork = FALSE) else normalizePath("..")
RES    <- file.path(ROOT, "results")
FIGS   <- file.path(RES, "figures")
R_DATA_DIR_NAME <- Sys.getenv("E4_R_DATA_DIR", "r_data")
OUT_SUFFIX      <- Sys.getenv("E4_OUT_SUFFIX", "")
R_DATA <- file.path(RES, R_DATA_DIR_NAME)
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

# Minimum models per group to treat inferential DIF as valid.
MIN_PER_GROUP <- 10

cat("E4: Loading item_matrix and DIF groups ...\n")
raw      <- read.csv(file.path(R_DATA, "item_matrix.csv"), row.names = 1,
                     check.names = FALSE)
item_mat <- raw[, sapply(raw, is.numeric), drop = FALSE]

dif_grp  <- read.csv(file.path(R_DATA, "dif_groups.csv"),
                     stringsAsFactors = FALSE)
cat("  Item matrix:", nrow(item_mat), "models ×", ncol(item_mat), "items\n")

# Align group labels to matrix row order
dif_grp_aligned <- dif_grp[match(rownames(item_mat), dif_grp$model), ]
group_vec <- dif_grp_aligned$objaverse_group

cat("  Group distribution (0 = non-Objaverse, 1 = Objaverse):\n")
print(table(group_vec))

n_grp0 <- sum(group_vec == 0, na.rm = TRUE)
n_grp1 <- sum(group_vec == 1, na.rm = TRUE)
n_total <- n_grp0 + n_grp1

# ── Sample-size / group-balance gate ──────────────────────────────────────
too_small <- (n_grp0 < MIN_PER_GROUP) || (n_grp1 < MIN_PER_GROUP)

if (too_small) {
  cat("\n")
  cat("  *** SAMPLE SIZE WARNING ***\n")
  cat("  Inferential DIF (Mantel-Haenszel, logistic regression) requires\n")
  cat("  at least", MIN_PER_GROUP, "models per group for valid chi-squared inference.\n")
  cat("  Current data: group 0 =", n_grp0, "model(s),",
      "group 1 =", n_grp1, "model(s), total =", n_total, ".\n")
  cat("  With", n_grp0, "non-Objaverse and", n_grp1, "Objaverse models,\n")
  cat("  MH and logistic-regression DIF are not statistically reliable.\n")
  cat("  Running DESCRIPTIVE Objaverse sensitivity analysis instead.\n\n")
}

# ── Descriptive Objaverse sensitivity analysis (always computed) ──────────
# Compute per-item pass rates separately for each group.
idx0 <- which(group_vec == 0)
idx1 <- which(group_vec == 1)

p_non_objaverse <- if (length(idx0) > 0) colMeans(item_mat[idx0, , drop = FALSE], na.rm = TRUE) else rep(NA_real_, ncol(item_mat))
p_objaverse     <- if (length(idx1) > 0) colMeans(item_mat[idx1, , drop = FALSE], na.rm = TRUE) else rep(NA_real_, ncol(item_mat))

gap  <- p_objaverse - p_non_objaverse
absg <- abs(gap)

sensitivity_df <- data.frame(
  item                      = colnames(item_mat),
  p_non_objaverse           = round(p_non_objaverse, 3),
  p_objaverse               = round(p_objaverse,     3),
  gap_objaverse_minus_non   = round(gap,  3),
  abs_gap                   = round(absg, 3)
)
sensitivity_df <- sensitivity_df[order(-sensitivity_df$abs_gap), ]

write.csv(sensitivity_df,
          file.path(RES, paste0("T4_objaverse_sensitivity", OUT_SUFFIX, ".csv")),
          row.names = FALSE)
cat("  Saved T4_objaverse_sensitivity.csv (",
    nrow(sensitivity_df), " items)\n", sep = "")
cat("  abs_gap summary: mean =", round(mean(absg, na.rm = TRUE), 3),
    " | max =", round(max(absg, na.rm = TRUE), 3), "\n")

# Interpretation note
cat("\n  INTERPRETATION NOTE:\n")
cat("  - This is a descriptive group pass-rate contrast, NOT formal DIF.\n")
cat("  - The current data has only", n_total, "models (", n_grp0,
    "non-Objaverse,", n_grp1, "Objaverse).\n")
cat("  - Any '0 flagged items' result should NOT be interpreted as\n")
cat("    evidence of no DIF; the analysis is severely underpowered.\n")
cat("  - Treat 'abs_gap' as an item-level sensitivity screen,\n")
cat("    not as measurement invariance evidence.\n\n")

# ── Inferential DIF (only when sample size is adequate) ───────────────────
dif_df <- NULL
if (!too_small) {
  cat("Running Mantel-Haenszel DIF (N per group is adequate) ...\n")
  data_wg <- as.data.frame(cbind(item_mat, group = group_vec))

  mh_result <- tryCatch(
    difMH(data_wg, group = "group", focal.name = 1, alpha = 0.05),
    error = function(e) { cat("  MH DIF failed:", e$message, "\n"); NULL })

  lr_result <- tryCatch(
    difLogistic(data_wg, group = "group", focal.name = 1,
                type = "both", alpha = 0.05),
    error = function(e) { cat("  LR DIF failed:", e$message, "\n"); NULL })

  item_names <- colnames(item_mat)
  n_items    <- length(item_names)
  mh_stats <- mh_pvals <- rep(NA_real_, n_items)
  lr_chi2  <- lr_pvals <- rep(NA_real_, n_items)

  if (!is.null(mh_result)) {
    tryCatch({
      raw_s <- mh_result$statistic
      if (is.numeric(raw_s) && length(raw_s) == n_items) {
        mh_stats <- as.numeric(raw_s)
        mh_pvals <- pchisq(abs(mh_stats), df = 1, lower.tail = FALSE)
      } else cat("  MH statistic has unexpected format; treating as NA\n")
    }, error = function(e) cat("  MH stat extraction failed:", e$message, "\n"))
  }

  if (!is.null(lr_result)) {
    tryCatch({
      raw_lr <- lr_result$Stat
      raw_pv <- lr_result$P.value
      if (is.numeric(raw_lr) && length(raw_lr) == n_items) {
        lr_chi2  <- as.numeric(raw_lr)
        lr_pvals <- if (is.matrix(raw_pv)) as.numeric(raw_pv[, 1]) else as.numeric(raw_pv)
      } else cat("  LR statistic has unexpected format; treating as NA\n")
    }, error = function(e) cat("  LR stat extraction failed:", e$message, "\n"))
  }

  mh_padj <- p.adjust(mh_pvals, method = "BH")
  lr_padj <- p.adjust(lr_pvals, method = "BH")

  dif_df <- data.frame(
    item        = item_names,
    MH_stat     = round(mh_stats, 3),
    MH_pval     = round(mh_pvals, 4),
    MH_padj_BH  = round(mh_padj,  4),
    LR_chi2     = round(lr_chi2,  3),
    LR_pval     = round(lr_pvals, 4),
    LR_padj_BH  = round(lr_padj,  4),
    dif_flagged = (!is.na(mh_padj) & mh_padj < 0.05) |
                  (!is.na(lr_padj)  & lr_padj  < 0.05)
  )
  write.csv(dif_df, file.path(RES, paste0("T4_dif_flags", OUT_SUFFIX, ".csv")), row.names = FALSE)
  cat("  Flagged items (padj < 0.05):",
      sum(dif_df$dif_flagged, na.rm = TRUE), "/", n_items, "\n")
}

# ── Plots ─────────────────────────────────────────────────────────────────
png(file.path(FIGS, paste0("F4_dif_analysis", OUT_SUFFIX, ".png")), width = 1100, height = 500, res = 100)
par(mfrow = c(1, 2))

# Panel 1: histogram of gap distribution
hist(gap, breaks = 30, col = "steelblue", border = "white",
     main = "Objaverse Group Pass-Rate Contrast\n(Descriptive — not formal DIF)",
     xlab = "gap = p_objaverse - p_non_objaverse",
     ylab = "Number of items")
abline(v = 0, lty = 2, col = "gray40")
legend("topright",
       legend = c(
         paste0("mean = ", round(mean(gap, na.rm=TRUE), 3)),
         paste0("N groups: ", n_grp0, " vs ", n_grp1, " models")
       ),
       bty = "n", cex = 0.85)

# Panel 2: top-20 items by absolute gap (bar chart)
top20 <- head(sensitivity_df, 20)
if (nrow(top20) > 0) {
  bp <- barplot(top20$abs_gap,
                names.arg = top20$item,
                las = 2, col = "steelblue", border = "white",
                main = "Top 20 Items by |Gap| (Group Pass-Rate Contrast)",
                ylab = "|gap|",
                cex.names = 0.6,
                ylim = c(0, max(top20$abs_gap, na.rm = TRUE) * 1.15))
  abline(h = 0.2, lty = 2, col = "gray40")
  mtext("dashed line = |gap| = 0.20 reference", side = 1,
        line = 4.5, cex = 0.75, col = "gray40")
} else {
  plot.new(); title("No items to display")
}

par(mfrow = c(1, 1))
dev.off()
cat("  Saved F4_dif_analysis.png\n")

# ── Summary note file ─────────────────────────────────────────────────────
note_lines <- c(
  "E4 Objaverse Sensitivity Analysis — Interpretation Notes",
  "=========================================================",
  "",
  paste0("Total models: ", n_total,
         " (", n_grp0, " non-Objaverse, ", n_grp1, " Objaverse)"),
  paste0("Minimum per-group threshold for valid inferential DIF: ", MIN_PER_GROUP),
  paste0("Inferential DIF run: ", ifelse(too_small, "NO (underpowered)", "YES")),
  "",
  "Key results (descriptive group pass-rate contrast):",
  paste0("  Mean |gap|  : ", round(mean(absg, na.rm=TRUE), 3)),
  paste0("  Max  |gap|  : ", round(max(absg,  na.rm=TRUE), 3)),
  paste0("  Items with |gap| > 0.20: ",
         sum(absg > 0.20, na.rm=TRUE), " / ", length(absg)),
  "",
  "WARNINGS:",
  "  - '0 flagged items' does NOT constitute evidence of no DIF.",
  "  - MH and logistic-regression DIF require >=10 models per group;",
  paste0("    this dataset has only ", n_grp0, " non-Objaverse model(s)."),
  "  - Results should be described as 'item-level Objaverse sensitivity',",
  "    not as formal measurement invariance evidence.",
  "  - Replicate with additional non-Objaverse models before drawing",
  "    conclusions about benchmark contamination."
)
writeLines(note_lines, file.path(RES, paste0("T4_interpretation_note", OUT_SUFFIX, ".txt")))
cat("  Saved T4_interpretation_note.txt\n")

cat("Done: e4_dif_analysis.R\n")
