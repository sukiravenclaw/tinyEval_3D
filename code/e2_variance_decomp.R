suppressPackageStartupMessages({
  library(lme4); library(irr)
})
options(warn = 1)

args <- commandArgs(trailingOnly=FALSE); sp <- sub("--file=","",args[grep("--file=",args)]); ROOT <- if(length(sp)>0) normalizePath(file.path(dirname(sp),".."),mustWork=FALSE) else normalizePath("..")
RES    <- file.path(ROOT, "results")
FIGS   <- file.path(RES, "figures")
# R_DATA_DIR / OUT_SUFFIX let this script be re-run against either data stream
# analyzed in the paper: the automated-score stream (results/r_data_dev,
# Table 3) or the Delphi stream (results/r_data, default). Override via env
# vars, e.g. `E2_R_DATA_DIR=r_data_dev E2_OUT_SUFFIX=_automated Rscript e2_variance_decomp.R`.
R_DATA_DIR_NAME <- Sys.getenv("E2_R_DATA_DIR", "r_data")
OUT_SUFFIX      <- Sys.getenv("E2_OUT_SUFFIX", "")
R_DATA <- file.path(RES, R_DATA_DIR_NAME)
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)

CRITERIA <- c("geometric_consistency","structural_consistency",
              "semantic_consistency","aesthetics","text_3d_alignment")

cat("E2: Loading long_tensor.csv from", R_DATA, "...\n")
df <- read.csv(file.path(R_DATA, "long_tensor.csv"), stringsAsFactors = FALSE)
df$score     <- as.numeric(df$score)
df$model     <- as.factor(df$model)
df$prompt_id <- as.factor(df$prompt_id)
df$judge     <- as.factor(df$judge)
df$criterion <- as.factor(df$criterion)
df$round     <- as.factor(df$round)
cat("  Rows:", nrow(df), "| Models:", nlevels(df$model),
    "| Judges:", nlevels(df$judge), "| Rounds:", nlevels(df$round), "\n")

# as.data.frame(VarCorr(<lmer fit>)) already returns one row per grouping
# factor PLUS a final grp=NA row for the residual (vcov == attr(.,"sc")^2).
# A previous version of this function additionally appended a second,
# separately-computed residual row and summed it into the denominator,
# double-counting residual variance in both the table and total_var. That
# produced the duplicate "Residual" rows in T2_variance_components.csv and
# the fabricated "Other interactions" line in paper Table 3. Do not add a
# manual residual row here -- it is already present in `vc`.
tidy_varcorr <- function(fit) {
  vc <- as.data.frame(VarCorr(fit))
  vc$facet    <- ifelse(is.na(vc$grp), "Residual", as.character(vc$grp))
  vc$variance <- vc$vcov
  total_var   <- sum(vc$variance, na.rm = TRUE)
  vc$pct_total <- round(100 * vc$variance / total_var, 2)
  vc_df <- vc[order(-vc$pct_total), c("facet","variance","pct_total")]
  rownames(vc_df) <- NULL
  vc_df
}

fit_formula_full   <- score ~ 1 + (1|model) + (1|prompt_id) + (1|judge) +
                                   (1|criterion) + (1|round) + (1|model:prompt_id)
fit_formula_simple <- score ~ 1 + (1|model) + (1|prompt_id) + (1|criterion) + (1|model:prompt_id)

fit_random_effects <- function(data) {
  tryCatch({
    lmer(fit_formula_full, data = data, REML = TRUE,
         control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  }, error = function(e) {
    cat("  Full model failed:", e$message, "\nTrying simpler model...\n")
    tryCatch(
      lmer(fit_formula_simple, data = data, REML = TRUE),
      error = function(e2) { cat("  Simple model also failed:", e2$message, "\n"); NULL }
    )
  })
}

# ── 1a. Random-effects model, raw scale (current spec; criterion random) ──
cat("Fitting lme4 MFRM model (raw scale) ...\n")
fit <- fit_random_effects(df)

vc_df <- NULL
if (!is.null(fit)) {
  vc_df <- tidy_varcorr(fit)
  write.csv(vc_df, file.path(RES, paste0("T2_variance_components", OUT_SUFFIX, ".csv")), row.names = FALSE)
  cat("  Variance components saved.\n")
  print(vc_df)
  cat("  Sum of %:", sum(vc_df$pct_total), "\n")

  # G-coefficient
  sigma2_model <- vc_df$variance[vc_df$facet == "model"]
  resid_var    <- vc_df$variance[vc_df$facet == "Residual"]
  if (length(sigma2_model) && !is.na(sigma2_model)) {
    n_judges    <- max(1, nlevels(df$judge))
    n_prompts   <- max(1, nlevels(df$prompt_id))
    G_coeff <- sigma2_model / (sigma2_model + resid_var / (n_judges * n_prompts))
    cat("  G-coefficient (approx):", round(G_coeff, 3), "\n")
  }

  # ── Variance shares bar plot ────────────────────────────────────────────
  png(file.path(FIGS, paste0("F2_variance_shares", OUT_SUFFIX, ".png")), width = 700, height = 500, res = 100)
  barplot(vc_df$pct_total, names.arg = vc_df$facet,
          col = "steelblue", las = 2, cex.names = 0.8,
          main = "Variance Decomposition (% of Total)",
          ylab = "% Variance", xlab = "")
  dev.off()
} else {
  cat("  WARNING: lme4 model failed; skipping variance decomposition.\n")
}

# ── 1b. Same random-effects spec, per-criterion z-scored scores ───────────
# Standardizes each criterion to mean 0 / SD 1 before pooling, so that the
# variance decomposition is not dominated by differences in each criterion's
# raw scale (mean/range). Because z-scoring forces each criterion's mean to
# 0 within itself, the criterion variance component is ~0 by construction;
# what matters is how the OTHER shares move once that scale artefact is
# removed. Reported alongside the raw-scale table, not as a replacement.
cat("Fitting lme4 MFRM model (per-criterion z-scored) ...\n")
df_z <- df
df_z$score <- ave(df_z$score, df_z$criterion,
                   FUN = function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
fit_z <- fit_random_effects(df_z)
vc_z_df <- NULL
if (!is.null(fit_z)) {
  vc_z_df <- tidy_varcorr(fit_z)
  write.csv(vc_z_df, file.path(RES, paste0("T2_variance_components_zscored", OUT_SUFFIX, ".csv")), row.names = FALSE)
  cat("  Z-scored variance components saved.\n")
  print(vc_z_df)
  cat("  Sum of %:", sum(vc_z_df$pct_total), "\n")
} else {
  cat("  WARNING: z-scored lme4 model failed; skipping.\n")
}

# ── 1c. Criterion (and its interactions) as FIXED effects ─────────────────
# The five criteria are a fixed set specified by the benchmark rubric, not a
# sample from a population of criteria, so a random intercept (1|criterion)
# is the wrong G-theory framing (a fixed facet should not be treated as a
# random facet of generalization; cf. Cronbach et al. 1972; Brennan 2001).
# We refit with criterion as a fixed main effect and add model:criterion,
# prompt:criterion as random interactions (interactions of a fixed factor
# with a random facet are themselves random). The fixed effect's share of
# total variance is reported as eta^2 from a one-way ANOVA on criterion
# (unambiguous Type I = Type III here, since criterion is the only fixed
# factor); remaining facets are variance components from the mixed model,
# rescaled to add up to (1 - eta^2) of total so the full table sums to 100%.
cat("Fitting fixed-criterion model with model:criterion, prompt:criterion interactions ...\n")
aov_fit <- aov(score ~ criterion, data = df)
ss <- summary(aov_fit)[[1]][["Sum Sq"]]
eta2_criterion <- ss[1] / sum(ss)
cat("  eta^2 (criterion, fixed):", round(eta2_criterion, 4), "\n")

fixed_formula_full   <- score ~ criterion + (1|model) + (1|prompt_id) + (1|judge) + (1|round) +
                                   (1|model:prompt_id) + (1|model:criterion) + (1|prompt_id:criterion)
fixed_formula_simple <- score ~ criterion + (1|model) + (1|prompt_id) +
                                   (1|model:prompt_id) + (1|model:criterion) + (1|prompt_id:criterion)
fit_fixed <- tryCatch({
  lmer(fixed_formula_full, data = df, REML = TRUE,
       control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
}, error = function(e) {
  cat("  Full fixed-criterion model failed:", e$message, "\nTrying simpler model...\n")
  tryCatch(
    lmer(fixed_formula_simple, data = df, REML = TRUE),
    error = function(e2) { cat("  Simple fixed-criterion model also failed:", e2$message, "\n"); NULL }
  )
})

vc_fixed_df <- NULL
if (!is.null(fit_fixed)) {
  vc_resid <- tidy_varcorr(fit_fixed)
  total_resid <- sum(vc_resid$variance, na.rm = TRUE)
  vc_resid$pct_total <- round(100 * (vc_resid$variance / total_resid) * (1 - eta2_criterion), 2)
  crit_row <- data.frame(facet = "Criterion (fixed effect)", variance = NA,
                          pct_total = round(100 * eta2_criterion, 2))
  vc_fixed_df <- rbind(crit_row, vc_resid[, c("facet","variance","pct_total")])
  vc_fixed_df <- vc_fixed_df[order(-vc_fixed_df$pct_total), ]
  rownames(vc_fixed_df) <- NULL
  write.csv(vc_fixed_df, file.path(RES, paste0("T2_variance_components_fixed_criterion", OUT_SUFFIX, ".csv")), row.names = FALSE)
  cat("  Fixed-criterion variance components saved.\n")
  print(vc_fixed_df)
  cat("  Sum of %:", sum(vc_fixed_df$pct_total), "\n")
} else {
  cat("  WARNING: fixed-criterion model failed; skipping.\n")
}

# ── 2. Reliability per criterion ──────────────────────────────────────────
cat("Computing reliability coefficients per criterion ...\n")
rel_rows <- list()
for (crit in CRITERIA) {
  sub <- df[df$criterion == crit, ]
  judges <- unique(sub$judge)
  if (length(judges) < 2) {
    cat("  ", crit, ": only 1 judge, skipping inter-rater stats\n")
    rel_rows[[crit]] <- data.frame(criterion = crit, alpha_kripp = NA,
                                   ICC = NA, ICC_lower = NA, ICC_upper = NA)
    next
  }
  # Pivot to judges × items matrix
  mat <- tryCatch({
    wide <- reshape(sub[, c("judge","prompt_id","score")],
                    idvar = "prompt_id", timevar = "judge", direction = "wide")
    score_cols <- grep("^score\\.", names(wide))
    as.matrix(wide[, score_cols])
  }, error = function(e) { cat("  pivot failed for", crit, "\n"); NULL })
  if (is.null(mat) || nrow(mat) < 5) {
    rel_rows[[crit]] <- data.frame(criterion = crit, alpha_kripp = NA, ICC = NA,
                                   ICC_lower = NA, ICC_upper = NA)
    next
  }
  mat <- mat[complete.cases(mat), ]
  alpha_val <- tryCatch(kripp.alpha(t(mat), method = "interval")$value, error = function(e) NA)
  icc_res   <- tryCatch({
    r <- icc(mat, model = "twoway", type = "agreement", unit = "average")
    c(r$value, r$lbound, r$ubound)
  }, error = function(e) c(NA, NA, NA))
  rel_rows[[crit]] <- data.frame(criterion = crit, alpha_kripp = round(alpha_val, 3),
                                  ICC = round(icc_res[1], 3),
                                  ICC_lower = round(icc_res[2], 3),
                                  ICC_upper = round(icc_res[3], 3))
  cat("  ", crit, "α=", round(alpha_val,3), "ICC=", round(icc_res[1],3), "\n")
}
rel_df <- do.call(rbind, rel_rows)
write.csv(rel_df, file.path(RES, paste0("T2_reliability", OUT_SUFFIX, ".csv")), row.names = FALSE)

# ── 3. Judge severity plot ─────────────────────────────────────────────────
judge_means <- aggregate(score ~ judge + criterion, data = df, FUN = mean)
png(file.path(FIGS, paste0("F2_judge_severity", OUT_SUFFIX, ".png")), width = 800, height = 500, res = 100)
par(mar = c(8, 4, 3, 1))
overall_mean <- aggregate(score ~ judge, data = df, FUN = mean)
overall_mean <- overall_mean[order(overall_mean$score), ]
dotchart(overall_mean$score,
         labels = overall_mean$judge,
         main = "Judge Severity (Mean Score Across All Items)",
         xlab = "Mean Score (0–9 scale)",
         pch = 19, col = "steelblue")
abline(v = mean(df$score, na.rm = TRUE), lty = 2, col = "gray50")
dev.off()

# ── 4. Delphi convergence effect ──────────────────────────────────────────
# `df` above is deduplicated to one row per (model, prompt_id, judge,
# criterion) at whichever round it finalized (see prep_r_data.py). Grouping
# THAT by round and taking sd(score) conflates disjoint item populations
# (fast- vs slow-converging items end up tagged with different rounds) and
# pools variance across models/prompts, not just inter-judge disagreement --
# it does not track the same item's disagreement shrinking across rounds,
# and does not reproduce the convergence numbers reported in the paper.
# The correct input is the RAW pre-dedup per-round ratings (one row per
# model x prompt_id x judge x round x criterion), from which we compute
# inter-judge SD for each (model, prompt_id, criterion, round) and then
# average across items -- optionally restricted to items present at every
# round, for a genuine same-item paired comparison.
raw_path <- file.path(R_DATA, "raw_round_tensor.csv")
if (file.exists(raw_path)) {
  cat("Computing Delphi convergence effect (raw per-round tensor) ...\n")
  raw_df <- read.csv(raw_path, stringsAsFactors = FALSE)
  raw_df$score <- as.numeric(raw_df$score)

  sd_by_round <- aggregate(score ~ model + prompt_id + criterion + round, data = raw_df,
                            FUN = function(x) if (length(x) >= 2) sd(x) else NA_real_)
  names(sd_by_round)[names(sd_by_round) == "score"] <- "sd_score"
  sd_by_round <- sd_by_round[!is.na(sd_by_round$sd_score), ]
  rounds_present <- sort(unique(sd_by_round$round))

  if (length(rounds_present) >= 2) {
    first_round <- min(rounds_present)
    last_round  <- max(rounds_present)

    # Primary estimate: population-level mean SD at each round (item
    # population differs across rounds because items stop once converged).
    r1_mean_sd <- mean(sd_by_round$sd_score[sd_by_round$round == first_round])
    r3_mean_sd <- mean(sd_by_round$sd_score[sd_by_round$round == last_round])
    convergence_effect <- 1 - r3_mean_sd / r1_mean_sd
    cat("  [Population-level, item mix differs by round] Round", first_round,
        "mean SD:", round(r1_mean_sd, 3), "(n=", sum(sd_by_round$round == first_round), ")",
        "| Round", last_round, "mean SD:", round(r3_mean_sd, 3),
        "(n=", sum(sd_by_round$round == last_round), ")",
        "| Convergence effect:", round(convergence_effect * 100, 1), "%\n")

    # Robustness check: PAIRED comparison restricted to the same items
    # tracked across every round from first_round to last_round.
    wide <- reshape(sd_by_round, idvar = c("model","prompt_id","criterion"),
                     timevar = "round", direction = "wide")
    sd_cols <- paste0("sd_score.", rounds_present)
    paired <- wide[complete.cases(wide[, sd_cols]), ]
    if (nrow(paired) > 0) {
      r1_paired <- mean(paired[[paste0("sd_score.", first_round)]])
      r3_paired <- mean(paired[[paste0("sd_score.", last_round)]])
      paired_effect <- 1 - r3_paired / r1_paired
      cat("  [Paired, same items, n=", nrow(paired), "] Round", first_round,
          "mean SD:", round(r1_paired, 3), "| Round", last_round, "mean SD:",
          round(r3_paired, 3), "| Convergence effect:", round(paired_effect * 100, 1), "%\n")
    }

    conv_df <- data.frame(
      round = rounds_present,
      mean_judge_sd = sapply(rounds_present, function(r) mean(sd_by_round$sd_score[sd_by_round$round == r])),
      n_items = sapply(rounds_present, function(r) sum(sd_by_round$round == r)))
    write.csv(conv_df, file.path(RES, paste0("T2_convergence_by_round", OUT_SUFFIX, ".csv")), row.names = FALSE)
  } else {
    cat("  Single round in raw tensor — skipping Delphi convergence analysis.\n")
  }
} else {
  cat("  raw_round_tensor.csv not found in", R_DATA,
      "-- skipping Delphi convergence analysis (needs pre-dedup per-round",
      "ratings, not the final-round-only long_tensor.csv).\n")
}

cat("Done: e2_variance_decomp.R\n")
