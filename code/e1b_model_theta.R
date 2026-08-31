suppressPackageStartupMessages({ library(mirt) })
options(warn = 1)

# ── Model-ability estimation via IRT-weighted composite (E1's MIRT, reused) ─
# E3's flipped 2PL (prompts=persons, models=items) exists to solve small-N
# (only 6 model "persons" would be far below the N>=200 stability threshold),
# and that is unrelated to and unaffected by this script -- models never get
# a theta there, only b_m (stringency) and a_m (discrimination).
# This script answers a different question: when models trade off across the
# 5 criteria (e.g. one wins geometry, another wins aesthetics), what is each
# model's overall ability once criteria are weighted by how informative they
# actually are (IRT discrimination), instead of a naive equal-weighted mean?
# It reuses E1's existing MIRT design -- persons = (model, prompt) assets
# (hundreds of them, never subject to the small-N problem), items = the 5
# criteria -- and extracts the per-asset ability score (theta) that E1
# currently discards after using the same fit only for BIC model selection.

args <- commandArgs(trailingOnly=FALSE); sp <- sub("--file=","",args[grep("--file=",args)]); ROOT <- if(length(sp)>0) normalizePath(file.path(dirname(sp),".."),mustWork=FALSE) else normalizePath("..")
RES    <- file.path(ROOT, "results")
R_DATA_DIR_NAME <- Sys.getenv("E1B_R_DATA_DIR", "r_data")
OUT_SUFFIX      <- Sys.getenv("E1B_OUT_SUFFIX", "")
R_DATA <- file.path(RES, R_DATA_DIR_NAME)

CRITERIA <- c("geometric_consistency","structural_consistency",
              "semantic_consistency","aesthetics","text_3d_alignment")

cat("E1b: Loading wide_final.csv from", R_DATA, "...\n")
df <- read.csv(file.path(R_DATA, "wide_final.csv"), stringsAsFactors = FALSE)
dat <- df[, CRITERIA]
keep <- complete.cases(dat)
dat <- dat[keep, ]
df_keep <- df[keep, ]
cat("  Complete (model,prompt) assets:", nrow(dat), "| Models:",
    paste(sort(unique(df_keep$model)), collapse=", "), "\n")

# mirt graded model requires 0-based consecutive integer categories (same
# recoding E1 uses, kept identical for consistency with the reported factor
# structure).
dat_mirt <- as.data.frame(lapply(dat, function(x) {
  pmin(4L, as.integer(x) - as.integer(min(x, na.rm = TRUE)))
}))

sign_fix <- function(theta_col, raw_criteria_cols) {
  # Orient theta so that higher theta = higher raw scores (mirt's sign is
  # otherwise arbitrary and can flip between refits/rotations).
  ref <- rowMeans(raw_criteria_cols, na.rm = TRUE)
  if (cor(theta_col, ref, use = "complete.obs") < 0) -theta_col else theta_col
}

cat("Fitting graded MIRT k=1 (single IRT-weighted composite ability) ...\n")
fit1 <- mirt(dat_mirt, model = 1, itemtype = "graded", verbose = FALSE,
             technical = list(NCYCLES = 2000))
theta1 <- as.numeric(fscores(fit1, method = "EAP"))
theta1 <- sign_fix(theta1, dat)

cat("Fitting graded MIRT k=2 (structural-coherence vs. content-accuracy) ...\n")
fit2 <- mirt(dat_mirt, model = 2, itemtype = "graded", verbose = FALSE,
             technical = list(NCYCLES = 2000))
theta2 <- fscores(fit2, method = "EAP")

# Identify which of the two extracted factors is "structural coherence" vs
# "content accuracy" from the rotated loadings (labels are arbitrary output
# order from mirt, not guaranteed to match E1's reported F1/F2 labels).
load_mat <- summary(fit2, verbose = FALSE)$rotF
cat("Rotated loadings:\n"); print(round(load_mat, 3))
structural_col <- which.max(abs(load_mat["structural_consistency", ]))
content_col    <- setdiff(1:2, structural_col)

theta_structural <- sign_fix(theta2[, structural_col], dat[, "structural_consistency", drop = FALSE])
theta_content    <- sign_fix(theta2[, content_col],
                              dat[, c("text_3d_alignment","geometric_consistency","semantic_consistency")])

out <- data.frame(model = df_keep$model, prompt_id = df_keep$prompt_id,
                   theta_composite = theta1,
                   theta_structural = theta_structural,
                   theta_content = theta_content)
write.csv(out, file.path(RES, paste0("T1b_theta_by_item", OUT_SUFFIX, ".csv")), row.names = FALSE)

model_theta <- aggregate(cbind(theta_composite, theta_structural, theta_content) ~ model,
                          data = out, FUN = mean)
model_theta <- model_theta[order(-model_theta$theta_composite), ]
cat("\n=== Model ability (IRT-weighted theta), by model ===\n")
print(model_theta, digits = 3)
write.csv(model_theta, file.path(RES, paste0("T1b_model_theta", OUT_SUFFIX, ".csv")), row.names = FALSE)

# Naive equal-weighted composite mean, for direct comparison against theta.
naive <- aggregate(score ~ model,
                    data = data.frame(model = df_keep$model, score = rowMeans(dat, na.rm = TRUE)),
                    FUN = mean)
names(naive)[2] <- "naive_mean"
naive <- naive[order(-naive$naive_mean), ]
cat("\n=== Naive equal-weighted composite mean, by model (comparison) ===\n")
print(naive, digits = 3)

comparison <- merge(model_theta, naive, by = "model")
comparison <- comparison[order(-comparison$theta_composite), ]
write.csv(comparison, file.path(RES, paste0("T1b_theta_vs_naive", OUT_SUFFIX, ".csv")), row.names = FALSE)
cat("\nRank by IRT theta_composite  :", paste(comparison$model[order(-comparison$theta_composite)], collapse=" > "), "\n")
cat("Rank by naive equal-weighted :", paste(comparison$model[order(-comparison$naive_mean)], collapse=" > "), "\n")

cat("\nDone: e1b_model_theta.R\n")
