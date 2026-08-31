#!/usr/bin/env Rscript
# Master script: run all 5 psychometric analyses in sequence.
# Run from project root:  Rscript code/run_psychometrics.R
options(warn = 1)

SCRIPTS <- c("e1_factor_structure.R", "e2_variance_decomp.R",
             "e3_item_params.R",      "e4_dif_analysis.R",
             "e5_adaptive_subset.R")

script_dir <- file.path(normalizePath("."), "code")

for (s in SCRIPTS) {
  path <- file.path(script_dir, s)
  cat("\n", strrep("=", 60), "\n")
  cat("Running:", s, "\n")
  cat(strrep("=", 60), "\n")
  tryCatch(
    source(path, echo = FALSE, local = FALSE),
    error = function(e) cat("ERROR in", s, ":", conditionMessage(e), "\n")
  )
}
cat("\n=== All psychometric analyses complete ===\n")
cat("Figures saved to: results/figures/\n")
cat("Tables saved to:  results/\n")
