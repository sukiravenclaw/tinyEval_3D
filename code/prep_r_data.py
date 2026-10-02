"""
Prepare R-ready CSV files from the rating tensor (or Hi3DBench dev data).

Outputs to results/r_data/ (default) or results/r_data_dev/ (--dev):
  - wide_final.csv      — (model×prompt) × 5 criteria, judge-averaged final scores
  - long_tensor.csv     — full long format for MFRM/lme4
  - item_matrix.csv     — items × models binary matrix for IRT (dichotomized at threshold)
  - dif_groups.csv      — per-model Objaverse exposure group label

Run BEFORE the R scripts. --dev uses Hi3DBench automated scores (6 models, incl. crm)
instead of the real Delphi tensor (results/rating_tensor.parquet, 5 models) and writes
to results/r_data_dev/ so it never overwrites the production Delphi-derived data in
results/r_data/. The R scripts pick between the two streams via env vars
(E{1b,2,3,4,5}_R_DATA_DIR=r_data|r_data_dev; E1 has no such override, always r_data).
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"

CRITERIA = [
    "geometric_consistency",
    "structural_consistency",
    "semantic_consistency",
    "aesthetics",
    "text_3d_alignment",
]

# Objaverse exposure grouping for E4 DIF analysis.
# Group A (1) = primarily trained on Objaverse
# Group B (0) = not / minimal Objaverse training
# Sources: original model papers + Hi3DEval paper (Table 1)
OBJAVERSE_GROUP = {
    "spard":    1,  # SPARD / Zero123++ training includes Objaverse
    "trellis":  1,  # TRELLIS uses Objaverse
    "crm":      1,  # CRM trained on Objaverse
    "unique3d": 1,  # Unique3D uses Objaverse
    "triposr":  0,  # TripoSR primarily trained on G-Objaverse (filtered, different distribution)
    "hunyuan":  1,  # HunyuanDiT 3D uses Objaverse
}


def load_tensor(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prep_from_tensor(df: pd.DataFrame, dichotomize_threshold: int = 5, out_dir: str = "r_data") -> None:
    out = RESULTS / out_dir
    out.mkdir(exist_ok=True)

    # Use final round scores (max round per model/prompt/judge/criterion)
    df_final = (df.sort_values("round")
                  .groupby(["model", "prompt_id", "judge", "criterion"], as_index=False)
                  .last())

    # ── Long tensor for MFRM / lme4 ──────────────────────────────────────
    df_final.to_csv(out / "long_tensor.csv", index=False)
    print(f"long_tensor.csv: {len(df_final)} rows")

    # ── Wide final: judge-averaged, one row per (model, prompt), cols = 5 criteria ──
    wide = (df_final.groupby(["model", "prompt_id", "criterion"])["score"]
                    .mean()
                    .unstack("criterion")
                    .reset_index())
    wide.columns.name = None
    wide.to_csv(out / "wide_final.csv", index=False)
    print(f"wide_final.csv: {len(wide)} rows × {len(wide.columns)} cols")

    # ── Binary pass/fail item matrices for E3/E4/E5 (flipped 2PL IRT etc.) ──
    # item_matrix.csv must be a genuine binary (0/1) matrix: e3_item_params.R
    # treats colMeans/rowMeans as pass-rate proportions and mirt's 2PL requires
    # exactly 2 categories per item. A prior version of this function produced
    # a 0-4 ordinal composite instead, which silently broke the 2PL fit
    # (falls back to classical stats) and made "pass rate" not actually a
    # proportion -- despite this module's own docstring documenting
    # item_matrix.csv as a "binary matrix" from the start.
    #
    # A model with no row for a prompt_id that OTHER models in this stream
    # were rated on (e.g. a generation that failed) is scored as an
    # automatic fail (0) on every criterion for that prompt, rather than
    # excluded. CRM in particular has automated scores for only 247 of the
    # 510 prompts; treating the other 263 as missing (and excluding them)
    # would silently shrink the eligible sample instead of reflecting a real
    # benchmark outcome (failure to generate). This must be disclosed
    # wherever N or coverage is reported.
    #
    # The prompt universe is the union of prompt_ids actually attempted by
    # ANY model in df_final -- NOT the full 510-prompt shared library.
    # The Delphi stream deliberately samples only 100 of the 510 prompts by
    # design (cost); treating the other 410 as automatic fails would
    # conflate "never attempted" with "attempted and failed to generate",
    # wrongly forcing 410/510 = 80% zero-variance items. The automated
    # stream's union still equals all 510 (every model but CRM has full
    # coverage), so this reduces to the intended fill for CRM specifically.
    all_prompts = sorted(df_final["prompt_id"].unique())
    models_present = sorted(df_final["model"].unique())

    def raw_matrix(crit: str) -> pd.DataFrame:
        sub = df_final[df_final["criterion"] == crit]
        mat = sub.groupby(["model", "prompt_id"])["score"].mean().unstack("prompt_id")
        return mat.reindex(index=models_present, columns=all_prompts).fillna(0.0)

    crit_mat_raw = {crit: raw_matrix(crit) for crit in CRITERIA}

    # The 5 criteria have very different native ranges even after rescaling
    # (e.g. geometric_consistency spans ~2-7, but structural_consistency and
    # aesthetics cap at 3-4, semantic_consistency at 0-1) -- averaging them
    # and thresholding at a fixed value like 5 is not meaningful (most
    # criteria can never reach 5). E3's flipped 2PL therefore dichotomizes
    # geometric_consistency alone -- matching the paper's own Table 4 caption
    # ("scores >= 5 on geometric consistency") -- not a cross-criterion
    # composite. This is a real, criterion-scale-driven constraint, not an
    # arbitrary choice: it is the one criterion whose observed range (2-7)
    # actually straddles a threshold of 5.
    main_crit = "geometric_consistency"
    main_raw = crit_mat_raw[main_crit]

    # Per-criterion matrices for the E3 supplementary discrimination plot use
    # each criterion's own observed midpoint as its dichotomization threshold,
    # since a shared threshold of 5 would be degenerate (always 0) for the
    # criteria capped below it.
    crit_thresholds = {}
    for crit in CRITERIA:
        lo, hi = crit_mat_raw[crit].values.min(), crit_mat_raw[crit].values.max()
        thresh = round((lo + hi) / 2)
        crit_thresholds[crit] = thresh
        (crit_mat_raw[crit] >= thresh).astype(int).to_csv(out / f"item_matrix_{crit}.csv")
    print("  Per-criterion dichotomization thresholds (own range midpoint):", crit_thresholds)

    for thresh in sorted({dichotomize_threshold, 3, 4, 6, 7}):
        suffix = "" if thresh == dichotomize_threshold else f"_thresh{thresh}"
        (main_raw >= thresh).astype(int).to_csv(out / f"item_matrix{suffix}.csv")

    zero_var_pct = {
        t: float(((main_raw >= t).astype(int).var(axis=0) == 0).mean())
        for t in (3, 4, 5, 6, 7)
    }
    print(f"item_matrix.csv: {main_raw.shape[0]} models x {main_raw.shape[1]} prompts "
          f"(binary on {main_crit}, threshold={dichotomize_threshold})")
    print("  Zero-variance item % by threshold:",
          {k: round(v * 100, 1) for k, v in zero_var_pct.items()})
    print("  Per-model pass rate (main threshold):")
    main_bin = (main_raw >= dichotomize_threshold).astype(int)
    for m in main_bin.index:
        print(f"    {m}: {main_bin.loc[m].mean():.3f}")

    # ── DIF groups ────────────────────────────────────────────────────────
    dif_df = pd.DataFrame({
        "model": models_present,
        "objaverse_group": [OBJAVERSE_GROUP.get(m, -1) for m in models_present],
    })
    dif_df.to_csv(out / "dif_groups.csv", index=False)
    print(f"dif_groups.csv: {len(dif_df)} models")

    print(f"\nAll R data written to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", default=str(RESULTS / "rating_tensor.parquet"),
                        help="Path to rating tensor (parquet or csv)")
    parser.add_argument("--dev", action="store_true",
                        help="Use Hi3DBench automated scores as development data")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Dichotomization threshold (0-9 scale)")
    args = parser.parse_args()

    if args.dev:
        from data_utils import reshape_hi3dbench_to_tensor
        hi3d_path = str(ROOT / "data" / "hi3dbench_object_level.json")
        df = reshape_hi3dbench_to_tensor(hi3d_path)
        print(f"Using Hi3DBench development data: {len(df)} rows")
        out_dir = "r_data_dev"
    else:
        df = load_tensor(args.tensor)
        print(f"Loaded tensor: {len(df)} rows")
        out_dir = "r_data"

    prep_from_tensor(df, args.threshold, out_dir)
