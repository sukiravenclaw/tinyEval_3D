# tinyEval3D

Code and data for "Delphi-Psychometric Evaluation of 3D Generative Models: Combining
Structured LLM Elicitation with Item Response Theory," presented at the Stanford AIMS
Workshop on AI Measurement Science @ COLM 2026.

Applies Item Response Theory, Many-Facet Rasch Modelling, and a Delphi-structured
multi-round LLM judging protocol to Hi3DBench (six image-to-3D generation models,
510 prompts, five quality criteria). The Delphi LLM-judging pipeline itself covers
five of the six models (100 of the 510 prompts, cost-sampled) — `crm` has automated
Hi3DBench scores only (and only for 247/510 prompts), never LLM-judged. The sixth
model and full 510-prompt coverage appear in the automated-score stream used for
comparison throughout.

## Contents

- `code/` — pipeline scripts
  - `llm_client.py`, `delphi_pipeline.py` — Delphi-structured multi-round LLM judging
  - `data_utils.py`, `prep_r_data.py` — reshape Hi3DBench data into R-ready CSVs
  - `render_glb_to_png.py` — render 3D assets to multi-view images
  - `e1_factor_structure.R` … `e5_adaptive_subset.R`, `run_psychometrics.R` — the five
    psychometric analyses (factor structure, variance decomposition, IRT, DIF,
    adaptive item subset) and their master runner. `run_psychometrics.R` runs E1–E5
    only — **not** `e1b_model_theta.R` (below), which must be run separately.
  - `e1b_model_theta.R`, `analyze_control_experiment.py`, `run_control_experiment.py`,
    `check_model_overlap.py` — supporting analyses
  - `make_delphi_flowchart.py` — generates the Delphi protocol figure
- `data/` — analysis-ready data
  - `hi3dbench_object_level.json` / `.parquet` — object-level automated scores
  - `hi3dbench_material_subject.json` — material-level scores (cross-level analysis)
  - `text_prompts.json` — the 510 prompt strings
- `images_prompts_510/` — multi-view renderings for all 510 prompts, used as Delphi
  judge input
- `results/` — output tables (T1–T5) referenced in the paper
- `ai_measurement_r/R-validate.Rmd` — the most complete reproduction path: sources the
  actual `code/e*.R` scripts (E1→E1b→E2→E3→E4→E5) for the Delphi stream, plus an
  automated-stream comparison section for E3–E5
- `paper/` — COLM manuscript (`main.tex`, `supplementary.tex`) and the submission PDFs

## Not included

Raw per-model rendered-image and mesh archives (`data/zips/`, `data/rendered_zips/`,
tens of GB total) are excluded due to size; available on request.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Python 3.9+; pinned to what's verified working here
```

R 4.5+ with `mirt`, `lme4`, `lavaan`, `psych`, `difR`, `irr`, `jsonlite` installed
(no lockfile — `install.packages(c('mirt','lme4','lavaan','psych','difR','irr','jsonlite'))`).

## Reproducing the analysis

The repo ships with `results/r_data/` already populated from the real Delphi ratings
(`results/rating_tensor.parquet`) — you don't need to regenerate it to explore the
paper's headline numbers; just run:

```bash
Rscript code/run_psychometrics.R     # E1-E5 against results/r_data/ (Delphi stream)
Rscript code/e1b_model_theta.R       # E1b (not included in run_psychometrics.R)
```

To reshape the data yourself, or to compare against the automated/Hi3DBench-score
stream (6 models incl. `crm`, all 510 prompts) used throughout as a robustness check:

```bash
python code/prep_r_data.py           # real Delphi tensor -> results/r_data/
python code/prep_r_data.py --dev     # automated scores   -> results/r_data_dev/ (never touches r_data/)

# then, per analysis script, to run it against the automated stream instead:
E2_R_DATA_DIR=r_data_dev E2_OUT_SUFFIX=_automated Rscript code/e2_variance_decomp.R
# (same pattern for e1b, e3, e4, e5 — see ai_measurement_r/R-validate.Rmd)
```

The Delphi LLM judging pipeline itself (`code/delphi_pipeline.py`, which produces
`results/rating_tensor.parquet` in the first place) requires API credentials supplied
via a local `.env` file (not included) and is run separately.
