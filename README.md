# tinyEval3D

Code and data for "Delphi-Psychometric Evaluation of 3D Generative Models: Combining
Structured LLM Elicitation with Item Response Theory," presented at the Stanford AIMS
Workshop on AI Measurement Science @ COLM 2026.

Applies Item Response Theory, Many-Facet Rasch Modelling, and a Delphi-structured
multi-round LLM judging protocol to Hi3DBench (six image-to-3D generation models,
510 prompts, five quality criteria).

## Contents

- `code/` — pipeline scripts
  - `llm_client.py`, `delphi_pipeline.py` — Delphi-structured multi-round LLM judging
  - `data_utils.py`, `prep_r_data.py` — reshape Hi3DBench data into R-ready CSVs
  - `render_glb_to_png.py` — render 3D assets to multi-view images
  - `e1_factor_structure.R` … `e5_adaptive_subset.R`, `run_psychometrics.R` — the five
    psychometric analyses (factor structure, variance decomposition, IRT, DIF,
    adaptive item subset) and their master runner
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
- `ai_measurement_r/R-validate.Rmd` — R validation notebook
- `pipeline_walkthrough.ipynb` — end-to-end walkthrough of the pipeline

## Not included

Raw per-model rendered-image and mesh archives (`data/zips/`, `data/rendered_zips/`,
tens of GB total) are excluded due to size; available on request.

## Reproducing the analysis

```bash
python code/prep_r_data.py --dev   # reshape Hi3DBench scores into R-ready CSVs
Rscript code/run_psychometrics.R   # run all five psychometric analyses
```

The Delphi LLM judging pipeline (`code/delphi_pipeline.py`) requires API credentials
supplied via a local `.env` file (not included) and is run separately.
