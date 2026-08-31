"""
Data utilities: tensor construction, statistics, convergence checks, Parquet I/O.
"""
import json, statistics, re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

CRITERIA = [
    "geometric_consistency",
    "structural_consistency",
    "semantic_consistency",
    "aesthetics",
    "text_3d_alignment",
]

# Hi3DBench → Eval3D criterion name mapping
HI3D_TO_EVAL3D = {
    "geometry_score":    "geometric_consistency",
    "geo_detail_score":  "structural_consistency",
    "geo_texture_score": "semantic_consistency",
    "texture_score":     "aesthetics",
    "alignment_score":   "text_3d_alignment",
}

# Observed score ranges in Hi3DBench (from dataset inspection).
# Used to give judges interpretive context alongside raw scores.
HI3D_SCORE_RANGES = {
    "geometric_consistency": (2.5, 7.7, 10),   # (min_obs, max_obs, scale_max)
    "structural_consistency": (1.0, 8.3, 10),
    "semantic_consistency":   (0.0, 7.0, 10),
    "aesthetics":             (0.5, 7.5, 10),
    "text_3d_alignment":      (1.2, 4.0,  4),  # alignment tops out at 4, not 10
}


def get_automated_scores(hi3d_scores: dict, model: str, prompt_idx: int) -> dict:
    """
    Return Hi3DBench automated scores for a (model, prompt) pair,
    mapped to Eval3D criterion names.
    Returns {} if the entry is not found (e.g. crm incomplete coverage).
    """
    key = f"image2shape_{model}_{prompt_idx}"
    entry = hi3d_scores.get(key)
    if not entry:
        return {}
    return {
        eval3d_crit: entry[hi3d_crit]
        for hi3d_crit, eval3d_crit in HI3D_TO_EVAL3D.items()
        if hi3d_crit in entry
    }

# Generic fallback (not used in Delphi runs — each judge gets a distinct persona below)
SYSTEM_PROMPT = (
    "You are an expert evaluator of 3D generative model outputs, with deep knowledge of "
    "geometry, computer graphics, perception, and language–image alignment. "
    "You will rate a single 3D asset on five criteria. Respond with structured JSON only, "
    "after a brief reasoning step."
)

# Per-judge personas that give each model a distinct evaluative lens.
# Diversity is the key Delphi ingredient — each judge emphasises different failure modes.
JUDGE_PERSONAS = {
    "claude": (
        "You are a senior 3D technical artist with 15 years of production experience in film VFX "
        "and real-time game assets. Your eye is trained on surface quality: you immediately notice "
        "broken normals, texture seams, inconsistent shading, and the telltale 'baked-flat' look "
        "of poor geometry. You hold aesthetics to a high standard and reward assets that look "
        "hand-crafted. You are methodical and conservative — you do not give high scores unless "
        "the asset genuinely earns them. You will rate a single 3D asset on five criteria. "
        "Respond with structured JSON only, after a brief reasoning step."
    ),
    "gpt": (
        "You are a computer vision researcher specialising in 3D reconstruction and neural "
        "rendering quality metrics. You think in terms of multi-view consistency, geometric "
        "fidelity, and structural plausibility. You are acutely sensitive to the Janus problem "
        "(repeated facial features), view-dependent artefacts, and topology errors. You evaluate "
        "text–3D alignment by decomposing the prompt into verifiable visual predicates and "
        "checking each one systematically. You are precise and analytical. You will rate a single "
        "3D asset on five criteria. Respond with structured JSON only, after a brief reasoning step."
    ),
    "gemini": (
        "You are a product designer and UX researcher who evaluates 3D assets for use in "
        "e-commerce, AR/VR product visualisation, and educational media. You care most about "
        "whether the object is immediately recognisable, whether it communicates the right "
        "semantic content, and whether a non-expert viewer would find it convincing. You are "
        "tolerant of minor geometric imperfections if the overall impression is clear and "
        "appealing, but you penalise hard anything that would mislead a user about what the "
        "object is. You will rate a single 3D asset on five criteria. Respond with structured "
        "JSON only, after a brief reasoning step."
    ),
}

ROUND1_USER_TEMPLATE = """\
You are shown a multi-view rendering of a single 3D asset generated from the text prompt: "{prompt_text}"

Rate this asset on each of the following five criteria using a 0–9 integer scale \
where 0 = severe failure and 9 = ideal quality. \
For each criterion, first write 2–3 sentences of reasoning, then output the integer score.

Criteria definitions:
- geometric_consistency: Do the rendered surface normals appear consistent with the visible \
texture and lighting? Penalize floating fragments, jagged surfaces, normal–texture mismatches.
- structural_consistency: Is the asset globally coherent across views? Look for Janus issues \
(faces appearing on multiple sides), implausible part placement, view-dependent geometry collapse.
- semantic_consistency: Does the asset maintain consistent semantic content across viewpoints \
(always recognizably the same object), without content drift or hallucinated features?
- aesthetics: Is the asset visually appealing in terms of color, composition, and surface detail? \
Rate honestly.
- text_3d_alignment: Does the asset accurately depict the input prompt? Consider: \
{vqa_questions} Penalize missing prompt elements and incorrect attributes.

Output format: Reply with a single JSON object exactly matching this schema:
{{"geometric_consistency": {{"reasoning": "...", "score": 0}},
  "structural_consistency": {{"reasoning": "...", "score": 0}},
  "semantic_consistency": {{"reasoning": "...", "score": 0}},
  "aesthetics": {{"reasoning": "...", "score": 0}},
  "text_3d_alignment": {{"reasoning": "...", "score": 0}}}}

Do not include any text outside the JSON object. Do not use markdown code fences. \
Use integers only for scores. If genuinely undecided, prefer the lower score and explain."""

ROUND2_USER_TEMPLATE = """\
You are shown a multi-view rendering of a single 3D asset generated from the text prompt: "{prompt_text}"

This is Round {round_num} of a Delphi elicitation. You previously rated this asset on five criteria. \
Re-examine the image carefully. You may revise or confirm each score — either is fine. \
If you revise, explain what you noticed that changed your view.

--- YOUR PREVIOUS SCORES AND REASONING ---
{own_scores_reasoning_json}

--- GROUP STATISTICS (median and IQR across all judges, previous round) ---
{group_stats_json}

--- AUTOMATED REFERENCE SCORES (Hi3DBench, validated against human judgments) ---
{automated_scores_json}

Scale note: geometric_consistency / structural_consistency / semantic_consistency / aesthetics \
are on a 0–10 scale (observed range ~0–8); text_3d_alignment is on a 0–4 scale. \
These represent human-validated baseline quality assessments. Consider whether your ratings \
are directionally consistent with this baseline. If you diverge substantially (>2 points on \
the equivalent 0–9 scale), explain why your visual assessment differs from the automated score.

--- ANONYMIZED DISSENT SUMMARY ---
{dissent_summary}

Output the same JSON schema as Round 1, adding a "revised" field (true if you changed the score):
{{"geometric_consistency": {{"reasoning": "...", "score": 0, "revised": false}},
  "structural_consistency": {{"reasoning": "...", "score": 0, "revised": false}},
  "semantic_consistency": {{"reasoning": "...", "score": 0, "revised": false}},
  "aesthetics": {{"reasoning": "...", "score": 0, "revised": false}},
  "text_3d_alignment": {{"reasoning": "...", "score": 0, "revised": false}}}}"""


def parse_judge_response(text: str):
    """Extract JSON object from judge response, handling minor formatting issues."""
    text = text.strip()
    # Strip thinking tags (Gemini 2.5 Pro emits <think>...</think> before output)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try each '{' as a potential JSON start (handles preamble text before the object)
        for m in re.finditer(r'\{', text):
            try:
                return json.loads(text[m.start():])
            except json.JSONDecodeError:
                continue
    return None


def compute_round_stats(scores_by_judge: dict) -> dict[str, dict]:
    """Given {judge: {criterion: score}}, compute per-criterion median, IQR, SD."""
    stats = {}
    for criterion in CRITERIA:
        vals = [scores_by_judge[j][criterion] for j in scores_by_judge
                if criterion in scores_by_judge[j]]
        if not vals:
            stats[criterion] = {"median": None, "iqr": None, "sd": None}
            continue
        vals_sorted = sorted(vals)
        median = statistics.median(vals)
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[criterion] = {"median": median, "iqr": round(q3 - q1, 3), "sd": round(sd, 3)}
    return stats


def should_skip_round3(round2_stats: dict) -> bool:
    """Return True if >= 4 of 5 criteria have inter-judge SD < 1.0 after round 2."""
    below = sum(1 for c in CRITERIA
                if round2_stats.get(c, {}).get("sd", 999) < 1.0)
    return below >= 4


def build_long_tensor(all_ratings: list) -> pd.DataFrame:
    """
    Convert list of rating dicts to long-format DataFrame.
    Each dict: {model, prompt_id, judge, round, criterion, score, reasoning, revised}
    """
    return pd.DataFrame(all_ratings)


def compute_convergence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-round, per-criterion inter-judge SD table."""
    return (df.groupby(["round", "criterion"])["score"]
              .std()
              .unstack("criterion")
              .round(3)
              .rename_axis(index=None, columns=None))


def save_tensor(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def load_tensor(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def reshape_hi3dbench_to_tensor(hi3d_path: str) -> pd.DataFrame:
    """
    Convert Hi3DBench object-level.json into the same long tensor format.
    Treats automated scores as a single judge ('automated'), single round (0).
    Scores are on a 0–9 scale (Hi3DBench uses 0–10; we rescale).
    """
    with open(hi3d_path) as f:
        raw = json.load(f)

    rows = []
    pattern = re.compile(r'image2shape_(\w+)_(\d+)')
    for key, scores in raw.items():
        m = pattern.match(key)
        if not m:
            continue
        model, prompt_idx = m.group(1), int(m.group(2))
        for hi3d_crit, eval3d_crit in HI3D_TO_EVAL3D.items():
            score_raw = scores.get(hi3d_crit)
            if score_raw is None:
                continue
            # Rescale 0-10 → 0-9
            score = min(9, round(score_raw * 9 / 10))
            rows.append({
                "model": model,
                "prompt_id": prompt_idx,
                "judge": "automated",
                "round": 0,
                "criterion": eval3d_crit,
                "score": score,
                "reasoning": "",
                "revised": False,
            })
    return pd.DataFrame(rows)
