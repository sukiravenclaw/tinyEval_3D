"""
Delphi-structured LLM judging pipeline for 3D generative model evaluation.

Usage (from project root, with ML gateway credentials in .env):
    python code/delphi_pipeline.py [--models spard trellis unique3d triposr hunyuan]
                                   [--n-prompts 160]
                                   [--workers 5]       # Parallel items (default 5)
                                   [--sample-test]     # Run 2 items to verify setup
                                   [--resume]          # Resume from checkpoint
"""
from __future__ import annotations
import argparse, json, logging, os, random, shutil, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from llm_client import call_judge, summarize_dissent, test_all_judges, JUDGES
from data_utils import (
    CRITERIA, JUDGE_PERSONAS, ROUND1_USER_TEMPLATE, ROUND2_USER_TEMPLATE,
    parse_judge_response, compute_round_stats, should_skip_round3,
    build_long_tensor, compute_convergence_table, save_tensor,
    get_automated_scores,
)

ROOT = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
RENDERED_DIR  = DATA_DIR / "rendered_zips"
RESULTS       = ROOT / "results"
LOGS          = ROOT / "logs"
TMP_IMGS      = ROOT / "tmp_images"

MODELS_DEFAULT = ["spard", "trellis", "unique3d", "triposr", "hunyuan"]
HF_BASE = "https://huggingface.co/datasets/3DTopia/Hi3DBench/resolve/main"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGS / "delphi_pipeline.log"),
              logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_prompts() -> dict[int, str]:
    with open(DATA_DIR / "text_prompts.json") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {i: v for i, v in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


def download_model_zip(model: str, dest: Path) -> Path:
    zip_path = dest / f"{model}.zip"
    if zip_path.exists():
        # A previous interrupted download can leave a corrupt file behind.
        if zipfile.is_zipfile(zip_path):
            logger.info("Zip already downloaded: %s", zip_path)
            return zip_path
        logger.warning("Invalid zip detected, re-downloading: %s", zip_path)
        zip_path.unlink(missing_ok=True)
    url = f"{HF_BASE}/{model}.zip"
    logger.info("Downloading %s ...", url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print()
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Downloaded file is not a valid zip: {zip_path}")
    return zip_path


def extract_png(zip_path: Path, model: str, prompt_idx: int, out_dir: Path) -> Path | None:
    out_path = out_dir / f"{model}_{prompt_idx}.png"
    if out_path.exists():
        return out_path
    try:
        with zipfile.ZipFile(zip_path) as z:
            png_name = f"{model}/{prompt_idx}.png"
            if png_name not in z.namelist():
                return None
            with z.open(png_name) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    except zipfile.BadZipFile:
        logger.error("Corrupted zip encountered: %s", zip_path)
        return None
    return out_path


def archive_contains_images(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                lower = name.lower()
                if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def select_prompts(n: int, models: list[str], hi3d_scores: dict) -> list[int]:
    """Select n prompt IDs that have rendered images for all specified models."""
    import re
    available: dict[str, set[int]] = {}
    pattern = re.compile(r'image2shape_(\w+)_(\d+)')
    for key in hi3d_scores:
        m = pattern.match(key)
        if m and m.group(1) in models:
            available.setdefault(m.group(1), set()).add(int(m.group(2)))

    common = set.intersection(*available.values()) if available else set()
    return sorted(list(common)[:n])


# ── Per-item Delphi elicitation ───────────────────────────────────────────────

def elicit_round1(model: str, prompt_idx: int, image_path: Path,
                  prompt_text: str) -> dict:
    user_msg = ROUND1_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        vqa_questions="Does the 3D asset match the text prompt?")

    def _call(judge: str) -> tuple[str, dict | None]:
        logger.debug("Round 1 %s/%s judge=%s", model, prompt_idx, judge)
        resp = call_judge(judge, JUDGE_PERSONAS[judge], user_msg, str(image_path))
        if not resp["ok"]:
            logger.error("Judge %s failed: %s", judge, resp["error"])
            return judge, None
        parsed = parse_judge_response(resp["text"])
        if parsed is None:
            logger.warning("Could not parse JSON from %s round 1. Raw response:\n%s",
                           judge, resp["text"][:2000])
        return judge, parsed

    with ThreadPoolExecutor(max_workers=len(JUDGES)) as ex:
        return dict(ex.map(_call, JUDGES))


def elicit_round_n(round_num: int, model: str, prompt_idx: int, image_path: Path,
                   prompt_text: str, prev_round_results: dict,
                   dissent_summaries: dict[str, str],
                   automated_scores: dict) -> dict:
    scores_by_judge = {
        judge: {c: data[c]["score"] for c in CRITERIA if c in data}
        for judge, data in prev_round_results.items() if data is not None
    }
    group_stats = compute_round_stats(scores_by_judge)

    def _call(judge: str) -> tuple[str, dict | None]:
        if prev_round_results.get(judge) is None:
            return judge, None
        own_scores_reasoning = {
            c: {
                "score":     prev_round_results[judge][c]["score"],
                "reasoning": prev_round_results[judge][c].get("reasoning", ""),
            }
            for c in CRITERIA if c in prev_round_results[judge]
        }
        auto_note = automated_scores if automated_scores else {"note": "not available for this model/prompt"}
        user_msg = ROUND2_USER_TEMPLATE.format(
            prompt_text=prompt_text,
            round_num=round_num,
            own_scores_reasoning_json=json.dumps(own_scores_reasoning, indent=2),
            group_stats_json=json.dumps(group_stats, indent=2),
            automated_scores_json=json.dumps(auto_note, indent=2),
            dissent_summary=json.dumps(dissent_summaries, indent=2),
        )
        resp = call_judge(judge, JUDGE_PERSONAS[judge], user_msg, str(image_path))
        if not resp["ok"]:
            logger.error("Judge %s round %d failed: %s", judge, round_num, resp["error"])
            return judge, None
        return judge, parse_judge_response(resp["text"])

    with ThreadPoolExecutor(max_workers=len(JUDGES)) as ex:
        return dict(ex.map(_call, JUDGES))


def build_dissent_summaries(round_results: dict) -> dict[str, str]:
    def _summarize(criterion: str) -> tuple[str, str]:
        scores, reasonings = [], []
        for judge, data in round_results.items():
            if data and criterion in data:
                scores.append(data[criterion]["score"])
                reasonings.append(data[criterion].get("reasoning", ""))
        if len(set(scores)) > 1:
            return criterion, summarize_dissent(criterion, scores, reasonings)
        return criterion, "Judges were in agreement on this criterion."

    with ThreadPoolExecutor(max_workers=len(CRITERIA)) as ex:
        return dict(ex.map(_summarize, CRITERIA))


# ── Flatten to rows ───────────────────────────────────────────────────────────

def flatten_round(model: str, prompt_idx: int, round_num: int, round_results: dict) -> list[dict]:
    rows = []
    for judge, data in round_results.items():
        if data is None:
            continue
        for criterion in CRITERIA:
            if criterion not in data:
                continue
            rows.append({
                "model": model,
                "prompt_id": prompt_idx,
                "judge": judge,
                "round": round_num,
                "criterion": criterion,
                "score": data[criterion]["score"],
                "reasoning": data[criterion].get("reasoning", ""),
                "revised": data[criterion].get("revised", False),
            })
    return rows


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def checkpoint_path(model: str, prompt_idx: int) -> Path:
    return RESULTS / "checkpoints" / f"{model}_{prompt_idx}.json"


def load_checkpoint(model: str, prompt_idx: int) -> dict | None:
    p = checkpoint_path(model, prompt_idx)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_checkpoint(model: str, prompt_idx: int, data: dict) -> None:
    p = checkpoint_path(model, prompt_idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


# ── Per-item orchestration ────────────────────────────────────────────────────

def _process_item(
    model: str, prompt_idx: int,
    prompts: dict, model_zips: dict, hi3d_scores: dict, resume: bool,
) -> tuple[list[dict], list[dict], int, int]:
    """Run all Delphi rounds for one (model, prompt_idx). Returns (rows, conv_entries, skipped_r3, missing)."""
    if resume:
        ckpt = load_checkpoint(model, prompt_idx)
        if ckpt:
            logger.info("Resumed checkpoint: %s/%s", model, prompt_idx)
            return ckpt["rows"], [], 0, 0

    image_path = extract_png(model_zips[model], model, prompt_idx, TMP_IMGS)
    if image_path is None:
        logger.warning("No image for %s prompt %d, skipping", model, prompt_idx)
        return [], [], 0, 1

    prompt_text = prompts.get(prompt_idx, f"prompt_{prompt_idx}")

    # Human-validated automated scores — shown to judges in Round 2/3 as anchor,
    # not in Round 1 (preserves Delphi independence on first pass).
    auto_scores = get_automated_scores(hi3d_scores, model, prompt_idx)

    r1 = elicit_round1(model, prompt_idx, image_path, prompt_text)
    rows_r1 = flatten_round(model, prompt_idx, 1, r1)
    dissent_1 = build_dissent_summaries(r1)

    r2 = elicit_round_n(2, model, prompt_idx, image_path, prompt_text, r1, dissent_1, auto_scores)
    rows_r2 = flatten_round(model, prompt_idx, 2, r2)

    scores_r2 = {j: {c: r2[j][c]["score"] for c in CRITERIA if c in r2[j]} for j in r2 if r2[j]}
    stats_r2 = compute_round_stats(scores_r2)
    conv_entries = [{"model": model, "prompt_id": prompt_idx, "round": 2,
                     **{f"{c}_sd": stats_r2[c]["sd"] for c in CRITERIA}}]

    all_rows = rows_r1 + rows_r2
    skipped = 0

    if should_skip_round3(stats_r2):
        skipped = 1
        logger.debug("Skipping round 3 for %s/%s (converged)", model, prompt_idx)
    else:
        dissent_2 = build_dissent_summaries(r2)
        r3 = elicit_round_n(3, model, prompt_idx, image_path, prompt_text, r2, dissent_2, auto_scores)
        rows_r3 = flatten_round(model, prompt_idx, 3, r3)
        all_rows += rows_r3
        scores_r3 = {j: {c: r3[j][c]["score"] for c in CRITERIA if c in r3[j]} for j in r3 if r3[j]}
        stats_r3 = compute_round_stats(scores_r3)
        conv_entries.append({"model": model, "prompt_id": prompt_idx, "round": 3,
                              **{f"{c}_sd": stats_r3[c]["sd"] for c in CRITERIA}})

    save_checkpoint(model, prompt_idx, {"rows": rows_r1 + rows_r2})
    return all_rows, conv_entries, skipped, 0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(models: list[str], n_prompts: int, sample_test: bool,
                 resume: bool, workers: int = 5,
                 rendered_zips_dir: Path | None = None) -> None:
    RESULTS.mkdir(exist_ok=True)
    TMP_IMGS.mkdir(exist_ok=True)
    (RESULTS / "checkpoints").mkdir(exist_ok=True)

    logger.info("=== Phase 1: Delphi rating pipeline ===")
    logger.info("Models: %s | Prompts: %d | Workers: %d | Sample: %s",
                models, n_prompts, workers, sample_test)

    logger.info("Testing API access...")
    if not test_all_judges():
        raise SystemExit("One or more LLM judges are unreachable. Check ML gateway credentials.")

    prompts = load_prompts()
    with open(DATA_DIR / "hi3dbench_object_level.json") as f:
        hi3d_scores = json.load(f)

    prompt_ids = select_prompts(n_prompts, models, hi3d_scores)
    if sample_test:
        prompt_ids = prompt_ids[:2]
        logger.info("Sample test mode: using prompts %s", prompt_ids)

    rdir = rendered_zips_dir if rendered_zips_dir is not None else RENDERED_DIR
    zip_cache = DATA_DIR / "zips"
    zip_cache.mkdir(exist_ok=True)
    model_zips = {}
    for model in models:
        rendered_path = rdir / f"{model}.zip"
        if rendered_path.exists() and archive_contains_images(rendered_path):
            logger.info("Using rendered zip for %s: %s", model, rendered_path)
            model_zips[model] = rendered_path
        else:
            if rendered_path.exists():
                logger.warning(
                    "Rendered zip exists for %s but contains no images — "
                    "re-run render_glb_to_png.py, then retry.", model)
            model_zips[model] = download_model_zip(model, zip_cache)

    if not any(archive_contains_images(model_zips[m]) for m in models):
        raise SystemExit(
            "No render images found. Run first:\n"
            "  python code/render_glb_to_png.py\n"
            "then retry the Delphi pipeline."
        )

    # ── Parallel rating loop ───────────────────────────────────────────────
    all_rows: list[dict] = []
    convergence_stats: list[dict] = []
    skipped_r3 = 0
    missing_images = 0

    items = [(m, p) for m in models for p in prompt_ids]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_item, m, p, prompts, model_zips, hi3d_scores, resume): (m, p)
            for m, p in items
        }
        with tqdm(total=len(items), desc="Delphi elicitation") as pbar:
            for future in as_completed(futures):
                rows, conv_entries, skipped, missing = future.result()
                all_rows.extend(rows)
                convergence_stats.extend(conv_entries)
                skipped_r3 += skipped
                missing_images += missing
                pbar.update(1)

    if not all_rows:
        raise SystemExit(
            f"No rating rows were produced. Missing images for {missing_images}/{len(items)} items. "
            "Check asset archives and expected PNG paths before rerunning."
        )

    tensor = build_long_tensor(all_rows)
    tensor_path = RESULTS / "rating_tensor.parquet"
    save_tensor(tensor, str(tensor_path))
    logger.info("Tensor saved: %s (%d rows)", tensor_path, len(tensor))

    conv_df = pd.DataFrame(convergence_stats)
    conv_df.to_csv(RESULTS / "convergence_diagnostics.csv", index=False)
    logger.info("Convergence table saved. Round-3 skipped for %d/%d items.", skipped_r3, len(items))

    print("\n=== Phase 1 Complete ===")
    print(f"Tensor: {tensor_path}  ({len(tensor)} rows)")
    print(f"Models: {models}  |  Prompts: {len(prompt_ids)}")
    print(f"Round-3 skipped (converged): {skipped_r3}/{len(items)}")
    print("\nPer-round inter-judge SD (mean across prompts):")
    if not conv_df.empty:
        sd_cols = [c for c in conv_df.columns if c.endswith("_sd")]
        print(conv_df.groupby("round")[sd_cols].mean().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS_DEFAULT)
    parser.add_argument("--n-prompts", type=int, default=160)
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel items (default 5); raise if API rate limits allow")
    parser.add_argument("--sample-test", action="store_true",
                        help="Run 2 items only to verify setup")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items with existing checkpoints")
    parser.add_argument("--rendered-zips-dir", type=Path, default=None,
                        help="Directory containing rendered PNG zips (default: data/rendered_zips/)")
    args = parser.parse_args()
    run_pipeline(args.models, args.n_prompts, args.sample_test, args.resume,
                 workers=args.workers, rendered_zips_dir=args.rendered_zips_dir)
