"""
Render .glb/.obj files from data/zips/ into multi-view composite PNGs and
repackage them into data/rendered_zips/{model}.zip with the structure

    {model}/{prompt_idx}.png

expected by delphi_pipeline.py's extract_png().

Usage (from project root):
    python code/render_glb_to_png.py                          # all 5 models
    python code/render_glb_to_png.py --models hunyuan trellis # subset
    python code/render_glb_to_png.py --sample 5              # first 5 prompts only (smoke test)
    python code/render_glb_to_png.py --resume                 # skip prompts already in zip

Each output image is a 1024×1024 composite of 4 views (front/right/back/left
at 30° elevation), suitable for vision-capable LLM judges.

Dependencies: trimesh, pyrender, Pillow, numpy (all in requirements or venv).
On macOS, pyrender uses the native OpenGL backend — no EGL/OSMesa needed.
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RENDERED_DIR = DATA_DIR / "rendered_zips"

# Maps model name → (zip filename, folder prefix inside zip, file extension)
MODEL_CONFIG: dict[str, tuple[str, str, str]] = {
    "hunyuan":  ("hunyuan.zip",  "hunyuan",    "glb"),
    "spard":    ("spard.zip",    "spard_glb",  "glb"),
    "trellis":  ("trellis.zip",  "Trellis_glb","glb"),
    "triposr":  ("triposr.zip",  "triposr",    "obj"),
    "unique3d": ("unique3d.zip", "unique3d",   "glb"),
}

VIEW_AZIMUTHS  = [0, 90, 180, 270]   # degrees, 4 cardinal views
VIEW_ELEVATION = 30                   # degrees above horizon
IMG_SIZE       = 512                  # per-view pixel size; composite = 1024×1024
JPEG_QUALITY   = 92                   # PNG is used but quality flag kept for reference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _load_mesh(data: bytes, file_type: str) -> trimesh.Scene | trimesh.Trimesh:
    return trimesh.load(io.BytesIO(data), file_type=file_type, process=False)


def _to_pyrender_scene(mesh: trimesh.Scene | trimesh.Trimesh) -> pyrender.Scene:
    if isinstance(mesh, trimesh.scene.scene.Scene):
        return pyrender.Scene.from_trimesh_scene(mesh, ambient_light=[0.4, 0.4, 0.4])
    # Single Trimesh — wrap in a Scene
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])
    scene.add(pr_mesh)
    return scene


def _scene_bounds(mesh: trimesh.Scene | trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, trimesh.scene.scene.Scene):
        bounds = mesh.bounds
    else:
        bounds = mesh.bounds
    return bounds[0], bounds[1]


def _camera_pose(center: np.ndarray, dist: float,
                 azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cx = dist * np.cos(el) * np.sin(az) + center[0]
    cy = dist * np.sin(el)              + center[1]
    cz = dist * np.cos(el) * np.cos(az) + center[2]
    cam_pos = np.array([cx, cy, cz])

    fwd   = center - cam_pos
    fwd  /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:          # camera directly above/below
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd
    pose[:3, 3] = cam_pos
    return pose


def render_multiview(mesh_data: bytes, file_type: str,
                     renderer: pyrender.OffscreenRenderer) -> Image.Image:
    """Render a 3D mesh into a 2×2 composite multi-view PNG."""
    mesh = _load_mesh(mesh_data, file_type)
    lo, hi = _scene_bounds(mesh)
    center  = (lo + hi) / 2.0
    extents = hi - lo
    cam_dist = max(extents) * 2.2    # pull back enough to frame the object

    scene = _to_pyrender_scene(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    light  = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    views: list[Image.Image] = []
    for az in VIEW_AZIMUTHS:
        pose     = _camera_pose(center, cam_dist, az, VIEW_ELEVATION)
        cam_node = scene.add(camera, pose=pose)
        lit_node = scene.add(light,  pose=pose)

        color, _ = renderer.render(scene,
                                   flags=pyrender.RenderFlags.RGBA)
        # Composite over white background (color is H×W×4 uint8 RGBA)
        rgba  = Image.fromarray(color)
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        views.append(white.convert("RGB"))

        scene.remove_node(cam_node)
        scene.remove_node(lit_node)

    # 2×2 grid
    W = H = IMG_SIZE
    composite = Image.new("RGB", (W * 2, H * 2), (255, 255, 255))
    for i, img in enumerate(views):
        composite.paste(img.resize((W, H)), ((i % 2) * W, (i // 2) * H))
    return composite


# ── Per-model processing ──────────────────────────────────────────────────────

def rendered_prompts_in_zip(out_zip: Path, model: str) -> set[int]:
    """Return prompt indices already rendered in the output zip."""
    if not out_zip.exists():
        return set()
    try:
        with zipfile.ZipFile(out_zip, "r") as z:
            done = set()
            for name in z.namelist():
                # expect  model/42.png
                parts = name.split("/")
                if len(parts) == 2 and parts[1].endswith(".png"):
                    try:
                        done.add(int(parts[1][:-4]))
                    except ValueError:
                        pass
            return done
    except zipfile.BadZipFile:
        return set()


def process_model(model: str, sample: int | None, resume: bool) -> None:
    zip_name, folder_prefix, ext = MODEL_CONFIG[model]
    src_zip  = DATA_DIR / "zips" / zip_name
    out_zip  = RENDERED_DIR / f"{model}.zip"

    if not src_zip.exists():
        logger.error("Source zip not found: %s — skipping %s", src_zip, model)
        return

    # Enumerate all prompt indices present in the source zip
    with zipfile.ZipFile(src_zip, "r") as z:
        all_indices = sorted(
            int(n.split("/")[1].rsplit(".", 1)[0])
            for n in z.namelist()
            if not n.endswith("/") and n.startswith(f"{folder_prefix}/")
        )

    if sample is not None:
        all_indices = all_indices[:sample]

    if resume:
        done = rendered_prompts_in_zip(out_zip, model)
        todo = [i for i in all_indices if i not in done]
        logger.info("%s: %d already done, %d remaining", model, len(done), len(todo))
    else:
        todo = all_indices

    if not todo:
        logger.info("%s: nothing to render — all done.", model)
        return

    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)

    # Open source zip once; open output zip in append mode
    mode = "a" if (resume and out_zip.exists()) else "w"
    errors = 0

    with zipfile.ZipFile(src_zip, "r") as src, \
         zipfile.ZipFile(out_zip, mode, compression=zipfile.ZIP_DEFLATED) as dst:

        for idx in tqdm(todo, desc=model, unit="mesh"):
            src_name = f"{folder_prefix}/{idx}.{ext}"
            try:
                mesh_data = src.read(src_name)
                img = render_multiview(mesh_data, ext, renderer)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                dst.writestr(f"{model}/{idx}.png", buf.getvalue())

            except KeyError:
                logger.warning("%s: %s not found in zip, skipping", model, src_name)
                errors += 1
            except Exception as exc:
                logger.warning("%s/%d render failed: %s", model, idx, exc)
                errors += 1

    renderer.delete()
    logger.info("%s: done. %d rendered, %d errors. Output: %s",
                model, len(todo) - errors, errors, out_zip)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 3D model zips → multi-view PNG zips for Delphi pipeline."
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_CONFIG.keys()),
        choices=list(MODEL_CONFIG.keys()),
        help="Which models to render (default: all 5)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Render only the first N prompts per model (smoke test)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip prompts already present in output zip",
    )
    args = parser.parse_args()

    logger.info("Models: %s | Sample: %s | Resume: %s",
                args.models, args.sample, args.resume)
    logger.info("Output directory: %s", RENDERED_DIR)

    for model in args.models:
        logger.info("=== Rendering %s ===", model)
        process_model(model, args.sample, args.resume)

    logger.info("All done. Rendered zips are in %s/", RENDERED_DIR)
    logger.info("Run the Delphi pipeline with:")
    logger.info("  python code/delphi_pipeline.py --rendered-zips-dir data/rendered_zips")


if __name__ == "__main__":
    main()
