import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from mrc.io import get_tomo, save_tomo

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"
DOWNLOAD_DIR = "/home/liushuo/isensee/temp"


def ensure_model_folder() -> str:
    model_folder = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
    if os.path.isdir(model_folder):
        return model_folder

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{MODEL_NAME}/*"],
        local_dir=DOWNLOAD_DIR,
        local_dir_use_symlinks=False,
    )
    return model_folder


def list_images(image_dir: str, exts: Iterable[str]) -> List[str]:
    paths = []
    for entry in sorted(os.listdir(image_dir)):
        lower = entry.lower()
        if any(lower.endswith(ext) for ext in exts):
            paths.append(os.path.join(image_dir, entry))
    return paths


def _strip_known_suffixes(name: str) -> str:
    suffixes = [
        "-wbp",
        "_wbp",
        "-bin8-wbp",
        "-bin4-wbp",
        "-bin2-wbp",
        "-bin8-20i",
        "_bin8-20i",
        "-bin8",
        "-bin4",
        "-bin2",
        "-corrected",
        "_corrected",
    ]
    for suffix in suffixes:
        if suffix and suffix in name:
            name = name.replace(suffix, "")
    # remove patterns like -bin8-20i or _bin4-15i when they are not leading prefixes
    name = re.sub(r"(?<!^)[-_]?bin\d+(?:-\d+i)?", "", name)
    return name


def _strip_known_prefixes(name: str) -> str:
    prefixes = [
        "bin8_rec_",
        "bin4_rec_",
        "bin2_rec_",
        "rec_",
        "bin8_",
        "bin4_",
        "bin2_",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _normalize_token(token: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", token.lower())


def guess_point_file(image_name: str, coords_dir: str) -> str:
    base = os.path.splitext(os.path.basename(image_name))[0]
    candidates = set()

    # Raw base
    candidates.add(base)

    stripped = _strip_known_suffixes(base)
    candidates.add(stripped)

    prefix_stripped = _strip_known_prefixes(stripped)
    candidates.add(prefix_stripped)

    # Replace t### with _ts_###
    converted = re.sub(r"t(\d{3})", r"_ts_\1", prefix_stripped)
    candidates.add(converted)

    # Remove duplicated separators
    double_norm = converted.replace("--", "-").replace("__", "_")
    candidates.add(double_norm)

    for cand in list(candidates):
        cand = cand.strip("-_ ")
        if not cand:
            continue
        possible = [
            os.path.join(coords_dir, f"{cand}.point"),
            os.path.join(coords_dir, f"{cand}_select.point"),
        ]
        for path in possible:
            if os.path.isfile(path):
                return path

    # Fallback to fuzzy search
    norm_base = _normalize_token(base)
    fallback = None
    for fname in sorted(os.listdir(coords_dir)):
        if not fname.lower().endswith(".point"):
            continue
        name_core = _normalize_token(os.path.splitext(fname)[0])
        if not name_core:
            continue
        if norm_base in name_core or name_core in norm_base:
            fallback = os.path.join(coords_dir, fname)
            break

    if fallback:
        return fallback

    raise FileNotFoundError(
        f"Could not find matching .point file for {image_name} in {coords_dir}"
    )


def load_points(point_path: str, scale: float) -> List[Tuple[int, int, int]]:
    pts: List[Tuple[int, int, int]] = []
    with open(point_path, "r") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                if len(parts) >= 5:
                    x_ori, y_ori, z_ori = map(float, parts[-3:])
                else:
                    x_ori, y_ori, z_ori = map(float, parts[:3])
            except ValueError:
                continue
            x = int(round(x_ori * scale))
            y = int(round(y_ori * scale))
            z = int(round(z_ori * scale))
            pts.append((x, y, z))
    return pts


def clamp_point(pt: Sequence[int], shape_xyz: Sequence[int]) -> Tuple[int, int, int]:
    x = min(max(pt[0], 0), shape_xyz[0] - 1)
    y = min(max(pt[1], 0), shape_xyz[1] - 1)
    z = min(max(pt[2], 0), shape_xyz[2] - 1)
    return x, y, z


def prepare_session(device: str) -> nnInteractiveInferenceSession:
    model_folder = ensure_model_folder()
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(model_folder)
    return session


def run_inference_on_image(
    session: nnInteractiveInferenceSession,
    image_path: str,
    point_path: str,
    scale: float,
    output_dir: Optional[str],
    target_pixel_size: float,
    overwrite: bool,
    output_path: Optional[str] = None,
) -> str:
    if output_path is None:
        if output_dir is None:
            raise ValueError("Either output_dir or output_path must be provided.")
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{stem}_nninteractive.mrc")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if os.path.exists(output_path) and not overwrite:
        print(f"[skip] {output_path} exists. Use --overwrite to recompute.")
        return output_path

    tomo = get_tomo(image_path)
    if tomo.ndim != 3:
        raise ValueError(f"Tomogram must be 3D. Got shape {tomo.shape} for {image_path}")

    # mrc axes are (z, y, x); nnInteractive expects (1, x, y, z)
    img = np.transpose(tomo, (2, 1, 0))[None]
    session.set_image(img)

    target_shape = img.shape[1:]
    points = load_points(point_path, scale)
    if not points:
        print(f"[warn] No valid points in {point_path}, skipping {image_path}")
        return output_path

    results = np.zeros_like(img, dtype=np.uint16)
    for label_idx, pt in enumerate(points, start=1):
        target = torch.zeros(target_shape, dtype=torch.uint8, device=session.device)
        session.set_target_buffer(target)
        session.add_point_interaction(clamp_point(pt, target_shape), include_interaction=True)
        mask = (session.target_buffer > 0).cpu().numpy()
        new_region = mask & (results[0] == 0)
        results[0][new_region] = label_idx
        session.reset_interactions()

    # convert back to (z, y, x)
    out_volume = np.transpose(results[0], (2, 1, 0))
    save_tomo(out_volume, output_path, voxel_size=target_pixel_size, datetype=np.uint16)
    print(f"[done] Saved {output_path}")
    return output_path
