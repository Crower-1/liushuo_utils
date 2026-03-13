"""Utility to split vesicle labels into spherical and ellipsoidal masks."""
import argparse
import json
import re
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from mrc.io import get_tomo_with_voxel_size, save_tomo

_NAME_PATTERN = re.compile(r"vesicle_(\d+)$")


def _load_label_mrc(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load label volume and capture voxel spacing."""
    data, voxel_size = get_tomo_with_voxel_size(path)
    return data, voxel_size


def _save_mask(mask: np.ndarray, path: str, voxel_size: Tuple[float, float, float]) -> None:
    """Persist a binary mask as an MRC file."""
    save_tomo(mask, path, voxel_size=17.14, datetype=np.uint8)


def _parse_vesicle_id(name: str) -> int:
    match = _NAME_PATTERN.search(name)
    if not match:
        raise ValueError(f"Unable to parse vesicle id from name '{name}'")
    return int(match.group(1))


def _categorize_vesicles(vesicles: Iterable[dict], ratio_threshold: float) -> Tuple[Set[int], Set[int]]:
    """Split vesicle ids into ellipsoidal and spherical groups."""
    ellipsoidal: Set[int] = set()
    spherical: Set[int] = set()

    for vesicle in vesicles:
        name = vesicle.get("name")
        if not name:
            print("Skipping vesicle entry without name.")
            continue

        try:
            vesicle_id = _parse_vesicle_id(name)
        except ValueError as exc:
            print(f"Warning: {exc}; skipping.")
            continue

        radii = vesicle.get("radii")
        if not radii or len(radii) != 3:
            print(f"Warning: vesicle '{name}' missing three radii values; skipping.")
            continue

        longest = max(radii)
        shortest = min(radii)
        if shortest <= 0:
            print(f"Warning: vesicle '{name}' has non-positive shortest radius; skipping.")
            continue

        ratio = longest / shortest
        if ratio > ratio_threshold:
            ellipsoidal.add(vesicle_id)
        else:
            spherical.add(vesicle_id)

    return ellipsoidal, spherical


def _build_mask(labels: np.ndarray, ids: Set[int]) -> np.ndarray:
    if not ids:
        return np.zeros_like(labels, dtype=np.uint8)
    return np.isin(labels, list(ids)).astype(np.uint8)


def split_vesicles(
    label_path: str,
    json_path: str,
    ellipsoidal_out: str,
    spherical_out: str,
    ratio_threshold: float = 1.18,
) -> None:
    labels, voxel_size = _load_label_mrc(label_path)

    with open(json_path, "r") as fh:
        vesicles = json.load(fh).get("vesicles", [])

    ellipsoidal_ids, spherical_ids = _categorize_vesicles(vesicles, ratio_threshold)
    print(f"Ellipsoidal ids ({len(ellipsoidal_ids)}): {sorted(ellipsoidal_ids)}")
    print(f"Spherical ids ({len(spherical_ids)}): {sorted(spherical_ids)}")

    ellipsoidal_mask = _build_mask(labels, ellipsoidal_ids)
    spherical_mask = _build_mask(labels, spherical_ids)

    _save_mask(ellipsoidal_mask, ellipsoidal_out, voxel_size)
    _save_mask(spherical_mask, spherical_out, voxel_size)
    print(f"Saved ellipsoidal mask to {ellipsoidal_out}")
    print(f"Saved spherical mask to {spherical_out}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split vesicle mask into spherical and ellipsoidal subsets.")
    parser.add_argument("label_mrc", help="Path to the input label mask MRC file.")
    parser.add_argument("json_path", help="Path to the vesicle metadata JSON file.")
    parser.add_argument("ellipsoidal_output", help="Path to save the ellipsoidal binary mask MRC file.")
    parser.add_argument("spherical_output", help="Path to save the spherical binary mask MRC file.")
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.18,
        help="Axis ratio threshold separating ellipsoidal from spherical vesicles (default: 1.18).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    split_vesicles(
        label_path=args.label_mrc,
        json_path=args.json_path,
        ellipsoidal_out=args.ellipsoidal_output,
        spherical_out=args.spherical_output,
        ratio_threshold=args.ratio_threshold,
    )


if __name__ == "__main__":
    main()
