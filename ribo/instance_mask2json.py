import argparse
import json
import os
from typing import Dict, List

import numpy as np

from mrc.io import get_tomo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute centroid, volume, and max cross-sectional area for each instance id."
        )
    )
    parser.add_argument(
        "--input-mrc",
        required=True,
        help="Path to input instance label MRC (3D).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--output-excel",
        default="",
        help="Path to output Excel (.xlsx). Default: same as JSON with .xlsx extension.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include background id 0 in the output.",
    )
    return parser.parse_args()


def _validate_labels(data: np.ndarray) -> None:
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}")
    if data.min() < 0:
        raise ValueError("Instance labels must be non-negative integers.")


def compute_instance_stats(data: np.ndarray, include_zero: bool) -> List[Dict[str, object]]:
    _validate_labels(data)
    data = np.asarray(data)
    max_id = int(data.max())
    if max_id == 0 and not include_zero:
        return []

    z_size, y_size, x_size = data.shape
    counts_total = np.zeros(max_id + 1, dtype=np.int64)
    sum_x = np.zeros(max_id + 1, dtype=np.float64)
    sum_y = np.zeros(max_id + 1, dtype=np.float64)
    sum_z = np.zeros(max_id + 1, dtype=np.float64)
    max_area = np.zeros(max_id + 1, dtype=np.int64)

    x_coords_flat = np.tile(np.arange(x_size, dtype=np.float64), y_size)
    y_coords_flat = np.repeat(np.arange(y_size, dtype=np.float64), x_size)

    for z in range(z_size):
        slice_labels = data[z]
        flat = slice_labels.ravel()
        counts = np.bincount(flat, minlength=max_id + 1)
        counts_total += counts
        max_area = np.maximum(max_area, counts)
        sum_x += np.bincount(flat, weights=x_coords_flat, minlength=max_id + 1)
        sum_y += np.bincount(flat, weights=y_coords_flat, minlength=max_id + 1)
        sum_z += counts * z

    ids = np.nonzero(counts_total)[0]
    if not include_zero:
        ids = ids[ids != 0]

    results = []
    for instance_id in ids:
        volume = int(counts_total[instance_id])
        if volume == 0:
            continue
        center_x = float(sum_x[instance_id] / volume)
        center_y = float(sum_y[instance_id] / volume)
        center_z = float(sum_z[instance_id] / volume)
        results.append(
            {
                "id": int(instance_id),
                "center": [center_x, center_y, center_z],
                "volume": volume,
                "max_area": int(max_area[instance_id]),
            }
        )
    return results


def write_json(data: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_excel(data: List[Dict[str, object]], output_path: str) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to write Excel. Please install pandas (and openpyxl)."
        ) from exc

    rows = []
    for item in data:
        center = item.get("center", [None, None, None])
        rows.append(
            {
                "id": item.get("id"),
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "volume": item.get("volume"),
                "max_area": item.get("max_area"),
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)


def main() -> None:
    args = parse_args()
    volume = get_tomo(args.input_mrc)
    stats = compute_instance_stats(volume, include_zero=args.include_zero)
    write_json(stats, args.output_json)
    output_excel = args.output_excel.strip()
    if not output_excel:
        base, _ext = os.path.splitext(args.output_json)
        output_excel = f"{base}.xlsx"
    write_excel(stats, output_excel)


if __name__ == "__main__":
    main()
