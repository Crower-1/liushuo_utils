import argparse
import os

import numpy as np
import pandas as pd
import mrcfile as mf
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt


def read_mrc(mrc_path):
    with mf.open(mrc_path, permissive=True) as mrc:
        data = mrc.data.copy()
        voxel = mrc.voxel_size
    voxel_size = float(voxel.x) if hasattr(voxel, "x") else float(voxel)
    if hasattr(voxel, "y") and hasattr(voxel, "z"):
        sizes = np.array([float(voxel.x), float(voxel.y), float(voxel.z)], dtype=float)
        if not np.allclose(sizes, sizes[0]):
            voxel_size = float(np.mean(sizes))
    return data, voxel_size


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_value_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if value is None:
        return []
    parts = str(value).split(",")
    values = []
    for part in parts:
        part = part.strip()
        if part == "":
            continue
        values.append(int(part))
    return values


def build_mito_mask(data, membrane_values, body_values):
    values = list(membrane_values) + list(body_values)
    if not values:
        return np.zeros_like(data, dtype=bool)
    return np.isin(data, values)


def clean_and_label(mask, min_voxels=3000):
    labeled, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=np.int32)
    cleaned = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, num + 1):
        instance = labeled == label_id
        if int(np.count_nonzero(instance)) < min_voxels:
            continue
        filled = ndimage.binary_fill_holes(instance)
        cleaned |= filled
    relabeled, _ = ndimage.label(cleaned)
    return relabeled.astype(np.int32)


def surface_area(mask, pixel_size_nm):
    if np.count_nonzero(mask) < 4:
        return np.nan
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(pixel_size_nm,) * 3
    )
    if verts.size == 0 or faces.size == 0:
        return np.nan
    return float(measure.mesh_surface_area(verts, faces))


def compute_metrics(labeled, data, pixel_size_nm, membrane_values):
    rows = []
    voxel_volume = pixel_size_nm ** 3
    membrane_values = set(int(v) for v in membrane_values)
    ids = np.unique(labeled)
    ids = ids[ids != 0]
    for label_id in ids:
        instance = labeled == label_id
        voxel_count = int(np.count_nonzero(instance))
        if membrane_values:
            memb_vox = int(np.count_nonzero(np.isin(data, list(membrane_values)) & instance))
        else:
            memb_vox = 0
        memb_ratio = (memb_vox / voxel_count) if voxel_count > 0 else np.nan
        volume_nm3 = voxel_count * voxel_volume
        area_nm2 = surface_area(instance, pixel_size_nm)
        rows.append(
            {
                "id": int(label_id),
                "voxel_count": voxel_count,
                "memb_vox": memb_vox,
                "memb_vox_ratio": memb_ratio,
                "volume_nm3": volume_nm3,
                "surface_area_nm2": area_nm2,
            }
        )
    return pd.DataFrame(rows)


def make_boxplot(values, label, title, output_path, jitter=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    fig, ax = plt.subplots(figsize=(4.6, 6))
    ax.boxplot(
        values,
        vert=True,
        widths=0.45,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#2f2f2f", "linewidth": 1.6},
        boxprops={"linewidth": 1.2, "edgecolor": "#3a3a3a", "facecolor": "#8c8adf"},
        whiskerprops={"linewidth": 1.1, "color": "#3a3a3a"},
        capprops={"linewidth": 1.1, "color": "#3a3a3a"},
    )
    rng = np.random.default_rng(0)
    xs = 1.0 + rng.uniform(-jitter, jitter, size=values.size)
    ax.scatter(xs, values, s=22, facecolor="white", edgecolor="#1f1f1f", linewidth=0.7, zorder=3)
    ax.set_ylabel(label)
    ax.set_title(title, pad=10)
    ax.set_xlim(0.5, 1.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mitochondria surface area and volume from a semantic MRC mask."
    )
    parser.add_argument("--mask-path", type=str, required=True, help="Input mito mask MRC path.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--membrane-values",
        type=str,
        default="7",
        help="Comma-separated membrane values in mask (default: 7).",
    )
    parser.add_argument(
        "--body-values",
        type=str,
        default="2",
        help="Comma-separated mito body values in mask (default: 2).",
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=300,
        help="Minimum voxel count to keep an instance.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    membrane_values = parse_value_list(args.membrane_values)
    body_values = parse_value_list(args.body_values)
    data, pixel_size_a = read_mrc(args.mask_path)
    pixel_size_nm = pixel_size_a * 1e1  # Å to nm
    mito_mask = build_mito_mask(data, membrane_values, body_values)
    labeled = clean_and_label(mito_mask, args.min_voxels)
    if np.count_nonzero(labeled) == 0:
        print("No mito instances found after cleaning. No output generated.")
        return

    df = compute_metrics(labeled, data, pixel_size_nm, membrane_values)
    if df.empty:
        print("No mito instances found. No output generated.")
        return

    base_name = os.path.splitext(os.path.basename(args.mask_path))[0]
    excel_path = os.path.join(args.out_dir, f"{base_name}_mito_area_volume.xlsx")
    df.to_excel(excel_path, index=False)
    metrics = {
        "memb_vox_ratio": "Membrane voxels / total voxels",
        "volume_nm3": "Volume (nm^3)",
        "surface_area_nm2": "Surface area (nm^2)",
    }
    for metric, label in metrics.items():
        values = df[metric].dropna().values
        if values.size == 0:
            continue
        plot_path = os.path.join(args.out_dir, f"{base_name}_{metric}_boxplot.png")
        make_boxplot(values, label, metric, plot_path)
    print(f"Saved Excel: {excel_path}")


if __name__ == "__main__":
    main()
