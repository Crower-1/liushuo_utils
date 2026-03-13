import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mrcfile as mf
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize_3d


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


def make_boxplot(values, label, title, output_path, jitter=0.05):
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(values, vert=True, widths=0.5, showfliers=False)
    rng = np.random.default_rng()
    xs = 1.0 + rng.uniform(-jitter, jitter, size=len(values))
    ax.scatter(xs, values, s=28, facecolor="white", edgecolor="black", linewidth=0.8)
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_er_mask(data, membrane_value=6, lumen_value=1):
    return np.logical_or(data == membrane_value, data == lumen_value)


def clean_and_label(er_mask, min_voxels=3000):
    labeled, num = ndimage.label(er_mask)
    if num == 0:
        return np.zeros_like(er_mask, dtype=np.int32)
    cleaned = np.zeros_like(er_mask, dtype=bool)
    for label_id in range(1, num + 1):
        instance = labeled == label_id
        if int(np.count_nonzero(instance)) < min_voxels:
            continue
        filled = ndimage.binary_fill_holes(instance)
        cleaned |= filled
    relabeled, _ = ndimage.label(cleaned)
    return relabeled.astype(np.int32)


def _shift_overlap(mask, dz, dy, dx):
    z0 = max(0, dz)
    z1 = mask.shape[0] + min(0, dz)
    y0 = max(0, dy)
    y1 = mask.shape[1] + min(0, dy)
    x0 = max(0, dx)
    x1 = mask.shape[2] + min(0, dx)
    a = mask[z0:z1, y0:y1, x0:x1]
    b = mask[z0 - dz : z1 - dz, y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return a, b


def skeleton_length(skeleton, pixel_size_nm):
    if not np.any(skeleton):
        return 0.0
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0):
                    offsets.append((dz, dy, dx))
    total = 0.0
    for dz, dy, dx in offsets:
        a, b = _shift_overlap(skeleton, dz, dy, dx)
        if a.size == 0:
            continue
        count = np.count_nonzero(a & b)
        if count == 0:
            continue
        dist = math.sqrt(dz * dz + dy * dy + dx * dx) * pixel_size_nm
        total += count * dist
    return float(total)


def branch_point_count(skeleton):
    if not np.any(skeleton):
        return 0
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode="constant", cval=0)
    return int(np.count_nonzero(skeleton & (neighbors >= 3)))


def thickness_stats(mask, pixel_size_nm):
    if not np.any(mask):
        return {
            "thickness_mean_nm": np.nan,
            "thickness_median_nm": np.nan,
            "tube_diameter_nm": np.nan,
            "sheet_width_nm": np.nan,
        }
    dist = ndimage.distance_transform_edt(mask)
    thickness_nm = 2.0 * dist * pixel_size_nm
    values = thickness_nm[mask]
    if values.size == 0:
        return {
            "thickness_mean_nm": np.nan,
            "thickness_median_nm": np.nan,
            "tube_diameter_nm": np.nan,
            "sheet_width_nm": np.nan,
        }
    return {
        "thickness_mean_nm": float(np.mean(values)),
        "thickness_median_nm": float(np.median(values)),
        "tube_diameter_nm": float(np.percentile(values, 25)),
        "sheet_width_nm": float(np.percentile(values, 90)),
    }


def surface_curvature(mask, pixel_size_nm, radius_nm):
    try:
        import trimesh
    except ImportError:
        return np.nan
    if np.count_nonzero(mask) < 4:
        return np.nan
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(pixel_size_nm, pixel_size_nm, pixel_size_nm)
    )
    if verts.size == 0 or faces.size == 0:
        return np.nan
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.is_empty:
        return np.nan
    radius = radius_nm if radius_nm > 0 else 3.0 * pixel_size_nm
    curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius)
    if curvature is None or len(curvature) == 0:
        return np.nan
    return float(np.mean(np.abs(curvature)))


def compute_metrics(labeled, pixel_size_nm, curvature_radius_nm):
    rows = []
    voxel_volume = pixel_size_nm ** 3
    ids = np.unique(labeled)
    ids = ids[ids != 0]
    for label_id in ids:
        instance = labeled == label_id
        voxel_count = int(np.count_nonzero(instance))
        volume_nm3 = voxel_count * voxel_volume
        skeleton = skeletonize_3d(instance)
        length_nm = skeleton_length(skeleton, pixel_size_nm)
        branch_points = branch_point_count(skeleton)
        thickness = thickness_stats(instance, pixel_size_nm)
        curvature_mean = surface_curvature(instance, pixel_size_nm, curvature_radius_nm)
        rows.append(
            {
                "id": int(label_id),
                "voxel_count": voxel_count,
                "volume_nm3": volume_nm3,
                "length_nm": length_nm,
                "branch_points": branch_points,
                "thickness_mean_nm": thickness["thickness_mean_nm"],
                "thickness_median_nm": thickness["thickness_median_nm"],
                "tube_diameter_nm": thickness["tube_diameter_nm"],
                "sheet_width_nm": thickness["sheet_width_nm"],
                "curvature_mean_1_per_nm": curvature_mean,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute ER quantitative metrics from a semantic MRC mask."
    )
    parser.add_argument("--mask-path", type=str, required=True, help="Input ER mask MRC path.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--membrane-value", type=int, default=6, help="Membrane value in mask.")
    parser.add_argument("--lumen-value", type=int, default=1, help="Lumen value in mask.")
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=3000,
        help="Minimum voxel count to keep an instance.",
    )
    parser.add_argument(
        "--curvature-radius-nm",
        type=float,
        default=10.0,
        help="Radius (nm) for surface curvature estimation.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    data, pixel_size_nm = read_mrc(args.mask_path)
    er_mask = build_er_mask(data, args.membrane_value, args.lumen_value)
    labeled = clean_and_label(er_mask, args.min_voxels)
    if np.count_nonzero(labeled) == 0:
        print("No ER instances found after cleaning. No output generated.")
        return

    df = compute_metrics(labeled, pixel_size_nm, args.curvature_radius_nm)
    if df.empty:
        print("No ER instances found. No output generated.")
        return

    base_name = os.path.splitext(os.path.basename(args.mask_path))[0]
    excel_path = os.path.join(args.out_dir, f"{base_name}_ER_quantitative.xlsx")
    df.to_excel(excel_path, index=False)

    metrics = {
        "volume_nm3": "Volume (nm^3)",
        "length_nm": "Length (nm)",
        "tube_diameter_nm": "Tube diameter (nm)",
        "sheet_width_nm": "Sheet width (nm)",
        "branch_points": "Branch points",
        "curvature_mean_1_per_nm": "Curvature (1/nm)",
    }
    for metric, label in metrics.items():
        values = df[metric].dropna().values
        if values.size == 0:
            continue
        plot_path = os.path.join(args.out_dir, f"{base_name}_{metric}_boxplot.png")
        make_boxplot(values, label, metric, plot_path)

    print(f"Saved Excel: {excel_path}")
    print(f"Saved boxplots to: {args.out_dir}")


if __name__ == "__main__":
    main()
