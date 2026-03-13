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


def feret_diameter_slices(mask, pixel_size_nm):
    values = []
    for z in range(mask.shape[0]):
        slice_mask = mask[z]
        if not np.any(slice_mask):
            continue
        props = measure.regionprops(slice_mask.astype(np.uint8))
        if not props:
            continue
        values.append(props[0].feret_diameter_max)
    if not values:
        return np.nan, np.nan
    values = np.asarray(values, dtype=float) * pixel_size_nm
    return float(np.mean(values)), float(np.max(values))


def surface_curvature(mask, pixel_size_nm, radius_nm):
    try:
        import trimesh
    except ImportError:
        return np.nan
    if np.count_nonzero(mask) < 4:
        return np.nan
    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=(pixel_size_nm,) * 3
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


def cristae_spacing_nm(instance, membrane_mask, pixel_size_nm):
    if membrane_mask is None or not np.any(membrane_mask):
        return np.nan
    interior = instance & ~membrane_mask
    if np.count_nonzero(interior) < 10:
        return np.nan
    dist_to_membrane = ndimage.distance_transform_edt(~membrane_mask) * pixel_size_nm
    skeleton = skeletonize_3d(interior)
    values = dist_to_membrane[skeleton]
    if values.size == 0:
        return np.nan
    return float(2.0 * np.mean(values))


def cristae_connection_count(instance, membrane_mask):
    if membrane_mask is None or not np.any(membrane_mask):
        return np.nan
    skeleton = skeletonize_3d(membrane_mask)
    if not np.any(skeleton):
        return 0
    eroded = ndimage.binary_erosion(instance)
    boundary = instance if not np.any(eroded) else (instance & ~eroded)
    boundary = ndimage.binary_dilation(boundary, iterations=1)
    labeled, num = ndimage.label(skeleton)
    if num == 0:
        return 0
    count = 0
    for label_id in range(1, num + 1):
        comp = labeled == label_id
        if np.any(comp & boundary):
            count += 1
    return int(count)


def compute_metrics(labeled, membrane_mask, pixel_size_nm, curvature_radius_nm):
    rows = []
    voxel_volume = pixel_size_nm ** 3
    ids = np.unique(labeled)
    ids = ids[ids != 0]
    for label_id in ids:
        instance = labeled == label_id
        voxel_count = int(np.count_nonzero(instance))
        volume_nm3 = voxel_count * voxel_volume
        area_nm2 = surface_area(instance, pixel_size_nm)
        feret_mean_nm, feret_max_nm = feret_diameter_slices(instance, pixel_size_nm)
        membrane_instance = membrane_mask & instance if membrane_mask is not None else None
        curvature = surface_curvature(
            membrane_instance if membrane_instance is not None and np.any(membrane_instance) else instance,
            pixel_size_nm,
            curvature_radius_nm,
        )
        cristae_spacing = cristae_spacing_nm(instance, membrane_instance, pixel_size_nm)
        cristae_connections = cristae_connection_count(instance, membrane_instance)
        rows.append(
            {
                "id": int(label_id),
                "voxel_count": voxel_count,
                "volume_nm3": volume_nm3,
                "surface_area_nm2": area_nm2,
                "feret_diameter_mean_nm": feret_mean_nm,
                "feret_diameter_max_nm": feret_max_nm,
                "membrane_curvature_mean_1_per_nm": curvature,
                "cristae_spacing_nm": cristae_spacing,
                "cristae_connection_count": cristae_connections,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute mitochondria quantitative metrics from a semantic MRC mask."
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
        help="Comma-separated mito body values in mask (default: 5).",
    )
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
    membrane_values = parse_value_list(args.membrane_values)
    body_values = parse_value_list(args.body_values)
    data, pixel_size_nm = read_mrc(args.mask_path)
    mito_mask = build_mito_mask(data, membrane_values, body_values)
    labeled = clean_and_label(mito_mask, args.min_voxels)
    if np.count_nonzero(labeled) == 0:
        print("No mito instances found after cleaning. No output generated.")
        return

    membrane_mask = np.isin(data, membrane_values) if membrane_values else None
    df = compute_metrics(labeled, membrane_mask, pixel_size_nm, args.curvature_radius_nm)
    if df.empty:
        print("No mito instances found. No output generated.")
        return

    base_name = os.path.splitext(os.path.basename(args.mask_path))[0]
    excel_path = os.path.join(args.out_dir, f"{base_name}_mito_quantitative.xlsx")
    df.to_excel(excel_path, index=False)

    metrics = {
        "volume_nm3": "Volume (nm^3)",
        "surface_area_nm2": "Surface area (nm^2)",
        "feret_diameter_mean_nm": "Feret diameter mean (nm)",
        "feret_diameter_max_nm": "Feret diameter max (nm)",
        "membrane_curvature_mean_1_per_nm": "Membrane curvature (1/nm)",
        "cristae_spacing_nm": "Cristae spacing (nm)",
        "cristae_connection_count": "Cristae connection count",
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
