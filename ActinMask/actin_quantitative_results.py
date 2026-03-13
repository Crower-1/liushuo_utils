import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_actin_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", [])
    if not isinstance(data, list):
        raise ValueError("JSON content must be a list of actin objects.")
    return data


def compute_length(points):
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_curvature(points):
    if len(points) < 3:
        return 0.0
    curvatures = []
    for i in range(1, len(points) - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if a <= 0 or b <= 0 or c <= 0:
            continue
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area <= 0:
            curv = 0.0
        else:
            curv = 4.0 * area / (a * b * c)
        curvatures.append(curv)
    if not curvatures:
        return 0.0
    return float(np.mean(curvatures))


def compute_orientation_vector(points):
    if len(points) < 2:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    centered = points - points.mean(axis=0)
    if np.allclose(centered, 0):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v = vt[0]
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v / norm


def compute_center(points):
    if len(points) == 1:
        return points[0]
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg_lengths))
    if total == 0:
        return points.mean(axis=0)
    half = total / 2.0
    cum = np.cumsum(seg_lengths)
    idx = int(np.searchsorted(cum, half))
    if idx >= len(seg_lengths):
        return points[-1]
    prev_cum = 0.0 if idx == 0 else cum[idx - 1]
    seg_len = seg_lengths[idx]
    if seg_len == 0:
        return points[idx]
    t = (half - prev_cum) / seg_len
    return points[idx] * (1.0 - t) + points[idx + 1] * t


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


def make_vector_field_plot(df, output_path, scale):
    data = df.dropna(
        subset=[
            "center_x_nm",
            "center_y_nm",
            "center_z_nm",
            "orientation_x",
            "orientation_y",
            "orientation_z",
        ]
    )
    if data.empty:
        return
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    x = data["center_x_nm"].to_numpy()
    y = data["center_y_nm"].to_numpy()
    z = data["center_z_nm"].to_numpy()
    u = data["orientation_x"].to_numpy() * scale
    v = data["orientation_y"].to_numpy() * scale
    w = data["orientation_z"].to_numpy() * scale
    vec = np.stack([data["orientation_x"], data["orientation_y"], data["orientation_z"]], axis=1)
    azimuth = (np.arctan2(vec[:, 1], vec[:, 0]) + 2 * np.pi) % (2 * np.pi)
    azimuth_norm = azimuth / (2 * np.pi)
    colors = cm.plasma(azimuth_norm)
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        pivot='tip',
        length=0.5,
        normalize=False,
        linewidth=2,
        colors=colors,
    )
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title("Actin orientation vectors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_metrics(data, pixel_size_nm, diameter_nm):
    rows = []
    radius_nm = diameter_nm / 2.0
    cross_area_nm2 = math.pi * radius_nm * radius_nm
    for item in data:
        points = np.asarray(item.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        # Input points are in z, y, x order; convert to x, y, z for calculations.
        points = points[:, [2, 1, 0]]
        length_px = compute_length(points)
        length_nm = length_px * pixel_size_nm
        curvature_1_per_px = compute_curvature(points)
        curvature_1_per_nm = curvature_1_per_px / pixel_size_nm if pixel_size_nm > 0 else np.nan
        orientation_vec = compute_orientation_vector(points)
        center_px = compute_center(points)
        center_nm = center_px * pixel_size_nm
        volume_nm3 = length_nm * cross_area_nm2
        rows.append(
            {
                "id": item.get("id"),
                "num_points": int(points.shape[0]),
                "length_px": length_px,
                "length_nm": length_nm,
                "curvature_1_per_px": curvature_1_per_px,
                "curvature_1_per_nm": curvature_1_per_nm,
                "orientation_x": float(orientation_vec[0]),
                "orientation_y": float(orientation_vec[1]),
                "orientation_z": float(orientation_vec[2]),
                "center_x_px": float(center_px[0]),
                "center_y_px": float(center_px[1]),
                "center_z_px": float(center_px[2]),
                "center_x_nm": float(center_nm[0]),
                "center_y_nm": float(center_nm[1]),
                "center_z_nm": float(center_nm[2]),
                "volume_nm3": volume_nm3,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute actin quantitative metrics from JSON points."
    )
    parser.add_argument("--pixel-size", type=float, required=True, help="Pixel size in nm.")
    parser.add_argument("--json-path", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--diameter-nm",
        type=float,
        default=7.0,
        help="Actin filament diameter in nm (default: 7).",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=2500.0,
        help="Scale factor for orientation vectors in the 3D plot.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    data = load_actin_json(args.json_path)
    df = compute_metrics(data, args.pixel_size, args.diameter_nm)
    if df.empty:
        print("No valid actin entries found. No output generated.")
        return

    base_name = os.path.splitext(os.path.basename(args.json_path))[0]
    excel_path = os.path.join(args.out_dir, f"{base_name}_quantitative.xlsx")
    df.to_excel(excel_path, index=False)

    metrics = {
        "length_nm": "Length (nm)",
        "curvature_1_per_nm": "Curvature (1/nm)",
        "volume_nm3": "Volume (nm^3)",
    }
    for metric, label in metrics.items():
        values = df[metric].dropna().values
        if values.size == 0:
            continue
        plot_path = os.path.join(args.out_dir, f"{base_name}_{metric}_boxplot.png")
        make_boxplot(values, label, metric, plot_path)

    vector_path = os.path.join(args.out_dir, f"{base_name}_orientation_vector.png")
    make_vector_field_plot(df, vector_path, args.vector_scale)

    print(f"Saved Excel: {excel_path}")
    print(f"Saved boxplots to: {args.out_dir}")


if __name__ == "__main__":
    main()
