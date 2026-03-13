import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_vesicle_id(name, fallback):
    if not name:
        return fallback
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else fallback


def ellipsoid_volume(a, b, c):
    return (4.0 / 3.0) * math.pi * a * b * c


def ellipsoid_surface_area(a, b, c):
    # Knud Thomsen approximation
    p = 1.6075
    return 4.0 * math.pi * ((a ** p * b ** p + a ** p * c ** p + b ** p * c ** p) / 3.0) ** (
        1.0 / p
    )


def load_vesicles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("vesicles", [])


def build_dataframe(vesicles, pixel_size):
    rows = []
    for idx, vesicle in enumerate(vesicles, start=1):
        center = vesicle.get("center", [np.nan, np.nan, np.nan])
        radii = vesicle.get("radii", [np.nan, np.nan, np.nan])
        evecs = vesicle.get("evecs", [[np.nan, np.nan, np.nan]] * 3)
        name = vesicle.get("name", "")

        center_nm = [float(v) * pixel_size for v in center]
        radii_nm = [float(v) * pixel_size for v in radii]
        a, b, c = radii_nm
        volume = ellipsoid_volume(a, b, c)
        area = ellipsoid_surface_area(a, b, c)
        mean_radius = float(np.mean(radii_nm))

        rows.append(
            {
                "id": parse_vesicle_id(name, idx),
                "x": center_nm[0],
                "y": center_nm[1],
                "z": center_nm[2],
                "r1": radii_nm[0],
                "r2": radii_nm[1],
                "r3": radii_nm[2],
                "evec1_x": float(evecs[0][0]),
                "evec1_y": float(evecs[0][1]),
                "evec1_z": float(evecs[0][2]),
                "evec2_x": float(evecs[1][0]),
                "evec2_y": float(evecs[1][1]),
                "evec2_z": float(evecs[1][2]),
                "evec3_x": float(evecs[2][0]),
                "evec3_y": float(evecs[2][1]),
                "evec3_z": float(evecs[2][2]),
                "volume": volume,
                "surface_area": area,
                "mean_radius": mean_radius,
            }
        )
    return pd.DataFrame(rows)


def plot_single_boxplot(values, title, ylabel, out_path, jitter=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(values, vert=True, widths=0.5, showfliers=False)
    if values.size:
        rng = np.random.default_rng()
        jitter_x = 1.0 + rng.uniform(-jitter, jitter, size=values.size)
        ax.scatter(jitter_x, values, s=28, facecolor="white", edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Convert vesicle JSON to Excel results and boxplots."
    )
    parser.add_argument("--json-path", type=str, required=True, help="Input vesicle JSON path.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size (same unit will be used for output).",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    vesicles = load_vesicles(args.json_path)
    if not vesicles:
        print("No vesicles found in JSON. No output generated.")
        return

    df = build_dataframe(vesicles, args.pixel_size)
    if df.empty:
        print("No vesicles found after parsing. No output generated.")
        return

    base_name = os.path.splitext(os.path.basename(args.json_path))[0]
    excel_path = os.path.join(args.out_dir, f"{base_name}_vesicle_metrics.xlsx")
    df.to_excel(excel_path, index=False)

    volume_path = os.path.join(args.out_dir, f"{base_name}_vesicle_volume_boxplot.png")
    surface_path = os.path.join(args.out_dir, f"{base_name}_vesicle_surface_area_boxplot.png")
    mean_radius_path = os.path.join(args.out_dir, f"{base_name}_vesicle_mean_radius_boxplot.png")

    plot_single_boxplot(df["volume"], "Volume", "Volume", volume_path, jitter=0.05)
    plot_single_boxplot(
        df["surface_area"], "Surface Area", "Surface Area", surface_path, jitter=0.05
    )
    plot_single_boxplot(df["mean_radius"], "Mean Radius", "Radius", mean_radius_path, jitter=0.05)

    print(f"Saved Excel: {excel_path}")
    print(f"Saved plots: {volume_path}")
    print(f"Saved plots: {surface_path}")
    print(f"Saved plots: {mean_radius_path}")


if __name__ == "__main__":
    main()
