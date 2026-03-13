import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_vesicles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("vesicles", [])


def compute_metrics(vesicles, pixel_size):
    mean_diameters = []
    axis_ratios = []
    for vesicle in vesicles:
        radii = vesicle.get("radii")
        if not radii or len(radii) != 3:
            continue
        try:
            radii_nm = [float(r) * pixel_size for r in radii]
        except (TypeError, ValueError):
            continue

        mean_diameters.append(2.0 * float(np.mean(radii_nm)))
        axis_ratios.append(float(np.max(radii_nm) / np.min(radii_nm)))

    return np.asarray(mean_diameters, dtype=float), np.asarray(axis_ratios, dtype=float)


def gaussian_fit_curve(values, grid):
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    if sigma <= 0:
        return np.zeros_like(grid)
    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return coeff * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


def density_curve(values, grid):
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        return kde(grid)
    except Exception:
        return gaussian_fit_curve(values, grid)


def plot_distributions(mean_diameters, axis_ratios, out_path):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 4))

    if mean_diameters.size:
        bins = max(10, int(np.sqrt(mean_diameters.size)))
        ax_a.hist(mean_diameters, bins=bins, density=True, color="red", alpha=0.75)
        x_grid = np.linspace(mean_diameters.min(), mean_diameters.max(), 300)
        y_grid = density_curve(mean_diameters, x_grid)
        ax_a.plot(x_grid, y_grid, color="black", linewidth=1.5)
        peak_idx = int(np.argmax(y_grid)) if y_grid.size else 0
        peak_val = float(x_grid[peak_idx]) if x_grid.size else float(np.mean(mean_diameters))
        ax_a.axvline(peak_val, color="black", linestyle="--", linewidth=1.0)
        ax_a.text(
            0.98,
            0.95,
            f"peak: {peak_val:.2f} nm",
            transform=ax_a.transAxes,
            ha="right",
            va="top",
        )
    else:
        ax_a.text(0.5, 0.5, "No data", ha="center", va="center")

    ax_a.set_xlabel("SVs mean diameter (nm)")
    ax_a.set_ylabel("Relative Density")
    ax_a.set_title("A")

    if axis_ratios.size:
        x_grid = np.linspace(axis_ratios.min(), axis_ratios.max(), 300)
        y_grid = density_curve(axis_ratios, x_grid)
        ax_b.plot(x_grid, y_grid, color="red", linewidth=1.8)
        peak_idx = int(np.argmax(y_grid)) if y_grid.size else 0
        peak_val = float(x_grid[peak_idx]) if x_grid.size else float(np.mean(axis_ratios))
        ax_b.axvline(peak_val, color="black", linestyle="--", linewidth=1.0)
        ax_b.text(
            0.98,
            0.95,
            f"peak: {peak_val:.2f}",
            transform=ax_b.transAxes,
            ha="right",
            va="top",
        )
    else:
        ax_b.text(0.5, 0.5, "No data", ha="center", va="center")

    ax_b.set_xlabel("Long/short axis")
    ax_b.set_ylabel("Relative Density")
    ax_b.set_title("B")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot vesicle diameter and axis ratio distributions.")
    parser.add_argument("--json-path", required=True, help="Input vesicle JSON path.")
    parser.add_argument("--out-path", required=True, help="Output image path (png/jpg).")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=17.14,
        help="Pixel size in nm to scale radii (default: 1.0).",
    )
    args = parser.parse_args()

    vesicles = load_vesicles(args.json_path)
    if not vesicles:
        raise SystemExit("No vesicles found in JSON.")

    mean_diameters, axis_ratios = compute_metrics(vesicles, args.pixel_size)
    if mean_diameters.size == 0 or axis_ratios.size == 0:
        raise SystemExit("No valid vesicle radii found.")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_distributions(mean_diameters, axis_ratios, args.out_path)
    print(f"Saved plot: {args.out_path}")


if __name__ == "__main__":
    main()
