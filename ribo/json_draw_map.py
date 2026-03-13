import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw scatter plot of max_area (x) vs volume (y) from multiple JSON files."
        )
    )
    parser.add_argument(
        "--input-json",
        nargs="+",
        required=True,
        help="Input JSON files, e.g. type1.json type2.json ...",
    )
    parser.add_argument(
        "--output-image",
        required=True,
        help="Output image path, e.g. scatter.png.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output image. Default: 300.",
    )
    return parser.parse_args()


def load_points(path: str) -> Tuple[List[float], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    xs: List[float] = []
    ys: List[float] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        max_area = item.get("max_area")
        volume = item.get("volume")
        if max_area is None or volume is None:
            continue
        xs.append(float(max_area))
        ys.append(float(volume))
    return xs, ys


def _label_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    return name or base


def plot_scatter(files: Sequence[str], output_path: str, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for path in files:
        xs, ys = load_points(path)
        if not xs:
            continue
        ax.scatter(xs, ys, s=20, alpha=0.8, label=_label_from_path(path))

    ax.set_xlabel("max_area")
    ax.set_ylabel("volume")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_scatter(args.input_json, args.output_image, args.title.strip(), args.dpi)


if __name__ == "__main__":
    main()
