import argparse
from typing import Callable, Dict

import numpy as np

from mrc.io import get_tomo_with_voxel_size, save_tomo


_OPS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "gt": lambda data, th: data > th,
    "ge": lambda data, th: data >= th,
    "lt": lambda data, th: data < th,
    "le": lambda data, th: data <= th,
}


def binarize_volume(data: np.ndarray, threshold: float, op: str) -> np.ndarray:
    if op not in _OPS:
        raise ValueError(f"Unsupported op '{op}'. Choose from: {', '.join(sorted(_OPS.keys()))}")
    return _OPS[op](data, threshold).astype(np.uint8)


def binarize_mrc(
    input_path: str,
    output_path: str,
    threshold: float = 0.0,
    op: str = "gt",
    foreground: int = 1,
    background: int = 0,
) -> None:
    data, voxel_size = get_tomo_with_voxel_size(input_path)
    binary = binarize_volume(data, threshold=threshold, op=op)

    if foreground != 1 or background != 0:
        binary = np.where(binary == 1, foreground, background).astype(np.uint8)

    save_tomo(binary, output_path, voxel_size=voxel_size, datetype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binarize a 3D MRC volume by threshold.")
    parser.add_argument("-i", "--input", required=True, help="Input MRC file path.")
    parser.add_argument("-o", "--output", required=True, help="Output binary MRC file path.")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold value used for binarization (default: 0.0).",
    )
    parser.add_argument(
        "--op",
        choices=sorted(_OPS.keys()),
        default="gt",
        help="Comparison operator: gt/ge/lt/le (default: gt).",
    )
    parser.add_argument(
        "--foreground",
        type=int,
        default=1,
        help="Value for foreground voxels after binarization (default: 1).",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=0,
        help="Value for background voxels after binarization (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    binarize_mrc(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        op=args.op,
        foreground=args.foreground,
        background=args.background,
    )
    print(f"Saved binary MRC: {args.output}")


if __name__ == "__main__":
    main()
