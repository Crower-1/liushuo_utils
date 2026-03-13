import argparse
import os
import sys
from typing import Callable

import numpy as np

from mrc.io import get_tomo

try:
    from tifffile import imread as _read_tiff
except ImportError:  # pragma: no cover - fallback for environments without tifffile
    _read_tiff = None


def _ensure_file_exists(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")


def _load_mrc_mask(path: str) -> np.ndarray:
    data = np.asarray(get_tomo(path))
    return data != 0


def _load_tiff_mask(path: str, reader: Callable[[str], np.ndarray]) -> np.ndarray:
    data = np.asarray(reader(path))
    return data != 0


def _compute_ratio(actin_mask: np.ndarray, synapse_mask: np.ndarray) -> float:
    if actin_mask.shape != synapse_mask.shape:
        raise ValueError(
            "Actin mask and synapse mask must have the same shape; "
            f"got {actin_mask.shape} and {synapse_mask.shape}."
        )

    synapse_volume = synapse_mask.sum(dtype=np.int64)
    if synapse_volume == 0:
        raise ValueError("Synapse mask volume is zero; cannot compute ratio.")

    and_volume = np.logical_and(actin_mask, synapse_mask).sum(dtype=np.int64)
    return  and_volume / synapse_volume


# def parse_args(argv: list[str]) -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Compute actin coverage ratio within a synapse mask."
#     )
#     parser.add_argument("actin", help="Path to the actin mask MRC file.")
#     parser.add_argument("synapse", help="Path to the synapse mask TIFF file.")
#     parser.add_argument(
#         "--precision",
#         type=int,
#         default=6,
#         help="Number of decimal places to print (default: 6).",
#     )
#     return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # args = parse_args(argv or sys.argv[1:])
    tomo_name = 'pp064'
    actin_path = f'/media/liushuo/data3/lwd/{tomo_name}/synapse_seg/actin/{tomo_name}_actin_filament.mrc'
    synapse_path = f'/media/liushuo/data3/lwd/{tomo_name}/synapse_seg/volume/nnInteractive - Label Layer.tif'

    _ensure_file_exists(actin_path)
    _ensure_file_exists(synapse_path)

    if _read_tiff is None:
        raise ImportError(
            "tifffile is required to read synapse masks; please install it before running."
        )

    actin_mask = _load_mrc_mask(actin_path)
    synapse_mask = _load_tiff_mask(synapse_path, _read_tiff)

    ratio = _compute_ratio(actin_mask, synapse_mask)
    precision = max(6, 0)
    print(f"{ratio:.{precision}f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
