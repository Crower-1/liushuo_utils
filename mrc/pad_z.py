#!/usr/bin/env python
import argparse
import os

import numpy as np

from mrc.io import get_tomo_with_voxel_size, save_tomo


def pad_z(input_path, target_z, output_path=None, pad_value=0, mode="post"):
    data, voxel_size = get_tomo_with_voxel_size(input_path)
    current_z = data.shape[0]

    if target_z < current_z:
        raise ValueError(
            f"target_z ({target_z}) must be >= current z ({current_z})"
        )

    if output_path is None:
        root, ext = os.path.splitext(input_path)
        suffix = f"-padz{target_z}"
        if mode != "post":
            suffix = f"{suffix}-{mode}"
        output_path = f"{root}{suffix}{ext or '.mrc'}"

    if mode == "post":
        pad_before = 0
        pad_after = target_z - current_z
    elif mode == "symmetric":
        total_pad = target_z - current_z
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")

    pad_width = [(pad_before, pad_after), (0, 0), (0, 0)]
    padded = np.pad(data, pad_width, mode="constant", constant_values=pad_value)

    save_tomo(padded, output_path, voxel_size=voxel_size, datetype=data.dtype)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Pad Z dimension of a 3D MRC volume.")
    parser.add_argument("-i", "--input", required=True, help="Input MRC path")
    parser.add_argument("-z", "--target-z", required=True, type=int, help="Target Z size")
    parser.add_argument("-o", "--output", default=None, help="Output MRC path")
    parser.add_argument(
        "-p",
        "--pad-value",
        type=float,
        default=0,
        help="Pad value for Z extension",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["post", "symmetric"],
        default="post",
        help="Pad mode: append at end (post) or pad equally on both ends (symmetric)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = pad_z(
        args.input,
        args.target_z,
        output_path=args.output,
        pad_value=args.pad_value,
        mode=args.mode,
    )
    print(f"Saved padded volume to {out_path}")


if __name__ == "__main__":
    main()
