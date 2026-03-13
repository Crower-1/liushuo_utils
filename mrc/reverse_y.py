#!/usr/bin/env python
import argparse
import os

import numpy as np

from mrc.io import get_tomo_with_voxel_size, save_tomo


def reverse_y(input_path, output_path=None):
    data, voxel_size = get_tomo_with_voxel_size(input_path)

    reversed_data = data[:, ::-1, ::-1]

    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}-reversey{ext or '.mrc'}"

    save_tomo(reversed_data, output_path, voxel_size=voxel_size, datetype=data.dtype)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Reverse Y axis of a 3D MRC volume.")
    parser.add_argument("-i", "--input", required=True, help="Input MRC path")
    parser.add_argument("-o", "--output", default=None, help="Output MRC path")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = reverse_y(args.input, output_path=args.output)
    print(f"Saved reversed volume to {out_path}")


if __name__ == "__main__":
    main()
