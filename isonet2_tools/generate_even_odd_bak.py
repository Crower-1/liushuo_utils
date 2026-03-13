import argparse
import os
from typing import List, Tuple

import mrcfile
import numpy as np


def read_tilt_angles(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        return [float(line.strip()) for line in f if line.strip()]


def write_tilt_angles(path: str, angles: List[float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for angle in angles:
            f.write(f"{angle}\n")


def split_stack(
    data: np.ndarray, angles: List[float]
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Split tilt series into odd/even frames (1-based indexing)."""
    odd_data = data[0::2]
    even_data = data[1::2]
    odd_angles = angles[0::2]
    even_angles = angles[1::2]
    return odd_data, even_data, odd_angles, even_angles


def save_stack(path: str, data: np.ndarray, voxel_size) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data))
        try:
            mrc.voxel_size = voxel_size
        except Exception:
            # Keep default voxel size if header assignment fails.
            pass


def process_pair(ali_path: str, tlt_path: str, output_root: str) -> None:
    base_name = os.path.splitext(os.path.basename(ali_path))[0]

    with mrcfile.open(ali_path, permissive=True) as mrc:
        data = np.copy(mrc.data)
        voxel_size = mrc.voxel_size

    angles = read_tilt_angles(tlt_path)

    if len(angles) != data.shape[0]:
        print(
            f"Warning: frame count mismatch for {base_name} "
            f"({data.shape[0]} slices vs {len(angles)} tilt angles). "
            "Truncating to the shortest length."
        )
        min_len = min(data.shape[0], len(angles))
        data = data[:min_len]
        angles = angles[:min_len]

    odd_data, even_data, odd_angles, even_angles = split_stack(data, angles)

    odd_dir = os.path.join(output_root, "ODD")
    even_dir = os.path.join(output_root, "EVEN")

    odd_ali = os.path.join(odd_dir, f"{base_name}_ODD.mrc")
    odd_tlt = os.path.join(odd_dir, f"{base_name}_ODD.tlt")
    even_ali = os.path.join(even_dir, f"{base_name}_EVN.mrc")
    even_tlt = os.path.join(even_dir, f"{base_name}_EVN.tlt")

    save_stack(odd_ali, odd_data, voxel_size)
    write_tilt_angles(odd_tlt, odd_angles)

    save_stack(even_ali, even_data, voxel_size)
    write_tilt_angles(even_tlt, even_angles)

    print(
        f"Processed {base_name}: odd frames -> {odd_ali}, even frames -> {even_ali}"
    )


def find_pairs(input_dir: str, pairs_method: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if pairs_method == "ali":
        for entry in sorted(os.listdir(input_dir)):
            if not entry.lower().endswith(".ali"):
                continue
            base = os.path.splitext(entry)[0]
            ali_path = os.path.join(input_dir, entry)
            tlt_path = os.path.join(input_dir, f"{base}.tlt")
            if os.path.isfile(tlt_path):
                pairs.append((ali_path, tlt_path))
            else:
                print(f"Skipping {entry}: missing {base}.tlt")
    elif pairs_method == "bin2_mrc":
        # Prefer bin2/ if present, otherwise fall back to bin/
        candidate_bins = [os.path.join(input_dir, "bin2"), os.path.join(input_dir, "bin")]
        bin_dir = next((p for p in candidate_bins if os.path.isdir(p)), candidate_bins[-1])
        for entry in sorted(os.listdir(input_dir)):
            if not entry.lower().endswith(".tlt"):
                continue
            base = os.path.splitext(entry)[0]
            mrc_path = os.path.join(bin_dir, f"{base}_bin2.mrc")
            tlt_path = os.path.join(input_dir, entry)
            if os.path.isfile(mrc_path):
                pairs.append((mrc_path, tlt_path))
            else:
                print(f"Skipping {entry}: missing {os.path.relpath(mrc_path, input_dir)}")
    else:
        raise ValueError(f"Unknown pairs_method: {pairs_method}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split tilt stacks (e.g., .ali or bin2 .mrc) and tilt angles into odd/even subsets."
    )
    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing inputs. For pairs_method=ali, expects <name>.ali and <name>.tlt "
            "in this directory. For pairs_method=bin2_mrc, expects <name>.tlt here and "
            "bin/<name>_bin2.mrc."
        ),
    )
    parser.add_argument(
        "--pairs_method",
        default="ali",
        choices=["ali", "bin2_mrc"],
        help=(
            "How to pair tilt stacks with tilt angles. "
            "'ali' expects <name>.ali with <name>.tlt in the same folder (default). "
            "'bin2_mrc' pairs bin/<name>_bin2.mrc with <name>.tlt in input_dir."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    pairs = find_pairs(input_dir, args.pairs_method)
    if not pairs:
        print(f"No pairs found using pairs_method '{args.pairs_method}'.")
        return

    for ali_path, tlt_path in pairs:
        process_pair(ali_path, tlt_path, input_dir)


if __name__ == "__main__":
    main()
