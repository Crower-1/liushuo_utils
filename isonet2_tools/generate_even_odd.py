import argparse
import os
import shutil
from typing import List, Optional, Tuple

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
    odd_angles, even_angles = split_angles(angles)
    return odd_data, even_data, odd_angles, even_angles


def split_angles(angles: List[float]) -> Tuple[List[float], List[float]]:
    return angles[0::2], angles[1::2]


def save_stack(path: str, data: np.ndarray, voxel_size) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data))
        try:
            mrc.voxel_size = voxel_size
        except Exception:
            # Keep default voxel size if header assignment fails.
            pass


def process_pair(
    stack_path: str,
    tlt_path: str,
    xtilt_path: Optional[str],
    tilt_com_path: Optional[str],
    output_root: str,
) -> None:
    base_name = os.path.splitext(os.path.basename(stack_path))[0]

    with mrcfile.open(stack_path, permissive=True) as mrc:
        data = np.copy(mrc.data)
        voxel_size = mrc.voxel_size

    angles = read_tilt_angles(tlt_path)
    if xtilt_path:
        xtilt_angles = read_tilt_angles(xtilt_path)
    else:
        xtilt_angles = [0.0] * len(angles)
        print(f"Info: {base_name}.xtilt not found, using 0.0 for all xtilt angles.")

    if len(angles) != data.shape[0] or len(xtilt_angles) != data.shape[0]:
        print(
            f"Warning: frame count mismatch for {base_name} "
            f"({data.shape[0]} slices vs {len(angles)} tilt angles, "
            f"{len(xtilt_angles)} xtilt angles). "
            "Truncating to the shortest length."
        )
        min_len = min(data.shape[0], len(angles), len(xtilt_angles))
        data = data[:min_len]
        angles = angles[:min_len]
        xtilt_angles = xtilt_angles[:min_len]

    odd_data, even_data, odd_angles, even_angles = split_stack(data, angles)
    odd_xtilt_angles, even_xtilt_angles = split_angles(xtilt_angles)

    odd_dir = os.path.join(output_root, "ODD")
    even_dir = os.path.join(output_root, "EVEN")
    os.makedirs(odd_dir, exist_ok=True)
    os.makedirs(even_dir, exist_ok=True)

    odd_ali = os.path.join(odd_dir, f"{base_name}_ODD.mrc")
    odd_tlt = os.path.join(odd_dir, f"{base_name}_ODD.tlt")
    odd_xtilt = os.path.join(odd_dir, f"{base_name}_ODD.xtilt")
    odd_tilt_com = os.path.join(odd_dir, f"{base_name}_tilt.com")
    even_ali = os.path.join(even_dir, f"{base_name}_EVN.mrc")
    even_tlt = os.path.join(even_dir, f"{base_name}_EVN.tlt")
    even_xtilt = os.path.join(even_dir, f"{base_name}_EVN.xtilt")
    even_tilt_com = os.path.join(even_dir, f"{base_name}_tilt.com")

    save_stack(odd_ali, odd_data, voxel_size)
    write_tilt_angles(odd_tlt, odd_angles)
    write_tilt_angles(odd_xtilt, odd_xtilt_angles)
    if tilt_com_path:
        shutil.copyfile(tilt_com_path, odd_tilt_com)

    save_stack(even_ali, even_data, voxel_size)
    write_tilt_angles(even_tlt, even_angles)
    write_tilt_angles(even_xtilt, even_xtilt_angles)
    if tilt_com_path:
        shutil.copyfile(tilt_com_path, even_tilt_com)

    print(
        f"Processed {base_name}: odd frames -> {odd_ali}, even frames -> {even_ali}"
    )


def find_pairs(input_dir: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    pairs: List[Tuple[str, str, Optional[str], Optional[str]]] = []
    seen_bases = set()
    for entry in sorted(os.listdir(input_dir)):
        if not entry.lower().endswith((".ali", ".mrc")):
            continue
        base = os.path.splitext(entry)[0]
        if base in seen_bases:
            print(
                f"Skipping {entry}: duplicated stack base name '{base}', "
                "already matched by another stack file."
            )
            continue
        stack_path = os.path.join(input_dir, entry)
        tlt_path = os.path.join(input_dir, f"{base}.tlt")
        xtilt_path = os.path.join(input_dir, f"{base}.xtilt")
        tilt_com_path = os.path.join(input_dir, f"{base}_tilt.com")
        if not os.path.isfile(tlt_path):
            print(f"Skipping {entry}: missing {base}.tlt")
            continue

        matched_xtilt = xtilt_path if os.path.isfile(xtilt_path) else None
        matched_tilt_com = tilt_com_path if os.path.isfile(tilt_com_path) else None
        if not matched_tilt_com:
            fallback_tilt_com = os.path.join(input_dir, "tilt.com")
            if os.path.isfile(fallback_tilt_com):
                matched_tilt_com = fallback_tilt_com

        pairs.append((stack_path, tlt_path, matched_xtilt, matched_tilt_com))
        seen_bases.add(base)
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split tilt stacks (e.g., .ali or bin2 .mrc) and tilt angles into odd/even subsets."
    )
    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing RAW_ALI_DATA. Expects <name>.ali/.mrc and <name>.tlt "
            "(<name>.xtilt and <name>_tilt.com are optional)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = os.path.abspath(os.path.join(args.input_dir, "RAW_ALI_DATA"))

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    pairs = find_pairs(input_dir)
    if not pairs:
        print("No stack/tlt pairs found.")
        return

    for stack_path, tlt_path, xtilt_path, tilt_com_path in pairs:
        process_pair(stack_path, tlt_path, xtilt_path, tilt_com_path, input_dir)


if __name__ == "__main__":
    main()
