"""Create the YYT pre/post nnUNet dataset and its bin4 derivative."""

from __future__ import annotations

import argparse
from pathlib import Path

from create_bin4_dataset import create_bin4_dataset
from create_dataset import (
    DEFAULT_LABEL_REMAPS,
    DEFAULT_SOURCE_ROOT,
    create_dataset,
)


DEFAULT_RAW_OUTPUT = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/"
    "Dataset004_synapseseg_roiprepost"
)
DEFAULT_BIN4_OUTPUT = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/"
    "Dataset005_synapseseg_roiprepost_34tomo_bin4"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--bin4-output", type=Path, default=DEFAULT_BIN4_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=34)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dataset = create_dataset(
        args.source_root,
        args.raw_output,
        expected_count=args.expected_count,
        label_remaps={
            case: mapping.copy() for case, mapping in DEFAULT_LABEL_REMAPS.items()
        },
    )
    create_bin4_dataset(
        raw_dataset,
        args.bin4_output,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":
    main()
