import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from mrc.io import get_tomo_with_voxel_size, save_tomo


DEFAULT_SOURCE_DIR = Path(
    "/media/liushuo/data3/nnUNet_dataset/nnUNet_raw/Dataset017_1tomo_dna"
)
DEFAULT_TARGET_DIR = Path(
    "/media/liushuo/data3/nnUNet_dataset/nnUNet_raw/Dataset018_1tomo_crop_dna"
)
IMAGE_NAME = "P79_0000.mrc"
LABEL_NAME = "P79.mrc"
CASE_PREFIX = "P79_crop"


SliceSpec = Tuple[slice, slice, slice]


def center_split_slices(shape: Sequence[int]) -> List[SliceSpec]:
    """Split a 3D volume into 8 blocks by the center of z, y, x axes."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D volume, got shape: {tuple(shape)}")

    axis_slices: List[Tuple[slice, slice]] = []
    for axis_size in shape:
        center = axis_size // 2
        axis_slices.append((slice(0, center), slice(center, axis_size)))

    blocks: List[SliceSpec] = []
    for z_slice in axis_slices[0]:
        for y_slice in axis_slices[1]:
            for x_slice in axis_slices[2]:
                blocks.append((z_slice, y_slice, x_slice))
    return blocks


def prepare_target_dir(target_dir: Path, overwrite: bool) -> None:
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target dataset already exists: {target_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(target_dir)

    (target_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (target_dir / "labelsTr").mkdir(parents=True, exist_ok=True)


def write_dataset_json(source_dir: Path, target_dir: Path, num_training: int) -> None:
    source_json = source_dir / "dataset.json"
    target_json = target_dir / "dataset.json"

    with source_json.open("r", encoding="utf-8") as f:
        dataset: Dict[str, object] = json.load(f)

    dataset["numTraining"] = num_training

    with target_json.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        f.write("\n")


def crop_pair(source_dir: Path, target_dir: Path) -> int:
    image_path = source_dir / "imagesTr" / IMAGE_NAME
    label_path = source_dir / "labelsTr" / LABEL_NAME

    image, image_voxel_size = get_tomo_with_voxel_size(str(image_path))
    label, label_voxel_size = get_tomo_with_voxel_size(str(label_path))

    if image.shape != label.shape:
        raise ValueError(
            f"Image and label shapes do not match: {image.shape} vs {label.shape}"
        )

    blocks = center_split_slices(image.shape)
    for block_id, block_slices in enumerate(blocks):
        case_name = f"{CASE_PREFIX}{block_id:03d}"
        image_crop = np.ascontiguousarray(image[block_slices])
        label_crop = np.ascontiguousarray(label[block_slices])

        save_tomo(
            image_crop,
            str(target_dir / "imagesTr" / f"{case_name}_0000.mrc"),
            voxel_size=image_voxel_size,
            datetype=image.dtype,
        )
        save_tomo(
            label_crop,
            str(target_dir / "labelsTr" / f"{case_name}.mrc"),
            voxel_size=label_voxel_size,
            datetype=label.dtype,
        )

        print(
            f"Saved {case_name}: "
            f"shape={image_crop.shape}, slices={_format_slices(block_slices)}"
        )

    return len(blocks)


def _format_slices(slices: Iterable[slice]) -> str:
    return ", ".join(f"{s.start}:{s.stop}" for s in slices)


def crop_dataset(source_dir: Path, target_dir: Path, overwrite: bool = False) -> None:
    prepare_target_dir(target_dir, overwrite=overwrite)
    num_training = crop_pair(source_dir, target_dir)
    write_dataset_json(source_dir, target_dir, num_training=num_training)
    print(f"Created cropped dataset: {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Dataset016 by splitting Dataset015 TS_199 into 8 center crops."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source nnUNet dataset directory (default: {DEFAULT_SOURCE_DIR}).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Target nnUNet dataset directory (default: {DEFAULT_TARGET_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the target dataset directory before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
