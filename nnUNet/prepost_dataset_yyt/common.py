"""Shared nnUNet dataset validation and transactional output helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from typing import Iterator

import mrcfile
import numpy as np


ALLOWED_LABELS = frozenset({0, 1, 2})


@contextmanager
def atomic_dataset_directory(output_dir: Path) -> Iterator[Path]:
    """Yield a sibling staging directory and publish it atomically on success."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
        yield staging
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def paired_training_cases(dataset_dir: Path, expected_count: int | None = None) -> list[str]:
    """Return sorted nnUNet case names after checking image/label pairing."""

    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    image_cases = {
        path.name.removesuffix("_0000.mrc")
        for path in images_dir.glob("*_0000.mrc")
    }
    label_cases = {path.stem for path in labels_dir.glob("*.mrc")}
    if image_cases != label_cases:
        raise ValueError(
            "Unpaired training files: "
            f"images_only={sorted(image_cases - label_cases)}, "
            f"labels_only={sorted(label_cases - image_cases)}"
        )
    cases = sorted(image_cases)
    if not cases:
        raise ValueError(f"No paired training cases found in {dataset_dir}")
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {len(cases)}")
    return cases


def validate_mrc_pair(image_path: Path, label_path: Path) -> dict[int, int]:
    """Validate geometry and label values for one MRC image/label pair."""

    with mrcfile.mmap(image_path, mode="r", permissive=True) as image, mrcfile.mmap(
        label_path, mode="r", permissive=True
    ) as label:
        if image.data.shape != label.data.shape:
            raise ValueError(
                f"{image_path.stem}: shape mismatch "
                f"{image.data.shape} != {label.data.shape}"
            )
        if not np.allclose(
            image.voxel_size.tolist(),
            label.voxel_size.tolist(),
            rtol=0,
            atol=1e-4,
        ):
            raise ValueError(f"{image_path.stem}: image/label voxel size mismatch")

        counts = {value: 0 for value in ALLOWED_LABELS}
        unexpected: set[int] = set()
        for z_index in range(label.data.shape[0]):
            values, value_counts = np.unique(label.data[z_index], return_counts=True)
            for value, count in zip(values.tolist(), value_counts.tolist()):
                integer_value = int(value)
                if integer_value in counts:
                    counts[integer_value] += int(count)
                else:
                    unexpected.add(integer_value)
        if unexpected:
            raise ValueError(
                f"{label_path.stem}: unexpected label values {sorted(unexpected)}"
            )
        return counts


def write_dataset_json(output_dir: Path, num_training: int) -> None:
    """Write the two-class nnUNet v2 dataset metadata."""

    metadata = {
        "channel_names": {"0": "cryoET"},
        "labels": {"background": 0, "pre": 1, "post": 2},
        "numTraining": num_training,
        "file_ending": ".mrc",
    }
    with (output_dir / "dataset.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=4)
        stream.write("\n")


def copy_spatial_header(
    source: mrcfile.mrcobject.MrcObject,
    target: mrcfile.mrcobject.MrcObject,
    voxel_scale: float,
) -> None:
    """Copy spatial MRC metadata while scaling voxel spacing."""

    target.voxel_size = tuple(
        float(value) * voxel_scale for value in source.voxel_size.tolist()
    )
    target.header.origin.x = source.header.origin.x
    target.header.origin.y = source.header.origin.y
    target.header.origin.z = source.header.origin.z
    target.header.mapc = source.header.mapc
    target.header.mapr = source.header.mapr
    target.header.maps = source.header.maps
