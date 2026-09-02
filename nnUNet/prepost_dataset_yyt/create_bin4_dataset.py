"""Create a bin4 nnUNet dataset using SynapseSeg's resampling utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

from synapseseg.domain.image_utils import (
    resample_image_by_bin4,
    resample_label_with_output_shape,
)

from common import (
    atomic_dataset_directory,
    copy_spatial_header,
    paired_training_cases,
    validate_mrc_pair,
)


DEFAULT_SOURCE = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/"
    "Dataset004_synapseseg_roiprepost"
)
DEFAULT_OUTPUT = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/"
    "Dataset005_synapseseg_roiprepost_34tomo_bin4"
)


def create_bin4_dataset(
    source_dataset: Path,
    output_dir: Path,
    *,
    expected_count: int | None = None,
) -> Path:
    """Bin images by block mean and labels by nearest-neighbour resampling."""

    source_dataset = source_dataset.resolve()
    output_dir = output_dir.resolve()
    cases = paired_training_cases(source_dataset, expected_count)
    print(f"Found {len(cases)} paired cases", flush=True)

    with atomic_dataset_directory(output_dir) as staging:
        images_out = staging / "imagesTr"
        labels_out = staging / "labelsTr"
        images_out.mkdir()
        labels_out.mkdir()

        for index, case in enumerate(cases, start=1):
            source_image = source_dataset / "imagesTr" / f"{case}_0000.mrc"
            source_label = source_dataset / "labelsTr" / f"{case}.mrc"
            target_image = images_out / f"{case}_0000.mrc"
            target_label = labels_out / f"{case}.mrc"

            print(f"[{index}/{len(cases)}] {case}: binning image", flush=True)
            with mrcfile.mmap(source_image, mode="r", permissive=True) as image:
                source_shape = tuple(image.data.shape)
                expected_shape = tuple(size // 4 for size in source_shape)
                binned_image = np.asarray(
                    resample_image_by_bin4(image.data), dtype=np.float32
                )
                if binned_image.shape != expected_shape:
                    raise ValueError(
                        f"{case}: image shape {binned_image.shape}, expected {expected_shape}"
                    )
                with mrcfile.new(target_image, overwrite=False) as output:
                    output.set_data(np.ascontiguousarray(binned_image))
                    copy_spatial_header(image, output, voxel_scale=4)
                    output.update_header_stats()
                del binned_image

            print(f"[{index}/{len(cases)}] {case}: binning label", flush=True)
            with mrcfile.mmap(source_label, mode="r", permissive=True) as label:
                if tuple(label.data.shape) != source_shape:
                    raise ValueError(f"{case}: source image/label shape mismatch")
                binned_label = np.asarray(
                    resample_label_with_output_shape(label.data, expected_shape),
                    dtype=np.int8,
                )
                values = {int(value) for value in np.unique(binned_label)}
                if not values.issubset({0, 1, 2}):
                    raise ValueError(f"{case}: unexpected labels after bin4: {sorted(values)}")
                with mrcfile.new(target_label, overwrite=False) as output:
                    output.set_data(np.ascontiguousarray(binned_label))
                    copy_spatial_header(label, output, voxel_scale=4)
                    output.update_header_stats()
                del binned_label

            counts = validate_mrc_pair(target_image, target_label)
            print(
                f"[{index}/{len(cases)}] {case}: {source_shape} -> "
                f"{expected_shape}, labels={counts}",
                flush=True,
            )

        with (source_dataset / "dataset.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        metadata["numTraining"] = len(cases)
        with (staging / "dataset.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=4)
            stream.write("\n")

    print(f"Created {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_bin4_dataset(args.source, args.output, expected_count=args.expected_count)


if __name__ == "__main__":
    main()
