"""Create a pre/post nnUNet raw dataset from YYT ROI mask archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import tempfile

import mrcfile
import numpy as np
import py7zr

from common import atomic_dataset_directory, validate_mrc_pair, write_dataset_json


DEFAULT_SOURCE_ROOT = Path("/share/data/CryoET_Data/yanyt/4ls")
DEFAULT_OUTPUT = Path(
    "/share/data/CryoET_Data/liushuo/dataset/nnUNet/nnUNet_raw/"
    "Dataset004_synapseseg_roiprepost"
)
DEFAULT_LABEL_REMAPS = {
    "pp366": {3: 2},  # third ROI is post-synaptic
    "pp4001": {3: 1},  # third ROI is pre-synaptic
}


def discover_label_groups(source_root: Path) -> dict[str, dict[str, Path]]:
    """Discover base and optional ``-1`` ROI-mask archives by case name."""

    groups: dict[str, dict[str, Path]] = {}
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"(.+?)(-1)?", directory.name)
        if match is None:
            continue
        case = match.group(1)
        archive = directory / "synapse_seg" / "roi" / f"{case}_roi_mask.mrc.7z"
        if archive.is_file():
            groups.setdefault(case, {})[directory.name] = archive

    if not groups:
        raise ValueError(f"No ROI mask archives found under {source_root}")
    for case, entries in groups.items():
        names = set(entries)
        if case not in names or not names.issubset({case, f"{case}-1"}):
            raise ValueError(f"{case}: unsupported/missing label directories {sorted(names)}")
    return groups


def select_image(
    source_root: Path,
    case: str,
    special_images: dict[str, Path],
) -> Path:
    """Resolve the source image, including the known pp387 Isonet exception."""

    if case in special_images:
        image = special_images[case]
    elif case == "pp387":
        image = (
            source_root
            / case
            / "synapse_seg"
            / "isonet"
            / "tomo_deconv"
            / f"{case}_wbp_resample.mrc"
        )
    else:
        image = source_root / case / "synapse_seg" / f"{case}_wbp_corrected.mrc"
    if not image.is_file():
        raise FileNotFoundError(f"{case}: source image not found: {image}")
    return image


def extract_single_mrc(archive: Path, destination: Path, staging: Path) -> None:
    """Extract the single MRC member of a 7z archive."""

    with tempfile.TemporaryDirectory(prefix="extract_", dir=staging) as temp_name:
        temp_dir = Path(temp_name)
        with py7zr.SevenZipFile(archive, mode="r") as seven_zip:
            names = seven_zip.getnames()
            if len(names) != 1:
                raise ValueError(f"{archive}: expected one entry, found {names}")
            seven_zip.extractall(path=temp_dir)
        files = [path for path in temp_dir.rglob("*") if path.is_file()]
        if len(files) != 1:
            raise ValueError(f"{archive}: expected one extracted file, found {files}")
        shutil.move(str(files[0]), destination)


def remap_labels(label_path: Path, mapping: dict[int, int]) -> dict[int, int]:
    """Apply explicit per-case semantic label remapping in-place."""

    changed: dict[int, int] = {}
    with mrcfile.mmap(label_path, mode="r+", permissive=True) as mrc:
        for source_value, target_value in mapping.items():
            count = 0
            for z_index in range(mrc.data.shape[0]):
                mask = mrc.data[z_index] == source_value
                count += int(np.count_nonzero(mask))
                mrc.data[z_index][mask] = target_value
            changed[source_value] = count
        mrc.update_header_stats()
        mrc.flush()
    return changed


def merge_background_fill(primary: Path, secondary: Path) -> tuple[int, int]:
    """Fill primary background from secondary while preserving primary conflicts."""

    filled = 0
    conflicts = 0
    with mrcfile.mmap(primary, mode="r+", permissive=True) as dst, mrcfile.mmap(
        secondary, mode="r", permissive=True
    ) as src:
        if dst.data.shape != src.data.shape:
            raise ValueError(f"Merge shape mismatch: {dst.data.shape} != {src.data.shape}")
        for z_index in range(dst.data.shape[0]):
            dst_slice = dst.data[z_index]
            src_slice = src.data[z_index]
            fill_mask = (dst_slice == 0) & (src_slice != 0)
            conflict_mask = (
                (dst_slice != 0) & (src_slice != 0) & (dst_slice != src_slice)
            )
            filled += int(np.count_nonzero(fill_mask))
            conflicts += int(np.count_nonzero(conflict_mask))
            dst_slice[fill_mask] = src_slice[fill_mask]
        dst.update_header_stats()
        dst.flush()
    return filled, conflicts


def create_dataset(
    source_root: Path,
    output_dir: Path,
    *,
    expected_count: int | None = None,
    label_remaps: dict[str, dict[int, int]] | None = None,
    special_images: dict[str, Path] | None = None,
) -> Path:
    """Create an nnUNet raw dataset from all discovered YYT tomo directories."""

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    groups = discover_label_groups(source_root)
    cases = sorted(groups)
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, discovered {len(cases)}")
    label_remaps = label_remaps or {}
    special_images = special_images or {}
    print(f"Discovered {len(cases)} cases: {cases}", flush=True)

    with atomic_dataset_directory(output_dir) as staging:
        images_dir = staging / "imagesTr"
        labels_dir = staging / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()
        dataset_counts = {1: 0, 2: 0}

        for index, case in enumerate(cases, start=1):
            image_source = select_image(source_root, case, special_images)
            target_image = images_dir / f"{case}_0000.mrc"
            target_label = labels_dir / f"{case}.mrc"
            print(f"[{index}/{len(cases)}] {case}: copying {image_source.name}", flush=True)
            shutil.copy2(image_source, target_image)
            extract_single_mrc(groups[case][case], target_label, staging)

            mapping = label_remaps.get(case)
            if mapping:
                changed = remap_labels(target_label, mapping)
                print(f"[{index}/{len(cases)}] {case}: remapped {mapping}, voxels={changed}", flush=True)

            suffix_archive = groups[case].get(f"{case}-1")
            if suffix_archive is not None:
                with tempfile.TemporaryDirectory(prefix=f"merge_{case}_", dir=staging) as temp_name:
                    suffix_label = Path(temp_name) / f"{case}-1.mrc"
                    extract_single_mrc(suffix_archive, suffix_label, staging)
                    if mapping:
                        remap_labels(suffix_label, mapping)
                    filled, conflicts = merge_background_fill(target_label, suffix_label)
                print(
                    f"[{index}/{len(cases)}] {case}: merged -1, "
                    f"filled={filled}, conflicts_kept={conflicts}",
                    flush=True,
                )

            counts = validate_mrc_pair(target_image, target_label)
            dataset_counts[1] += counts[1]
            dataset_counts[2] += counts[2]
            print(f"[{index}/{len(cases)}] {case}: labels={counts}", flush=True)

        if dataset_counts[1] == 0 or dataset_counts[2] == 0:
            raise ValueError(f"Dataset is missing a foreground class: {dataset_counts}")
        write_dataset_json(staging, len(cases))

    print(f"Created {output_dir}", flush=True)
    return output_dir


def parse_remap(value: str) -> tuple[str, int, int]:
    """Parse ``CASE:SOURCE:TARGET`` from the command line."""

    try:
        case, source, target = value.split(":", maxsplit=2)
        return case, int(source), int(target)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "label remap must have the form CASE:SOURCE:TARGET"
        ) from error


def parse_special_image(value: str) -> tuple[str, Path]:
    """Parse ``CASE=PATH`` from the command line."""

    try:
        case, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("special image must have the form CASE=PATH") from error
    return case, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--label-remap",
        action="append",
        type=parse_remap,
        default=[],
        metavar="CASE:SOURCE:TARGET",
        help="Add or override a per-case label mapping; may be repeated.",
    )
    parser.add_argument(
        "--no-default-remaps",
        action="store_true",
        help="Disable the known pp366 and pp4001 label-3 mappings.",
    )
    parser.add_argument(
        "--special-image",
        action="append",
        type=parse_special_image,
        default=[],
        metavar="CASE=PATH",
        help="Override a case's source image; may be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    remaps = {} if args.no_default_remaps else {
        case: mapping.copy() for case, mapping in DEFAULT_LABEL_REMAPS.items()
    }
    for case, source, target in args.label_remap:
        remaps.setdefault(case, {})[source] = target
    create_dataset(
        args.source_root,
        args.output,
        expected_count=args.expected_count,
        label_remaps=remaps,
        special_images=dict(args.special_image),
    )


if __name__ == "__main__":
    main()
