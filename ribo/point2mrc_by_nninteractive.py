import argparse
import os

from point2mrc_nninteractive_lib import (
    guess_point_file,
    list_images,
    prepare_session,
    run_inference_on_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert points (.point) to nnInteractive masks for every tomogram."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory with input tomograms (.mrc/.rec).",
    )
    parser.add_argument(
        "--coords-dir",
        required=True,
        help="Directory with point files (.point).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store the segmentation volumes.",
    )
    parser.add_argument(
        "--image-pixel-size",
        type=float,
        required=True,
        help="Pixel size (nm) of the tomograms.",
    )
    parser.add_argument(
        "--coords-pixel-size",
        type=float,
        required=True,
        help="Pixel size (nm) of coordinates stored in .point files.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for nnInteractive (default: cuda:0).",
    )
    parser.add_argument(
        "--allowed-exts",
        default=".mrc,.rec,.mrcs",
        help="Comma separated list of valid image extensions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image_dir = os.path.abspath(args.image_dir)
    coords_dir = os.path.abspath(args.coords_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(coords_dir):
        raise FileNotFoundError(f"Coordinate directory not found: {coords_dir}")

    scale = args.coords_pixel_size / args.image_pixel_size
    allowed_exts = tuple(ext.strip().lower() for ext in args.allowed_exts.split(",") if ext.strip())
    image_files = list_images(image_dir, allowed_exts)

    if not image_files:
        raise RuntimeError(f"No image files found in {image_dir} with extensions {allowed_exts}")

    session = prepare_session(args.device)

    for image_path in image_files:
        try:
            point_path = guess_point_file(image_path, coords_dir)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue

        try:
            run_inference_on_image(
                session=session,
                image_path=image_path,
                point_path=point_path,
                scale=scale,
                output_dir=output_dir,
                target_pixel_size=args.image_pixel_size,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[error] Failed on {image_path}: {exc}")


if __name__ == "__main__":
    main()
