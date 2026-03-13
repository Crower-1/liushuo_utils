import argparse
import os

from point2mrc_nninteractive_lib import prepare_session, run_inference_on_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one .point file to nnInteractive mask for a single tomogram."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input tomogram (.mrc/.rec).",
    )
    parser.add_argument(
        "--point",
        required=True,
        help="Path to input point file (.point).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .mrc path (default: <image>_nninteractive.mrc in cwd).",
    )
    parser.add_argument(
        "--image-pixel-size",
        type=float,
        required=True,
        help="Pixel size (nm) of the tomogram.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing output if present.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    image_path = os.path.abspath(args.image)
    point_path = os.path.abspath(args.point)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(point_path):
        raise FileNotFoundError(f"Point file not found: {point_path}")

    scale = args.coords_pixel_size / args.image_pixel_size
    session = prepare_session(args.device)

    output_path = args.output
    if output_path is None:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(os.getcwd(), f"{stem}_nninteractive.mrc")
    else:
        output_path = os.path.abspath(output_path)

    run_inference_on_image(
        session=session,
        image_path=image_path,
        point_path=point_path,
        scale=scale,
        output_dir=None,
        target_pixel_size=args.image_pixel_size,
        overwrite=args.overwrite,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
