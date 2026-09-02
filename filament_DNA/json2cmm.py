import argparse
import json
import os
from xml.sax.saxutils import escape


DEFAULT_COLORS = [
    (0.94118, 0.66275, 0.61569),
    (0.92549, 0.94118, 0.99608),
    (0.88627, 0.91373, 0.82745),
    (0.99216, 0.99608, 0.72549),
    (0.90196, 0.60000, 0.92941),
    (0.97647, 0.91373, 0.89412),
    (0.65098, 0.87843, 0.92549),
]


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("filaments", "actins", "dnas", "objects"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list, or a dict containing a filament list")

    return data


def default_output_path(json_path):
    base, _ = os.path.splitext(json_path)
    return base + ".cmm"


def load_mrc_origin(mrc_path):
    """Return MRC header origin as (x, y, z)."""
    try:
        import mrcfile
    except ImportError as exc:
        raise RuntimeError(
            "mrcfile is required for --origin-mrc. Install it or use --origin X Y Z."
        ) from exc

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        origin = mrc.header.origin
        return (float(origin.x), float(origin.y), float(origin.z))


def _format_float(value, digits=6):
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _points_equal(point_a, point_b, tolerance):
    if len(point_a) != len(point_b):
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(point_a, point_b))


def _point_to_xyz(point, pixel_size, input_order, coordinate_origin):
    if len(point) != 3:
        raise ValueError(f"Point must have 3 coordinates, got: {point}")

    values = {axis: float(value) for axis, value in zip(input_order, point)}
    origin_x, origin_y, origin_z = coordinate_origin
    return (
        values["x"] * pixel_size + origin_x,
        values["y"] * pixel_size + origin_y,
        values["z"] * pixel_size + origin_z,
    )


def _iter_filament_points(json_data):
    for index, item in enumerate(json_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} is not a JSON object")

        points = item.get("points")
        if points is None:
            raise ValueError(f"Item {index} has no 'points' field")
        if not isinstance(points, list):
            raise ValueError(f"Item {index} field 'points' must be a list")

        filament_id = item.get("id", index)
        yield filament_id, points


def json_to_cmm_text(
    json_data,
    pixel_size,
    input_order="zyx",
    coordinate_origin=(0.0, 0.0, 0.0),
    marker_radius=10.0,
    link_radius=4.59,
    color_mode="by-object",
    color=(0.94118, 0.66275, 0.61569),
    close_tolerance=1e-6,
    marker_set_name="markers",
):
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than 0")

    lines = [f'<marker_set name="{escape(marker_set_name)}">']
    next_marker_id = 1
    marker_count = 0
    link_count = 0
    filament_count = 0

    for filament_index, (_filament_id, points) in enumerate(_iter_filament_points(json_data)):
        if not points:
            continue

        is_closed = len(points) > 2 and _points_equal(points[0], points[-1], close_tolerance)
        unique_points = points[:-1] if is_closed else points
        if not unique_points:
            continue

        if color_mode == "by-object":
            current_color = DEFAULT_COLORS[filament_index % len(DEFAULT_COLORS)]
        else:
            current_color = color

        r, g, b = current_color
        marker_ids = []

        for point in unique_points:
            x, y, z = _point_to_xyz(point, pixel_size, input_order, coordinate_origin)
            marker_id = next_marker_id
            next_marker_id += 1
            marker_ids.append(marker_id)
            marker_count += 1

            lines.append(
                "  "
                f'<marker id="{marker_id}" '
                f'x="{_format_float(x)}" y="{_format_float(y)}" z="{_format_float(z)}" '
                f'r="{_format_float(r, 5)}" g="{_format_float(g, 5)}" '
                f'b="{_format_float(b, 5)}" radius="{_format_float(marker_radius)}"/>'
            )

        for id1, id2 in zip(marker_ids[:-1], marker_ids[1:]):
            lines.append(
                "  "
                f'<link id1="{id1}" id2="{id2}" '
                f'r="{_format_float(r, 5)}" g="{_format_float(g, 5)}" '
                f'b="{_format_float(b, 5)}" radius="{_format_float(link_radius)}"/>'
            )
            link_count += 1

        if is_closed and len(marker_ids) > 2:
            lines.append(
                "  "
                f'<link id1="{marker_ids[-1]}" id2="{marker_ids[0]}" '
                f'r="{_format_float(r, 5)}" g="{_format_float(g, 5)}" '
                f'b="{_format_float(b, 5)}" radius="{_format_float(link_radius)}"/>'
            )
            link_count += 1

        filament_count += 1

    lines.append("</marker_set>")
    text = "\n".join(lines) + "\n"
    stats = {
        "filaments": filament_count,
        "markers": marker_count,
        "links": link_count,
        "coordinate_origin": coordinate_origin,
    }
    return text, stats


def save_cmm(cmm_text, cmm_path):
    with open(cmm_path, "w", encoding="utf-8") as f:
        f.write(cmm_text)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert filament JSON to Chimera/ChimeraX CMM marker links."
    )
    parser.add_argument("json_path", help="Input filament .json file")
    parser.add_argument("pixel_size", type=float, help="Pixel size used to convert pixels to CMM coordinates")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .cmm path. Default: same basename as input JSON",
    )
    parser.add_argument(
        "--input-order",
        choices=("zyx", "xyz"),
        default="zyx",
        help="Coordinate order in JSON. Default: zyx, matching this repo's filament JSON convention",
    )
    origin_group = parser.add_mutually_exclusive_group()
    origin_group.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Physical coordinate origin to add after multiplying by pixel_size",
    )
    origin_group.add_argument(
        "--origin-mrc",
        default=None,
        help="Read coordinate origin from this MRC header and add it after scaling",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=10.0,
        help="CMM marker radius. Default: 10",
    )
    parser.add_argument(
        "--link-radius",
        type=float,
        default=4.59,
        help="CMM link radius. Default: 4.59",
    )
    parser.add_argument(
        "--color-mode",
        choices=("by-object", "single"),
        default="by-object",
        help="Use cycling colors per object or one color for all markers/links",
    )
    parser.add_argument(
        "--color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.94118, 0.66275, 0.61569),
        help="RGB color for --color-mode single, each value in 0-1",
    )
    parser.add_argument(
        "--close-tolerance",
        type=float,
        default=1e-6,
        help="Treat first and last JSON points as the same closed-loop point within this tolerance",
    )
    parser.add_argument(
        "--marker-set-name",
        default="markers",
        help="CMM marker_set name. Default: markers",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.origin_mrc:
        coordinate_origin = load_mrc_origin(args.origin_mrc)
    elif args.origin:
        coordinate_origin = tuple(args.origin)
    else:
        coordinate_origin = (0.0, 0.0, 0.0)

    json_data = load_json(args.json_path)
    cmm_text, stats = json_to_cmm_text(
        json_data=json_data,
        pixel_size=args.pixel_size,
        input_order=args.input_order,
        coordinate_origin=coordinate_origin,
        marker_radius=args.marker_radius,
        link_radius=args.link_radius,
        color_mode=args.color_mode,
        color=tuple(args.color),
        close_tolerance=args.close_tolerance,
        marker_set_name=args.marker_set_name,
    )

    cmm_path = args.output or default_output_path(args.json_path)
    save_cmm(cmm_text, cmm_path)

    print(f"[OK] Saved CMM to {cmm_path}")
    print(
        "[INFO] "
        f"filaments={stats['filaments']}, markers={stats['markers']}, links={stats['links']}"
    )
    print(f"[INFO] coordinate_origin={stats['coordinate_origin']}")


if __name__ == "__main__":
    main()
