import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def _sort_key(marker_id):
    try:
        return (0, int(marker_id))
    except ValueError:
        return (1, marker_id)


def _local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _edge_key(id1, id2):
    return tuple(sorted((id1, id2), key=_sort_key))


def parse_cmm(cmm_path):
    """Read markers and links from a Chimera/ChimeraX CMM file."""
    tree = ET.parse(cmm_path)
    root = tree.getroot()

    markers = {}
    links = []

    for elem in root.iter():
        tag = _local_name(elem.tag)

        if tag == "marker":
            marker_id = elem.attrib.get("id")
            if marker_id is None:
                raise ValueError("Found a marker without an id attribute")
            if marker_id in markers:
                raise ValueError(f"Duplicate marker id: {marker_id}")

            try:
                x = float(elem.attrib["x"])
                y = float(elem.attrib["y"])
                z = float(elem.attrib["z"])
            except KeyError as exc:
                raise ValueError(f"Marker {marker_id} is missing {exc.args[0]!r}") from exc

            markers[marker_id] = (x, y, z)

        elif tag == "link":
            id1 = elem.attrib.get("id1")
            id2 = elem.attrib.get("id2")
            if id1 is None or id2 is None:
                raise ValueError("Found a link without id1 or id2")
            links.append((id1, id2))

    return markers, links


def build_graph(markers, links):
    """Build an undirected graph from CMM links."""
    adjacency = defaultdict(set)
    edge_count = 0
    duplicate_count = 0
    missing_links = []
    seen_edges = set()

    for id1, id2 in links:
        if id1 not in markers or id2 not in markers:
            missing_links.append((id1, id2))
            continue
        if id1 == id2:
            duplicate_count += 1
            continue

        key = _edge_key(id1, id2)
        if key in seen_edges:
            duplicate_count += 1
            continue

        seen_edges.add(key)
        adjacency[id1].add(id2)
        adjacency[id2].add(id1)
        edge_count += 1

    return adjacency, edge_count, duplicate_count, missing_links


def _walk_path(start, next_node, adjacency, visited_edges):
    """Walk one non-branching path from an endpoint or branch node."""
    path = [start]
    prev = start
    curr = next_node

    visited_edges.add(_edge_key(prev, curr))
    path.append(curr)

    while len(adjacency[curr]) == 2:
        candidates = [
            node
            for node in sorted(adjacency[curr], key=_sort_key)
            if node != prev and _edge_key(curr, node) not in visited_edges
        ]
        if not candidates:
            break

        next_node = candidates[0]
        visited_edges.add(_edge_key(curr, next_node))
        prev, curr = curr, next_node
        path.append(curr)

    return path


def _walk_cycle(start, next_node, adjacency, visited_edges):
    """Walk a closed loop; repeat the start node at the end to preserve closure."""
    path = [start]
    prev = start
    curr = next_node

    visited_edges.add(_edge_key(prev, curr))
    path.append(curr)

    while curr != start:
        candidates = [
            node
            for node in sorted(adjacency[curr], key=_sort_key)
            if node != prev and _edge_key(curr, node) not in visited_edges
        ]
        if not candidates:
            break

        next_node = candidates[0]
        visited_edges.add(_edge_key(curr, next_node))
        prev, curr = curr, next_node
        path.append(curr)

    if path[-1] != start:
        raise ValueError(f"Could not close loop that starts at marker {start}")

    return path


def link_markers_to_paths(adjacency):
    """
    Convert the link graph into ordered marker-id paths.

    Linear components become one path from endpoint to endpoint.
    Closed components become one path with the first marker repeated at the end.
    Branching components are split at branch markers because the JSON format stores
    one ordered polyline per object.
    """
    paths = []
    visited_edges = set()

    start_nodes = sorted(
        [node for node, neighbors in adjacency.items() if len(neighbors) != 2],
        key=_sort_key,
    )

    for start in start_nodes:
        for next_node in sorted(adjacency[start], key=_sort_key):
            if _edge_key(start, next_node) in visited_edges:
                continue
            paths.append(_walk_path(start, next_node, adjacency, visited_edges))

    for start in sorted(adjacency, key=_sort_key):
        for next_node in sorted(adjacency[start], key=_sort_key):
            if _edge_key(start, next_node) in visited_edges:
                continue
            paths.append(_walk_cycle(start, next_node, adjacency, visited_edges))

    return paths


def convert_point(
    point_xyz,
    pixel_size,
    output_order,
    round_digits=None,
    coordinate_origin=(0.0, 0.0, 0.0),
):
    x, y, z = point_xyz
    origin_x, origin_y, origin_z = coordinate_origin
    scaled = {
        "x": (x - origin_x) / pixel_size,
        "y": (y - origin_y) / pixel_size,
        "z": (z - origin_z) / pixel_size,
    }
    point = [scaled[axis] for axis in output_order]

    if round_digits is not None:
        point = [round(value, round_digits) for value in point]

    return point


def calculate_length(points):
    if len(points) < 2:
        return 0.0

    length = 0.0
    for point_a, point_b in zip(points[:-1], points[1:]):
        length += math.dist(point_a, point_b)
    return float(length)


def cmm_to_json_data(
    cmm_path,
    pixel_size,
    output_order="zyx",
    round_digits=None,
    include_isolated=False,
    coordinate_origin=(0.0, 0.0, 0.0),
):
    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than 0")

    markers, links = parse_cmm(cmm_path)
    adjacency, edge_count, duplicate_count, missing_links = build_graph(markers, links)
    marker_id_paths = link_markers_to_paths(adjacency)

    if include_isolated:
        linked_marker_ids = set(adjacency)
        for marker_id in sorted(markers, key=_sort_key):
            if marker_id not in linked_marker_ids:
                marker_id_paths.append([marker_id])

    data = []
    for json_id, marker_id_path in enumerate(marker_id_paths, start=1):
        points = [
            convert_point(
                markers[marker_id],
                pixel_size,
                output_order,
                round_digits,
                coordinate_origin,
            )
            for marker_id in marker_id_path
        ]
        data.append(
            {
                "id": json_id,
                "points": points,
                "length": calculate_length(points),
            }
        )

    stats = {
        "markers": len(markers),
        "links": len(links),
        "used_links": edge_count,
        "duplicate_or_self_links": duplicate_count,
        "missing_links": missing_links,
        "paths": len(data),
        "coordinate_origin": coordinate_origin,
    }
    return data, stats


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def default_output_path(cmm_path):
    base, _ = os.path.splitext(cmm_path)
    return base + ".json"


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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert Chimera/ChimeraX CMM marker links to filament JSON."
    )
    parser.add_argument("cmm_path", help="Input .cmm file")
    parser.add_argument("pixel_size", type=float, help="Pixel size used by the CMM coordinates")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .json path. Default: same basename as input CMM",
    )
    parser.add_argument(
        "--output-order",
        choices=("zyx", "xyz"),
        default="zyx",
        help="Coordinate order written to JSON. Default: zyx, matching this repo's filament JSON convention",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=None,
        help="Optionally round output coordinates to this many decimal places",
    )
    parser.add_argument(
        "--include-isolated",
        action="store_true",
        help="Also save unlinked markers as single-point objects with length 0",
    )
    origin_group = parser.add_mutually_exclusive_group()
    origin_group.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Physical coordinate origin to subtract before dividing by pixel_size",
    )
    origin_group.add_argument(
        "--origin-mrc",
        default=None,
        help="Read coordinate origin from this MRC header and subtract it before scaling",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    json_path = args.output or default_output_path(args.cmm_path)
    if args.origin_mrc:
        coordinate_origin = load_mrc_origin(args.origin_mrc)
    elif args.origin:
        coordinate_origin = tuple(args.origin)
    else:
        coordinate_origin = (0.0, 0.0, 0.0)

    data, stats = cmm_to_json_data(
        cmm_path=args.cmm_path,
        pixel_size=args.pixel_size,
        output_order=args.output_order,
        round_digits=args.round_digits,
        include_isolated=args.include_isolated,
        coordinate_origin=coordinate_origin,
    )
    save_json(data, json_path)

    print(f"[OK] Saved {stats['paths']} paths to {json_path}")
    print(
        "[INFO] "
        f"markers={stats['markers']}, links={stats['links']}, used_links={stats['used_links']}"
    )
    print(f"[INFO] coordinate_origin={stats['coordinate_origin']}")
    if stats["duplicate_or_self_links"]:
        print(f"[WARN] Skipped {stats['duplicate_or_self_links']} duplicate/self links")
    if stats["missing_links"]:
        print(f"[WARN] Skipped {len(stats['missing_links'])} links with missing markers")


if __name__ == "__main__":
    main()
