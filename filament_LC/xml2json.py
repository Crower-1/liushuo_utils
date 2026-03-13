import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET


def _load_xml_root(xml_path):
    try:
        tree = ET.parse(xml_path)
        return tree.getroot()
    except ValueError as exc:
        if "multi-byte encodings are not supported" not in str(exc):
            raise

    with open(xml_path, "rb") as f:
        data = f.read()

    match = re.match(br'\s*<\?xml[^>]*encoding=[\'"]([^\'"]+)[\'"]', data)
    encoding = match.group(1).decode("ascii", errors="ignore").lower() if match else "utf-8"

    try:
        text = data.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        text = data.decode("utf-8", errors="replace")

    # ElementTree does not allow encoding declarations in decoded strings.
    text = re.sub(r'^\s*<\?xml[^>]*\?>\s*', '', text, count=1)
    return ET.fromstring(text)


def avizo_filament_xml_to_json(xml_path, json_path, flip_y=False, mrc_shape=None):
    """
    Convert Avizo filament Excel-XML to JSON:
    one segment -> ordered 3D keypoints
    """

    root = _load_xml_root(xml_path)

    # Excel XML namespace
    ns = {
        "ss": "urn:schemas-microsoft-com:office:spreadsheet"
    }

    if flip_y:
        if not mrc_shape:
            raise ValueError("mrc_shape is required when flip_y is True")
        y_max = mrc_shape[1] - 1
    else:
        y_max = None

    # --------------------------------------------------
    # 1. Parse Points table
    # --------------------------------------------------
    point_dict = {}
    min_x = min_y = min_z = None
    max_x = max_y = max_z = None

    for ws in root.findall(".//ss:Worksheet", ns):
        if ws.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Name") == "Points":
            rows = ws.findall(".//ss:Row", ns)[1:]  # skip header

            for row in rows:
                cells = row.findall("ss:Cell/ss:Data", ns)
                if len(cells) < 5:
                    continue

                pid = int(float(cells[0].text))
                x = float(cells[2].text)
                y = float(cells[3].text)
                z = float(cells[4].text)

                if y_max is not None:
                    y = y_max - y

                point_dict[pid] = [z, y, x]
                if min_x is None:
                    min_x = max_x = x
                    min_y = max_y = y
                    min_z = max_z = z
                else:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_z = min(min_z, z)
                    max_z = max(max_z, z)

    if min_x is not None:
        print(f"[INFO] X min/max: {min_x} / {max_x}")
        print(f"[INFO] Y min/max: {min_y} / {max_y}")
        print(f"[INFO] Z min/max: {min_z} / {max_z}")

    # --------------------------------------------------
    # 2. Parse Segments table
    # --------------------------------------------------
    segments = []

    for ws in root.findall(".//ss:Worksheet", ns):
        if ws.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Name") == "Segments":
            rows = ws.findall(".//ss:Row", ns)[1:]  # skip header

            for row in rows:
                cells = row.findall("ss:Cell/ss:Data", ns)
                if len(cells) < 18:
                    continue

                seg_id = int(float(cells[0].text))
                point_ids_str = cells[17].text  # "0,1,2,3,..."
                point_ids = [int(p) for p in point_ids_str.split(",") if p.strip()]

                points = []
                for pid in point_ids:
                    if pid in point_dict:
                        points.append(point_dict[pid])

                json_id = seg_id + 1  # JSON IDs start from 1
                segments.append({
                    "id": json_id,
                    "points": points
                })

    # --------------------------------------------------
    # 3. Write JSON
    # --------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4)

    print(f"[OK] Converted {len(segments)} segments → {json_path}")

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert Avizo filament Excel-XML to JSON.")
    parser.add_argument("xml_path", help="Path to Avizo Excel-XML file")
    parser.add_argument("json_path", help="Output JSON path")
    parser.add_argument("--flip-y", action="store_true", help="Flip Y using the original MRC shape")
    parser.add_argument("--ori-mrc-path", help="Path to original MRC file (required with --flip-y)")
    return parser


def _load_mrc_shape(mrc_path):
    import mrcfile

    with mrcfile.open(mrc_path, "r") as mrc:
        return mrc.data.shape


def main():
    xml_path = f"/media/liushuo/data1/data/liucong/TS_009/TS_009_9.30Apx.CorrelationLines.xml"
    json_path = xml_path.replace(".xml", ".json")
    # json_path = "/media/liushuo/data1/data/liucong/test_demo/TS_115_9.30Apx.json"
    # avizo_filament_xml_to_json(xml_path, json_path)

    # parser = _build_arg_parser()
    # args = parser.parse_args()
    ori_mrc_path = f"/media/liushuo/data1/data/liucong/TS_009/TS_009_9.30Apx.mrc"
    mrc_shape = _load_mrc_shape(ori_mrc_path)

    avizo_filament_xml_to_json(
        xml_path,
        json_path,
        flip_y=True,
        mrc_shape=mrc_shape,
    )


if __name__ == "__main__":
    main()
