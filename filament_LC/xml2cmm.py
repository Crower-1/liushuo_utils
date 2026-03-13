import argparse
import re
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

    match = re.match(br"\s*<\?xml[^>]*encoding=['\"]([^'\"]+)['\"]", data)
    encoding = match.group(1).decode("ascii", errors="ignore").lower() if match else "utf-8"

    try:
        text = data.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        text = data.decode("utf-8", errors="replace")

    # ElementTree does not allow encoding declarations in decoded strings.
    text = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", text, count=1)
    return ET.fromstring(text)


def xml_points_to_cmm(xml_path, pixel_size, cmm_output_path, radius=10, rgb=(1, 1, 0)):
    root = _load_xml_root(xml_path)

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    points = []

    for ws in root.findall(".//ss:Worksheet", ns):
        if ws.attrib.get("{urn:schemas-microsoft-com:office:spreadsheet}Name") == "Points":
            rows = ws.findall(".//ss:Row", ns)[1:]
            for row in rows:
                cells = row.findall("ss:Cell/ss:Data", ns)
                if len(cells) < 5:
                    continue
                x = float(cells[2].text) * pixel_size
                y = float(cells[3].text) * pixel_size
                z = float(cells[4].text) * pixel_size
                points.append((x, y, z))

    r, g, b = rgb
    lines = ["<marker_set name=\"markers\">"]
    for idx, (x, y, z) in enumerate(points, start=1):
        lines.append(
            f'<marker id="{idx}" x="{x:.2f}" y="{y:.2f}" z="{z:.2f}" '
            f'r="{r}" g="{g}" b="{b}" radius="{radius}"/>'
        )
    lines.append("</marker_set>")

    with open(cmm_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote {len(points)} markers -> {cmm_output_path}")


# def _build_arg_parser():
#     parser = argparse.ArgumentParser(description="Convert Avizo filament XML Points to CMM markers.")
#     parser.add_argument("xml_path", help="Path to Avizo Excel-XML file")
#     parser.add_argument("pixel_size", type=float, help="Pixel size scale factor")
#     parser.add_argument(
#         "-o",
#         "--output",
#         dest="output",
#         help="Output CMM path (default: same name with .cmm)",
#     )
#     parser.add_argument("--radius", type=float, default=10, help="Marker radius (default: 10)")
#     return parser


def main():

    xml_path = f'/media/liushuo/data1/data/liucong/test_demo/TS_115_9.30Apx.CorrelationLines.xml'
    output_path = xml_path.replace('.xml', '.cmm')
    pixel_size = 9.3  # Example pixel size

    xml_points_to_cmm(xml_path, pixel_size, output_path, radius=10)


if __name__ == "__main__":
    main()
