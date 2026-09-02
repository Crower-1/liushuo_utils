#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path


def _normalize_header(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _find_column_indices(header):
    normalized = [_normalize_header(h) for h in header]

    id_idx = x_idx = y_idx = z_idx = None
    for i, col in enumerate(normalized):
        if id_idx is None and col in {"ids", "id", "segmentid", "fiberid", "actinid"}:
            id_idx = i
        if x_idx is None and col.startswith("x"):
            x_idx = i
        if y_idx is None and col.startswith("y"):
            y_idx = i
        if z_idx is None and col.startswith("z"):
            z_idx = i

    if None in (id_idx, x_idx, y_idx, z_idx):
        raise ValueError(
            "无法识别列名，CSV 需包含 IDs/X/Y/Z 四列。"
            f" 当前表头: {header}"
        )

    return id_idx, x_idx, y_idx, z_idx


def _is_numeric_row(row):
    if len(row) < 4:
        return False
    try:
        float(row[0])
        float(row[1])
        float(row[2])
        float(row[3])
        return True
    except ValueError:
        return False


def _distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def convert_actin_csv_to_json(input_csv: Path, output_json: Path, pixel_size: float = 1.0):
    if pixel_size <= 0:
        raise ValueError("pixel_size 必须大于 0。")

    grouped = OrderedDict()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("输入 CSV 为空。")

        if _is_numeric_row(first_row):
            # 无表头：默认前四列分别为 IDs, X, Y, Z
            id_idx, x_idx, y_idx, z_idx = 0, 1, 2, 3
            data_rows = [(1, first_row)]
            row_start = 2
        else:
            # 有表头：自动识别 IDs/X/Y/Z 列
            id_idx, x_idx, y_idx, z_idx = _find_column_indices(first_row)
            data_rows = []
            row_start = 2

        for row_idx, row in data_rows:
            if not row or all(str(cell).strip() == "" for cell in row):
                continue
            if max(id_idx, x_idx, y_idx, z_idx) >= len(row):
                raise ValueError(f"第 {row_idx} 行列数不足: {row}")

            try:
                raw_id = int(float(row[id_idx]))
                x = float(row[x_idx])
                y = float(row[y_idx])
                z = float(row[z_idx])
            except ValueError as exc:
                raise ValueError(f"第 {row_idx} 行存在无法转换的数值: {row}") from exc

            grouped.setdefault(raw_id, []).append([x, y, z])

        for row_idx, row in enumerate(reader, start=row_start):
            if not row or all(str(cell).strip() == "" for cell in row):
                continue
            if max(id_idx, x_idx, y_idx, z_idx) >= len(row):
                raise ValueError(f"第 {row_idx} 行列数不足: {row}")

            try:
                raw_id = int(float(row[id_idx]))
                x = float(row[x_idx])
                y = float(row[y_idx])
                z = float(row[z_idx])
            except ValueError as exc:
                raise ValueError(f"第 {row_idx} 行存在无法转换的数值: {row}") from exc

            grouped.setdefault(raw_id, []).append([x, y, z])

    results = []
    scale = 1.0 / pixel_size
    for new_id, (_, points_xyz) in enumerate(grouped.items(), start=1):
        # 将物理坐标转换为像素坐标，单位统一为 pixel
        points_xyz_px = [[p[0] * scale, p[1] * scale, p[2] * scale] for p in points_xyz]

        length = 0.0
        for i in range(len(points_xyz_px) - 1):
            length += _distance(points_xyz_px[i], points_xyz_px[i + 1])

        points_zyx = [[p[2], p[1], p[0]] for p in points_xyz_px]

        results.append({
            "id": new_id,
            "points": points_zyx,
            "length": length,
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return len(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 actin 轨迹 CSV 转换为 JSON（id 从 1 开始重新编号）"
    )
    parser.add_argument("input_csv", type=Path, help="输入 CSV 路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出 JSON 路径，默认与输入同名 .json",
    )
    parser.add_argument(
        "-p",
        "--pixel-size",
        "--Pixel_size",
        type=float,
        default=17.14,
        help="像素尺寸（与 CSV 坐标同单位），输出 JSON 会转换为像素坐标，默认 1.0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = args.input_csv
    output_json = args.output or input_csv.with_suffix(".json")

    count = convert_actin_csv_to_json(input_csv, output_json, pixel_size=args.pixel_size)
    print(f"[OK] 已转换 {count} 条 actin: {output_json}")


if __name__ == "__main__":
    main()
