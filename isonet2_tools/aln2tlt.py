#!/usr/bin/env python3
"""
Convert AreTomo .aln file to .tlt file
Extract only the TILT column from the global alignment table
"""

import sys
from pathlib import Path


def aln_to_tlt(aln_path, tlt_path):
    tilts = []
    in_global_table = False

    with open(aln_path, "r") as f:
        for line in f:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 遇到 Local Alignment，立即停止解析
            if line.startswith("# Local Alignment"):
                break

            # 跳过注释行
            if line.startswith("#"):
                continue

            parts = line.split()

            # SEC 行至少应包含 10 列（SEC ... TILT）
            if len(parts) < 10:
                continue

            # 第一列必须是整数（SEC）
            try:
                int(parts[0])
            except ValueError:
                continue

            # 最后一列是 TILT
            tilt = parts[-1]
            tilts.append(tilt)

    if not tilts:
        raise RuntimeError("No TILT values found in aln file.")

    with open(tlt_path, "w") as f:
        for t in tilts:
            f.write(f"{t}\n")

    print(f"[OK] Extracted {len(tilts)} tilt angles")
    print(f"[OK] Output written to: {tlt_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: aln_to_tlt.py input.aln output.tlt")
        sys.exit(1)

    aln_file = Path(sys.argv[1])
    tlt_file = Path(sys.argv[2])

    aln_to_tlt(aln_file, tlt_file)
