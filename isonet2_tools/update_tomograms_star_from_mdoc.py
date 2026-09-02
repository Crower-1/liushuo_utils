#!/usr/bin/env python3
import os
import re
import glob
import shutil


def extract_tomo_base_from_mdoc(mdoc_path):
    """
    Example:
        p36.mrc.mdoc -> p36
        p36.mdoc     -> p36
    """
    name = os.path.basename(mdoc_path)

    if name.endswith(".mrc.mdoc"):
        return name.replace(".mrc.mdoc", "")
    elif name.endswith(".mdoc"):
        return name.replace(".mdoc", "")
    else:
        return os.path.splitext(name)[0]


def extract_tomo_base_from_star_path(path):
    """
    Example:
        ./EVN//p36_EVN.mrc -> p36
        ./ODD//p36_ODD.mrc -> p36
    """
    name = os.path.basename(path)

    name = name.replace("_EVN.mrc", "")
    name = name.replace("_ODD.mrc", "")
    name = name.replace(".mrc", "")

    return name


def parse_mdoc_file(mdoc_path):
    records = []

    current_tilt = None
    current_defocus = None

    with open(mdoc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("[ZValue"):
                if current_tilt is not None:
                    records.append({
                        "TiltAngle": current_tilt,
                        "Defocus": current_defocus,
                    })

                current_tilt = None
                current_defocus = None

            elif line.startswith("TiltAngle"):
                match = re.search(r"TiltAngle\s*=\s*([-+]?\d+\.?\d*)", line)
                if match:
                    current_tilt = float(match.group(1))

            elif line.startswith("Defocus"):
                match = re.search(r"Defocus\s*=\s*([-+]?\d+\.?\d*)", line)
                if match:
                    current_defocus = float(match.group(1))

        # Save the final ZValue block
        if current_tilt is not None:
            records.append({
                "TiltAngle": current_tilt,
                "Defocus": current_defocus,
            })

    if not records:
        return None

    min_record = min(records, key=lambda x: x["TiltAngle"])
    max_record = max(records, key=lambda x: x["TiltAngle"])
    zero_record = min(records, key=lambda x: abs(x["TiltAngle"]))

    if zero_record["Defocus"] is None:
        print(f"[Warning] Defocus not found near zero tilt in {mdoc_path}")
        corrected_defocus = None
    else:
        corrected_defocus = zero_record["Defocus"] * -10000

    return {
        "mdoc_path": mdoc_path,
        "tilt_min": min_record["TiltAngle"],
        "tilt_max": max_record["TiltAngle"],
        "zero_tilt": zero_record["TiltAngle"],
        "defocus_raw": zero_record["Defocus"],
        "defocus_star": corrected_defocus,
    }


def collect_mdoc_info():
    mdoc_files = glob.glob("./**/*.mdoc", recursive=True)

    mdoc_info_dict = {}

    for mdoc_path in mdoc_files:
        tomo_base = extract_tomo_base_from_mdoc(mdoc_path)
        info = parse_mdoc_file(mdoc_path)

        if info is None:
            print(f"[Warning] No TiltAngle found in {mdoc_path}")
            continue

        mdoc_info_dict[tomo_base] = info

    return mdoc_info_dict


def parse_star_header(lines):
    """
    Find STAR column names and their column indices.
    Returns:
        header_start_index
        data_start_index
        column_map
    """
    column_map = {}
    header_start_index = None
    data_start_index = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("_rln"):
            if header_start_index is None:
                header_start_index = i

            parts = stripped.split()
            column_name = parts[0]

            if len(parts) >= 2 and parts[1].startswith("#"):
                column_index = int(parts[1].replace("#", "")) - 1
                column_map[column_name] = column_index

        elif header_start_index is not None and stripped and not stripped.startswith("_rln"):
            data_start_index = i
            break

    return header_start_index, data_start_index, column_map


def update_tomograms_star(star_path, output_path=None):
    if output_path is None:
        output_path = star_path

    mdoc_info_dict = collect_mdoc_info()

    if not mdoc_info_dict:
        print("[Error] No valid .mdoc information found.")
        return

    print(f"Found {len(mdoc_info_dict)} valid .mdoc files.")

    with open(star_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    _, data_start_index, column_map = parse_star_header(lines)

    required_columns = [
        "_rlnTomoReconstructedTomogramHalf1",
        "_rlnDefocus",
        "_rlnTiltMin",
        "_rlnTiltMax",
    ]

    for col in required_columns:
        if col not in column_map:
            raise ValueError(f"Required column not found in STAR file: {col}")

    half1_idx = column_map["_rlnTomoReconstructedTomogramHalf1"]
    defocus_idx = column_map["_rlnDefocus"]
    tilt_min_idx = column_map["_rlnTiltMin"]
    tilt_max_idx = column_map["_rlnTiltMax"]

    updated_count = 0
    new_lines = []

    for i, line in enumerate(lines):
        if i < data_start_index:
            new_lines.append(line)
            continue

        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        parts = stripped.split()

        if len(parts) <= max(half1_idx, defocus_idx, tilt_min_idx, tilt_max_idx):
            new_lines.append(line)
            continue

        half1_path = parts[half1_idx]
        tomo_base = extract_tomo_base_from_star_path(half1_path)

        if tomo_base not in mdoc_info_dict:
            print(f"[Warning] No matched mdoc found for STAR row: {half1_path}")
            new_lines.append(line)
            continue

        info = mdoc_info_dict[tomo_base]

        if info["defocus_star"] is None:
            print(f"[Warning] Skip defocus update for {tomo_base}, because Defocus is missing.")
        else:
            parts[defocus_idx] = f"{info['defocus_star']:.6f}"

        parts[tilt_min_idx] = f"{info['tilt_min']:.6f}"
        parts[tilt_max_idx] = f"{info['tilt_max']:.6f}"

        new_lines.append("\t".join(parts) + "\n")

        updated_count += 1

        print(
            f"[Updated] {tomo_base}: "
            f"TiltMin={info['tilt_min']:.6f}, "
            f"TiltMax={info['tilt_max']:.6f}, "
            f"ZeroTilt={info['zero_tilt']:.6f}, "
            f"DefocusRaw={info['defocus_raw']}, "
            f"DefocusSTAR={info['defocus_star']}"
        )

    # Backup original STAR file if overwriting
    if output_path == star_path:
        backup_path = star_path + ".bak"
        shutil.copy2(star_path, backup_path)
        print(f"\nBackup saved to: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\nUpdated STAR file saved to: {output_path}")
    print(f"Updated rows: {updated_count}")


def main():
    star_path = "tomograms.star"

    if not os.path.exists(star_path):
        print(f"[Error] STAR file not found: {star_path}")
        return

    update_tomograms_star(star_path)


if __name__ == "__main__":
    main()