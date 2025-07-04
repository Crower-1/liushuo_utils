import argparse
import json
import mrcfile
import os
import sys
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Match vesicles in the JSON file with masks in the MRC file based on center coordinates.")
    parser.add_argument("mrc_path", type=str, help="Path to the MRC file with masks")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing vesicle information")
    return parser.parse_args()

def load_mrc(mrc_path):
    if not os.path.isfile(mrc_path):
        print(f"Error: MRC file '{mrc_path}' does not exist.")
        sys.exit(1)
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.copy()
        return data
    except Exception as e:
        print(f"Error: Unable to read MRC file '{mrc_path}'. Details: {e}")
        sys.exit(1)

def load_json(json_path):
    if not os.path.isfile(json_path):
        print(f"Error: JSON file '{json_path}' does not exist.")
        sys.exit(1)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error: JSON file '{json_path}' is improperly formatted. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unable to read JSON file '{json_path}'. Details: {e}")
        sys.exit(1)

def extract_id(vesicle_name):
    try:
        # Assuming the format is 'vesicle_<ID>'
        return int(vesicle_name.split('_')[-1])
    except (IndexError, ValueError):
        print(f"Warning: Could not extract ID from name '{vesicle_name}'.")
        return None

def check_matching_center(mrc_data, vesicles):
    # Use numpy.unique to find unique mask IDs, excluding 0 (assumed to be background)
    unique_mask_ids = np.unique(mrc_data)
    unique_mask_ids = unique_mask_ids[unique_mask_ids != 0]  # Exclude background

    matched_mask_ids = set()
    unmatched_json_ids = []

    z_dim, y_dim, x_dim = mrc_data.shape

    for vesicle in vesicles:
        name = vesicle.get("name", "")
        vesicle_id = extract_id(name)
        if vesicle_id is None:
            continue

        center = vesicle.get("center", [])
        if len(center) != 3:
            print(f"Warning: Vesicle '{name}' has incomplete center coordinates.")
            unmatched_json_ids.append(vesicle_id)
            continue

        # Convert floating-point coordinates to integer indices
        z, y, x = [int(round(coord)) for coord in center]

        # Check if the indices are within bounds
        if not (0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim):
            print(f"Warning: Vesicle '{name}' has center coordinates ({z}, {y}, {x}) out of MRC data bounds.")
            unmatched_json_ids.append(vesicle_id)
            continue

        voxel_value = mrc_data[z, y, x]
        if voxel_value != 0:
            matched_mask_ids.add(voxel_value)
        else:
            unmatched_json_ids.append(vesicle_id)

    # Find unmatched mask IDs
    unmatched_mask_ids = set(unique_mask_ids) - matched_mask_ids

    return unmatched_mask_ids, unmatched_json_ids

def main():
    args = parse_arguments()
    mrc_data = load_mrc(args.mrc_path)
    json_data = load_json(args.json_path)

    vesicles = json_data.get("vesicles", [])
    if not vesicles:
        print("Warning: No 'vesicles' information found in the JSON file.")
        sys.exit(1)

    unmatched_mask_ids, unmatched_json_ids = check_matching_center(mrc_data, vesicles)

    # Using numpy.unique to calculate the total number of non-background masks
    total_masks = len(np.unique(mrc_data)) - 1  # Exclude background (0)
    total_vesicles = len(vesicles)

    unmatched_masks_count = len(unmatched_mask_ids)
    unmatched_vesicles_count = len(unmatched_json_ids)

    unmatched_masks_percent = (unmatched_masks_count / total_masks) * 100 if total_masks > 0 else 0
    unmatched_vesicles_percent = (unmatched_vesicles_count / total_vesicles) * 100 if total_vesicles > 0 else 0

    print("Unmatched mask ID statistics:")
    print(f"Number of unmatched mask IDs: {unmatched_masks_count}")
    print(f"Percentage of unmatched mask IDs: {unmatched_masks_percent:.2f}%")
    print(f"List of unmatched mask IDs: {sorted(unmatched_mask_ids)}\n")

    print("Unmatched JSON vesicle ID statistics:")
    print(f"Number of unmatched JSON vesicle IDs: {unmatched_vesicles_count}")
    print(f"Percentage of unmatched JSON vesicle IDs: {unmatched_vesicles_percent:.2f}%")
    print(f"List of unmatched JSON vesicle IDs: {sorted(unmatched_json_ids)}")

if __name__ == "__main__":
    main()
