import os
import numpy as np
import pandas as pd
import json

def euclidean_distance(A, B):
    """Calculate the Euclidean distance between two points A and B."""
    return np.linalg.norm(A - B)

# Parameters
distance_threshold = 50  # Threshold to distinguish different fibers

def default(obj):
    if isinstance(obj, np.float32):
        return round(float(obj),2)
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 转换 numpy 数组为列表
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def process_coords_to_json(coords_path):
    """Process a .coords file and save the result as a .json file.

    Args:
        coords_path (str): Path to the input .coords file.

    Returns:
        str: Path to the output .json file.
    """
    # Load coordinate data
    points = pd.read_csv(coords_path, header=None, sep='\s+', dtype=np.float32).values

    # Initialize data structure
    actins = []
    current_actin = {
        "id": 1,
        "points": [],
        "length": 0.0
    }

    # Iterate through all points
    for i in range(len(points) - 1):
        A = points[i]
        B = points[i + 1]
        distance = euclidean_distance(A, B)

        if distance < distance_threshold:
            # Current point pair belongs to the same fiber
            current_actin["points"].append([round(A[2], 2), round(A[1], 2), round(A[0], 2)])  # Convert to (z, y, x) with 2 decimal places
            current_actin["length"] += distance
        else:
            # Current point pair does not belong to the same fiber
            current_actin["points"].append([round(A[2], 2), round(A[1], 2), round(A[0], 2)])  # Convert to (z, y, x) with 2 decimal places
            actins.append(current_actin)

            current_actin = {
                "id": current_actin["id"] + 1,
                "points": [],
                "length": 0.0
            }

    # Handle the last point
    if current_actin["points"]:
        current_actin["points"].append([round(points[-1][2], 2), round(points[-1][1], 2), round(points[-1][0], 2)])  # Convert to (z, y, x) with 2 decimal places
        actins.append(current_actin)

    # Determine output path
    output_dir = os.path.dirname(coords_path)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(coords_path))[0] + ".json")

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(actins, f, indent=4, default=default)

    print(f"Actins data saved to {output_path}")
    return output_path

# Example usage
coords_path = "/media/liushuo/新加卷/data/actin trace/actintomo/20200820/tomo/pp4001/actin.coords"
process_coords_to_json(coords_path)
