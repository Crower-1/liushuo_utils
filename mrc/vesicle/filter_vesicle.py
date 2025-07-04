import mrcfile
import numpy as np
import json
import os

def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data
    return data

def save_tomo(data, path, voxel_size=17.14):
    """
    Save a 3D numpy array as an MRC file.
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(np.int8)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def load_json(json_path):
    """
    Load vesicle information from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['vesicles']

def get_vesicle_id_from_center(tomo_data, center):
    """
    Get the vesicle ID from tomo_data using the center coordinates.

    Parameters:
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - center: list or tuple of three floats
        The (z, y, x) coordinates of the vesicle center.

    Returns:
    - vesicle_id: int
        The ID of the vesicle at the given center coordinates.
    """
    z, y, x = center
    # Round the coordinates to the nearest integer
    z = int(round(z))
    y = int(round(y))
    x = int(round(x))
    
    # Ensure the coordinates are within the bounds of tomo_data
    if (0 <= z < tomo_data.shape[0] and
        0 <= y < tomo_data.shape[1] and
        0 <= x < tomo_data.shape[2]):
        vesicle_id = tomo_data[z, y, x]
        return vesicle_id
    else:
        raise ValueError(f"Center coordinates {center} are out of bounds for tomo_data with shape {tomo_data.shape}")

def categorize_vesicles(vesicles, tomo_data, threshold=34):
    """
    Categorize vesicles into small and other based on average radius.

    Parameters:
    - vesicles: list of dict
        The list of vesicle information.
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - threshold: float
        The radius threshold to categorize vesicles.

    Returns:
    - small_vesicle_ids: list of int
        List of IDs for small vesicles.
    - other_vesicle_ids: list of int
        List of IDs for other vesicles.
    """
    small_vesicle_ids = []
    other_vesicle_ids = []
    
    for vesicle in vesicles:
        radii = vesicle['radii']
        avg_radius = np.mean(radii)
        
        center = vesicle['center']
        try:
            vesicle_id = get_vesicle_id_from_center(tomo_data, center)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        
        if avg_radius < threshold:
            small_vesicle_ids.append(vesicle_id)
        else:
            other_vesicle_ids.append(vesicle_id)
    
    return small_vesicle_ids, other_vesicle_ids

def create_mask(tomo_data, vesicle_ids):
    """
    Create a mask for the given vesicle IDs.

    Parameters:
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - vesicle_ids: list of int
        The list of vesicle IDs to include in the mask.

    Returns:
    - mask: ndarray
        A binary mask where voxels belonging to the specified vesicle IDs are 1, else 0.
    """
    mask = np.isin(tomo_data, vesicle_ids).astype(np.float32)
    return mask

def generate_vesicle_mrc(mask, output_path):
    """
    Save the vesicle mask as an MRC file.
    """
    save_tomo(mask, output_path)

def process_vesicle_data(mrc_path, json_path, small_vesicle_output, other_vesicle_output, threshold=34):
    """
    Process vesicle data and save small and other vesicle MRC files.

    Parameters:
    - mrc_path: str
        Path to the input MRC segmentation mask.
    - json_path: str
        Path to the JSON file containing vesicle information.
    - small_vesicle_output: str
        Path to save the small vesicles MRC file.
    - other_vesicle_output: str
        Path to save the other vesicles MRC file.
    - threshold: float
        Radius threshold to categorize vesicles.
    """
    
    # Ensure output directories exist
    small_vesicle_dir = os.path.dirname(small_vesicle_output)
    other_vesicle_dir = os.path.dirname(other_vesicle_output)
    
    if not os.path.exists(small_vesicle_dir):
        os.makedirs(small_vesicle_dir)
        print(f"Created directory: {small_vesicle_dir}")
    
    if not os.path.exists(other_vesicle_dir):
        os.makedirs(other_vesicle_dir)
        print(f"Created directory: {other_vesicle_dir}")
    # Load the MRC file
    print("Loading MRC file...")
    tomo_data = get_tomo(mrc_path)
    print("MRC file loaded.")

    # Load the vesicle information from JSON
    print("Loading JSON file...")
    vesicles = load_json(json_path)
    print("JSON file loaded.")

    # Categorize vesicles based on average radius
    print("Categorizing vesicles...")
    small_vesicle_ids, other_vesicle_ids = categorize_vesicles(vesicles, tomo_data, threshold)
    print(f"Found {len(small_vesicle_ids)} small vesicles and {len(other_vesicle_ids)} other vesicles.")

    # Create masks
    print("Creating masks for small vesicles...")
    small_vesicles_mask = create_mask(tomo_data, small_vesicle_ids)
    print("Creating masks for other vesicles...")
    other_vesicles_mask = create_mask(tomo_data, other_vesicle_ids)

    # Save the masks as MRC files
    print(f"Saving small vesicles to {small_vesicle_output}...")
    generate_vesicle_mrc(small_vesicles_mask, small_vesicle_output)
    print(f"Saving other vesicles to {other_vesicle_output}...")
    generate_vesicle_mrc(other_vesicles_mask, other_vesicle_output)

    print("Processing complete.")

if __name__ == "__main__":
    # Example usage
    mrc_path = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/pp0039_label_vesicle.mrc'
    json_path = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/pp0039_vesicle.json'
    small_vesicle_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/small_vesicles.mrc'
    other_vesicle_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/other_vesicles.mrc'

    threshold = 34/1.714/2
    process_vesicle_data(mrc_path, json_path, small_vesicle_output, other_vesicle_output, threshold)
