import os
import numpy as np
import mrcfile
import argparse
import math
import json
from glob import glob

def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.

    Parameters:
    - path: str
        Path to the MRC file.

    Returns:
    - data: ndarray
        The 3D data loaded from the MRC file.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data.copy()
    return data

def save_tomo(data, path, voxel_size=17.14):
    """
    Save a 3D numpy array as an MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - path: str
        Path where the MRC file will be saved.
    - voxel_size: float
        The voxel size of the data.
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(np.float32)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def pad_image(image, patch_size=(128, 128, 128)):
    """
    Pad the image so that its dimensions are divisible by the patch size.

    Parameters:
    - image: ndarray
        The image array with shape (z, y, x).
    - patch_size: tuple
        The size of each patch (z, y, x).

    Returns:
    - padded_image: ndarray
        The padded image array.
    - pad_width: tuple of tuples
        The padding applied to each dimension.
    """
    pad_width = []
    for dim, size in zip(image.shape, patch_size):
        pad = (0, (size - dim % size) if dim % size != 0 else 0)
        pad_width.append(pad)
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    return padded_image, pad_width

def extract_patches(image, patch_size=(128, 128, 128)):
    """
    Extract patches from the image volume, padding if necessary.

    Parameters:
    - image: ndarray
        The image array with shape (z, y, x).
    - patch_size: tuple
        The size of each patch (z, y, x).

    Returns:
    - patches: list of ndarray
        List of image patches.
    - patch_positions: list of tuple
        List of patch starting indices (z_start, y_start, x_start).
    """
    z, y, x = image.shape
    patch_z, patch_y, patch_x = patch_size

    num_patches_z = math.ceil(z / patch_z)
    num_patches_y = math.ceil(y / patch_y)
    num_patches_x = math.ceil(x / patch_x)

    patches = []
    patch_positions = []

    for i in range(num_patches_z):
        for j in range(num_patches_y):
            for k in range(num_patches_x):
                z_start = i * patch_z
                y_start = j * patch_y
                x_start = k * patch_x

                patch = image[z_start:z_start+patch_z,
                             y_start:y_start+patch_y,
                             x_start:x_start+patch_x]
                
                # If the patch is smaller than patch_size, pad it
                pad_z = patch_z - patch.shape[0]
                pad_y = patch_y - patch.shape[1]
                pad_x = patch_x - patch.shape[2]

                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    patch = np.pad(patch, 
                                  ((0, pad_z), (0, pad_y), (0, pad_x)), 
                                  mode='constant', constant_values=0)
                
                patches.append(patch)
                patch_positions.append((z_start, y_start, x_start))

    return patches, patch_positions

def save_patches(patches, patch_positions, output_dir, image_name):
    """
    Save image patches to the specified directory with proper naming.

    Parameters:
    - patches: list of ndarray
        List of image patches.
    - patch_positions: list of tuple
        List of patch starting indices (z_start, y_start, x_start).
    - output_dir: str
        The root directory where the patches will be saved.
    - image_name: str
        The base name of the original image file (without extension).

    Returns:
    - metadata: dict
        Metadata containing patch information and original image size.
    """
    image_folder = os.path.join(output_dir, image_name)
    os.makedirs(image_folder, exist_ok=True)

    metadata = {
        "original_image_shape": [],
        "patch_size": [128, 128, 128],
        "patches": []
    }

    for idx, (patch, pos) in enumerate(zip(patches, patch_positions)):
        patch_id = f"{idx:04d}"
        image_filename = f"Synapse_{patch_id}_0000.mrc"
        patch_path = os.path.join(image_folder, image_filename)
        save_tomo(patch, patch_path)

        # Record metadata
        metadata["patches"].append({
            "id": patch_id,
            "filename": image_filename,
            "position": pos
        })

    return metadata

def save_metadata(metadata, output_dir, image_name):
    """
    Save the metadata to a JSON file.

    Parameters:
    - metadata: dict
        Metadata to save.
    - output_dir: str
        The root directory where the dataset is saved.
    - image_name: str
        The base name of the original image file (without extension).

    Returns:
    - None
    """
    metadata["original_image_shape"] = list(metadata["patches"][0].get("original_shape", []))
    json_path = os.path.join(output_dir, image_name, "metadata.json")
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

def load_metadata(metadata_path):
    """
    Load metadata from a JSON file.

    Parameters:
    - metadata_path: str
        Path to the metadata JSON file.

    Returns:
    - metadata: dict
        Loaded metadata.
    """
    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

def stitch_patches(input_dir, output_path, patch_size=(128, 128, 128)):
    """
    Stitch patches back into the original image using metadata.

    Parameters:
    - input_dir: str
        Directory containing the patches and metadata.json.
    - output_path: str
        Path to save the reconstructed MRC file.
    - patch_size: tuple
        The size of each patch (z, y, x).

    Returns:
    - None
    """
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = load_metadata(metadata_path)
    patches_info = metadata["patches"]
    patch_size = tuple(metadata.get("patch_size", patch_size))

    # Determine the size of the reconstructed image
    max_z = max(pos[0] for pos in [p["position"] for p in patches_info]) + patch_size[0]
    max_y = max(pos[1] for pos in [p["position"] for p in patches_info]) + patch_size[1]
    max_x = max(pos[2] for pos in [p["position"] for p in patches_info]) + patch_size[2]

    reconstructed = np.zeros((max_z, max_y, max_x), dtype=np.float32)

    for patch_info in patches_info:
        patch_path = os.path.join(input_dir, patch_info["filename"])
        patch = get_tomo(patch_path)
        z_start, y_start, x_start = patch_info["position"]
        reconstructed[z_start:z_start+patch_size[0],
                      y_start:y_start+patch_size[1],
                      x_start:x_start+patch_size[2]] = patch

    # Remove padding if original image size is recorded
    if "original_image_shape" in metadata and metadata["original_image_shape"]:
        original_z, original_y, original_x = metadata["original_image_shape"]
        reconstructed = reconstructed[:original_z, :original_y, :original_x]

    save_tomo(reconstructed, output_path)
    print(f"Reconstructed image saved to {output_path}")

def process_split(image_path, output_dir):
    """
    Process the image by splitting it into patches with padding.

    Parameters:
    - image_path: str
        Path to the image MRC file.
    - output_dir: str
        Directory to save the patches.

    Returns:
    - None
    """
    image = get_tomo(image_path)
    padded_image, pad_width = pad_image(image)
    patches, patch_positions = extract_patches(padded_image)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    metadata = save_patches(patches, patch_positions, output_dir, image_name)

    # Record original image size
    metadata["original_image_shape"] = list(image.shape)

    # Save metadata
    save_metadata(metadata, output_dir, image_name)
    print(f"Image split into {len(patches)} patches and saved to {os.path.join(output_dir, image_name)}")
    print(f"Metadata saved to {os.path.join(output_dir, image_name, 'metadata.json')}")

def process_stitch(input_dir, output_path):
    """
    Process the patches by stitching them back into the original image.

    Parameters:
    - input_dir: str
        Directory containing the patches and metadata.json.
    - output_path: str
        Path to save the reconstructed MRC file.

    Returns:
    - None
    """
    stitch_patches(input_dir, output_path)
    print("Stitching completed.")

def main():
    parser = argparse.ArgumentParser(description="Split and stitch 3D MRC images into/from patches with padding.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # Split command
    split_parser = subparsers.add_parser('split', help='Split a 3D MRC image into patches with padding.')
    split_parser.add_argument("image_path", type=str, help="Path to the image MRC file.")
    split_parser.add_argument("output_dir", type=str, help="Directory to save the patches.")

    # Stitch command
    stitch_parser = subparsers.add_parser('stitch', help='Stitch patches back into a 3D MRC image.')
    stitch_parser.add_argument("input_dir", type=str, help="Directory containing the patches and metadata.json.")
    stitch_parser.add_argument("output_path", type=str, help="Path to save the reconstructed MRC file.")

    args = parser.parse_args()

    if args.command == 'split':
        process_split(args.image_path, args.output_dir)
    elif args.command == 'stitch':
        process_stitch(args.input_dir, args.output_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
