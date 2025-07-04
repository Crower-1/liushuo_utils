import os
import numpy as np
import mrcfile
import json
from math import floor
import argparse
import math

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

def find_valid_z_slices(label):
    """
    Find the top and bottom z-indices where the label is not all zeros.

    Parameters:
    - label: ndarray
        The label array with shape (z, y, x).

    Returns:
    - z_min: int
        The first z-index with non-zero labels.
    - z_max: int
        The last z-index with non-zero labels.
    """
    z_indices = np.where(label.any(axis=(1, 2)))[0]
    if z_indices.size == 0:
        raise ValueError("Label file contains only zeros.")
    z_min, z_max = z_indices[0], z_indices[-1]
    return z_min, z_max

def crop_and_pad_volume(image, label, z_min, z_max, patch_size):
    """
    Crop the image and label volumes along the z-axis and pad if necessary.

    Parameters:
    - image: ndarray
        The image array with shape (z, y, x).
    - label: ndarray
        The label array with shape (z, y, x).
    - z_min: int
        The starting z-index for cropping.
    - z_max: int
        The ending z-index for cropping.
    - patch_size: tuple
        The size of each patch (z, y, x).

    Returns:
    - cropped_padded_image: ndarray
        The cropped and padded image array.
    - cropped_padded_label: ndarray
        The cropped and padded label array.
    """
    cropped_image = image[z_min:z_max+1, :, :]
    cropped_label = label[z_min:z_max+1, :, :]

    # Calculate padding needed for each dimension
    current_shape = cropped_image.shape
    pad_z = (patch_size[0] - current_shape[0] % patch_size[0]) % patch_size[0]
    pad_y = (patch_size[1] - current_shape[1] % patch_size[1]) % patch_size[1]
    pad_x = (patch_size[2] - current_shape[2] % patch_size[2]) % patch_size[2]

    padding = (
        (0, pad_z),
        (0, pad_y),
        (0, pad_x)
    )

    cropped_padded_image = np.pad(cropped_image, padding, mode='constant', constant_values=0)
    cropped_padded_label = np.pad(cropped_label, padding, mode='constant', constant_values=0)

    return cropped_padded_image, cropped_padded_label

def normalize_image(image):
    """
    Normalize the image data to the range [0, 1].

    Parameters:
    - image: ndarray
        The image array.

    Returns:
    - normalized_image: ndarray
        The normalized image array.
    """
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def extract_patches(image, label, patch_size):
    """
    Extract non-overlapping patches from the image and label volumes.

    Parameters:
    - image: ndarray
        The image array with shape (z, y, x).
    - label: ndarray
        The label array with shape (z, y, x).
    - patch_size: tuple
        The size of each patch (z, y, x).

    Returns:
    - image_patches: list of ndarray
        List of image patches.
    - label_patches: list of ndarray
        List of label patches.
    """
    z, y, x = image.shape
    patch_z, patch_y, patch_x = patch_size

    # Calculate number of patches (ceil to include all areas)
    num_patches_z = math.ceil(z / patch_z)
    num_patches_y = math.ceil(y / patch_y)
    num_patches_x = math.ceil(x / patch_x)

    # Calculate required padding
    pad_z = (num_patches_z * patch_z) - z
    pad_y = (num_patches_y * patch_y) - y
    pad_x = (num_patches_x * patch_x) - x

    # Pad image and label
    padding = (
        (0, pad_z),  # Z-axis padding
        (0, pad_y),  # Y-axis padding
        (0, pad_x),  # X-axis padding
    )
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    padded_label = np.pad(label, padding, mode='constant', constant_values=0)

    image_patches = []
    label_patches = []

    # Extract patches
    for i in range(num_patches_z):
        for j in range(num_patches_y):
            for k in range(num_patches_x):
                z_start = i * patch_z
                y_start = j * patch_y
                x_start = k * patch_x

                image_patch = padded_image[z_start:z_start+patch_z,
                                           y_start:y_start+patch_y,
                                           x_start:x_start+patch_x]
                label_patch = padded_label[z_start:z_start+patch_z,
                                           y_start:y_start+patch_y,
                                           x_start:x_start+patch_x]

                image_patches.append(image_patch)
                label_patches.append(label_patch)

    return image_patches, label_patches

def save_patches(image_patches, label_patches, output_dir):
    """
    Save image and label patches to the specified directories.

    Parameters:
    - image_patches: list of ndarray
        List of image patches.
    - label_patches: list of ndarray
        List of label patches.
    - output_dir: str
        The root directory where the dataset will be saved.

    Returns:
    - num_patches: int
        The number of patches saved.
    """
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    num_patches = len(image_patches)
    for idx in range(num_patches):
        patch_id = f"{idx:04d}"
        image_filename = f"Synapse_{patch_id}_0000.mrc"
        label_filename = f"Synapse_{patch_id}.mrc"

        image_path = os.path.join(images_dir, image_filename)
        label_path = os.path.join(labels_dir, label_filename)

        save_tomo(image_patches[idx], image_path)
        save_tomo(label_patches[idx], label_path)

    return num_patches

def create_dataset_json(output_dir, num_training, file_ending=".mrc"):
    """
    Create the dataset.json file in the output directory.

    Parameters:
    - output_dir: str
        The root directory where the dataset is saved.
    - num_training: int
        The number of training samples.
    - file_ending: str
        The file extension for the dataset files.
    """
    dataset_info = {
        "channel_names": {
            "0": "cryoET"
        },
        "labels": {
            "background": 0,
            "membrane": 1,
            "ER": 2,
            "mitochondria": 3,
            "MT": 4,
            "vesicle": 5,
            "actin": 6
        },
        "numTraining": num_training,
        "file_ending": file_ending
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as json_file:
        json.dump(dataset_info, json_file, indent=4)

def process_multiple_datasets(image_paths, output_dir, patch_size):
    all_image_patches = []
    all_label_patches = []
    
    for image_path in image_paths:
        # Generate corresponding label path by replacing file name
        label_path = image_path.replace(".mrc", "_merged_label.mrc")

        if not os.path.exists(label_path):
            print(f"Label file for {image_path} not found. Skipping this pair.")
            continue

        image = get_tomo(image_path)
        label = get_tomo(label_path)
        image = normalize_image(image)
        image_patches, label_patches = extract_patches(image, label, patch_size)
        all_image_patches.extend(image_patches)
        all_label_patches.extend(label_patches)

    if not all_image_patches:
        raise ValueError("No patches were extracted. Check the volume size and patch size.")

    num_patches = save_patches(all_image_patches, all_label_patches, output_dir)
    create_dataset_json(output_dir, num_training=num_patches)

    print(f"Dataset creation complete. {num_patches} patches saved.")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset from multiple image paths.")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True, help="List of image paths.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output dataset.")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 512, 512],
                        help="Size of each patch (z y x). Default is 128 256 256.")

    args = parser.parse_args()

    process_multiple_datasets(args.image_paths, args.output_dir, tuple(args.patch_size))

if __name__ == "__main__":
    main()
