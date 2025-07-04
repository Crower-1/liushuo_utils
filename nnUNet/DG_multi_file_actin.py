import os
import numpy as np
import mrcfile
import json
from math import floor
import argparse
import math

def get_tomo(path):
    with mrcfile.open(path) as mrc:
        data = mrc.data.copy()
    return data

def save_tomo(data, path, voxel_size=17.14):
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(np.float32)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def crop_and_pad_volume(image, label, z_min, z_max, patch_size):
    cropped_image = image[z_min:z_max+1, :, :]
    cropped_label = label[z_min:z_max+1, :, :]

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
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def extract_patches(image, label, patch_size):
    z, y, x = image.shape
    patch_z, patch_y, patch_x = patch_size

    num_patches_z = math.ceil(z / patch_z)
    num_patches_y = math.ceil(y / patch_y)
    num_patches_x = math.ceil(x / patch_x)

    pad_z = (num_patches_z * patch_z) - z
    pad_y = (num_patches_y * patch_y) - y
    pad_x = (num_patches_x * patch_x) - x

    padding = (
        (0, pad_z),
        (0, pad_y),
        (0, pad_x),
    )
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    padded_label = np.pad(label, padding, mode='constant', constant_values=0)

    image_patches = []
    label_patches = []

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

                # Step 2: Set label values greater than 0 to 1
                label_patch[label_patch > 0] = 1
                label_patch[label_patch <= 0] = 0

                image_patches.append(image_patch)
                label_patches.append(label_patch)

    return image_patches, label_patches

def save_patches(image_patches, label_patches, output_dir):
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    num_patches = len(image_patches)
    patches_saved = 0
    for idx in range(num_patches):
        label_patch = label_patches[idx]
        # Step 3: Save patches where the label has more than 1000 pixels with value 1
        if np.sum(label_patch == 1) > 1000:
            patch_id = f"{idx:04d}"
            image_filename = f"Synapse_{patch_id}_0000.mrc"
            label_filename = f"Synapse_{patch_id}.mrc"

            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, label_filename)

            save_tomo(image_patches[idx], image_path)
            save_tomo(label_patches[idx], label_path)

            patches_saved += 1

    return patches_saved

def create_dataset_json(output_dir, num_training, file_ending=".mrc"):
    dataset_info = {
        "channel_names": {
            "0": "cryoET"
        },
        "labels": {
            "background": 0,
            "actin": 1
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
        base_name = os.path.basename(image_path).split('.')[0]
        # Step 1: Generate the label path
        label_path = os.path.join(os.path.dirname(image_path), 'actin', f"{base_name}_actin_label.mrc")

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
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64],
                        help="Size of each patch (z y x). Default is 128 128 128.")

    args = parser.parse_args()

    process_multiple_datasets(args.image_paths, args.output_dir, tuple(args.patch_size))

if __name__ == "__main__":
    main()
