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
        data = data.astype(np.float32)
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
            "membrane": 5,
            "er": 1,
            "er_memb": 6,
            "mito": 2,
            "mito_memb": 7,
            "mt": 3,
            "mt_memb": 8,
            "vesicle": 4,
            "vesicle_memb": 9,
            "actin": 10,
            "ribo": 11
        },
        "numTraining": num_training,
        "file_ending": file_ending
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, 'w') as json_file:
        json.dump(dataset_info, json_file, indent=4)

def process_multiple_datasets(image_paths, output_dir, patch_size):
    """
    对于给定的每个 image_path，生成对应 label_path（替换文件名后缀），
    加载图像与标签，归一化、提取 patch，并保存到 output_dir 中。

    参数:
    - image_paths: list of str
        图像路径列表。
    - output_dir: str
        输出数据集保存目录。
    - patch_size: tuple
        每个 patch 的尺寸 (z, y, x)。
    """
    all_image_patches = []
    all_label_patches = []
    
    for image_path in image_paths:
        # 根据文件名生成 label 路径（替换 .mrc 为 _merged_label.mrc）
        # 构造输出目录：在 tomo_path 同目录下新建 ribo 文件夹
        input_dir = os.path.dirname(image_path)
        ribo_dir  = os.path.join(input_dir, "ribo")
        os.makedirs(ribo_dir, exist_ok=True)

        # 构造输出文件名：将原名 xx_wbp_corrected.mrc 改为 xx_ribo_volumn.mrc
        base  = os.path.basename(image_path)                   # e.g. "pp267_wbp_corrected.mrc"
        stem  = os.path.splitext(base)[0]                    # e.g. "pp267_wbp_corrected"
        suffix = "_wbp_corrected"
        if stem.endswith(suffix):
            prefix = stem[:-len(suffix)]
        else:
            prefix = stem
            
        out_name = f"{prefix}_ribo_semantic_label.mrc"                # e.g. "pp267_ribo_volumn.mrc"
        
        label_path = os.path.join(ribo_dir, out_name)
        if not os.path.exists(label_path):
            print(f"Label 文件 {label_path} 不存在，跳过 {image_path}。")
            continue

        image = get_tomo(image_path)
        label = get_tomo(label_path)
        image = normalize_image(image)
        image_patches, label_patches = extract_patches(image, label, patch_size)
        all_image_patches.extend(image_patches)
        all_label_patches.extend(label_patches)

    if not all_image_patches:
        raise ValueError("没有提取到任何 patch，请检查体数据尺寸与 patch_size。")

    num_patches = save_patches(all_image_patches, all_label_patches, output_dir)
    create_dataset_json(output_dir, num_training=num_patches)

    print(f"数据集创建完成，共保存 {num_patches} 个 patch。")

def generate_image_paths(base_path, base_names):
    """
    根据 base_path 和 base_names 列表生成 image_paths 列表，
    每个 image_path 的格式为:
        {base_path}/{base_name}/{base_name}_merged_label.mrc

    参数:
    - base_path: str
        数据的基础路径。
    - base_names: list of str
        样本名称列表。

    返回:
    - image_paths: list of str
    """
    image_paths = [f"{base_path}/{base_name}/synapse_seg/{base_name}_wbp_corrected.mrc" for base_name in base_names]
    return image_paths

# def main():
#     parser = argparse.ArgumentParser(description="从多个 image_path 生成数据集。")
#     # 修改输入参数：不再直接传入 image_paths，而是传入 base_path 与 base_names
#     parser.add_argument('--base_path', type=str, required=True,
#                         help="基础路径，例如：'/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo'")
#     parser.add_argument('--base_names', type=str, nargs='+', required=True,
#                         help="样本名称列表，例如：'pp3266 pp3267'")
#     parser.add_argument('--output_dir', type=str, required=True,
#                         help="输出数据集保存的目录。")
#     parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 512, 512],
#                         help="每个 patch 的尺寸 (z y x)。默认值为 128 512 512。")

#     args = parser.parse_args()

#     image_paths = generate_image_paths(args.base_path, args.base_names)
#     process_multiple_datasets(image_paths, args.output_dir, tuple(args.patch_size))
    
def main():
    base_path = f'/media/liushuo/data1/data/synapse_seg'
    base_names = ['pp267', 'pp1776' ,'pp1033', 'pp4001', 'pp366', 'pp387']
    output_dir = '/home/liushuo/Documents/data/nnUNet/nnUNet_raw/Dataset010_6tomo_11classes'
    patch_size = tuple([128, 512, 512])

    image_paths = generate_image_paths(base_path, base_names)
    process_multiple_datasets(image_paths, output_dir, patch_size)

if __name__ == "__main__":
    main()