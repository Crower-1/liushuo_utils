import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label

def process_label_data(label_np):
    # 对不同的值分别进行连通性分析
    processed_label = np.zeros_like(label_np)
    unique_labels = np.unique(label_np)

    instance_id = 1  # 实例ID从1开始
    for lbl in unique_labels:
        if lbl == 0:
            continue  # 跳过背景

        # 对当前label的连通区域进行标记
        mask = (label_np == lbl)
        labeled_array, num_features = label(mask)

        # 每个连通区域赋予不同的实例值
        processed_label[labeled_array > 0] = labeled_array[labeled_array > 0] + instance_id - 1
        instance_id += num_features

    return processed_label.astype(np.uint16)

def crop_image_and_label(image_path, label_path, output_image_dir, output_label_dir, crop_size=1024):
    # 使用tifffile读取图像和标签
    image = tiff.imread(image_path)
    label_img = tiff.imread(label_path)

    # 获取原始图像的尺寸
    height, width = image.shape

    # 获取文件名并去除-spine或_mask后缀
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = os.path.splitext(os.path.basename(label_path))[0]

    if label_filename.endswith('-spine'):
        label_filename = label_filename[:-6]
    elif label_filename.endswith('_mask'):
        label_filename = label_filename[:-5]

    # 图像和标签尺寸大于crop_size的情况
    if height > crop_size or width > crop_size:
        # 按crop_size进行分块
        for i in range(0, width, crop_size):
            for j in range(0, height, crop_size):
                # 裁剪图像和标签
                image_crop = image[j:j + crop_size, i:i + crop_size]
                label_crop = label_img[j:j + crop_size, i:i + crop_size]

                # 处理标签
                processed_label = process_label_data(label_crop)

                # 保存裁剪后的图像和处理后的标签
                image_crop_filename = f'{image_filename}_{i}_{j}.tif'
                label_crop_filename = f'{label_filename}_{i}_{j}.tif'

                tiff.imwrite(os.path.join(output_image_dir, image_crop_filename), image_crop)
                tiff.imwrite(os.path.join(output_label_dir, label_crop_filename), processed_label)
    else:
        # 计算需要填充的尺寸
        pad_height = crop_size - height
        pad_width = crop_size - width

        # 填充图像和标签
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        padded_label = np.pad(label_img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

        # 处理标签
        processed_label = process_label_data(padded_label)

        # 保存填充后的图像和处理后的标签
        image_padded_filename = f'{image_filename}.tif'
        label_padded_filename = f'{label_filename}.tif'

        tiff.imwrite(os.path.join(output_image_dir, image_padded_filename), padded_image)
        tiff.imwrite(os.path.join(output_label_dir, label_padded_filename), processed_label)

def process_folder(input_dir, output_image_dir, output_label_dir, crop_size=1024):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif"):
            filepath = os.path.join(input_dir, filename)
            # 跳过标签文件
            if filename.endswith('-spine.tif') or filename.endswith('_mask.tif'):
                continue

            # 图像文件路径
            image_path = filepath

            # 尝试找到对应的标签文件
            base_name = os.path.splitext(filename)[0]
            label_path_spine = os.path.join(input_dir, f"{base_name}-spine.tif")
            label_path_mask = os.path.join(input_dir, f"{base_name}_mask.tif")

            if os.path.exists(label_path_spine):
                label_path = label_path_spine
            elif os.path.exists(label_path_mask):
                label_path = label_path_mask
            else:
                print(f"找不到对应的标签文件：{filename}")
                continue

            # 剪裁或填充图像和标签并处理标签
            crop_image_and_label(image_path, label_path, output_image_dir, output_label_dir, crop_size)

# 示例调用
input_dir = "/media/liushuo/新加卷2/data/lz/small"  # 包含原始图像和标签的文件夹路径
output_image_dir = "/media/liushuo/新加卷2/data/lz/Val/images"  # 裁剪后的图像输出路径
output_label_dir = "/media/liushuo/新加卷2/data/lz/Val/labels"  # 裁剪并处理后的标签输出路径

process_folder(input_dir, output_image_dir, output_label_dir, crop_size=1024)
