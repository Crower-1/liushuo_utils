import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label

def crop_image_and_label(image_path, label_path, output_image_dir, output_label_dir, crop_size=1024):
    # 使用tifffile读取图像和标签
    image = tiff.imread(image_path)
    label = tiff.imread(label_path)

    # 获取原始图像的尺寸
    height, width = image.shape

    # 按crop_size进行分块
    for i in range(0, width, crop_size):
        for j in range(0, height, crop_size):
            # 裁剪图像和标签
            image_crop = image[j:j + crop_size, i:i + crop_size]
            label_crop = label[j:j + crop_size, i:i + crop_size]

            # 生成文件名
            image_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_filename = os.path.splitext(os.path.basename(label_path))[0]
            
            # 保存裁剪后的图像和标签，使用tifffile保存
            tiff.imwrite(os.path.join(output_image_dir, f'{image_filename}_{i}_{j}.tif'), image_crop)
            tiff.imwrite(os.path.join(output_label_dir, f'{label_filename}_{i}_{j}.tif'), label_crop)

def process_label(label_path, output_path):
    # 使用tifffile读取label并转换为numpy数组
    label_np = tiff.imread(label_path)

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

    # 保存处理后的label，使用tifffile保存
    tiff.imwrite(output_path, processed_label.astype(np.uint16))

def process_folder(input_dir, output_image_dir, output_label_dir, crop_size=1024):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".tif") and not '-spine' in filename:
            # 获取图像和对应的label文件
            image_path = os.path.join(input_dir, filename)
            label_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}-spine.tif")

            if os.path.exists(label_path):
                # 剪裁图像和标签
                crop_image_and_label(image_path, label_path, output_image_dir, output_label_dir, crop_size)

                # 处理标签的连通区域
                for i in range(0, 3072, crop_size):
                    for j in range(0, 3072, crop_size):
                        cropped_label_path = os.path.join(output_label_dir, f"{os.path.splitext(filename)[0]}-spine_{i}_{j}.tif")
                        processed_label_path = os.path.join(output_label_dir, f"{os.path.splitext(filename)[0]}-spine_{i}_{j}_processed.tif")
                        process_label(cropped_label_path, processed_label_path)

# 示例调用
input_dir = "/media/liushuo/新加卷2/data/lz"  # 包含原始图像和标签的文件夹路径
output_image_dir = "/media/liushuo/新加卷2/data/lz/images"  # 裁剪后的图像输出路径
output_label_dir = "/media/liushuo/新加卷2/data/lz/labels"  # 裁剪并处理后的标签输出路径

process_folder(input_dir, output_image_dir, output_label_dir, crop_size=1024)
