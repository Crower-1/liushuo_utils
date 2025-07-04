#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据转换脚本

功能：
1. 输入为两个文件夹，分别为image文件夹和label文件夹，遍历其下文件。image文件被命名为xxx.tif，label文件被命名为xxx_actin_result.tif。
2. 如果图像和label为2D，则将其视为(1, y, x)，与3D图像进行同样的处理。
3. 读入原始图像后，去除最大的99.5%的值。
4. 将图像和label剪裁为1024*1024*z的patch，剪裁步长为512，并储存为tif格式，图像储存在./tif_crop/images/，
   label储存在./tif_crop/labels/，图像和label名字保持一致（使用原始图像名加patch编号）。
5. 将剪裁后的图像沿z轴进行切片，切片后的图像储存在./train/JPEGImage/{crop_image_name}/，
   储存格式为jpg。label储存在./train/Annotations/{crop_image_name}/，储存格式为png。
6. jpg和png命名为00000.jpg/00000.png，数字代表z轴的值。

使用方法：
    python data_convert.py --image_dir /path/to/image_folder --label_dir /path/to/label_folder

依赖库：
    - numpy
    - tifffile
    - Pillow
    - tqdm
"""

import os
import argparse
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='将TIFF图像和mask批量剪裁并切片保存')
    parser.add_argument('--image_dir', type=str, required=True, help='输入的图像文件夹路径')
    parser.add_argument('--label_dir', type=str, required=True, help='对应的label文件夹路径')
    args = parser.parse_args()
    return args

def create_directories():
    # 创建剪裁后存储的目录
    os.makedirs('./tif_crop/images/', exist_ok=True)
    os.makedirs('./tif_crop/labels/', exist_ok=True)
    # 创建切片存储的目录
    os.makedirs('./train/JPEGImage/', exist_ok=True)
    os.makedirs('./train/Annotations/', exist_ok=True)

def get_file_pairs(image_dir, label_dir):
    """
    获取image和label的文件对，假设image文件名为xxx.tif，label文件名为xxx_actin_result.tif
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
    file_pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}_actin_result.tif"
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, img_file)
        if os.path.isfile(label_path):
            file_pairs.append((image_path, label_path, base_name))
        else:
            print(f"警告: 找不到对应的label文件 {label_file}，跳过 {img_file}")
    return file_pairs

def remove_top_percentile(image, percentile=99.5):
    """
    去除图像中最大的percentile%的值，保留最低(100 - percentile)%的值。
    即设置大于阈值的像素为阈值。
    """
    threshold = np.percentile(image, percentile)
    image_clipped = np.clip(image, None, threshold)
    return image_clipped

def crop_patches(image, mask, patch_size=(1024, 1024), step=512):
    """
    对3D图像和mask进行剪裁

    :param image: numpy array, shape (z, y, x)
    :param mask: numpy array, shape (z, y, x)
    :param patch_size: tuple, (y_size, x_size)
    :param step: int, 步长
    :return: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    """
    patches = []
    z_max, y_max, x_max = image.shape

    patch_id = 0
    for z in tqdm(range(0, z_max, 1), desc='Cropping patches (z-axis)'):
        for y in range(0, y_max - patch_size[0] + 1, step):
            for x in range(0, x_max - patch_size[1] + 1, step):
                # 使用切片 z:z+1 保持z维度
                image_patch = image[z:z+1, y:y + patch_size[0], x:x + patch_size[1]]
                mask_patch = mask[z:z+1, y:y + patch_size[0], x:x + patch_size[1]]
                patch_name = f'patch_{patch_id:06d}'
                patches.append((patch_name, image_patch, mask_patch))
                patch_id += 1

    return patches

def save_tif_patches(patches, base_name):
    """
    保存剪裁后的图像和mask为tif文件

    :param patches: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    :param base_name: str, 原始图像的基础名称，用于命名patch
    """
    for patch_name, image_patch, mask_patch in tqdm(patches, desc='Saving TIFF patches'):
        # 使用原始图像名加patch编号命名
        full_patch_name = f"{base_name}_{patch_name}"
        image_path = os.path.join('./tif_crop/images/', f'{full_patch_name}.tif')
        mask_path = os.path.join('./tif_crop/labels/', f'{full_patch_name}.tif')
        # 转置 (z, y, x) -> (x, y, z)
        image_transposed = image_patch.transpose(2, 1, 0)  # (z, y, x) -> (x, y, z)
        mask_transposed = mask_patch.transpose(2, 1, 0)
        tifffile.imwrite(image_path, image_transposed)
        tifffile.imwrite(mask_path, mask_transposed.astype(np.uint16))

def slice_and_save_patches(patches, base_name):
    """
    将剪裁后的patch沿z轴切片，并保存为jpg和png格式

    :param patches: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    :param base_name: str, 原始图像的基础名称，用于命名patch
    """
    for patch_name, image_patch, mask_patch in tqdm(patches, desc='Slicing and saving patches'):
        # 使用原始图像名加patch编号命名
        full_patch_name = f"{base_name}_{patch_name}"
        # 创建对应的目录
        jpeg_dir = os.path.join('./train/JPEGImage/', full_patch_name)
        png_dir = os.path.join('./train/Annotations/', full_patch_name)
        os.makedirs(jpeg_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        z_max = image_patch.shape[0]
        for z in range(z_max):
            # 处理图像切片
            img_slice = image_patch[z, :, :]
            img_min = img_slice.min()
            img_max = img_slice.max()
            if img_max - img_min != 0:
                img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_norm = np.zeros_like(img_slice, dtype=np.uint8)
            img_pil = Image.fromarray(img_norm)
            img_filename = f'{z:05d}.jpg'
            img_pil.save(os.path.join(jpeg_dir, img_filename), 'JPEG')

            # 处理mask切片
            mask_slice = mask_patch[z, :, :]
            # 转换为 uint16 类型，确保兼容 PIL
            mask_slice = mask_slice.astype(np.uint16)
            # 如果需要16位存储
            mask_pil = Image.fromarray(mask_slice).convert('I;16')
            mask_filename = f'{z:05d}.png'
            mask_pil.save(os.path.join(png_dir, mask_filename), 'PNG')

def process_image_label_pair(image_path, label_path, base_name):
    """
    处理单个image和label文件对

    :param image_path: str, 图像文件路径
    :param label_path: str, label文件路径
    :param base_name: str, 原始图像的基础名称
    """
    # 读取图像和mask
    print(f"读取图像: {image_path} 和 mask: {label_path}...")
    image = tifffile.imread(image_path)
    mask = tifffile.imread(label_path)

    # 检查是否为2D，如果是，扩展为3D (1, y, x)
    if image.ndim == 2:
        print("检测到2D图像，扩展为3D (1, y, x)...")
        image = image[np.newaxis, :, :]  # (y, x) -> (1, y, x)
    if mask.ndim == 2:
        print("检测到2D mask，扩展为3D (1, y, x)...")
        mask = mask[np.newaxis, :, :]

    if image.ndim != 3 or mask.ndim != 3:
        print("输入的图像和mask必须是二维或三维的")
        return

    if image.shape != mask.shape:
        print(f"图像和mask的形状必须一致，当前图像形状: {image.shape}, mask形状: {mask.shape}")
        return

    # 转置图像和mask从 (z, y, x) 到 (x, y, z)
    # print("转置图像和mask的维度...")
    # image = image.transpose(2, 1, 0)  # (z, y, x) -> (x, y, z)
    # mask = mask.transpose(2, 1, 0)

    # 去除最大的99.5%的值
    print("去除图像中的最大99.5%的值...")
    image = remove_top_percentile(image, percentile=99.5)

    # 剪裁图像和mask
    print("开始剪裁图像和mask...")
    patches = crop_patches(image, mask, patch_size=(1024, 1024), step=512)

    # 保存剪裁后的patches为tif文件
    print("保存剪裁后的patches为tif文件...")
    save_tif_patches(patches, base_name)

    # 沿z轴切片并保存为jpg和png
    print("开始沿z轴切片并保存为jpg和png...")
    slice_and_save_patches(patches, base_name)

def main():
    args = parse_arguments()
    image_dir = args.image_dir
    label_dir = args.label_dir

    # 检查文件夹是否存在
    if not os.path.isdir(image_dir):
        print(f"图像文件夹不存在: {image_dir}")
        return
    if not os.path.isdir(label_dir):
        print(f"label文件夹不存在: {label_dir}")
        return

    # 创建必要的目录
    create_directories()

    # 获取所有image和label的文件对
    file_pairs = get_file_pairs(image_dir, label_dir)
    if not file_pairs:
        print("没有找到任何匹配的image和label文件。")
        return

    print(f"找到 {len(file_pairs)} 对image和label文件。")

    # 处理每一对image和label文件
    for image_path, label_path, base_name in tqdm(file_pairs, desc='Processing image-label pairs'):
        process_image_label_pair(image_path, label_path, base_name)

    print("所有数据转换完成！")

if __name__ == '__main__':
    main()
