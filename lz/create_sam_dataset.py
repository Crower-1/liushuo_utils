#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据转换脚本

功能：
1. 输入为一张.tif图像（float32）和对应的mask.tif标记文件（uint16），数据为三维（x,y,z）。
2. 将图像和label剪裁为1024*1024*z的patch，剪裁步长为512，并储存为tif格式，图像储存在./tif_crop/images/，
   label储存在./tif_crop/labels/，图像和label名字保持一致。
3. 将剪裁后的三维图像沿z轴进行切片，切片后的图像储存在./train/JPEGImage/{crop_image_name}/，
   储存格式为jpg。label储存在./train/Annotations/{crop_image_name}/，储存格式为png。
4. jpg和png命名为00000.jpg/00000.png，数字代表z轴的值。

使用方法：
    python data_convert.py --image_path /path/to/image.tif --mask_path /path/to/mask.tif

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

def remove_top_percentile(image, percentile=99.5):
    """
    去除图像中最大的percentile%的值，保留最低(100 - percentile)%的值。
    即设置大于阈值的像素为阈值。
    """
    threshold = np.percentile(image, percentile)
    image_clipped = np.clip(image, None, threshold)
    return image_clipped

def parse_arguments():
    parser = argparse.ArgumentParser(description='将3D TIFF图像和mask剪裁并切片保存')
    parser.add_argument('--image_path', type=str, required=True, help='输入的.tif图像文件路径')
    parser.add_argument('--mask_path', type=str, required=True, help='对应的mask.tif标记文件路径')
    args = parser.parse_args()
    return args

def create_directories():
    # 创建剪裁后存储的目录
    os.makedirs('./tif_crop/images/', exist_ok=True)
    os.makedirs('./tif_crop/labels/', exist_ok=True)
    # 创建切片存储的目录
    os.makedirs('./train/JPEGImage/', exist_ok=True)
    os.makedirs('./train/Annotations/', exist_ok=True)

def crop_patches(image, mask, patch_size=(1024, 1024), step=512):
    """
    对3D图像和mask进行剪裁

    :param image: numpy array, shape (x, y, z)
    :param mask: numpy array, shape (x, y, z)
    :param patch_size: tuple, (x_size, y_size)
    :param step: int, 步长
    :return: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    """
    patches = []
    x_max, y_max, z_max = image.shape

    patch_id = 0
    for x in tqdm(range(0, x_max - patch_size[0] + 1, step), desc='Cropping patches (x-axis)'):
        for y in range(0, y_max - patch_size[1] + 1, step):
            image_patch = image[x:x + patch_size[0], y:y + patch_size[1], :]
            mask_patch = mask[x:x + patch_size[0], y:y + patch_size[1], :]
            patch_name = f'patch_{patch_id:04d}'
            patches.append((patch_name, image_patch, mask_patch))
            patch_id += 1

    return patches

def save_tif_patches(patches):
    """
    保存剪裁后的图像和mask为tif文件

    :param patches: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    """
    for patch_name, image_patch, mask_patch in tqdm(patches, desc='Saving TIFF patches'):
        # 将维度从 (x, y, z) 转回 (z, y, x)
        image_patch = image_patch.transpose(2, 1, 0)
        mask_patch = mask_patch.transpose(2, 1, 0)
        image_path = os.path.join('./tif_crop/images/', f'{patch_name}.tif')
        mask_path = os.path.join('./tif_crop/labels/', f'{patch_name}.tif')
        tifffile.imwrite(image_path, image_patch)
        tifffile.imwrite(mask_path, mask_patch.astype(np.uint16))

def slice_and_save_patches(patches):
    """
    将剪裁后的patch沿z轴切片，并保存为jpg和png格式

    :param patches: list of tuples, 每个元组包含 (patch_name, image_patch, mask_patch)
    """
    for patch_name, image_patch, mask_patch in tqdm(patches, desc='Slicing and saving patches'):
        # 创建对应的目录
        jpeg_dir = os.path.join('./train/JPEGImage/', patch_name)
        png_dir = os.path.join('./train/Annotations/', patch_name)
        os.makedirs(jpeg_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)

        z_max = image_patch.shape[2]
        for z in range(z_max):
            # 处理图像切片
            img_slice = image_patch[:, :, z]
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
            mask_slice = mask_patch[:, :, z]
            # 如果mask值超过255，需要转换为适合PNG的格式
            if mask_slice.dtype != np.uint8:
                # 检查mask的最大值以决定使用何种模式
                if mask_slice.max() > 65535:
                    raise ValueError(f"Mask值超过16位范围: {mask_slice.max()}")
                elif mask_slice.max() > 255:
                    mask_pil = Image.fromarray(mask_slice).convert('I;16')
                else:
                    mask_pil = Image.fromarray(mask_slice.astype(np.uint8))
            else:
                mask_pil = Image.fromarray(mask_slice)
            mask_filename = f'{z:05d}.png'
            mask_pil.save(os.path.join(png_dir, mask_filename), 'PNG')

def main():
    args = parse_arguments()
    image_path = args.image_path
    mask_path = args.mask_path

    # 检查文件是否存在
    if not os.path.isfile(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    if not os.path.isfile(mask_path):
        print(f"mask文件不存在: {mask_path}")
        return

    # 创建必要的目录
    create_directories()

    # 读取图像和mask
    print("读取图像和mask...")
    image = tifffile.imread(image_path)
    mask = tifffile.imread(mask_path)

    if image.ndim != 3 or mask.ndim != 3:
        print("输入的图像和mask必须是三维的（x, y, z）")
        return

    if image.shape != mask.shape:
        print("图像和mask的形状必须一致")
        return

    # 去除最大的99.5%的值
    print("去除图像中的最大99.5%的值...")
    image = remove_top_percentile(image, percentile=99.5)
    
    # 转置图像和mask从 (z, y, x) 到 (x, y, z)
    print("转置图像和mask的维度...")
    image = image.transpose(2, 1, 0)  # (z, y, x) -> (x, y, z)
    mask = mask.transpose(2, 1, 0)    # (z, y, x) -> (x, y, z)

    # 剪裁图像和mask
    print("开始剪裁图像和mask...")
    patches = crop_patches(image, mask, patch_size=(1024, 1024), step=512)

    # 保存剪裁后的patches为tif文件
    print("保存剪裁后的patches为tif文件...")
    save_tif_patches(patches)

    # 沿z轴切片并保存为jpg和png
    print("开始沿z轴切片并保存为jpg和png...")
    slice_and_save_patches(patches)

    print("数据转换完成！")

if __name__ == '__main__':
    main()
