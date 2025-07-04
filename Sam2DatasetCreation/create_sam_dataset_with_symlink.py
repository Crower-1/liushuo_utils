#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据转换脚本(改写版)

变化点：
1. 图像和label由tif变为mrc格式，需要使用mrcfile读取。
2. 不再进行patch裁剪，而是将图像和label resize到 1024*1024*z，并保存到 ./mrc_resized/images/ 和 ./mrc_resized/labels/ 下，文件格式继续使用mrc存储。
3. 对最终切片时，不再输出全部z层，只选择mask在z轴方向上有标记的最小和最大层之间的所有z层进行切片(包含最小和最大)，图像切片存放到 ./train/JPEGImage/{base_name}/，标签切片存放到 ./train/Annotations/{base_name}/。

使用方法：
    python data_convert.py --image_dir /path/to/image_folder --label_dir /path/to/label_folder

依赖库：
    - numpy
    - mrcfile
    - Pillow
    - tqdm
    - scipy (用于3D连通性分析)
"""

import os
import argparse
import numpy as np
import mrcfile
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='将MRC图像和mask进行resize和连通分割后切片保存')
    parser.add_argument('--image_dir', type=str, required=True, help='输入的图像文件夹路径（mrc文件）')
    parser.add_argument('--label_dir', type=str, required=True, help='对应的label文件夹路径（mrc文件）')
    args = parser.parse_args()
    return args

def create_directories():
    os.makedirs('./mrc_resized/images/', exist_ok=True)
    os.makedirs('./mrc_resized/labels/', exist_ok=True)
    os.makedirs('./train/JPEGImage/', exist_ok=True)
    os.makedirs('./train/Annotations/', exist_ok=True)

def get_file_pairs(image_dir, label_dir):
    """
    假设image文件名为xxx.mrc，label文件名为xxx_actin_result.mrc
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.mrc')]
    file_pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        # label_file = f"{base_name}_actin_result.mrc"
        label_file = f"{base_name}.mrc"
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
    即将大于阈值的像素截断到阈值。
    """
    threshold = np.percentile(image, percentile)
    image_clipped = np.clip(image, None, threshold)
    return image_clipped

def resize_volume(volume, new_size=(1024, 1024)):
    """
    将3D数据在xy平面resize为new_size。
    volume形状: (z, y, x)
    new_size: (new_y, new_x)
    使用最近邻插值对标签进行resize，使用双线性对图像进行resize。
    对于标签，使用最近邻插值，为保证分割正确性。
    对于图像，使用双线性插值。
    因为函数仅在主流程中根据数据类型调用不同插值方式，这里仅实现一个通用的resize。
    """
    z, y, x = volume.shape
    # 对于3D数据，逐层resize
    # 这里使用PIL对每一帧进行resize，保持z不变
    resized = np.zeros((z, new_size[0], new_size[1]), dtype=volume.dtype)
    for i in range(z):
        slice_img = volume[i]
        pil_img = Image.fromarray(slice_img)
        # 对于图像可使用双线性插值 Image.BILINEAR；对于标签用最近邻 Image.NEAREST
        # 这里先返回pil_img，以便在调用时决定插值方式
        resized_slice = np.array(pil_img.resize(new_size, resample=Image.BILINEAR if volume.dtype!=np.int32 else Image.NEAREST))
        resized[i] = resized_slice
    return resized

def process_image_label_pair(image_path, label_path, base_name):
    # 读取图像和mask (mrc格式)
    print(f"读取图像: {image_path} 和 mask: {label_path}...")
    with mrcfile.open(image_path, 'r') as mrc:
        image = mrc.data
    with mrcfile.open(label_path, 'r') as mrc:
        mask = mrc.data

    # 如果2D，则扩展为3D
    if image.ndim == 2:
        print("检测到2D图像，扩展为3D (1, y, x)...")
        image = image[np.newaxis, :, :]  # (y, x) -> (1, y, x)
    if mask.ndim == 2:
        print("检测到2D mask，扩展为3D (1, y, x)...")
        mask = mask[np.newaxis, :, :]

    if image.shape != mask.shape:
        print(f"图像和mask的形状必须一致，当前图像形状: {image.shape}, mask形状: {mask.shape}")
        return

    # 去除最大的99.5%值
    print("去除图像中的最大99.5%的值...")
    image = remove_top_percentile(image, percentile=99.5)

    # 将图像和mask resize到 1024*1024*z
    print("开始resize图像和mask至1024*1024*z...")
    z, y, x = image.shape
    # 对图像使用双线性插值，对mask使用最近邻插值
    def resize_3d(volume, new_size, is_label=False):
        resized = np.zeros((z, new_size[0], new_size[1]), dtype=volume.dtype)
        for i in range(z):
            slice_img = volume[i]
            pil_img = Image.fromarray(slice_img)
            # 对于label用最近邻
            mode = Image.NEAREST if is_label else Image.BILINEAR
            resized_slice = np.array(pil_img.resize(new_size, resample=mode))
            resized[i] = resized_slice
        return resized

    image_resized = resize_3d(image, (1024, 1024), is_label=False)
    mask_resized = resize_3d(mask, (1024, 1024), is_label=True)

    # 保存resize后的数据到 ./mrc_resized/
    print("保存resize后的mrc数据...")
    os.makedirs('./mrc_resized/images/', exist_ok=True)
    os.makedirs('./mrc_resized/labels/', exist_ok=True)
    image_mrc_path = os.path.join('./mrc_resized/images/', f'{base_name}.mrc')
    label_mrc_path = os.path.join('./mrc_resized/labels/', f'{base_name}.mrc')
    with mrcfile.new(image_mrc_path, overwrite=True) as mrc:
        mrc.set_data(image_resized.astype(np.float32))
    with mrcfile.new(label_mrc_path, overwrite=True) as mrc:
        mrc.set_data(mask_resized.astype(np.int16))

    # 找到label中有标记的z范围
    nonzero_z = np.where(mask_resized > 0)[0]
    if len(nonzero_z) == 0:
        print("mask中没有标记，无法进行切片保存。")
        return
    z_min, z_max = nonzero_z.min(), nonzero_z.max()

    # 对 z_min ~ z_max 的层进行切片并保存
    # 创建对应的目录
    jpeg_dir = os.path.join('./train/JPEGImage/', base_name)
    png_dir = os.path.join('./train/Annotations/', base_name)
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print(f"切片范围为 z={z_min} 到 z={z_max}，共 {z_max - z_min + 1} 个切片。")
    for z_idx in range(z_min, z_max + 1):
        # 处理图像切片
        img_slice = image_resized[z_idx, :, :]
        img_min = img_slice.min()
        img_max = img_slice.max()
        if img_max > img_min:
            img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_slice, dtype=np.uint8)
        img_pil = Image.fromarray(img_norm)
        img_filename = f'{z_idx:05d}.jpg'
        img_pil.save(os.path.join(jpeg_dir, img_filename), 'JPEG')

        # 处理mask切片
        mask_slice = mask_resized[z_idx, :, :].astype(np.uint16)
        mask_pil = Image.fromarray(mask_slice).convert('I;16')
        mask_filename = f'{z_idx:05d}.png'
        mask_pil.save(os.path.join(png_dir, mask_filename), 'PNG')

    print("处理完成！")

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
