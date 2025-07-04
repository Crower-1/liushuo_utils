#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to calculate volume and generate surface point files for _label.mrc files in a specified folder.

Usage:
    python calculate_volume_and_surface_point.py --input_dir <target folder path>
"""

import os
import argparse
import mrcfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_erosion
from scipy.ndimage import median_filter

def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Process _label.mrc files to calculate volume and generate surface point files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the target folder')
    return parser.parse_args()

def calculate_volume(mask):
    """
    Calculate volume (i.e., count of voxels with value 1)
    """
    volume = np.sum(mask)
    return volume

def extract_tomo_name(filename):
    """
    Extract the tomo name from the filename by removing the '_label.mrc' suffix
    """
    if filename.endswith('_label.mrc'):
        return filename[:-10]  # Remove the '_label.mrc' suffix (10 characters)
    else:
        return None

def compute_surface_points(mask_data):
    """
    Calculate surface points
    """
    edges = np.zeros_like(mask_data, dtype=bool)
    for z in range(mask_data.shape[0]):
        layer = mask_data[z, :, :]
        eroded_layer = binary_erosion(layer, iterations=1)
        edges[z, :, :] = layer & ~eroded_layer
    return edges

def save_surface_points(edges, tomo_name, output_dir):
    """
    Save surface point coordinates as a txt file in x y z format
    """
    points = np.argwhere(edges)
    # Reorder to x y z
    points = points[:, [2, 1, 0]]  # Change from z y x to x y z
    output_path = os.path.join(output_dir, f"{tomo_name}_surface.txt")
    # Save as txt file with format x y z, each line as a point
    np.savetxt(output_path, points, fmt='%d', delimiter='\t', comments='')
    # print(f"Surface points saved to: {output_path}")

def main():
    args = parse_arguments()
    input_dir = args.input_dir

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a valid folder.")
        return

    # Find all files ending with '_label.mrc'
    all_files = os.listdir(input_dir)
    label_files = [f for f in all_files if f.endswith('_label.mrc')]

    if not label_files:
        print(f"No '_label.mrc' files found in folder '{input_dir}'.")
        return

    # Prepare data structure to store volume information
    volume_data = {
        'Tomo Name': [],
        'Voxel Volume': []
    }

    # Create folder for saving surface points
    points_dir = os.path.join(input_dir, 'points')
    os.makedirs(points_dir, exist_ok=True)

    # Process each label file
    for label_file in tqdm(label_files, desc="Processing files"):
        tomo_name = extract_tomo_name(label_file)
        if not tomo_name:
            print(f"Skipping file '{label_file}', unable to extract tomo name.")
            continue

        label_path = os.path.join(input_dir, label_file)

        # Read MRC file
        try:
            with mrcfile.open(label_path, mode='r') as mrc:
                mask_data = mrc.data.astype(int)
                # mask_data = mrc.data.astype(bool)
        except Exception as e:
            print(f"Error reading file '{label_path}': {e}")
            continue

        # # 滤波参数
        # size = (3, 1, 1)  # 仅沿 z 轴进行滤波
        # iterations = 3    # 滤波迭代次数
        # label_data_filtered = mask_data
        # for _ in range(iterations):
        #     label_data_filtered = median_filter(label_data_filtered, size=size)
        # mask_data = label_data_filtered.astype(int)
        # Calculate volume
        voxel_volume = calculate_volume(mask_data)

        # Add to data structure
        volume_data['Tomo Name'].append(tomo_name)
        volume_data['Voxel Volume'].append(voxel_volume)

        # Compute surface points
        edges = compute_surface_points(mask_data)

        # Save surface points
        save_surface_points(edges, tomo_name, points_dir)

    # Create DataFrame and save as Excel
    df = pd.DataFrame(volume_data)
    excel_path = os.path.join(input_dir, 'volume.xlsx')
    try:
        df.to_excel(excel_path, index=False)
        print(f"Volume information saved to: {excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    main()
