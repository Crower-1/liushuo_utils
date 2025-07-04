import os
import argparse
import numpy as np
import mrcfile
from scipy.ndimage import zoom

from mrc.io import get_tomo, save_tomo

def resample_label(input_path, output_path, target_shape, voxel_size=None, dtype=np.int8):
    """
    将 label 体数据做最近邻重采样到目标尺寸，并保存结果。

    Parameters:
    - input_path: str
        原始 MRC 文件路径。
    - output_path: str
        重采样后 MRC 文件保存路径。
    - target_shape: tuple of int (Z, Y, X)
        目标尺寸。
    - voxel_size: float or None
        如果不为 None，则写入 header 的 voxel_size。
        None 时沿用原文件 header 中的值。
    - dtype: numpy dtype
        输出时的数据类型。
    """
    # 读取
    data = get_tomo(input_path)
    src_shape = np.array(data.shape)
    tgt_shape = np.array(target_shape)

    # 计算缩放因子
    zoom_factors = tgt_shape / src_shape

    # 最近邻插值（order=0）
    data_rs = zoom(data, zoom_factors, order=0)

    save_tomo(data_rs, output_path, voxel_size, datetype=dtype)
    print(f"Resampled from {tuple(src_shape)} → {tuple(tgt_shape)}, saved to {output_path}")

# def parse_args():
#     p = argparse.ArgumentParser(description="Resample a 3D label MRC using nearest-neighbor interpolation.")
#     p.add_argument('--input',  '-i', required=True, help="输入 MRC label 文件路径")
#     p.add_argument('--output', '-o', required=True, help="输出 resample 后 MRC 文件路径")
#     p.add_argument('--size',   '-s', required=True,
#                    help="目标尺寸，格式为 Z,Y,X。例：64,128,128")

#     return p.parse_args()

if __name__ == '__main__':
    # args = parse_args()
    # 解析目标尺寸
    working_dir = '/media/liushuo/data1/data/fig_demo_2/pp199/synapse_seg/'
    base_name = 'pp199'
    ori_tomo_path = working_dir + base_name + '/' + base_name + '.mrc'
    bin2_tomo_path = working_dir + base_name + '/'   + 'synapse/' + base_name + '_bin2.mrc'
    bin4_tomo_path = working_dir + base_name + '/'   + 'synapse/' + base_name + '_bin4.mrc'
    bin4_synapse_label_path = working_dir + base_name + '/'  + 'synapse/' + base_name + '_bin4_label.mrc'
    synapse_label_path = working_dir + base_name + '/'  + 'synapse/' + base_name + '_label.mrc'
    ori_tomo = get_tomo(ori_tomo_path)
    target_shape = ori_tomo.shape
    # input_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp0622/pp0622_bin_pre.mrc'
    # output_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp0622/pp0622_pre.mrc'
    # 将 dtype 名称转换为 numpy dtype
    # dtype = getattr(np, args.dtype)
    resample_label(bin4_synapse_label_path, synapse_label_path, target_shape, voxel_size=17.14)