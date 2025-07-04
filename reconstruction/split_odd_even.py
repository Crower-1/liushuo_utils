import mrcfile
import os
import numpy as np
from mrc.io import get_tomo, save_tomo


def load_tlt(tlt_path):
    """
    读取 .tlt 文件，返回角度列表
    """
    with open(tlt_path, 'r') as f:
        angles = [float(line.strip()) for line in f if line.strip()]
    return angles

def save_tlt(angles, tlt_path):
    """
    将角度列表保存为 .tlt 文件，每行一个角度（保留两位小数）
    """
    # 获取文件所在的目录
    directory = os.path.dirname(tlt_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(tlt_path, 'w') as f:
        for angle in angles:
            f.write(f"{angle:.2f}\n")

def split_odd_even(ali_path, tlt_path, odd_mrc_path, odd_tlt_path, even_mrc_path, even_tlt_path):
    """
    根据投影的序号（文件中第一幅图像为1号）将输入的.ali图像堆栈和对应.tlt角度文件，
    分成奇数序号和偶数序号两部分，并分别保存为新的 .mrc 和 .tlt 文件。

    Parameters:
    - ali_path: str
        输入 .ali 文件路径。
    - tlt_path: str
        输入 .tlt 文件路径，对应 .ali 文件的角度。
    - odd_mrc_path: str
        输出奇数序号图像的 .mrc 文件路径。
    - odd_tlt_path: str
        输出奇数序号角度的 .tlt 文件路径。
    - even_mrc_path: str
        输出偶数序号图像的 .mrc 文件路径。
    - even_tlt_path: str
        输出偶数序号角度的 .tlt 文件路径。
    """
    # 1. 读取角度和图像数据
    angles = load_tlt(tlt_path)
    tomo_data = get_tomo(ali_path)

    # 检查图像数量与角度数量是否匹配
    num_images = tomo_data.shape[0]
    if num_images != len(angles):
        raise ValueError("图像数量与角度数目不匹配，请检查输入文件！")

    # 2. 根据序号拆分数据（文件中第一幅图像为1号，属于奇数）
    odd_indices = list(range(0, num_images, 2))   # python索引 0,2,4,...
    even_indices = list(range(1, num_images, 2))    # python索引 1,3,5,...

    odd_images = tomo_data[odd_indices, :, :]
    even_images = tomo_data[even_indices, :, :]

    odd_angles = [angles[i] for i in odd_indices]
    even_angles = [angles[i] for i in even_indices]

    # 3. 保存奇数和偶数的数据
    save_tomo(odd_images, odd_mrc_path)
    save_tlt(odd_angles, odd_tlt_path)
    print(f"奇数序号的图像已保存到 {odd_mrc_path}，角度已保存到 {odd_tlt_path}")

    save_tomo(even_images, even_mrc_path)
    save_tlt(even_angles, even_tlt_path)
    print(f"偶数序号的图像已保存到 {even_mrc_path}，角度已保存到 {even_tlt_path}")

# 示例调用
if __name__ == '__main__':
    # 输入文件路径
    ali_file = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/pp1776.ali'
    tlt_file = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/pp1776.tlt'
    
    # 输出文件路径
    odd_mrc = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/output/odd.mrc'
    odd_tlt = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/output/odd.tlt'
    even_mrc = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/output/even.mrc'
    even_tlt = '/media/liushuo/data1/data/synapse_seg/pp1776/ori/output/even.tlt'
    
    split_odd_even(ali_file, tlt_file, odd_mrc, odd_tlt, even_mrc, even_tlt)
