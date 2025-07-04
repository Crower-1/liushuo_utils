import mrcfile
import pandas as pd
import numpy as np
import os

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
        data = mrc.data
    return data

# def save_tomo(data, path, voxel_size=17.14):
#     """
#     Save a 3D numpy array as an MRC file.

#     Parameters:
#     - data: ndarray
#         The 3D data to save.
#     - voxel_size: float
#         The voxel size of the data.
#     """
#     with mrcfile.new(path, overwrite=True) as mrc:
#         data = data.astype(np.int16)
#         mrc.set_data(data)
#         mrc.voxel_size = voxel_size

def save_tomo(data, path, voxel_size=17.14):
    # # 获取文件所在的目录
    # directory = os.path.dirname(path)
    
    # # 如果目录不存在，则创建
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.int16))
        mrc.voxel_size = voxel_size
        
        
        
def change_tomo_voxel_size(path, new_voxel_size=17.14):
    """
    Change the voxel size of a 3D MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - voxel_size: float
    """
    data = get_tomo(path)
    save_tomo(data, path, new_voxel_size)
    
def change_tomo_mask_to_bimask(mask):
    """
    Change the mask of a 3D MRC file to binary mask.

    Parameters:
    - mask: ndarray
        The 3D mask to save.
    """
    mask = mask.astype(np.bool_)
    return mask

def normalize_data(data):
    """
    Normalize the 3D data.

    Parameters:
    - data: ndarray
        The 3D data to normalize.
    """
    data = (data - data.min()) / (data.max() - data.min())
    lower_percentile = np.percentile(data, 1)  # 最小1%值
    upper_percentile = np.percentile(data, 99)  # 最大99%值
    new_data = np.clip(data, lower_percentile, upper_percentile)  # 将图像像素裁剪到1%~99%的范围
    return new_data
    
if __name__ == '__main__':
    tomo_path = f'/media/liushuo/data1/data/fig_demo/pp1776/ves_seg/pp1776_label_vesicle.mrc'
    # mask_path = f'/media/liushuo/新加卷/data/actin trace/actintomo/20200820/tomo/pp4001/predictions/pp4001_membrain_v9_528_merged_DA_DS_run2.ckpt_segmented.mrc'
    save_path = f'/media/liushuo/data1/data/fig_demo/pp1776/ves_seg/pp1776_label_vesicle2.mrc'
    tomo = get_tomo(tomo_path)
    save_tomo(tomo, save_path, 17.14)
    # # change_tomo_voxel_size(path, new_voxel_size=17.14)
    # tomo = get_tomo(tomo_path)
    # mask = get_tomo(mask_path)
    # normalized_tomo = normalize_data(tomo)
    # normalized_tomo[mask == 1] = 0.8
    
    # save_tomo(normalized_tomo, save_path, voxel_size=17.14)