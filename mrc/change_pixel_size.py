import mrcfile
import pandas as pd
import numpy as np
import os
from mrc.io import get_tomo, save_tomo


        
        
def change_tomo_voxel_size(path, new_voxel_size=17.14):
    """
    Change the voxel size of a 3D MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - voxel_size: float
    """
    data = get_tomo(path)
    data_type = data.dtype
    save_tomo(data, path, new_voxel_size, datetype=data_type)
    
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
    tomo_path = f'/media/liushuo/data1/data/liucong/TS_173/_isonet2-n2n_unet-medium_TS_173_10.96Apx.mrc'
    # mask_path = f'/media/liushuo/新加卷/data/actin trace/actintomo/20200820/tomo/pp4001/predictions/pp4001_membrain_v9_528_merged_DA_DS_run2.ckpt_segmented.mrc'
    save_path = f'/media/liushuo/data1/data/liucong/TS_173/TS_173_10.96Apx.mrc'
    tomo = get_tomo(tomo_path)
    save_tomo(tomo, save_path, 10.96, datetype=tomo.dtype)
    # # change_tomo_voxel_size(path, new_voxel_size=17.14)
    # tomo = get_tomo(tomo_path)
    # mask = get_tomo(mask_path)
    # normalized_tomo = normalize_data(tomo)
    # normalized_tomo[mask == 1] = 0.8
    
    # save_tomo(normalized_tomo, save_path, voxel_size=17.14)