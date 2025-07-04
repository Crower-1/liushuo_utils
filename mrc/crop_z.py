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
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
        
        
        
def crop_z(tomo, min_z, max_z):
    """
    Crop the z-axis of a 3D numpy array.

    Parameters:
    - tomo: ndarray
        The 3D data to crop.
    - min_z: int
        The minimum z index to keep.
    - max_z: int
        The maximum z index to keep.

    Returns:
    - cropped_tomo: ndarray
        The cropped 3D data.
    """
    return tomo[min_z:max_z, :, :]

def crop_z_2(tomo, min_z, max_z):
    """
    Crop the z-axis of a 3D numpy array.

    Parameters:
    - tomo: ndarray
        The 3D data to crop.
    - min_z: int
        The minimum z index to keep.
    - max_z: int
        The maximum z index to keep.

    Returns:
    - cropped_tomo: ndarray
        The cropped 3D data.
        Keep the shape the same as input
    """
    tomo_shape = tomo.shape
    # Create 0 array
    cropped_tomo = np.zeros(tomo_shape, dtype=tomo.dtype)
    for z in range(tomo_shape[0]):
        if z < min_z or z > max_z:
            cropped_tomo[z] = 0
        else:
            cropped_tomo[z] = tomo[z]
    return cropped_tomo
    
if __name__ == '__main__':
    tomo_path = f'/media/liushuo/data1/data/fig_demo_2/pp199/synapse_seg/pp199/active_zone/active_zone.mrc'
    save_path = f'/media/liushuo/data1/data/fig_demo_2/pp199/synapse_seg/pp199/active_zone/active_zone_92_151.mrc'
    new_tomo = crop_z_2(get_tomo(tomo_path), 92, 151)
    save_tomo(new_tomo, save_path)