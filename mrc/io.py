import mrcfile
import os
import numpy as np

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

def save_tomo(data, path, voxel_size=17.14,datetype=np.int8):
    """
    Save a 3D numpy array as an MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - voxel_size: float
        The voxel size of the data.
    """
        # 获取文件所在的目录
    directory = os.path.dirname(path)
    
    # 如果目录不存在，则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(datetype)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size