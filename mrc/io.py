import mrcfile
import os
import numpy as np

def _normalize_voxel_size(value):
    """Return a 3-tuple voxel size from various scalar/array inputs."""
    if value is None:
        return (1.0, 1.0, 1.0)

    components = None

    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        components = [float(value.x), float(value.y), float(value.z)]
    else:
        arr = np.array(value, copy=False)
        if arr.dtype.names:
            components = [float(arr[name]) for name in arr.dtype.names]
        else:
            flat = np.ravel(arr)
            if flat.size == 0:
                components = []
            else:
                components = [float(x) for x in flat]

    if not components:
        return (1.0, 1.0, 1.0)

    if len(components) == 1:
        val = components[0]
        return (val, val, val)

    if len(components) >= 3:
        return tuple(components[:3])

    # If fewer than 3 values are provided, repeat the last value.
    last_val = components[-1]
    padded = components + [last_val] * (3 - len(components))
    return tuple(padded)


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

def get_tomo_with_voxel_size(path):
    """Load a 3D MRC file and return both data and voxel size."""
    with mrcfile.open(path) as mrc:
        data = np.copy(mrc.data)
        voxel_size = _normalize_voxel_size(mrc.voxel_size)
    return data, voxel_size


def save_tomo(data, path, voxel_size=17.14, datetype=np.int8):
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
        mrc.voxel_size = _normalize_voxel_size(voxel_size)
