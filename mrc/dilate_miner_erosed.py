from mrc.io import get_tomo, save_tomo
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

mask = get_tomo('/media/liushuo/data1/data/fig_demo_2/pp370/pp370_roi.mrc')

def create_spherical_structure(radius):
    """
    创建一个球形的结构元，用于膨胀操作。

    参数:
    radius (int): 球的半径。

    返回:
    np.ndarray: 球形结构元。
    """
    # 生成球体网格
    size = 2 * radius + 1  # 球形结构元的大小
    z, y, x = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    distance = x**2 + y**2 + z**2
    # 将距离球心半径以内的点设为1，形成球形结构元
    spherical_structure = distance <= radius**2
    return spherical_structure

# 创建球形结构元
# spherical_structure = create_spherical_structure(6)  # 生成一个3D的结构元
# 对合并的mask进行球形膨胀
dilated_mask = binary_dilation(mask, structure=create_spherical_structure(2))
erosed_mask = binary_erosion(mask, structure=create_spherical_structure(2))
roi_mask = dilated_mask & ~erosed_mask
# 保存结果
save_tomo(roi_mask, '/media/liushuo/data1/data/fig_demo_2/pp370/pp370_roi_3.mrc', voxel_size=17.14, datetype=np.int8)