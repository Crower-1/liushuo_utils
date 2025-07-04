from mrc.io import get_tomo,save_tomo
import numpy as np
from scipy.ndimage import binary_dilation

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

def visualize_points(tomo_path, coords_path, output_path):
    # 读取体积数据
    tomo = get_tomo(tomo_path)
    tomo_shape = tomo.shape
    point_data = np.zeros(tomo_shape, dtype=np.uint8)

    points = []
    with open(coords_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    # 转换为 (z, y, x)
                    points.append((z, y, x))

    # 在体积数据上标记坐标点
    for coord in points:
        z, y, x = coord
        z, y, x = int(z), int(y), int(x)
        point_data[z, y, x] = 1

    # 创建球形结构元
    spherical_structure = create_spherical_structure(6)  # 生成一个3D的结构元
    # 对合并的mask进行球形膨胀
    dilated_point = binary_dilation(point_data, structure=spherical_structure)

    # 保存标记后的体积数据
    save_tomo(dilated_point, output_path)
    
def main():
    tomo_path = '/media/liushuo/data1/data/synapse_seg/pp4001/pp4001.mrc'
    coords_path = '/media/liushuo/data1/data/synapse_seg/pp4001/1-pp4001_bin4.coords'
    output_path = '/media/liushuo/data1/data/synapse_seg/pp4001/pp4001_points.mrc'
    visualize_points(tomo_path, coords_path, output_path)
    
if __name__ == '__main__':
    main()