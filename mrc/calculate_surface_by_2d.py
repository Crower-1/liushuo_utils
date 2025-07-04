import mrcfile
import numpy as np
from scipy.ndimage import binary_erosion

# 1. 读取 MRC 文件
with mrcfile.open('/media/liushuo/新加卷/data/all/20-p108_bin4_label.mrc', mode='r') as mrc:
    mask_data = mrc.data.astype(bool)  # 确保 mask 是布尔类型

# 2. 初始化一个空的三维数组用于存储边缘
edges = np.zeros_like(mask_data, dtype=bool)

# 3. 逐层进行2D腐蚀并计算边缘
for z in range(mask_data.shape[0]):
    layer = mask_data[z, :, :]  # 选择当前的 xy 平面
    eroded_layer = binary_erosion(layer, iterations=1)  # 对当前平面进行 2D 腐蚀
    edges[z, :, :] = layer & ~eroded_layer  # 计算边缘并存储到结果数组中

# 4. 将结果保存为新的 MRC 文件
with mrcfile.new('/home/liushuo/Documents/code/Utils/test_data/surface_mask_2dErosion.mrc', overwrite=True) as edge_mrc:
    edge_mrc.set_data(edges.astype(np.uint8))
