import mrcfile
import numpy as np

# 输入和输出文件名
input_filename = "/media/liushuo/data1/data/synapse_seg/pp1776/ori/reconstruction/odd.mrc"
output_filename = "/media/liushuo/data1/data/synapse_seg/pp1776/ori/reconstruction/mask.mrc"

# 读取 .mrc 图像
with mrcfile.open(input_filename, permissive=True) as mrc:
    data = mrc.data.copy()  # 获取图像数据，假设数据为三维数组 (z, x, y)
    z, x, y = data.shape
    print(f"图像尺寸: z={z}, x={x}, y={y}")

# 创建 ROI mask，初始全为0
mask = np.zeros((z, x, y), dtype=np.uint8)

# 计算保留中间区域的索引范围，15%对应 z 轴的上、下部分
start = int(z * 0.15)
end = int(z * 0.85)

# 将中间区域置为1
mask[start:end, :, :] = 1

# 保存生成的 mask 到新的 .mrc 文件中
with mrcfile.new(output_filename, overwrite=True) as mrc_out:
    mrc_out.set_data(mask)
    
print(f"ROI mask 已保存为 {output_filename}")
