import numpy as np
import mrcfile
import random

# 文件名（可根据需要修改）
ali_filename = '/media/liushuo/data1/data/synapse_seg/pp4001/reconstruction/pp4001.ali'
tlt_filename = '/media/liushuo/data1/data/synapse_seg/pp4001/reconstruction/pp4001.tlt'
output_mrc = '/media/liushuo/data1/data/synapse_seg/pp4001/reconstruction/output.mrc'
output_tlt = '/media/liushuo/data1/data/synapse_seg/pp4001/reconstruction/output.tlt'

# 1. 读取 .tlt 文件中的角度信息
with open(tlt_filename, 'r') as f:
    # 每一行存储一个角度，去除空白字符后转换为浮点数
    angles = [float(line.strip()) for line in f if line.strip()]

# 2. 读取 .ali 文件中的投影图像堆栈
# 这里假设 .ali 文件可以用 mrcfile 库读取，如果格式不同，
# 则需要使用合适的方法加载图像数据（例如 np.fromfile 并重塑维度）
try:
    ali_data = mrcfile.read(ali_filename)
except Exception as e:
    raise RuntimeError(f"读取 .ali 文件失败，请检查文件格式和路径: {e}")

# 检查图像数目与角度数是否一致
num_images = ali_data.shape[0]
if num_images != len(angles):
    raise ValueError("图像数量与角度数目不匹配，请检查输入文件！")

# 3. 筛选角度在 (-60, 60) 范围内的投影索引
valid_indices = [i for i, angle in enumerate(angles) if -60 < angle < 62]

if len(valid_indices) < 41:
    raise ValueError("符合条件的投影数量不足 41 个，请检查输入数据范围！")

# 随机抽取 41 个索引（可以选择保留原始顺序，也可以打乱顺序）
selected_indices = random.sample(valid_indices, 41)
selected_indices.sort()  # 如需要保持投影的顺序，则排序

# 根据选取的索引提取对应图像和角度
selected_images = ali_data[selected_indices, :, :]
selected_angles = [angles[i] for i in selected_indices]

# 4. 将选取的 41 个投影保存为新的 .mrc 文件
with mrcfile.new(output_mrc, overwrite=True) as mrc:
    # 建议转换为 float32 类型
    mrc.set_data(selected_images.astype(np.float32))
print(f"保存新的投影图像到 {output_mrc}")

# 同时生成新的 .tlt 文件，写入对应的角度信息
with open(output_tlt, 'w') as f:
    for angle in selected_angles:
        # 保留两位小数
        f.write(f"{angle:.2f}\n")
print(f"保存新的角度文件到 {output_tlt}")
