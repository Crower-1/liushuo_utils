import mrcfile
import numpy as np
import os

mrc_file_path = '/home/liushuo/Documents/data/stack-out_demo/pp1269/ves_seg/pp1269_wbp_corrected.mrc'
with mrcfile.open(mrc_file_path, permissive=True) as mrc:
    data = mrc.data
    
# 两点的坐标
point1 = (117, 205, 245)  # (z, y, x)
point2 = (159, 607, 699)  # (z, y, x)

z_min, y_min, x_min = np.minimum(point1, point2)
z_max, y_max, x_max = np.maximum(point1, point2)

# 剪裁图像
cropped_data = data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

# 获取保存的路径和文件名
directory, filename = os.path.split(mrc_file_path)
filename_without_ext = os.path.splitext(filename)[0]
cropped_filename = f"{filename_without_ext}_crop.mrc"
cropped_file_path = os.path.join(directory, cropped_filename)

# 将剪裁后的数据保存为新的MRC文件
with mrcfile.new(cropped_file_path, overwrite=True) as mrc:
    mrc.set_data(cropped_data.astype(np.int8))  # 确保数据类型一致

print(f"剪裁后的图像已保存至: {cropped_file_path}")