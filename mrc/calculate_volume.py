import mrcfile
import numpy as np
from scipy.ndimage import binary_dilation

def calculate_volume(mask):
    # 计算体积（即，mask中值为1的体素数量）
    volume = np.sum(mask)
    
    return volume

# 读取MRC文件
with mrcfile.open('/media/liushuo/新加卷/data/all/20-p108_bin4_label.mrc', mode='r') as mrc:
    mask_data = mrc.data.astype(bool)  # 转为布尔型，方便计算

# 计算体积和表面积
volume = calculate_volume(mask_data)

print(f"Volume: {volume} cubic units")
