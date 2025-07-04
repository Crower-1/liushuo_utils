import mrcfile
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

def calculate_surface_area_and_export_points(mask, output_file="surface_points.txt", output_mrc="surface_mask.mrc"):
    sigma=(2, 1, 1)
    
    # 使用高斯滤波进行平滑处理
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=sigma)
    
    # 将平滑结果二值化，确保还是一个二值mask
    binary_smoothed_mask = smoothed_mask > 0.5  # 根据需要调整阈值
    # 创建一个 3x3x3 的立方体结构元素
    structure = np.ones((3, 3, 3), dtype=bool)

    # 使用 1 像素大小的腐蚀
    eroded_once = binary_erosion(binary_smoothed_mask, structure=structure, iterations=1)
    # # 对原始mask进行一次腐蚀
    # eroded_once = binary_erosion(mask)
    
    # 计算表面体素：原始mask减去腐蚀一次后的结果
    surface_mask = binary_smoothed_mask & ~eroded_once
    
    # 导出表面点坐标到txt文件
    surface_points = np.argwhere(surface_mask)
    np.savetxt(output_file, surface_points, fmt="%d", header="z y x")
    
    # 计算表面积（表面体素的数量）
    surface_area = np.sum(surface_mask)
    
    # 导出surface_mask为MRC文件
    with mrcfile.new(output_mrc, overwrite=True) as mrc:
        mrc.set_data(surface_mask.astype(np.uint8))  # 转为 uint8 格式
    
    return surface_area

# 读取 MRC 文件
with mrcfile.open('/media/liushuo/新加卷/data/all/20-p108_bin4_label.mrc', mode='r') as mrc:
    mask_data = mrc.data.astype(bool)  # 转为布尔型，方便计算

# 计算表面积并导出表面坐标和表面mask的MRC文件
surface_area = calculate_surface_area_and_export_points(mask_data, "/home/liushuo/Documents/code/Utils/test_data/test.txt", "/home/liushuo/Documents/code/Utils/test_data/smooth_surface_mask.mrc")

print(f"Surface Area: {surface_area} square units")
print("Surface points saved to surface_points.txt")
print("Surface mask saved to surface_mask.mrc")
