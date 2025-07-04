import tifffile as tiff
import os

# 打开tif图像
img_array = tiff.imread('/media/liushuo/新加卷2/data/lz/3d/#Stablized AVG_roi2_seq3_3D-SIM561_RedCh_SIrecon_actin_result_overlap.tif')

# 定义裁剪的尺寸和保存路径
crop_size = 1024
z_dim = img_array.shape[0]  # 获取z维度大小
y_dim, x_dim = img_array.shape[1], img_array.shape[2]
save_folder = '/media/liushuo/新加卷2/data/lz/3d/label/'  # 替换为你的保存路径

# 确保保存路径存在
os.makedirs(save_folder, exist_ok=True)

# 循环裁剪图像
for i in range(0, y_dim, crop_size):
    for j in range(0, x_dim, crop_size):
        cropped_img = img_array[:, i:i+crop_size, j:j+crop_size]
        tiff.imsave(os.path.join(save_folder, f'cropped_{i}_{j}.tif'), cropped_img)
