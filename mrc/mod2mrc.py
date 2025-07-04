import numpy as np
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
import mrcfile
from skimage import draw
import matplotlib.path as mpath

def read_point(point_file, dtype_z=int):
    """读取IMOD点文件。"""
    point = np.loadtxt(point_file)
    if point.shape[1] != 5:
        raise ValueError("点文件应包含五列，分别对应object, contour, x, y, z.")
    
    cols = ["object", "contour", "x", "y", "z"]
    dtypes = [int, int, float, float, dtype_z]
    data = {
        cols[i]: pd.Series(point[:, i], dtype=dtypes[i])
        for i in range(5)
    }
    model = pd.DataFrame(data)
    return model

def read_model(model_file, dtype_z=int):
    """读取IMOD模型文件。"""
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
        point_file = temp_file.name
        cmd = f"model2point -ob {model_file} {point_file} >/dev/null"
        subprocess.run(cmd, shell=True, check=True)
        
        bak_file = Path(f"{point_file}~")
        if bak_file.exists():
            bak_file.unlink()

        model = read_point(point_file, dtype_z=dtype_z)
        return model

def create_closed_mask(contour1, contour2, img_shape):
    """
    根据两条曲线生成闭合的二维掩模。
    
    Parameters:
        contour1 (array-like): 第一条曲线的 (x, y) 点。
        contour2 (array-like): 第二条曲线的 (x, y) 点。
        img_shape (tuple): 掩模的形状 (height, width)。
    
    Returns:
        mask (ndarray): 二维掩模。
    """
    # 将两条曲线连接形成闭合多边形
    polygon = np.vstack([contour1, contour2[::-1]])  # 逆序连接形成闭环
    path = mpath.Path(polygon)

    # 创建网格并检查点是否在多边形内
    x, y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    grid = path.contains_points(points)
    mask = grid.reshape(img_shape)
    return mask

def mod_to_mrc(mod_file, output_mrc, voxel_size=(1.0, 1.0, 1.0)):
    """
    将 .mod 文件转换为 .mrc 图像。
    
    Parameters:
        mod_file (str): 输入的 .mod 文件路径。
        output_mrc (str): 输出的 .mrc 文件路径。
        voxel_size (tuple): 每个体素的大小 (z, y, x)，默认为 (1.0, 1.0, 1.0)。
    """
    # 读取模型数据
    model = read_model(mod_file, dtype_z=int)
    
    # 筛选object为2或3的点
    filtered = model[model['object'].isin([2, 3])]
    if filtered.empty:
        raise ValueError("没有找到object为2或3的点，请检查输入的 .mod 文件。")
    
    # 打印筛选后的数据
    print(f"筛选后的数据：\n{filtered.head()}")
    
    # 获取所有唯一的z层
    z_layers = filtered['z'].unique()
    z_layers.sort()
    
    # 确定图像的尺寸
    x_min, x_max = filtered['x'].min(), filtered['x'].max()
    y_min, y_max = filtered['y'].min(), filtered['y'].max()
    
    width = int(np.ceil(x_max - x_min)) + 1
    height = int(np.ceil(y_max - y_min)) + 1
    depth = int(np.ceil(z_layers.max() - z_layers.min())) + 1
    
    # 初始化三维掩模
    mask_3d = np.zeros((depth, height, width), dtype=np.uint8)
    
    # 遍历每个z层
    for idx, z in enumerate(z_layers):
        layer = filtered[filtered['z'] == z]
        contours = layer.groupby('contour')
        
        # 检查是否有object 2和3的轮廓
        contour_keys = contours.groups.keys()
        print(f"Z层 {z} 的轮廓：{list(contour_keys)}")
        
        if 2 not in contour_keys or 3 not in contour_keys:
            print(f"Z层 {z} 中缺少 object 2 或 3 的轮廓，跳过。")
            continue
        
        # 获取轮廓点
        contour2_points = contours.get_group(2)[['x', 'y']].values
        contour3_points = contours.get_group(3)[['x', 'y']].values
        
        # 平移坐标以适应掩模
        contour2_points[:, 0] -= x_min
        contour2_points[:, 1] -= y_min
        contour3_points[:, 0] -= x_min
        contour3_points[:, 1] -= y_min
        
        # 创建闭合的二维掩模
        mask = create_closed_mask(contour2_points, contour3_points, (height, width))
        
        # 将掩模赋值到三维掩模中
        mask_3d[idx, :, :] = mask.astype(np.uint8)
    
    # 保存为MRC文件
    with mrcfile.new(output_mrc, overwrite=True) as mrc:
        mrc.set_data(mask_3d)
        mrc.voxel_size = voxel_size  # 设置体素大小（可选）
        mrc.header.cella = [width * voxel_size[2], height * voxel_size[1], depth * voxel_size[0]]
        # mrc.header.cellb = [90.0, 90.0, 90.0]  # 假设为直角，视具体情况调整
        # mrc.update_header_from_data()
    
    print(f"MRC文件已保存至 {output_mrc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 .mod 文件转换为 .mrc 图像。")
    parser.add_argument("mod_file", help="输入的 .mod 文件路径")
    parser.add_argument("output_mrc", help="输出的 .mrc 文件路径")
    parser.add_argument("--voxel_size", nargs=3, type=float, default=(1.0, 1.0, 1.0),
                        help="每个体素的大小 (z, y, x)，默认是 1.0 立方单位")
    
    args = parser.parse_args()
    
    mod_to_mrc(args.mod_file, args.output_mrc, voxel_size=tuple(args.voxel_size))
