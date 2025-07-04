import os
import json
import numpy as np
import mrcfile as mf
from skimage.morphology import cube, dilation, ball


def calculate_bounds(coord1, coord2, margin, lower_bound, upper_bound):
    """计算坐标范围的边界，并添加安全边距。"""
    min_val = max(int(np.floor(min(coord1, coord2) - margin)), lower_bound)
    max_val = min(int(np.ceil(max(coord1, coord2) + margin)), upper_bound)
    return min_val, max_val


def generate_MT_mask(label_data, points, radius=8):
    """
    通过连接一系列点生成一个二值mask，表示微管。

    Args:
        label_data (np.ndarray): 用于定义mask形状的3D标签数据数组。
        points (list of list or np.ndarray): 每个点由 [z, y, x] 表示的列表。
        radius (int, optional): 圆柱形微管的半径。默认值为8。

    Returns:
        np.ndarray: 一个二值mask，其中微管表示为1。
    """
    # 获取label_data的维度
    nz, ny, nx = label_data.shape
    mask = np.zeros((nz, ny, nx), dtype=np.uint8)
    points = np.array(points, dtype=np.float32)

    for i in range(len(points) - 1):
        point_a, point_b = points[i], points[i + 1]

        # 计算包含安全边距的边界
        z_min, z_max = calculate_bounds(point_a[0], point_b[0], radius, 0, nz)
        y_min, y_max = calculate_bounds(point_a[1], point_b[1], radius, 0, ny)
        x_min, x_max = calculate_bounds(point_a[2], point_b[2], radius, 0, nx)

        # 生成边界框内的网格点
        z, y, x = np.meshgrid(
            np.arange(z_min, z_max),
            np.arange(y_min, y_max),
            np.arange(x_min, x_max),
            indexing='ij'
        )
        grid_points = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)  # 形状: (N, 3)

        # 计算向量和距离
        vector_pa = point_a - grid_points  # 从网格点到点A的向量
        vector_pb = point_b - grid_points  # 从网格点到点B的向量
        vector_ab = point_b - point_a      # 从点A到点B的向量

        cross_product = np.cross(vector_pa, vector_pb)
        norm_ab = np.linalg.norm(vector_ab)
        # 避免除以零
        if norm_ab == 0:
            continue
        perpendicular_distances = np.linalg.norm(cross_product, axis=1) / norm_ab

        # 判断网格点是否在线段上
        dot_pa_ab = np.einsum('ij,j->i', vector_pa, vector_ab)
        dot_pb_ab = np.einsum('ij,j->i', vector_pb, vector_ab)
        is_on_segment = (dot_pa_ab * dot_pb_ab <= 0)

        # 判断网格点是否在半径范围内
        within_radius = (perpendicular_distances <= radius)

        # 最终有效点
        valid_points = is_on_segment & within_radius

        # 添加端点覆盖
        start_points_distance = np.linalg.norm(grid_points - point_a, axis=1)
        start_points_within_radius = start_points_distance <= radius

        end_points_distance = np.linalg.norm(grid_points - point_b, axis=1)
        end_points_within_radius = end_points_distance <= radius

        valid_points |= start_points_within_radius
        valid_points |= end_points_within_radius

        # 标记有效点到mask
        valid_grid_points = grid_points[valid_points].astype(np.int32)
        mask[valid_grid_points[:, 0], valid_grid_points[:, 1], valid_grid_points[:, 2]] = 1

    return mask


def process_mt_to_mask(json_path, output_path):
    """读取JSON和MRC文件，生成相应的mask文件，其中mask值对应mt_id。"""
    # 1. 读取JSON文件
    with open(json_path, "r") as f:
        mts = json.load(f)

    # 提取tomo_name和MRC文件路径
    tomo_name = os.path.splitext(os.path.basename(json_path))[0]
    tomo_name = tomo_name.replace("_point", "")
    mrc_path = '/media/liushuo/data1/data/synapse_seg/pp463/Prediction/MT/center_distance_map.mrc'

    # 2. 读取MRC文件以获取z, y, x维度
    with mf.open(mrc_path, permissive=True) as mrc:
        z, y, x = mrc.data.shape
        voxel_size = mrc.voxel_size

    # 3. 创建一个大小为 (z, y, x) 的全零数组，类型为int16
    masks = np.zeros((z, y, x), dtype=np.int16)

    # 4. 处理每个mt，生成局部mask，并合并到全局mask中
    for mt in mts:
        mt_id = mt["id"]
        print(f"Processing mt {mt_id}...")
        seedlist = mt["points"]  # points 是 (z, y, x) 的列表

        # 生成当前mt的二值mask
        local_mask_binary = generate_MT_mask(masks, seedlist, radius=8)

        # 创建当前mt的mask，值为mt_id
        local_mask = local_mask_binary * mt_id

        # 合并当前mt的mask到全局mask中
        # 使用 np.where 优先保留已有的mt_id，或者覆盖（根据需求调整）
        masks = np.where(local_mask > 0, local_mask, masks)

    # # 5. 保存mask为MRC文件
    # output_path = os.path.join(os.path.dirname(json_path), f"{tomo_name}_mt_label.mrc")

    with mf.new(output_path, overwrite=True) as mrc:
        mrc.set_data(masks)
        mrc.voxel_size = 17.14  # 使用原始MRC的voxel_size，或根据需要设置

    print(f"Mask saved to {output_path}")


# 示例调用
json_path = f'/media/liushuo/data1/data/synapse_seg/pp463/Prediction/MT/mt_point.json'
output_path = f'/media/liushuo/data1/data/synapse_seg/pp463/Prediction/MT/mt_label.mrc'
process_mt_to_mask(json_path, output_path)
