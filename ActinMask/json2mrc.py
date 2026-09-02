import os
import json
import numpy as np
import mrcfile as mf


def calculate_bounds(coord1, coord2, margin, lower_bound, upper_bound):
    """计算坐标范围的边界，并添加安全边距。"""
    min_val = max(int(np.floor(min(coord1, coord2) - margin)), lower_bound)
    max_val = min(int(np.ceil(max(coord1, coord2) + margin)), upper_bound)
    return min_val, max_val


def generate_actin_mask(masks, points, actin_id, radius=2):
    """
    通过连接一系列点直接写入actin到全局mask。

    Args:
        masks (np.ndarray): 全局3D标签mask数组。
        points (list of list or np.ndarray): 每个点由 [z, y, x] 表示的列表。
        actin_id (int): 当前actin的标签值。
        radius (int, optional): 圆柱形actin的半径。默认值为2。
    """
    nz, ny, nx = masks.shape
    points = np.array(points, dtype=np.float32)

    for i in range(len(points) - 1):
        point_a, point_b = points[i], points[i + 1]

        z_min, z_max = calculate_bounds(point_a[0], point_b[0], radius, 0, nz)
        y_min, y_max = calculate_bounds(point_a[1], point_b[1], radius, 0, ny)
        x_min, x_max = calculate_bounds(point_a[2], point_b[2], radius, 0, nx)

        z, y, x = np.meshgrid(
            np.arange(z_min, z_max),
            np.arange(y_min, y_max),
            np.arange(x_min, x_max),
            indexing='ij'
        )
        grid_points = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)

        vector_pa = point_a - grid_points
        vector_pb = point_b - grid_points
        vector_ab = point_b - point_a

        cross_product = np.cross(vector_pa, vector_pb)
        norm_ab = np.linalg.norm(vector_ab)
        if norm_ab == 0:
            continue
        perpendicular_distances = np.linalg.norm(cross_product, axis=1) / norm_ab

        dot_pa_ab = np.einsum('ij,j->i', vector_pa, vector_ab)
        dot_pb_ab = np.einsum('ij,j->i', vector_pb, vector_ab)
        is_on_segment = (dot_pa_ab * dot_pb_ab <= 0)

        within_radius = (perpendicular_distances <= radius)
        valid_points = is_on_segment & within_radius

        start_points_distance = np.linalg.norm(grid_points - point_a, axis=1)
        start_points_within_radius = start_points_distance <= radius
        end_points_distance = np.linalg.norm(grid_points - point_b, axis=1)
        end_points_within_radius = end_points_distance <= radius

        valid_points |= start_points_within_radius
        valid_points |= end_points_within_radius

        valid_grid_points = grid_points[valid_points].astype(np.int32)
        masks[valid_grid_points[:, 0], valid_grid_points[:, 1], valid_grid_points[:, 2]] = actin_id

def process_actin_to_mask(json_path):
    """Read JSON and MRC files, generate a corresponding mask file."""
    # 1. Read the JSON file
    with open(json_path, "r") as f:
        actins = json.load(f)

    # Extract tomo_name and MRC file path
    tomo_name = os.path.splitext(os.path.basename(json_path))[0]
    tomo_name = tomo_name.replace("_point", "")
    mrc_path = '/media/liushuo/data1/data/synapse_seg/pp0312/pp0312.mrc'

    # 2. Read the MRC file to get z, y, x dimensions
    with mf.open(mrc_path, permissive=True) as mrc:
        z, y, x = mrc.data.shape

    # 3. Create a zero-filled array of size (z, y, x) in int16 format
    masks = np.zeros((z, y, x), dtype=np.int16)

    # 4. Process each actin, generate local masks, and merge them
    for actin in actins:
        actin_id = actin["id"]
        print(f"Processing actin {actin_id}...")
        seedlist = actin["points"]  # points 是 (z, y, x) 的列表

        # 直接写入当前actin到全局mask（覆盖语义与原 np.where 一致）
        generate_actin_mask(masks, seedlist, actin_id, radius=2)

    # 5. Save the mask as an MRC file
    output_path = os.path.join(os.path.dirname(json_path), f"{tomo_name}_label.mrc")
    
    with mf.new(output_path, overwrite=True) as mrc:
        data = masks.astype(np.int16)
        mrc.set_data(data)
        mrc.voxel_size = 17.14  # Set voxel size if required

    print(f"Mask saved to {output_path}")

# Example call
json_path = f"/media/liushuo/data1/data/synapse_seg/pp0312/actin/binary_instances.json"
process_actin_to_mask(json_path)
