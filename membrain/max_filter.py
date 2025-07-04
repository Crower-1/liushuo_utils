import numpy as np
from skimage.measure import label, regionprops
import mrcfile

def read_mrc_file(mrc_path):
    """
    读取.mrc文件并提取所有不为零的点坐标及其obj_id
    返回格式：[obj_id, 0, x, y, z]
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        shape = data.shape
        non_zero_points = np.argwhere(data != 0)  # 获取非零点的索引 (z, y, x)
        converted_points = non_zero_points[:, [2, 1, 0]]
        obj_id = data[non_zero_points[:, 0], non_zero_points[:, 1], non_zero_points[:, 2]]  # 获取对应的obj_id
        # 返回[obj_id, 0, x, y, z]格式
        points = np.hstack([obj_id[:, np.newaxis], np.zeros((obj_id.shape[0], 1)), converted_points])
        return points, shape

def max_filter(unfiltered):
    '''
    修复冲突，针对相同轮廓中的点（具有相同的x或y坐标）进行均值调整。
    '''
    filtered = []
    contours = []
    obj, _, _, _, _ = unfiltered[0]
    
    # 按照z坐标分割轮廓
    for z in sorted(list(set(unfiltered[:, -1].tolist()))):
        contours.append(unfiltered[unfiltered[:, -1] == z])
    
    for i, contour in enumerate(contours):
        idx = i + 1
        point_x_set = sorted(list(set(contour[:, 2].tolist())))
        point_y_set = sorted(list(set(contour[:, 3].tolist())))
        x_diff = max(point_x_set) - min(point_x_set)
        y_diff = max(point_y_set) - min(point_y_set)
        
        if x_diff < y_diff:  # 垂直膜，沿x轴均值
            length = int(x_diff + 5)
            for y in point_y_set:
                point_equ_y = contour[contour[:, 3] == y]
                obj, _, _, y, z = point_equ_y[0]
                idxs = point_equ_y[:, 2]
                idxs_mean = avg_for_1d(idxs, length)
                for x in idxs_mean:
                    filtered.append([obj, idx, x, y, z])
        else:  # 水平膜，沿y轴均值
            length = int(y_diff + 5)
            for x in point_x_set:
                point_equ_x = contour[contour[:, 2] == x]
                obj, _, x, _, z = point_equ_x[0]
                idxs = point_equ_x[:, 3]
                idxs_mean = avg_for_1d(idxs, length)
                for y in idxs_mean:
                    filtered.append([obj, idx, x, y, z])

    filtered = np.array(filtered)
    return filtered

def avg_for_1d(idxs, length):
    '''
    average for 1d slice of x(or y)
    '''
    arr = np.zeros((length, )).astype(np.int16)
    idxs_from0 = (np.array(idxs) - min(idxs)).astype(np.int16)
    arr[idxs_from0] = 1
    lbl = label(arr)
    regions = regionprops(np.stack([lbl, lbl]))
    idxs_mean = []
    for r in regions:
        rx = r.centroid
        idxs_mean.append(custom_round(rx[1] + min(idxs)))
    
    return idxs_mean

def custom_round(x, base=0.5):
    '''
    set coordinate to the nearest 0.5
    '''
    return np.round(x / base) * base

def save_as_mrc(filtered_points, output_mrc_path, shape):
    """
    将过滤后的坐标转化为3D数组，并保存为.mrc文件。
    """
    # 初始化一个空的3D数组（假设 shape 是 (z, y, x)）
    mrc_data = np.zeros(shape, dtype=np.float32)
    
    # 将filtered_points中的坐标转化为3D数组中的值
    for obj_id, _, x, y, z in filtered_points:
        x, y, z = int(x), int(y), int(z)
        mrc_data[z, y, x] = obj_id
    
    # 保存为.mrc文件
    with mrcfile.new(output_mrc_path, overwrite=True) as mrc:
        mrc.set_data(mrc_data)
        mrc.voxel_size = 17.14

# 主程序
def process_mrc(mrc_input_path, mrc_output_path):
    # 读取.mrc文件
    unfiltered_points, shape = read_mrc_file(mrc_input_path)
    
    # 调用max_filter处理数据
    filtered_points = max_filter(unfiltered_points)
    
    # 将处理结果保存为.mrc文件
    save_as_mrc(filtered_points, mrc_output_path, shape)

# 运行示例
mrc_input_path = "/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1776/membrane/pp1776_membrane_label.mrc"
mrc_output_path = "/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1776/membrane/max_filter.mrc"
# shape = (80, 3072, 3072)  # 假设的3D形状，根据实际情况调整
process_mrc(mrc_input_path, mrc_output_path)
