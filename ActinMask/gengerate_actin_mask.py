import numpy as np
import mrcfile as mf
import pandas as pd
import os

def d(A, B):
    """计算两个点A和B之间的欧氏距离。"""
    return np.linalg.norm(A - B)

radius = 3  # 像素
distance_threshold = 50  # 用于区分不同纤维的距离阈值
bounding_box_extension = 7  # 用于定义搜索体积的扩展范围

# 获取.mrc文件的尺寸
size_output = os.popen('header -size ./pp*.mrc').read().split()
size_z, size_y, size_x = int(size_output[2]), int(size_output[1]), int(size_output[0])
m = np.zeros((size_z, size_y, size_x)).astype(np.int16)  # 使用int16以支持多个标识符

# 将actin.mod转换为坐标文件actin.coords
os.system('model2point actin.mod actin.coords')

# 读取坐标数据
points = pd.read_csv('actin.coords', header=None, sep='\s+', dtype=np.float32).values

filament_number = 1  # 初始化纤维编号
current_filament = filament_number  # 当前纤维编号

for i in range(len(points) - 1):
    A = points[i].astype(np.int32)
    B = points[i + 1].astype(np.int32)
    distance = d(A, B)
    
    if distance < distance_threshold:
        # 当前点对属于同一根纤维
        for z in range(min(A[2],B[2])-7,max(A[2],B[2])+7):
            for y in range(min(A[1],B[1])-7,max(A[1],B[1])+7):
                for x in range(min(A[0],B[0])-7,max(A[0],B[0])+7):
        # for z in range(z_min, z_max):
        #     for y in range(y_min, y_max):
        #         for x in range(x_min, x_max):
                    P = np.array([x, y, z]).astype(np.int32)
                    PA = A - P
                    PB = B - P
                    AB = B - A
                    cross_product = np.cross(PA, PB)
                    h = np.linalg.norm(cross_product) / d(A, B) if d(A, B) != 0 else 0
                    
                    # 判断P是否在AB线段上或附近
                    if np.dot(PA, AB) * np.dot(PB, AB) <= 0 and h <= radius:
                        m[z, y, x] = current_filament
                    else:
                        if d(P, B) <= radius:
                            m[z, y, x] = current_filament
        # pass  # current_filament 保持不变
    else:
        # 当前点对不属于同一根纤维，开始新的纤维
        filament_number += 1
        current_filament = filament_number
    

# 保存3D掩模为.mrc文件
with mf.new('actinmask_instance.mrc', overwrite=True) as mrc:
    mrc.set_data(m)
