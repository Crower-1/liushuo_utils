import numpy as np
import subprocess
import os
import random

pixel_size = 21.75 / 2
file_name = "pp370.mod"
start_id = 2
level_density = 0.159

def Read_coords(file_name: str, pixel_size: float):
    """
    将 .mod 文件转换为 .txt，读取像素坐标并换算为物理坐标。

    参数:
    -----------
    file_name : str
        要处理的 .mod 文件路径，如 "example.mod"。
    pixel_size : float
        每个像素对应的物理长度单位（例如毫米/mm）。

    返回:
    -----------
    coords_phys : list of tuple
        形如 [(x1, y1, z1), (x2, y2, z2), ...] 的物理坐标列表。
    """

    # 1. 构造新的文件名
    if not file_name.endswith('.mod'):
        raise ValueError("输入文件名必须以 .mod 结尾")
    file_name_new = file_name.replace('.mod', '.txt')

    # 2. 调用外部命令 model2point 将 .mod 转为 .txt
    #    model2point <input.mod> <output.txt>
    try:
        subprocess.check_call(['model2point', file_name, file_name_new])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"执行 model2point 失败：{e}")

    # 3. 读取生成的 .txt 文件
    if not os.path.exists(file_name_new):
        raise FileNotFoundError(f"转换后文件未找到：{file_name_new}")

    coords_phys = []
    with open(file_name_new, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行是三个以空格分隔的浮点数
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"意外的行格式：'{line}'")
            x_px, y_px, z_px = map(float, parts)

            # 4. 转换为物理坐标
            x_phys = x_px * pixel_size - 696
            y_phys = y_px * pixel_size - 696
            z_phys = z_px * pixel_size - 696

            coords_phys.append((x_phys, y_phys, z_phys))

    # 5. 返回物理坐标列表
    return coords_phys

def generate_cxc_file(coords_phys, file_name, level_density, start_id=2):
    n = start_id
    tomo_name = file_name.replace('.mod', '')
    with open(tomo_name + ".cxc", "w") as tilt_line:
        for coord in coords_phys:
            n = n + 1   
            
            tilt_line.write('open cage_2175.mrc' + '\n')
            rot = random.uniform(-30, 30)
            tilt = random.uniform(-30, 30)
            psi = random.uniform(-30, 30)
                  
            #tilt_line.write('color #ffff0ea115f1 #'+ str(n) + '\n')
            #volume #1 level 0.344
            # 写入指令
            tilt_line.write(f"#volume #{n} level {level_density} \n")
            tilt_line.write(
                f"volume #{n} color #b2b2ff\n"
            )
            tilt_line.write(
                f"turn 0,0,1 {rot:.2f} center 696,696,696 model #{n} coord #{n}\n"
            )
            tilt_line.write(
                f"turn 0,1,0 {tilt:.2f} center 696,696,696 model #{n} coord #{n}\n"
            )
            tilt_line.write(
                f"turn 0,0,1 {psi:.2f} center 696,696,696 model #{n} coord #{n}\n"
            )
            tilt_line.write(
                f"move {coord[0]:.2f},{coord[1]:.2f},{coord[2]:.2f} model #{n}\n"
            )
                
     
if __name__ == "__main__":           
    coords_phys = Read_coords(file_name, pixel_size)
    print(coords_phys)
    generate_cxc_file(coords_phys, file_name, level_density, start_id)