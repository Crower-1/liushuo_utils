import numpy as np
import mrcfile as mf
import pandas as pd
import os
import subprocess

def generate_mt_mask(mrc_path, mod_path, radius=7):
    """
    根据给定的 .mrc 图像和 .mod 模型文件生成 MTmask_instance.mrc 掩码文件。
    
    参数:
    - mrc_path: str, 输入的 .mrc 图像文件路径。
    - mod_path: str, 输入的 .mod 模型文件路径。
    - radius: int, 掩码半径（默认为7）。
    
    输出:
    - 在输入的 .mrc 图像文件所在目录生成 MTmask_instance.mrc 文件。
    """
    
    def distance(A, B):
        return np.linalg.norm(A - B)
    
    # 确认输入文件存在
    if not os.path.isfile(mrc_path):
        raise FileNotFoundError(f"指定的 MRC 文件不存在: {mrc_path}")
    if not os.path.isfile(mod_path):
        raise FileNotFoundError(f"指定的 MOD 文件不存在: {mod_path}")
    
    # 提取 MRC 文件尺寸
    try:
        header_cmd = f"header -size {mrc_path}"
        size_output = subprocess.check_output(header_cmd, shell=True).decode().split()
        size = list(map(int, size_output))
        if len(size) < 3:
            raise ValueError("无法从 header 命令获取尺寸信息。")
        nx, ny, nz = size[:3]
    except Exception as e:
        raise RuntimeError(f"获取 MRC 文件尺寸失败: {e}")
    
    # 初始化掩码数组
    mask = np.zeros((nz, ny, nx), dtype=np.int32)
    
    # 将 .mod 文件转换为 .coords 文件
    coords_path = os.path.splitext(mod_path)[0] + ".coords"
    try:
        model2point_cmd = f"model2point {mod_path} {coords_path}"
        subprocess.check_call(model2point_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"执行 model2point 命令失败: {e}")
    
    # 读取坐标点
    try:
        points = pd.read_csv(coords_path, header=None, sep='\s+', dtype=np.float32).values
        points = points.astype(np.int32)
    except Exception as e:
        raise RuntimeError(f"读取坐标文件失败: {e}")
    
    # 将点集划分为多条线（假设每条线为连续点序列）
    # 这里简单假设所有连续点距离小于200为同一条线
    lines = []
    current_line = [points[0]]
    
    for i in range(1, len(points)):
        if distance(points[i], points[i-1]) < 200:
            current_line.append(points[i])
        else:
            lines.append(np.array(current_line))
            current_line = [points[i]]
    lines.append(np.array(current_line))  # 添加最后一条线
    
    print(f"共检测到 {len(lines)} 条线。")
    
    # 遍历每条线并生成掩码
    for line_idx, line in enumerate(lines, start=1):
        print(f"处理第 {line_idx} 条线，共有 {len(line)} 个点。")
        for i in range(len(line)-1):
            A = line[i]
            B = line[i+1]
            if distance(A, B) < 200:
                # 打印当前处理的点对的 Y 坐标
                print(f"线 {line_idx}: 点A Y={A[1]}, 点B Y={B[1]}")
                z_min = max(min(A[2], B[2]) - 13, 0)
                z_max = min(max(A[2], B[2]) + 13, nz)
                y_min = max(min(A[1], B[1]) - 13, 0)
                y_max = min(max(A[1], B[1]) + 13, ny)
                x_min = max(min(A[0], B[0]) - 13, 0)
                x_max = min(max(A[0], B[0]) + 13, nx)
                
                for z in range(z_min, z_max):
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            P = np.array([x, y, z])
                            PA = A - P
                            PB = B - P
                            AB = B - A
                            h = np.linalg.norm(np.cross(PA, PB)) / distance(A, B)
                            # 检查点 P 是否在 AB 线段上，并且距离小于等于 radius
                            if (np.dot(PA, AB) * np.dot(PB, AB) <= 0) and (h <= radius):
                                if mask[z, y, x] < line_idx:
                                    mask[z, y, x] = line_idx
                            else:
                                if distance(P, B) <= radius:
                                    if mask[z, y, x] < line_idx:
                                        mask[z, y, x] = line_idx
    
    # 保存掩码到 MRC 文件
    output_path = os.path.join(os.path.dirname(mrc_path), "MTmask_instance.mrc")
    try:
        with mf.new(output_path, overwrite=True) as mrc:
            mrc.set_data(mask.astype(np.float32))
        print(f"掩码文件已保存到: {output_path}")
    except Exception as e:
        raise RuntimeError(f"保存 MRC 文件失败: {e}")

# 示例调用
if __name__ == "__main__":
    # 示例路径，请根据实际情况修改
    mrc_file_path = "/home/liushuo/Documents/data/mt_demo/pp734/pp734.mrc"
    mod_file_path = "/home/liushuo/Documents/data/mt_demo/pp734/MT.mod"
    generate_mt_mask(mrc_file_path, mod_file_path, radius=7)
