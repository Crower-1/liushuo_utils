import os
import json
import numpy as np
import mrcfile

# 1. 读取 .mrc 文件
def load_mrc(file_path):
    with mrcfile.open(file_path, 'r') as mrc:
        return mrc.data

# 2. 根据平均半径计算 RGB 值
def get_RGB(radius):
    scale = 1.714
    radius = radius * scale
    median = 41 / 2
    mi = 25 / 2
    ma = 55 / 2
    precise = 3
    if radius <= median:
        b = 0
        r = 1 - (radius - mi) / (median - mi)
        g = (radius - mi) / (median - mi)
    else:
        r = 0
        g = 1 - (radius - median) / (ma - median)
        b = (radius - median) / (ma - median)
    r, g, b = r * 100, g * 100, b * 100
    return [np.round(r, precise), np.round(g, precise), np.round(b, precise)]

# 3. 将 mrc 中对应 mask 转换为 CMM 格式字符串
def mrc_mask_to_cmm(mrc_mask_data, rgb, scale_factor=17.14):
    r, g, b = rgb
    # mrc_mask_data 为布尔数组，np.where 找到所有 True 的坐标
    z, y, x = np.where(mrc_mask_data)
    cmm_data = '<marker_set name="markers">\n'
    for idx, (zi, yi, xi) in enumerate(zip(z, y, x), start=1):
        # 坐标乘以 scale_factor 并保留两位小数
        cmm_data += f'<marker id="{idx}" x="{xi * scale_factor:.2f}" y="{yi * scale_factor:.2f}" z="{zi * scale_factor:.2f}" r="{r}" g="{g}" b="{b}" radius="10"/>\n'
    cmm_data += '</marker_set>\n'
    return cmm_data

# 4. 保存 CMM 文件
def save_cmm(cmm_data, output_path):
    with open(output_path, 'w') as f:
        f.write(cmm_data)

# 主函数
def main(mrc_path, json_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取 mrc 数据
    mrc_data = load_mrc(mrc_path)
    
    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        vesicle_info = json.load(f)
    
    # 遍历每个囊泡
    for vesicle in vesicle_info.get("vesicles", []):
        # 从名称中提取 id（假设格式为 vesicle_数字）
        try:
            vesicle_id = int(vesicle["name"].split("_")[-1])
        except Exception as e:
            print(f"无法解析囊泡名称 {vesicle['name']}：{e}")
            continue
        
        # 计算 radii 的平均值
        radii = vesicle.get("radii", [])
        if not radii:
            print(f"囊泡 {vesicle['name']} 未提供 radii 信息，跳过。")
            continue
        avg_radius = np.mean(radii)
        
        # 通过平均半径计算 RGB
        rgb = get_RGB(avg_radius)
        
        # 从 mrc_data 中提取对应 mask（假设 mask 值等于 vesicle_id）
        mask_sub = (mrc_data == vesicle_id)
        # 检查是否有对应的 mask
        if np.sum(mask_sub) == 0:
            print(f"在 mrc 数据中找不到囊泡 {vesicle['name']} (id: {vesicle_id}) 的 mask，跳过。")
            continue
        
        # 转换 mask 为 CMM 格式字符串
        cmm_data = mrc_mask_to_cmm(mask_sub, rgb)
        
        # 构造输出文件路径，例如：output_folder/vesicle_10.cmm
        output_file = os.path.join(output_folder, f"vesicle_{vesicle_id}.cmm")
        save_cmm(cmm_data, output_file)
        print(f"保存 {output_file}")

if __name__ == "__main__":
    # 示例输入路径，请根据实际情况修改
    mrc_path = "/media/liushuo/data1/data/synapse_seg/pp1776/ves_seg/pp1776_label_vesicle.mrc"
    json_path = "/media/liushuo/data1/data/synapse_seg/pp1776/ves_seg/pp1776_vesicle.json"
    output_folder = "/media/liushuo/data1/data/synapse_seg/pp1776/vesicle/cmm"
    
    main(mrc_path, json_path, output_folder)
