import os
import xml.etree.ElementTree as ET
import numpy as np
import mrcfile
from matplotlib import cm
from tqdm import tqdm

# 1. 读取 .mrc 文件
def load_mrc(file_path):
    with mrcfile.open(file_path, 'r') as mrc:
        return mrc.data

# 2. 根据平均半径计算 RGB 值
def get_RGB(radius, pixel_size, mi=25/2, ma=55/2, precise=3):
    """
    将 radius 映射到 [mi, ma] 范围，线性插值生成浅绿->深绿的颜色。
    
    参数：
        radius: 原始半径值（未缩放）
        pixel_size: 像素大小（缩放因子）
        mi: 最小参考半径（默认 25/2）
        ma: 最大参考半径（默认 55/2）
        precise: 小数精度
    
    返回：
        [r, g, b]，范围 [0,100]，保留 precise 位小数
    """
    # 缩放
    r_scaled = radius * pixel_size
    
    # 截断到 [mi, ma]
    if r_scaled < mi:
        r_scaled = mi
    elif r_scaled > ma:
        r_scaled = ma
    
    # 归一化到 [0,1]
    t = (r_scaled - mi) / (ma - mi)
    
    # 浅绿和深绿在 [0,1] 空间的 RGB
    light = np.array([0.56, 0.93, 0.56])
    dark  = np.array([0.00, 0.39, 0.00])
    
    # 插值
    rgb_norm = light * (1 - t) + dark * t  # t=0 时浅绿，t=1 时深绿
    
    # 转到 [0,100] 并四舍五入
    rgb = np.round(rgb_norm * 100, precise)
    return rgb.tolist()

# 3. 将 mrc 中对应 mask 转换为 CMM 格式字符串
def mrc_mask_to_cmm(mrc_mask_data, rgb, scale_factor, radius_marker=10):
    scale_factor = scale_factor * 10  # 将 nm 转换为 A
    r, g, b = rgb
    z, y, x = np.where(mrc_mask_data)
    cmm_data = ['<marker_set name="markers">']
    for idx, (zi, yi, xi) in enumerate(zip(z, y, x), start=1):
        xw = xi * scale_factor
        yw = yi * scale_factor
        zw = zi * scale_factor
        cmm_data.append(
            f'  <marker id="{idx}" x="{xw:.2f}" y="{yw:.2f}" z="{zw:.2f}" '
            f'r="{r}" g="{g}" b="{b}" radius="{radius_marker}"/>'
        )
    cmm_data.append('</marker_set>')
    return '\n'.join(cmm_data)

# 4. 保存 CMM 文件
def save_cmm(cmm_data, output_path):
    with open(output_path, 'w') as f:
        f.write(cmm_data)

# 5. 从 XML 读取囊泡列表
def parse_vesicles_from_xml(xml_path):
    """
    返回：
      pixel_size: float
      vesicles: list of dict, each with keys 'id' (int) 和 'radius' (float)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 根节点 <VesicleList pixelSize="...">
    pixel_size = float(root.attrib.get('pixelSize', 1.714))
    vesicles = []
    for ves in root.findall('Vesicle'):
        vid = int(ves.attrib.get('vesicleId', -1))
        # 只处理 Type t="vesicle"
        t = ves.find('Type')
        if t is None or t.attrib.get('t') != 'vesicle':
            continue
        # Radius2D
        r2d = ves.find('Radius2D')
        if r2d is None:
            continue
        r1 = float(r2d.attrib.get('r1', 0))
        r2 = float(r2d.attrib.get('r2', 0))
        avg_r = (r1 + r2) / 2.0
        vesicles.append({'id': vid, 'radius': avg_r})
    return pixel_size, vesicles

# 主函数
def main(mrc_path, xml_path, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 读取 mrc 数据
    mrc_data = load_mrc(mrc_path)

    # 解析 XML
    pixel_size, vesicles = parse_vesicles_from_xml(xml_path)
    if not vesicles:
        print("未在 XML 中找到任何有效的 vesicle 条目。")
        return

    # 遍历并处理
    for ves in tqdm(vesicles, desc="Processing vesicles"):
        vid = ves['id']
        avg_r = ves['radius']
        rgb = get_RGB(avg_r, pixel_size)

        mask = (mrc_data == vid)
        if not mask.any():
            tqdm.write(f"Warning: 在 mrc 数据中未找到 id={vid} 的 mask，跳过。")
            continue

        cmm = mrc_mask_to_cmm(mask, rgb, scale_factor=pixel_size)
        out_file = os.path.join(output_folder, f"vesicle_{vid}.cmm")
        save_cmm(cmm, out_file)
        tqdm.write(f"Saved {out_file}")

if __name__ == "__main__":
    # 示例路径，请按需修改
    mrc_path = "/media/liushuo/data1/data/synapse_seg/pp1776/ves_seg/pp1776_label_vesicle.mrc"
    xml_path = "/media/liushuo/data1/data/fig_demo/pp1776/ves_seg/vesicle_analysis/pp1776_vesicle_class.xml"
    output_folder = "/media/liushuo/data1/data/fig_demo/pp1776/vesicle/cmm_green"
    main(mrc_path, xml_path, output_folder)
