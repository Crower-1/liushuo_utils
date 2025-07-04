import os
import xml.etree.ElementTree as ET
import numpy as np
import mrcfile
from tqdm import tqdm

# 1. 读取 .mrc 文件
def load_mrc(file_path):
    with mrcfile.open(file_path, 'r') as mrc:
        return mrc.data

# 2. 根据平均半径计算 RGB 值（浅绿->深绿）
def get_RGB(radius, pixel_size, mi, ma, precise=2):
    """
    将 radius 映射到 [mi, ma] 范围，线性插值生成浅绿->深绿的颜色。
    
    参数：
        radius: 原始半径值（未缩放）
        pixel_size: 像素大小（缩放因子）
        mi: 所有囊泡的最小缩放后半径
        ma: 所有囊泡的最大缩放后半径
        precise: 小数精度
    
    返回：
        [r, g, b]，范围 [0,100]，保留 precise 位小数
    """
    # 缩放
    r_scaled = radius * pixel_size
    
    # temp
    mi = 15
    
    # 截断到 [mi, ma]
    r_clipped = np.clip(r_scaled, mi, ma)
    
    # 2. 归一化到 [0,1]
    t = (r_clipped - mi) / (ma - mi) if ma > mi else 0.0
    
    # 3. 定义三点颜色（已在 0-100 空间）
    light = np.array([86, 8,   24], dtype=float)
    middle= np.array([ 49,  99,   0], dtype=float)
    dark  = np.array([  0,  54,  54], dtype=float)
    
    # 4. 分段插值
    if t <= 0.5:
        # 从 light 到 middle
        t2 = t / 0.5
        rgb = light * (1 - t2) + middle * t2
    else:
        # 从 middle 到 dark
        t2 = (t - 0.5) / 0.5
        rgb = middle * (1 - t2) + dark * t2
    
    # 5. 保留精度并返回
    rgb = np.round(rgb, precise)
    return rgb.tolist()

# 3. 将 mrc 中对应 mask 转换为 CMM 格式字符串
def mrc_mask_to_cmm(mrc_mask_data, rgb, scale_factor, radius_marker=10):
    scale_factor = scale_factor * 10  # 将 nm 转换为 Å
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
    pixel_size = float(root.attrib.get('pixelSize', 1.0))
    vesicles = []
    for ves in root.findall('Vesicle'):
        vid = int(ves.attrib.get('vesicleId', -1))
        t = ves.find('Type')
        if t is None or t.attrib.get('t') != 'vesicle':
            continue
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
    os.makedirs(output_folder, exist_ok=True)

    # 读取数据
    mrc_data = load_mrc(mrc_path)
    pixel_size, vesicles = parse_vesicles_from_xml(xml_path)
    if not vesicles:
        print("未在 XML 中找到任何有效的 vesicle 条目。")
        return

    # 计算所有囊泡的缩放后半径列表
    scaled_radii = [v['radius'] * pixel_size for v in vesicles]
    mi = min(scaled_radii)
    ma = max(scaled_radii)

    # 处理每个囊泡
    for ves in tqdm(vesicles, desc="Processing vesicles"):
        vid = ves['id']
        avg_r = ves['radius']
        rgb = get_RGB(avg_r, pixel_size, mi, ma)

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
    mrc_path = "/media/liushuo/data1/data/fig_demo/pp1776/vesicle/pp1776_vesicle_label.mrc"
    xml_path = "/media/liushuo/data1/data/fig_demo/pp1776/ves_seg/vesicle_analysis/pp1776_ori.xml"
    output_folder = "/media/liushuo/data1/data/fig_demo/pp1776/vesicle/cmm_green_all2"
    main(mrc_path, xml_path, output_folder)
