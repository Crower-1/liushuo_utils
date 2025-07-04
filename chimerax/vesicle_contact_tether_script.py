import xml.etree.ElementTree as ET
import numpy as np
from chimerax.core.commands import run


# 解析 XML 文件，读取囊泡的 Type、Center 和 Radius
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    vesicles = []

    # 获取像素大小
    pixel_size = float(root.attrib.get('pixelSize', 1.0))

    for vesicle in root.findall('Vesicle'):
        vesicle_data = {}

        # 获取 Type
        vesicle_type = vesicle.find('Type').attrib.get('t', '')
        if vesicle_type not in ['tether', 'contact']:  # 只保留tether和contact类型的囊泡
            continue
        vesicle_data['Type'] = vesicle_type

        # 获取 Center
        center = vesicle.find('Center')
        vesicle_data['Center'] = {
            'X': float(center.attrib.get('X', 0)),
            'Y': float(center.attrib.get('Y', 0)),
            'Z': float(center.attrib.get('Z', 0))
        }

        # 获取 Radius
        radius = vesicle.find('Radius')
        vesicle_data['Radius'] = float(radius.attrib.get('r', 0)) * 1.714  # 更新半径计算

        vesicles.append(vesicle_data)

    return vesicles, pixel_size


# 根据囊泡类型返回 RGB 颜色
def get_RGB(vesicle_type):
    if vesicle_type == 'tether':
        return [0, 0, 100]  # 蓝色
    elif vesicle_type == 'contact':
        return [100, 0, 0]  # 红色
    return [0, 0, 0]  # 默认值（黑色）


# 创建标记和颜色
def create_markers_and_color(xml_path):
    vesicles, scale = parse_xml(xml_path)
    scale = 17.14

    for i, vesicle in enumerate(vesicles):
        vesicle_type = vesicle['Type']
        x, y, z = vesicle['Center']['X'], vesicle['Center']['Y'], vesicle['Center']['Z']
        r = vesicle['Radius']
        rgb = get_RGB(vesicle_type)

        # 偏移坐标并缩放
        x = (x - 1) * scale
        y = (y - 1) * scale
        z = (z - 1) * scale

        # 创建标记命令
        marker_command = f'marker #{(i+1)*2} radius 10 position {x}, {y}, {z} color {rgb[0]},{rgb[1]},{rgb[2]}'
        run(session, marker_command)

        # 着色命令
        color_zone_command = f'color zone #{1} near #{(i+1)*2} distance {(r+10) * 10}'
        run(session, color_zone_command)

        # 分割命令
        split_command = f'vop splitbyzone #{1}'
        run(session, split_command)

        # 调整体数据等级命令
        level_command = f'volume #{(i+1)*2+1}.2 level 50'
        run(session, level_command)

        # 关闭命令
        close_command = f'close #{(i+1)*2+1}.1'
        run(session, close_command)


# 执行标记和颜色创建
create_markers_and_color('/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/vesicle_analysis/pp0039_vesicle_class.xml')
