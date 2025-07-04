import json

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 将 JSON 数据转换为 CMM 格式
def json_to_cmm(json_data):
    cmm_data = '<marker_set name="markers">\n'
    vesicles = json_data.get("vesicles", [])
    
    for idx, vesicle in enumerate(vesicles, start=1):
        center = vesicle.get("center", [])
        if len(center) == 3:
            z, y, x = center
            z = z * 17.14
            y = y * 17.14
            x = x * 17.14
            cmm_data += f'<marker id="{idx}" x="{x:.2f}" y="{y:.2f}" z="{z:.2f}" r="1" g="1" b="0" radius="300"/>\n'

    cmm_data += '</marker_set>\n'
    return cmm_data

# 保存 CMM 文件
def save_cmm(cmm_data, output_path):
    with open(output_path, 'w') as f:
        f.write(cmm_data)

# 主函数
def convert_json_to_cmm(json_path, cmm_output_path):
    json_data = load_json(json_path)
    cmm_data = json_to_cmm(json_data)
    save_cmm(cmm_data, cmm_output_path)
    print(f"Converted CMM file saved to: {cmm_output_path}")

# 示例使用
json_path = '/media/liushuo/data1/data/fig_demo_2/pp199/synapse_seg/pp199/actin/pp199_actin_label.mrc'  # 替换为实际的 JSON 文件路径
cmm_output_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1776/vesicle/output.cmm'  # 替换为输出的 CMM 文件路径

convert_json_to_cmm(json_path, cmm_output_path)
