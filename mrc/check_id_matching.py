import argparse
import json
import mrcfile
import os
import sys

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="检查MRC文件中mask与JSON文件中vesicle信息的匹配程度。")
#     parser.add_argument("mrc_path", type=str, default='/home/liushuo/Documents/data/stack-out_demo/p1/ves_seg/temp/label_2479082.mrc' , help="带mask的MRC文件路径")
#     parser.add_argument("json_path", type=str, default='/home/liushuo/Documents/data/stack-out_demo/p1/ves_seg/temp/vesicle_new_2479082.json', help="对应mask信息的JSON文件路径")
#     return parser.parse_args()

def parse_arguments():
    parser = argparse.ArgumentParser(description="检查MRC文件中mask与JSON文件中vesicle信息的匹配程度。")
    parser.add_argument("--mrc_path", type=str, default='/home/liushuo/Documents/data/stack-out_demo/pp1269/ves_seg/pp1269_label_vesicle.mrc' , help="带mask的MRC文件路径")
    parser.add_argument("--json_path", type=str, default='/home/liushuo/Documents/data/stack-out_demo/pp1269/ves_seg/pp1269_vesicle.json', help="对应mask信息的JSON文件路径")
    return parser.parse_args()

def load_mrc(mrc_path):
    if not os.path.isfile(mrc_path):
        print(f"错误：MRC文件 '{mrc_path}' 不存在。")
        sys.exit(1)
    try:
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data.copy()
        return data
    except Exception as e:
        print(f"错误：无法读取MRC文件 '{mrc_path}'。详细信息：{e}")
        sys.exit(1)

def load_json(json_path):
    if not os.path.isfile(json_path):
        print(f"错误：JSON文件 '{json_path}' 不存在。")
        sys.exit(1)
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件 '{json_path}' 格式不正确。详细信息：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：无法读取JSON文件 '{json_path}'。详细信息：{e}")
        sys.exit(1)

def extract_id(vesicle_name):
    try:
        # 假设名称格式为 'vesicle_<ID>'
        return int(vesicle_name.split('_')[-1])
    except (IndexError, ValueError):
        print(f"警告：无法从名称 '{vesicle_name}' 中提取ID。")
        return None

def check_matching(mrc_data, vesicles):
    total = 0
    matched = 0
    mismatched = 0
    mismatched_vesicles = []

    z_dim, y_dim, x_dim = mrc_data.shape

    for vesicle in vesicles:
        name = vesicle.get("name", "")
        vesicle_id = extract_id(name)
        if vesicle_id is None:
            continue

        center = vesicle.get("center", [])
        if len(center) != 3:
            print(f"警告：vesicle '{name}' 的中心坐标不完整。")
            continue

        # 将浮点坐标转换为整数索引
        z, y, x = [int(round(coord)) for coord in center]

        # 检查索引是否在范围内
        if not (0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim):
            print(f"警告：vesicle '{name}' 的中心坐标 ({z}, {y}, {x}) 超出MRC数据范围。")
            mismatched += 1
            mismatched_vesicles.append(name)
            total += 1
            continue

        voxel_value = mrc_data[z, y, x]
        if voxel_value == vesicle_id:
            matched += 1
        else:
            mismatched += 1
            mismatched_vesicles.append(name)
        total += 1

    return total, matched, mismatched, mismatched_vesicles

def main():
    args = parse_arguments()
    mrc_data = load_mrc(args.mrc_path)
    json_data = load_json(args.json_path)

    vesicles = json_data.get("vesicles", [])
    if not vesicles:
        print("警告：JSON文件中没有找到 'vesicles' 信息。")
        sys.exit(1)

    total, matched, mismatched, mismatched_vesicles = check_matching(mrc_data, vesicles)

    if total == 0:
        print("没有可检查的vesicles。")
        sys.exit(0)

    matched_percent = (matched / total) * 100
    mismatched_percent = (mismatched / total) * 100

    print("匹配统计信息：")
    print(f"总数：{total}")
    print(f"匹配数量：{matched} ({matched_percent:.2f}%)")
    print(f"不匹配数量：{mismatched} ({mismatched_percent:.2f}%)")

    if mismatched > 0:
        print("\n不匹配的vesicles列表：")
        for name in mismatched_vesicles:
            print(f"- {name}")

if __name__ == "__main__":
    main()