import os
import argparse
import json
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='在目标文件夹中生成工作文件夹子文件夹的软链接。')
    parser.add_argument('work_folder', type=str, help='工作文件夹路径')
    parser.add_argument('target_folder', type=str, help='目标文件夹路径')
    return parser.parse_args()

def load_json(json_path):
    if not os.path.exists(json_path):
        print(f"JSON文件不存在: {json_path}")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"读取JSON文件出错: {e}")
            return {}

def create_symlink(src, dest):
    try:
        if os.path.islink(dest) or os.path.exists(dest):
            os.remove(dest)  # 如果目标文件已存在，先删除
        os.symlink(src, dest)
        print(f"创建软链接: {dest} -> {src}")
    except OSError as e:
        print(f"无法创建软链接 {dest} -> {src}: {e}")

def main():
    args = parse_arguments()
    work_folder = os.path.abspath(args.work_folder)
    target_folder = os.path.abspath(args.target_folder)

    # 检查工作文件夹是否存在
    if not os.path.isdir(work_folder):
        print(f"工作文件夹不存在: {work_folder}")
        sys.exit(1)

    # 创建目标子文件夹 mrc_img 和 mrc_label
    mrc_img_folder = os.path.join(target_folder, 'mrc_img')
    mrc_label_folder = os.path.join(target_folder, 'mrc_label')
    os.makedirs(mrc_img_folder, exist_ok=True)
    os.makedirs(mrc_label_folder, exist_ok=True)

    # 加载 JSON 文件
    json_path = os.path.join(work_folder, 'segVesicle_heart_broken.json')
    broken_dict = load_json(json_path)

    # 遍历工作文件夹下的所有子文件夹
    for tomo_name in os.listdir(work_folder):
        tomo_path = os.path.join(work_folder, tomo_name)
        if not os.path.isdir(tomo_path):
            continue  # 跳过非文件夹

        # 检查 JSON 中是否标记为 true
        if broken_dict.get(tomo_name, False):
            print(f"跳过已标记为true的子文件夹: {tomo_name}")
            continue

        # 计算 base_tomo_name
        if '-1' in tomo_name:
            base_tomo_name = tomo_name.split('-1')[0]
        else:
            base_tomo_name = tomo_name

        # 构建图像和标签文件路径
        isonet_tomo_path = os.path.join(tomo_path, 'ves_seg', f"{base_tomo_name}_wbp_corrected.mrc")
        label_path = os.path.join(tomo_path, 'ves_seg', f"{base_tomo_name}_label_vesicle.mrc")

        # 检查文件是否存在
        if not os.path.isfile(isonet_tomo_path):
            print(f"图像文件不存在: {isonet_tomo_path}, 跳过 {tomo_name}")
            continue
        if not os.path.isfile(label_path):
            print(f"标签文件不存在: {label_path}, 跳过 {tomo_name}")
            continue

        # 目标软链接路径
        img_link = os.path.join(mrc_img_folder, f"{tomo_name}.mrc")
        label_link = os.path.join(mrc_label_folder, f"{tomo_name}.mrc")

        # 创建软链接
        create_symlink(isonet_tomo_path, img_link)
        create_symlink(label_path, label_link)

if __name__ == "__main__":
    main()
