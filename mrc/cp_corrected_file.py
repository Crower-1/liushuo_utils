import os
import shutil

def copy_corrected_mrc(src_dir, dest_dir):
    # 获取目标路径下的所有子文件夹名
    tomo_names = [folder for folder in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, folder))]

    for tomo_name in tomo_names:
        # 构造源文件路径
        src_file = os.path.join(src_dir, tomo_name, 'ves_seg', f'{tomo_name}_wbp_corrected.mrc')

        # 构造目标文件路径
        dest_file = os.path.join(dest_dir, tomo_name, f'{tomo_name}_wbp_corrected.mrc')

        # 检查源文件是否存在
        if os.path.exists(src_file):
            # 确保目标子文件夹存在
            os.makedirs(os.path.join(dest_dir, tomo_name), exist_ok=True)

            # 复制文件到目标路径
            shutil.copy(src_file, dest_file)
            print(f"复制了文件: {src_file} 到 {dest_file}")
        else:
            print(f"源文件不存在: {src_file}")
            
            
def copy_label_mrc(src_dir, dest_dir):
    # 获取目标路径下的所有子文件夹名
    tomo_names = [folder for folder in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, folder))]

    for tomo_name in tomo_names:
        # 构造源文件路径
        src_file = os.path.join(src_dir, tomo_name, 'ves_seg', f'{tomo_name}_label_vesicle.mrc')

        # 构造目标文件路径
        dest_file = os.path.join(dest_dir, tomo_name, f'{tomo_name}_label_vesicle.mrc')

        # 检查源文件是否存在
        if os.path.exists(src_file):
            # 确保目标子文件夹存在
            os.makedirs(os.path.join(dest_dir, tomo_name), exist_ok=True)

            # 复制文件到目标路径
            shutil.copy(src_file, dest_file)
            print(f"复制了文件: {src_file} 到 {dest_file}")
        else:
            print(f"源文件不存在: {src_file}")
            
def copy_json_file(src_dir, dest_dir):
    # 获取目标路径下的所有子文件夹名
    tomo_names = [folder for folder in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, folder))]

    for tomo_name in tomo_names:
        # 构造源文件路径
        src_file = os.path.join(src_dir, tomo_name, 'ves_seg', f'{tomo_name}_vesicle.json')

        # 构造目标文件路径
        dest_file = os.path.join(dest_dir, tomo_name, f'{tomo_name}_vesicle.json')

        # 检查源文件是否存在
        if os.path.exists(src_file):
            # 确保目标子文件夹存在
            os.makedirs(os.path.join(dest_dir, tomo_name), exist_ok=True)

            # 复制文件到目标路径
            shutil.copy(src_file, dest_file)
            print(f"复制了文件: {src_file} 到 {dest_file}")
        else:
            print(f"源文件不存在: {src_file}")
            
def copy_memb_file_back(src_dir, dest_dir):
    # 获取目标路径下的所有子文件夹名
    tomo_names = [folder for folder in os.listdir(src_dir) if os.path.isdir(os.path.join(dest_dir, folder))]

    for tomo_name in tomo_names:
        # 构造源文件路径
        dest_file = os.path.join(dest_dir, tomo_name, 'ves_seg', 'membrane',f'{tomo_name}_memb.mrc')

        # 构造目标文件路径
        src_file = os.path.join(src_dir, tomo_name, f'{tomo_name}_single_membrane_mask_2.mrc')

        # 检查源文件是否存在
        if os.path.exists(src_file):
            # 确保目标子文件夹存在
            os.makedirs(os.path.join(dest_dir, tomo_name, 'ves_seg', 'membrane'), exist_ok=True)

            # 复制文件到目标路径
            shutil.copy(src_file, dest_file)
            print(f"复制了文件: {src_file} 到 {dest_file}")
        else:
            print(f"源文件不存在: {src_file}")

# 示例用法
src_directory = "/home/liushuo/Documents/data/draw_memb/20190831_g4b2"  # 替换为原目录路径
dest_directory = "/home/liushuo/remote_mount/synapse/synapse2019VPP/20190831_g4b2_20190410_slot4/stack-out"  # 替换为目标目录路径

# copy_corrected_mrc(src_directory, dest_directory)
# copy_label_mrc(src_directory, dest_directory)
# copy_json_file(src_directory, dest_directory)
copy_memb_file_back(src_directory, dest_directory)