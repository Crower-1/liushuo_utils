import os
import subprocess

def run_membrain_for_all_tomos(dest_dir, model_path='/home/liushuo/Downloads/MemBrain_seg_v10_alpha.ckpt'):
    # 获取目标路径下的所有子文件夹名
    tomo_names = [folder for folder in os.listdir(dest_dir) if os.path.isdir(os.path.join(dest_dir, folder))]

    for tomo_name in tomo_names:
        tomo_file = os.path.join(dest_dir, tomo_name, f"{tomo_name}_wbp_corrected.mrc")
        out_folder = os.path.join(dest_dir, tomo_name, "membrain")

        # 确保输出文件夹存在
        os.makedirs(out_folder, exist_ok=True)

        # 构造 membrain 命令
        if os.path.exists(tomo_file):
            command = [
                'membrain', 'segment',
                '--tomogram-path', tomo_file,
                '--ckpt-path', model_path,
                '--out-folder', out_folder
            ]

            # 执行命令
            try:
                print(f"正在执行：{' '.join(command)}")
                subprocess.run(command, check=True)
                print(f"MemBrain 处理完成: {tomo_name}")
            except subprocess.CalledProcessError as e:
                print(f"执行错误: {e}")
        else:
            print(f"文件不存在: {tomo_file}")

# 示例用法
dest_directory = "/home/liushuo/Documents/data/draw_memb/20190826_g2b3"  # 替换为目标目录路径
run_membrain_for_all_tomos(dest_directory)