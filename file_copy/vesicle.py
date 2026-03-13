import os
import shutil

# 定义源文件夹路径和目标文件夹路径
source_root = "/run/user/1000/gvfs/sftp:host=172.20.175.234,user=liushuo/share/data/CryoET_Data/synapse/synapse2015/20150729-g1b3_20130828/stack-out"
target_root = "/media/liushuo/新加卷/data/corrected_tomo"

# 遍历当前文件夹下的所有文件夹
for folder_name in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder_name)
    
    if os.path.isdir(folder_path):
        # 构建要复制的文件路径
        wbp_corrected_path = os.path.join(folder_path, "ves_seg", f"{folder_name}_wbp_corrected.mrc")
        label_vesicle_path = os.path.join(folder_path, "ves_seg", f"{folder_name}_label_vesicle.mrc")
        
        # 检查文件是否存在
        if os.path.exists(wbp_corrected_path) and os.path.exists(label_vesicle_path):
            # 构建目标文件夹路径
            target_folder = os.path.join(target_root, folder_name)
            os.makedirs(target_folder, exist_ok=True)
            
            # 复制文件到目标文件夹
            shutil.copy(wbp_corrected_path, target_folder)
            shutil.copy(label_vesicle_path, target_folder)
            print(f"Copied {folder_name}_wbp_corrected.mrc and {folder_name}_label_vesicle.mrc to {target_folder}")
        else:
            print(f"Files for {folder_name} not found or incomplete, skipping...")