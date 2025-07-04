import os
import sys
import argparse
from pathlib import Path

# 假设你的脚本结构如下，调整导入路径根据实际情况
# 例如，如果filter_vesicle.py和process_masks.py在同一目录下的`vesicle`和`membrane`子目录中
from mrc.vesicle.filter_vesicle import process_vesicle_data
from mrc.membrane.get_single_membrane_mask2 import process_masks

def batch_process(target_folder, threshold):
    """
    批量处理目标文件夹下的所有子文件夹。
    
    Parameters:
    - target_folder: str
        目标文件夹路径。
    - thresholds: list of float
        用于分类小囊泡和其他囊泡的阈值列表。
    """
    target_path = Path(target_folder)
    
    if not target_path.exists() or not target_path.is_dir():
        print(f"目标文件夹不存在或不是一个目录: {target_folder}")
        sys.exit(1)
    
    # 获取所有子文件夹
    tomo_names = [d.name for d in target_path.iterdir() if d.is_dir()]
    
    if not tomo_names:
        print(f"在目标文件夹中未找到子文件夹: {target_folder}")
        sys.exit(1)
    
    print(f"在目标文件夹中找到 {len(tomo_names)} 个子文件夹。")
    
    for tomo_name in tomo_names:
        print(f"\n处理 {tomo_name} ...")
        tomo_dir = target_path / tomo_name
        
        # 构建关键路径
        membrane_mask_path = tomo_dir / "membrain" / f"{tomo_name}_wbp_corrected_MemBrain_seg_v10_alpha.ckpt_segmented.mrc"
        synapse_mask_path = tomo_dir / f"{tomo_name}_wbp_corrected_label.mrc"
        single_mask_path = tomo_dir / f"{tomo_name}_single_membrane_mask_2.mrc"
        vesicle_path = tomo_dir / f"{tomo_name}_label_vesicle.mrc"
        json_path = tomo_dir / f"{tomo_name}_vesicle.json"
        filter_vesicle_dir = tomo_dir / "filter_vesicle"
        small_vesicle_output = filter_vesicle_dir / "small_vesicles.mrc"
        other_vesicle_output = filter_vesicle_dir / "other_vesicles.mrc"
        
        # 检查必要文件是否存在
        required_files = [
            membrane_mask_path,
            synapse_mask_path,
            vesicle_path,
            json_path
        ]
        
        missing_files = [str(f) for f in required_files if not f.exists()]
        if missing_files:
            print(f"跳过 {tomo_name}，缺少以下文件:")
            for mf in missing_files:
                print(f"  - {mf}")
            continue
        
        # 创建filter_vesicle输出目录
        if not filter_vesicle_dir.exists():
            filter_vesicle_dir.mkdir(parents=True)
            print(f"创建目录: {filter_vesicle_dir}")
        
        # 处理膜相关的mask
        print("  处理膜相关的mask...")
        try:
            process_masks(
                membrane_mask_path,
                synapse_mask_path,
                single_mask_path
            )
            print("  膜相关的mask处理完成。")
        except Exception as e:
            print(f"  处理膜相关的mask时出错: {e}")
            continue
        
        
        print(f"  使用阈值 {threshold} 处理囊泡数据...")
        try:
            process_vesicle_data(
                vesicle_path,
                json_path,
                small_vesicle_output,
                other_vesicle_output,
                threshold
            )
            print(f"  囊泡数据处理完成，输出文件:")
            print(f"    小囊泡: {small_vesicle_output}")
            print(f"    其他囊泡: {other_vesicle_output}")
        except Exception as e:
            print(f"  处理囊泡数据时出错 (阈值 {threshold}): {e}")
            continue

def main():
    # parser = argparse.ArgumentParser(description="批量处理囊泡和膜的MRC文件。")
    # parser.add_argument(
    #     '--target_folder',
    #     type=str,
    #     required=True,
    #     help='目标文件夹路径，包含多个子文件夹，每个子文件夹对应一个tomo。'
    # )
    # parser.add_argument(
    #     '--thresholds',
    #     type=float,
    #     nargs='+',
    #     default=[34, 34/1.714, 34/2],
    #     help='用于分类小囊泡和其他囊泡的阈值列表。默认: 34, 34/1.714, 34/2'
    # )
    
    # args = parser.parse_args()
    threshold = 34/1.714/2
    batch_process('/home/liushuo/Documents/data/draw_memb/20190826_g2b3', threshold)

if __name__ == "__main__":
    main()
