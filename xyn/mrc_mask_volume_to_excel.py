import mrcfile
import numpy as np
import pandas as pd
import os

def save_mask_volume_to_excel(mask_file_path, output_excel_path):
    """
    统计mask文件中每个id的体积并保存为Excel文件。

    参数：
    - mask_file_path (str): mask文件的路径，格式为*.mrc。
    - output_excel_path (str): 保存统计结果的Excel文件路径。

    输出：
    - 生成的Excel文件路径。
    """
    try:
        # 确保输入的文件存在
        if not os.path.isfile(mask_file_path):
            raise ValueError(f"The mask file does not exist: {mask_file_path}")

        # 打开并读取mask文件
        with mrcfile.open(mask_file_path, permissive=True) as mrc:
            mask_data = mrc.data

        # 检查数据类型
        if not isinstance(mask_data, np.ndarray):
            raise ValueError("Invalid mask data format. Expected a numpy array.")

        # 统计每个id的体积
        unique_ids, counts = np.unique(mask_data, return_counts=True)
        
        # 创建结果数据表
        result_df = pd.DataFrame({
            'ID': unique_ids,
            'Volume (pixels)': counts
        })

        # 保存为Excel文件
        result_df.to_excel(output_excel_path, index=False)

        print(f"Volume statistics saved to: {output_excel_path}")
        return output_excel_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# 示例调用
if __name__ == "__main__":
    mask_path = f"/media/liushuo/新加卷1/data/demo_img/MT/pp684/MT/pp684_mt_label_actin.mrc"  # 替换为实际mask文件路径
    excel_path = f"/media/liushuo/新加卷1/data/demo_img/MT/pp684/MT/volume.xlsx"   # 替换为要保存的Excel文件路径

    save_mask_volume_to_excel(mask_path, excel_path)