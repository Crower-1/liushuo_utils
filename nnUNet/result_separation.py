from mrc.io import get_tomo, save_tomo
from chimerax.mrc2cmm import convert_mrc_to_cmm

def extract_sp_id(tomo_path, output_path, id):
    """从tomogram中提取指定ID并保存为新的MRC文件。

    参数:
        tomo_path (str): 输入tomogram文件路径。
        output_path (str): 输出MRC文件路径。
        id (int): 要提取的ID。
    """
    # 加载tomogram
    tomo = get_tomo(tomo_path)

    # 提取指定的ID
    mask = (tomo == id)

    # 保存提取结果
    save_tomo(mask, output_path)

    print(f"ID {id} 提取成功，保存至 {output_path}")

def generate_paths(base_name, label):
    """
    根据基本名称和标签生成输入tomogram路径和输出MRC路径。

    参数:
        base_name (str): 基本名称，例如 "pp1776"。
        label (str): 标签名称，如 "memb", "ER", 等。

    返回:
        (tomo_path, output_path) 两个字符串
    """
    tomo_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/ret1.mrc"
    output_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/predict_result/{base_name}_{label}_label.mrc"
    cmm_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/predict_result/cmm/{base_name}_{label}.cmm"
    return tomo_path, output_path, cmm_path

# 定义标签与对应ID的映射关系
labels = {
    1: "memb",
    2: "ER",
    3: "mito",
    4: "MT",
    5: "vesicle",
    6: "actin"
}

# 示例调用
base_name = "pp4001"

# 对每个标签进行处理
for id, label in labels.items():
    tomo_path, output_path, cmm_path = generate_paths(base_name, label)
    extract_sp_id(tomo_path, output_path, id)
    convert_mrc_to_cmm(output_path, cmm_path)
