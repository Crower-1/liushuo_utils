import os
import shutil
from pathlib import Path
import pandas as pd

def restore_files_from_excel(root_path, excel_filename='moved_redch_files.xlsx'):
    """
    从 root_path/<excel_filename> 中读取移动记录，将 TO_REMOVE 下的文件恢复到 original_path。
    
    参数:
        root_path (str or Path): 原始根目录，例如 '/share/data/CryoET_Data/yanglq/SIM/DATASTORAGE'
        excel_filename (str): 保存了移动记录的 Excel 文件名，默认为 'moved_redch_files.xlsx'
    
    返回:
        list of dict: 每条包含 'original_path', 'new_path', 'status'（'restored' 或 'skipped'）和 'error'（若有异常）
    """
    root_path = Path(root_path)
    excel_path = root_path / excel_filename
    to_remove_dir = root_path / "TO_REMOVE"

    if not excel_path.exists():
        print(f"找不到移动记录文件: {excel_path}")
        return

    # 读取 Excel
    try:
        df = pd.read_excel(excel_path, dtype=str)  # 确保路径读作字符串
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    results = []

    for idx, row in df.iterrows():
        original = Path(row['original_path'])
        new = Path(row['new_path'])

        record = {
            "original_path": str(original),
            "new_path": str(new),
            "status": None,
            "error": None
        }

        # 如果 new_path 不存在，说明文件可能已被手动删除或移动
        if not new.exists():
            record['status'] = 'skipped'
            record['error'] = f"目标文件不存在: {new}"
            print(f"[跳过] 行 {idx}：文件已不在 TO_REMOVE 下 → {new}")
            results.append(record)
            continue

        # 确保 original 的父目录存在
        try:
            parent_dir = original.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            record['status'] = 'skipped'
            record['error'] = f"无法创建父目录: {parent_dir}, 错误: {e}"
            print(f"[跳过] 行 {idx}：无法创建原始父目录 {parent_dir}，原因: {e}")
            results.append(record)
            continue

        # 如果 original 已经存在同名文件，跳过并警告
        if original.exists():
            record['status'] = 'skipped'
            record['error'] = f"原始位置已有同名文件: {original}"
            print(f"[跳过] 行 {idx}：原始位置已有同名文件 {original}")
            results.append(record)
            continue

        # 执行移动操作
        try:
            shutil.move(str(new), str(original))
            record['status'] = 'restored'
            print(f"[已恢复] {new} → {original}")
        except Exception as e:
            record['status'] = 'skipped'
            record['error'] = f"移动失败: {e}"
            print(f"[失败] 行 {idx}：{new} 无法移动回 {original}，原因: {e}")

        results.append(record)

    # 可以选择将恢复结果写回一个新的 Excel 或者打印出来
    result_df = pd.DataFrame(results)
    output_summary = root_path / 'restore_summary.xlsx'
    try:
        result_df.to_excel(output_summary, index=False)
        print(f"\n恢复完成，详情请参见: {output_summary}")
    except Exception as e:
        print(f"\n写入恢复摘要 Excel 失败: {e}")

    return results


def main():
    # 请替换为你的实际根目录路径
    root = '/media/liushuo/data2/data/demo_img/test_operation/'
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"路径无效或不存在: {root}")
        return

    # 调用恢复函数
    restore_files_from_excel(root_path)


if __name__ == '__main__':
    main()
