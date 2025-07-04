import os
import shutil
from pathlib import Path
import pandas as pd

def find_and_move_redch_files(root_path):
    """
    查找以 RedCh.mrc 结尾且大小大于 1MB 的文件，将其移动到 root_path/TO_REMOVE，并记录原始路径和新路径。

    参数:
        root_path (str or Path): 要遍历的根目录路径

    返回:
        list of dict: 每个字典包含文件的原始路径和新路径
    """
    records = []
    root_path = Path(root_path)
    to_remove_dir = root_path / "TO_REMOVE"
    to_remove_dir.mkdir(parents=True, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(root_path):
        
        if "TO_REMOVE" in dirnames:
            dirnames.remove("TO_REMOVE")
            
        for name in filenames:
            if name.endswith('RedCh.mrc'):
                file_path = Path(dirpath) / name
                
                if to_remove_dir in file_path.parents:
                # 如果 file_path 是 /…/TO_REMOVE/RedCh.mrc，也跳过
                    continue
                
                try:
                    size = file_path.stat().st_size
                except OSError:
                    continue  # 无法读取文件大小时跳过
                
                # 过滤：仅处理大小 > 1MB 的文件
                if size > 1024 * 1024:
                    # 构造新的文件路径
                    new_path = to_remove_dir / name
                    if new_path.exists():
                        stem = new_path.stem  # 去除后缀的文件名，例如 "RedCh"
                        suffix = new_path.suffix  # ".mrc"
                        counter = 1
                        # 循环直到找到一个不存在的文件名
                        while True:
                            candidate_name = f"{stem}({counter}){suffix}"
                            candidate_path = to_remove_dir / candidate_name
                            if not candidate_path.exists():
                                new_path = candidate_path
                                break
                            counter += 1
                    try:
                        shutil.move(str(file_path), str(new_path))
                        records.append({
                            "original_path": str(file_path),
                            "new_path": str(new_path)
                        })
                    except Exception as e:
                        print(f"移动文件失败: {file_path} -> {new_path}, 错误: {e}")
    return records

def export_to_excel(data, output_path):
    """
    将文件移动记录导出到 Excel 文档。

    参数:
        data (list of dict): 包含 original_path 和 new_path 的列表
        output_path (str or Path): 输出的 Excel 文件路径
    """
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

def main():
    # 指定要扫描的根目录
    root = '/share/data/CryoET_Data/yanglq/SIM/DATASTORAGE/'
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        print(f"路径无效或不存在: {root}")
        return

    # 查找并移动文件
    moved_records = find_and_move_redch_files(root_path)
    if not moved_records:
        print("未找到符合条件的文件，或文件移动失败。")
        return

    # 导出移动记录到 Excel
    output_file = root_path / 'moved_redch_files.xlsx'
    try:
        export_to_excel(moved_records, output_file)
        print(f'操作完成，已将记录保存至: {output_file}')
    except Exception as e:
        print(f'导出 Excel 失败，错误信息: {e}')

if __name__ == '__main__':
    main()