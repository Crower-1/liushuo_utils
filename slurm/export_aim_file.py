import os
from pathlib import Path
import pandas as pd

def find_hidden_files(root_path):
    """

    参数:
        root_path (str or Path): 要遍历的根目录路径

    返回:
        list of dict: 每个字典包含文件的绝对路径和大小（字节）
    """
    files = []
    root_path = Path(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if name.endswith('.mrc~') or name.endswith('.rec~') or name.endswith('.ali~'):
            # if name.endswith('bak.mrc') or name.endswith('ori.mrc'):
                file_path = Path(dirpath) / name
                try:
                    size = file_path.stat().st_size
                except OSError:
                    # 如果无法获取大小，则记录为 None
                    size = None
                files.append({
                    'path': str(file_path),
                    'size': size
                })
    return files


def export_to_excel(data, output_path):
    """
    将文件信息列表导出到 Excel 文档。

    参数:
        data (list of dict): 包含文件路径和大小的列表
        output_path (str or Path): 输出的 Excel 文件路径
    """
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)


def main():
    # 从用户输入获取目标路径
    root = f'/share/data/CryoET_Data/chongli'
    if not root:
        print('路径不能为空。')
        return

    # 查找隐藏文件
    results = find_hidden_files(root)
    if not results:
        print('未找到符合条件的文件。')
        return

    # 确定输出文件名
    # output_file = Path(root) / 'wait_to_remove.xlsx'
    output_file = Path('/share/data/CryoET_Data/liushuo/chongli_wait_to_remove.xlsx')

    # 导出为 Excel
    try:
        export_to_excel(results, output_file)
        print(f'导出完成，Excel 文件已保存至: {output_file}')
    except Exception as e:
        print(f'导出失败，错误信息: {e}')


if __name__ == '__main__':
    main()
