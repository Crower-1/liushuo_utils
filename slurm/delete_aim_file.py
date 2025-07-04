import os
from pathlib import Path

def delete_backup_files(root_path):
    """
    遍历给定路径下的所有文件及子目录，删除以 ~ 结尾的文件。

    参数:
        root_path (str or Path): 要遍历并删除备份文件的根目录路径

    返回:
        int: 删除的文件数量
    """
    root_path = Path(root_path)
    deleted_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        for name in filenames:
            if name.endswith('.mrc~') or name.endswith('.rec~') or name.endswith('.ali~'):
                file_path = Path(dirpath) / name
                try:
                    file_path.unlink()
                    deleted_count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    return deleted_count


def main():
    # 从用户输入获取目标路径
    path = input('请输入要扫描并删除备份文件的目标路径: ').strip()
    if not path:
        print('路径不能为空。')
        return

    # 删除以 ~ 结尾的备份文件
    count = delete_backup_files(path)
    print(f'共删除 {count} 个以 ~ 结尾的文件。')

if __name__ == '__main__':
    main()