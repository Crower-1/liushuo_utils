import os
import shutil
from pathlib import Path

def copy_folders_from_list(source_root, dest_root, list_file):
    """
    Copies folders listed in list_file from source_root to dest_root.

    If the destination folder exists, copies only missing files/subfolders,
    skipping any items with the same name.

    Args:
        source_root (str or Path): Root directory containing source folders.
        dest_root (str or Path): Destination root directory where folders will be copied.
        list_file (str or Path): Path to the file containing folder names to copy (one per line).
    """
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    list_file = Path(list_file)

    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root not found: {source_root}")
    if not list_file.is_file():
        raise FileNotFoundError(f"List file not found: {list_file}")
    dest_root.mkdir(parents=True, exist_ok=True)

    with list_file.open('r') as f:
        names = [line.strip() for line in f if line.strip()]

    for name in names:
        src_dir = source_root / name
        dst_dir = dest_root / name

        if not src_dir.exists():
            print(f"Warning: Source folder does not exist: {src_dir}")
            continue

        if not dst_dir.exists():
            # Destination folder doesn't exist: copy entire folder
            try:
                shutil.copytree(src_dir, dst_dir)
                print(f"Copied folder: {src_dir} -> {dst_dir}")
            except Exception as e:
                print(f"Error copying {src_dir} to {dst_dir}: {e}")
            continue

        # Destination exists: copy only missing contents
        for item in src_dir.iterdir():
            target_item = dst_dir / item.name
            if target_item.exists():
                print(f"Skipping existing item: {target_item}")
                continue
            try:
                if item.is_dir():
                    shutil.copytree(item, target_item)
                    print(f"Copied subfolder: {item} -> {target_item}")
                else:
                    # file or symlink
                    shutil.copy2(item, target_item)
                    print(f"Copied file: {item} -> {target_item}")
            except Exception as e:
                print(f"Error copying {item} to {target_item}: {e}")


if __name__ == '__main__':
    # Define your paths here
    SOURCE_ROOT = '/share/data/CryoET_Data/liushuo/data_transfer/20150729'
    DEST_ROOT = '/share/data/CryoET_Data/synapse/synapse2015/20150729-g1b3_20130828/stack-out/4'
    LIST_FILE = '/share/data/CryoET_Data/liushuo/data_transfer/code/destroyed_demo.batch'

    copy_folders_from_list(SOURCE_ROOT, DEST_ROOT, LIST_FILE)
