import os
import argparse

def is_file_owned_by_user(filepath):
    """判断文件是否为当前用户所有"""
    try:
        return os.stat(filepath).st_uid == os.getuid()
    except PermissionError:
        print(f"Skipping {filepath} due to PermissionError")
        return False

def set_permissions(filepath):
    """设置文件权限为 rwxrwxr-x (775)"""
    try:
        os.chmod(filepath, 0o775)
    except PermissionError:
        print(f"Failed to change permissions for {filepath} due to PermissionError")

def traverse_and_modify(directory):
    """遍历目录并修改权限"""
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if is_file_owned_by_user(filepath):
                print(f"Changing permissions for {filepath}")
                set_permissions(filepath)
            else:
                print(f"Skipping {filepath} (not owned by user or PermissionError)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遍历目录并将当前用户拥有的文件权限设置为rwxrwxr-x (775)")
    parser.add_argument("directory", type=str, help="要遍历的目录路径")
    args = parser.parse_args()

    directory_to_traverse = args.directory
    if os.path.isdir(directory_to_traverse):
        traverse_and_modify(directory_to_traverse)
    else:
        print(f"{directory_to_traverse} 不是一个有效的目录")
