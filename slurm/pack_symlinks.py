#!/usr/bin/env python3
import os
import argparse
import tarfile

def collect_symlinks(root_dir):
    """
    遍历 root_dir，返回所有符号链接的绝对路径列表。
    """
    symlinks = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查目录本身是不是链接
        if os.path.islink(dirpath):
            symlinks.append(dirpath)
        # 检查子目录名
        for d in dirnames:
            full_d = os.path.join(dirpath, d)
            if os.path.islink(full_d):
                symlinks.append(full_d)
        # 检查文件
        for f in filenames:
            full_f = os.path.join(dirpath, f)
            if os.path.islink(full_f):
                symlinks.append(full_f)
    return symlinks

def filter_symlinks(symlinks):
    """
    如果某个软链接指向目录，则保留该链接本身，但去掉该链接目录下的其他软链接。
    返回过滤后的软链接列表。
    """
    dir_links = {lnk for lnk in symlinks if os.path.isdir(lnk)}
    filtered = []
    for lnk in symlinks:
        # 如果 lnk 在某个目录型软链接的子路径下，则跳过
        if any(lnk != dlnk and lnk.startswith(dlnk + os.sep) for dlnk in dir_links):
            continue
        filtered.append(lnk)
    return filtered

def collect_dirs_for_symlinks(root_dir, symlinks):
    """
    对于每个软链接，收集从 root_dir 开始到它父目录的所有中间目录，
    以保证解包时能正确创建目录层级。
    """
    dirs = set()
    root_dir = os.path.abspath(root_dir)
    for lnk in symlinks:
        parent = os.path.dirname(os.path.abspath(lnk))
        # 沿着父目录一路向上，直到 root_dir
        while True:
            if not parent.startswith(root_dir):
                break
            dirs.add(parent)
            if parent == root_dir:
                break
            parent = os.path.dirname(parent)
    return dirs

def pack_symlinks_and_dirs(root_dir, dirs, symlinks, output_tar):
    """
    先创建必要的目录条目，再添加软链接本身（保留为链接）。
    """
    with tarfile.open(output_tar, mode='w') as tar:
        # 1. 添加目录（按路径长度排序，先短后长）
        for d in sorted(dirs, key=lambda p: p.count(os.sep)):
            arcdir = os.path.relpath(d, start=os.path.abspath(root_dir))
            if arcdir and arcdir != '.':
                tar.add(d, arcname=arcdir, recursive=False)
        # 2. 添加软链接
        for lnk in symlinks:
            arcname = os.path.relpath(lnk, start=os.path.abspath(root_dir))
            print(f"Packing symlink: {arcname}")
            tar.add(lnk, arcname=arcname, recursive=False)
    print(f"已将 {len(dirs)} 个目录和 {len(symlinks)} 个软链接打包到 {output_tar}")

def main():
    parser = argparse.ArgumentParser(description="打包指定目录下的软链接（仅保留含链接的目录）")
    parser.add_argument("-r", "--root",   required=True, help="要搜索的根目录路径")
    parser.add_argument("-o", "--output", required=True, help="输出的 tar 文件路径")
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        parser.error(f"根目录不存在或不是目录：{root}")

    # 1. 收集所有软链接
    all_links = collect_symlinks(root)
    if not all_links:
        print(f"在 {root} 未发现任何软链接。")
        return

    # 2. 过滤：跳过目录型链接下的其它链接
    links = filter_symlinks(all_links)


    # 3. 收集所有需要保留的目录层级
    dirs = collect_dirs_for_symlinks(root, links)

    # 4. 打包
    pack_symlinks_and_dirs(root, dirs, links, args.output)

if __name__ == "__main__":

    main()
