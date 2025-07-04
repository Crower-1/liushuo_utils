#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

def get_folder_size(path):
    """
    递归计算 path 及其所有子目录中所有文件的总大小（字节）。
    """
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            full = os.path.join(root, fname)
            try:
                total += os.path.getsize(full)
            except Exception as e:
                # 无法读取文件大小时跳过
                print(f"警告：无法获取 {full} 的大小，已跳过 ({e})")
    return total

def find_p_prefix_folders(base_path, size_limit=1*1024*1024):
    """
    遍历 base_path 下所有子目录（递归）。
    如果该子目录名以 p 或 P 开头，且总大小 < size_limit，
    则记录该目录的绝对路径。
    返回符合条件的目录列表。
    """
    matched = []
    for root, dirs, _ in os.walk(base_path):
        # 跳过根目录本身
        if root == base_path:
            continue

        # 只看当前目录名是否以 p/P 开头
        folder_name = os.path.basename(root)
        if not folder_name.lower().startswith('p'):
            continue

        # 计算该目录及其所有子目录的大小
        total_size = get_folder_size(root)
        if total_size < size_limit:
            matched.append(root)

        # （可选）如果不希望再进入更深层次去检查孙目录，
        # 可在此处清空 dirs：dirs.clear()

    return matched

def main():
    parser = argparse.ArgumentParser(
        description="查找以 p 开头且总大小 < 1MB 的子目录，并导出到 Excel"
    )
    parser.add_argument("path", help="要扫描的根目录路径")
    parser.add_argument(
        "-o", "--output",
        default="/share/data/CryoET_Data/liushuo/destoryed_folder.xlsx",
        help="输出的 Excel 文件名（默认为 results.xlsx）"
    )
    args = parser.parse_args()

    base_path = os.path.abspath(args.path)
    if not os.path.isdir(base_path):
        print(f"错误：{base_path} 不是有效目录。")
        return

    results = find_p_prefix_folders(base_path)
    if not results:
        print("未找到符合条件的子目录。")
        return

    # 将结果写入 Excel
    df = pd.DataFrame({"folder_path": results})
    try:
        df.to_excel(args.output, index=False)
        print(f"已将 {len(results)} 个目录路径保存到 {args.output}")
    except Exception as e:
        print(f"写 Excel 出错：{e}")

if __name__ == "__main__":
    main()
