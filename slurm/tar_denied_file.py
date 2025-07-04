#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import tarfile
import sys

CHECKPOINT_FILE = ".tar_denied_checkpoint"

def parse_denied_paths(error_txt_path):
    """
    解析错误日志文件，提取所有被 Permission denied 的相对路径。
    """
    denied = []
    pattern = re.compile(r"tar: (\S+): Cannot (?:open|savedir): Permission denied")
    with open(error_txt_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                path = m.group(1)
                if path.startswith("./"):
                    path = path[2:]
                denied.append(path)
    # 去重并保持原始顺序
    seen = set()
    return [p for p in denied if not (p in seen or seen.add(p))]

def load_checkpoint():
    """
    如果存在 CHECKPOINT_FILE，读取并返回上次出错时的索引；否则返回 0。
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                idx = int(f.read().strip())
            print(f"[INFO] Loaded checkpoint: starting from index {idx}")
            return idx
        except Exception:
            print("[WARN] 无法读取检查点，重新从头开始。")
    return 0

def save_checkpoint(idx):
    """
    将当前索引写入 CHECKPOINT_FILE。
    """
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(idx))

def remove_checkpoint():
    """
    打包成功后删除检查点文件。
    """
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def create_denied_tar(denied_paths, base_dir, output_tar, start_idx):
    """
    从 start_idx 开始打包，遇到 OSError 时保存检查点并退出。
    """
    with tarfile.open(output_tar, 'a') as tar:
        for idx in range(start_idx, len(denied_paths)):
            rel_path = denied_paths[idx]
            abs_path = os.path.join(base_dir, rel_path)

            # 更新检查点
            save_checkpoint(idx)

            if os.path.islink(abs_path):
                print(f"[SKIP] symlink: {rel_path}")
                continue
            if not os.path.exists(abs_path):
                print(f"[SKIP] not exists: {rel_path}")
                continue

            try:
                tar.add(abs_path, arcname=rel_path)
                print(f"[ADDED] ({idx}) {rel_path}")
            except OSError as e:
                print(f"[ERROR] ({idx}) {rel_path} -> {e}; saved checkpoint and exiting.")
                sys.exit(1)

    # 全部添加完毕，删除检查点
    remove_checkpoint()
    size = os.path.getsize(output_tar)
    print(f"\nDone! Created {output_tar} ({size} bytes)")

def main():
    parser = argparse.ArgumentParser(
        description="断点续传：打包所有因 Permission denied 无法访问的文件/文件夹。"
    )
    parser.add_argument("-d", "--dir", help="工作目录（默认当前目录）")
    parser.add_argument("-e", "--error-txt", required=True, help="Permission denied 日志文件")
    parser.add_argument("-o", "--output", help="输出 tar 文件名（默认 <dir>_denied.tar）")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dir) if args.dir else os.getcwd()
    if not os.path.isdir(base_dir):
        print(f"[FATAL] 工作目录不存在：{base_dir}")
        sys.exit(1)
    os.chdir(base_dir)

    dir_name = os.path.basename(base_dir.rstrip("/"))
    output_tar = args.output or f"{dir_name}_denied.tar"

    denied_paths = parse_denied_paths(args.error_txt)
    if not denied_paths:
        print("[INFO] 未找到被拒绝访问的路径，退出。")
        sys.exit(0)

    start_idx = load_checkpoint()
    create_denied_tar(denied_paths, base_dir, output_tar, start_idx)

if __name__ == "__main__":
    main()
