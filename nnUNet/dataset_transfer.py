import os
import shutil

def migrate_masks(operation_dir: str, base_name: str):
    """
    将旧的 segmentation 文件从 operation_dir/{base_name}/... 迁移到
    operation_dir/{base_name}/synapse_seg/... 并重命名为新的文件名。
    """
    # 构造旧文件根目录和新文件根目录
    old_root = os.path.join(operation_dir, base_name)
    new_root = os.path.join(operation_dir, base_name, 'synapse_seg')

    # 定义旧文件相对路径到新文件相对路径的映射
    mapping = {
        # isonet files
        f'{base_name}.mrc':                f'{base_name}_wbp_corrected.mrc',
        # final label
        f'{base_name}_label.mrc':    f'{base_name}_semantic_label.mrc',
        # volume masks
        f'er/{base_name}_er_nn.mrc':          f'er/{base_name}_er_volume.mrc',
        f'mito/{base_name}_mito_nn.mrc':      f'mito/{base_name}_mito_volume.mrc',
        f'vesicle/{base_name}_vesicle_label.mrc': f'vesicle/{base_name}_vesicle_volume.mrc',
        # MT
        f'mt/{base_name}_mt_label.mrc':       f'mt/{base_name}_mt_filament.mrc',
        # Actin
        f'actin/{base_name}_actin_label.mrc': f'actin/{base_name}_actin_filament.mrc',
        # membranes
        f'er/{base_name}_er_memb_label.mrc':       f'er/{base_name}_er_memb.mrc',
        f'mito/{base_name}_mito_memb_label.mrc':   f'mito/{base_name}_mito_memb.mrc',
        f'vesicle/{base_name}_vesicle_memb_label.mrc': f'vesicle/{base_name}_vesicle_memb.mrc',
        f'mt/{base_name}_mt_memb_label.mrc':       f'mt/{base_name}_mt_memb.mrc',
        # JSON
        f'mt/{base_name}_mt_point.json':     f'mt/{base_name}_mt.json',
        f'actin/{base_name}_actin_point.json': f'actin/{base_name}_actin.json',
    }

    for old_rel, new_rel in mapping.items():
        src = os.path.join(old_root, old_rel)
        dst = os.path.join(new_root, new_rel)

        # 如果源文件不存在，跳过并打印警告
        if not os.path.isfile(src):
            print(f'[WARN] 源文件不存在，已跳过：{src}')
            continue

        # 确保目标目录存在
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        # 执行移动
        try:
            shutil.copy(src, dst)
            print(f'[OK] 已移动: {src} → {dst}')
        except Exception as e:
            print(f'[ERROR] 移动失败: {src} → {dst}，原因：{e}')

if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser(description='迁移旧的 segmentation 文件到新目录并重命名')
    # parser.add_argument('operation_dir', help='旧文件父目录，例如 /media/liushuo/data1/data/synapse_seg')
    # parser.add_argument('base_name',     help='每个 tomo 的名称，比如 tomogram001')
    # args = parser.parse_args()
    operation_dir = '/media/liushuo/data1/data/synapse_seg/'
    base_name = 'pp387'
    migrate_masks(operation_dir, base_name)
