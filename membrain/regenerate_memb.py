from mrc.io import get_tomo, save_tomo

def regenerate_menb(semantic_label_path: str, memb_label_path: str):
    """
    从语义标签和膜标签生成新的膜数据。

    Parameters:
    - semantic_label_path: str
        语义标签 MRC 文件路径。
    - memb_label_path: str
        膜标签 MRC 文件路径。
    """
    # 读取语义标签和膜标签
    semantic_label = get_tomo(semantic_label_path)
    
    # 备份semantic_label_path数据
    bak_label_path = semantic_label_path.replace('.mrc', '_bak.mrc')
    save_tomo(semantic_label, bak_label_path)
    
    memb_label = get_tomo(memb_label_path)

    # 生成新的膜数据
    # 1. semantic_label == 5的数据置为0
    # 2. new_semantic_label 为0切memb_label大于0的数据置为5
    new_semantic_label = semantic_label.copy()
    new_semantic_label[semantic_label == 5] = 0
    new_semantic_label[(new_semantic_label == 0) & (memb_label > 0)] = 5

    # 保存新的膜数据
    save_tomo(new_semantic_label, semantic_label_path)
    print(f"Regenerated membrane data saved to {semantic_label_path}")
    
if __name__ == '__main__':
    operation_dir = "/media/liushuo/data1/data/synapse_seg/"
    base_name = "pp366"
    semantic_label_path = f"{operation_dir}/{base_name}/synapse_seg/{base_name}_semantic_label.mrc"
    memb_label_path = f'{operation_dir}/{base_name}/predictions/{base_name}_MemBrain_seg_v10_alpha.ckpt_segmented.mrc'
    regenerate_menb(semantic_label_path, memb_label_path)