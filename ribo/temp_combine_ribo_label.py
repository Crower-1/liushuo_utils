from mrc.io import get_tomo, save_tomo

operation_dir = '/media/liushuo/data1/data/synapse_seg/'
base_names = ['pp267', 'pp1776' ,'pp1033', 'pp4001', 'pp366', 'pp387']

for base_name in base_names:
    old_semantic_label_path = f"{operation_dir}/{base_name}/synapse_seg/{base_name}_semantic_label.mrc"
    new_semantic_label_path = f"{operation_dir}/{base_name}/synapse_seg/ribo/{base_name}_ribo_semantic_label.mrc"
    ribo_label_path = f"{operation_dir}/{base_name}/synapse_seg/ribo/{base_name}_ribo_volumn.mrc"
    
    old_semantic_label = get_tomo(old_semantic_label_path)
    ribo_label = get_tomo(ribo_label_path)
    
    output_label = old_semantic_label.copy()
    output_label[ribo_label > 0] = 11  # 将原语义标签中的 ribo (5) 设置为 0
    
    save_tomo(output_label, new_semantic_label_path, datetype=output_label.dtype)