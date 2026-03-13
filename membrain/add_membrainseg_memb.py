import os
import argparse
import numpy as np
import mrcfile


from mrc.io import get_tomo_with_voxel_size, save_tomo

def add_membrainseg_memb(input_semantic_label_path, output_semantic_label_path, membrainseg_resultc_path):
    # 读取语义标签数据
    semantic_label_data, voxel_size = get_tomo_with_voxel_size(input_semantic_label_path)
    # 读取MemBrainSeg结果数据
    membrainseg_data, voxel_size = get_tomo_with_voxel_size(membrainseg_resultc_path)
    
    # 将MemBrainSeg结果中的膜（标签值为2）添加到语义标签中，膜的标签值设为20
    semantic_label_data[(membrainseg_data == 1) & (semantic_label_data < 5)] = 5
    
    save_tomo(semantic_label_data, output_semantic_label_path, voxel_size=voxel_size, datetype=np.int8)
    print(f"Saved updated semantic label to {output_semantic_label_path}")

if __name__ == '__main__':
    # args = parse_args()
    # 解析目标尺寸
    input_semantic_label_path = '/media/liushuo/data1/data/synapse_seg/10002_00004/synapse_seg/10002_00004_semantic_label_bak.mrc'
    output_semantic_label_path = '/media/liushuo/data1/data/synapse_seg/10002_00004/synapse_seg/10002_00004_semantic_label.mrc'
    membrainseg_resultc_path = '/media/liushuo/data1/data/CET-MAP/actin/10002/00004/Reconstructions/VoxelSpacing13.480/Tomograms/100/predictions/00004_resample_MemBrain_seg_v10_alpha.ckpt_segmented.mrc'
    add_membrainseg_memb(input_semantic_label_path, output_semantic_label_path, membrainseg_resultc_path)