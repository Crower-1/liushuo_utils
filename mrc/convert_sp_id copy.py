from mrc.io import get_tomo_with_voxel_size, save_tomo
import numpy as np

tomo_path = '/media/liushuo/data1/data/CET-MAP/actin/10002/00004/Reconstructions/VoxelSpacing13.480/Annotations/100/00004_resample_label.mrc'
save_path = '/media/liushuo/data1/data/synapse_seg/10002_00004/synapse_seg/10002_00004_semantic_label.mrc'
tomo_data, pixel_size = get_tomo_with_voxel_size(tomo_path)
tomo_data = tomo_data.copy()
tomo_data[tomo_data == 1] = 10
save_tomo(tomo_data, save_path, voxel_size=pixel_size, datetype=np.int8)