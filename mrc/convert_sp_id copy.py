from mrc.io import get_tomo_with_voxel_size, save_tomo
import numpy as np

tomo_path = '/home/liushuo/Downloads/5FAD-Abeta-branch/TS_202/synapse_seg/TS_202_seg.mrc'
save_path = '/home/liushuo/Downloads/5FAD-Abeta-branch/TS_202/synapse_seg/TS_202_semantic_label.mrc'
tomo_data, pixel_size = get_tomo_with_voxel_size(tomo_path)
tomo_data = tomo_data.copy()
tomo_data[tomo_data == 1] = 10
save_tomo(tomo_data, save_path, voxel_size=pixel_size, datetype=np.int8)