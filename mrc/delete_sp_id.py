from mrc.io import get_tomo, save_tomo

tomo_path = '/media/liushuo/data1/data/synapse_seg/pp1776/synapse_seg/ribo/pp1776_ribo_volumn.mrc'
save_path = '/media/liushuo/data1/data/synapse_seg/pp1776/synapse_seg/ribo/pp1776_ribo_volumn_delete.mrc'
tomo_data = get_tomo(tomo_path).copy()
tomo_data[tomo_data == 55] = 0
save_tomo(tomo_data, tomo_path, datetype=tomo_data.dtype)