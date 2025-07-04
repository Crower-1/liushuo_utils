from mrc.io import get_tomo, save_tomo
from skimage.morphology import skeletonize_3d

def skelete_mt_mask(mask_path, save_path):
    mask_data = get_tomo(mask_path)
    skelete_mask = skeletonize_3d(mask_data)
    save_tomo(skelete_mask, save_path)
    
mask_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/MT/pp1033_semantic_MT_label.mrc'
save_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1033/MT/pp1033_skeleton_MT_label.mrc'
skelete_mt_mask(mask_path, save_path)
    