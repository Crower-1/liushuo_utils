from mrc.io import get_tomo, save_tomo
import mrcfile
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball

def get_outliner(mrc_input_path, mrc_output_path):
    # 读取.mrc文件
    data = get_tomo(mrc_input_path)
    
    binary_data = data > 0
    dilated_data = binary_dilation(binary_data, ball(1))
    erosion_data = binary_erosion(binary_data, ball(1))
    outliner = dilated_data & ~erosion_data
    
    
    # 保存为.mrc文件
    save_tomo(outliner, mrc_output_path, voxel_size=17.14)
    
    print(f"Outliner saved to: {mrc_output_path}")
    
mrc_input_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1776/vesicle/pp1776_vesicle_label.mrc'
mrc_output_path = '/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/pp1776/vesicle/pp1776_vesicle_outliner.mrc'
get_outliner(mrc_input_path, mrc_output_path)