from mrc.io import get_tomo, save_tomo
import numpy as np

def combine_masks(mask_1_path, mask_2_path, combine_mask_path):
    """
    Combine two masks by taking the maximum value at each voxel.
    
    Parameters:
    - mask_1_path: str
        Path to the first mask MRC file.
    - mask_2_path: str
        Path to the second mask MRC file.
    - combine_mask_path: str
        Path where the combined mask will be saved.
    """
    # Read mask volumes from disk
    mask_1 = get_tomo(mask_1_path)
    mask_2 = get_tomo(mask_2_path)
    
    binary_mask_1 = mask_1 > 0
    binary_mask_2 = mask_2 > 0
    # Merge foregrounds voxel-wise
    combined_mask = np.maximum(binary_mask_1, binary_mask_2)
    
    # Persist the merged mask
    save_tomo(combined_mask, combine_mask_path)
    
def combine_masks_with_ori_id(mask_1_path, mask_2_path, combine_mask_path):
    """
    Combine two masks by taking the maximum value at each voxel,
    preserving original IDs from the first mask where applicable.
    
    Parameters:
    - mask_1_path: str
        Path to the first mask MRC file.
    - mask_2_path: str
        Path to the second mask MRC file.
    - combine_mask_path: str
        Path where the combined mask will be saved.
    """
    # Read mask volumes from disk
    mask_1 = get_tomo(mask_1_path)
    mask_2 = get_tomo(mask_2_path)
    
    binary_mask_2 = mask_2 > 0
    
    combined_mask = mask_1.copy()
    max_id = mask_1.max()
    # Fill empty voxels in mask_1 wherever mask_2 has foreground
    combined_mask[(binary_mask_2 != 0) & (mask_1 == 0)] = max_id + 1
    
    # Persist the merged mask
    save_tomo(combined_mask, combine_mask_path)
    
def main():
    # Define the paths to the masks
    mask_1_path = '/media/liushuo/data1/data/synapse_seg/pp676/ret1_10tomo_no_crop.mrc'
    mask_2_path = '/media/liushuo/data1/data/synapse_seg/pp676/pp676_ribo_0716.mrc'
    combine_mask_path = '/media/liushuo/data1/data/synapse_seg/pp676/ret1_10tomo_no_crop_ribo.mrc'
    
    # Combine the masks
    combine_masks_with_ori_id(mask_1_path, mask_2_path, combine_mask_path)
    
if __name__ == "__main__":
    main()
