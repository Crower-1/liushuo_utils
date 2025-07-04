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
    # Load the masks
    mask_1 = get_tomo(mask_1_path)
    mask_2 = get_tomo(mask_2_path)
    
    binary_mask_1 = mask_1 > 0
    binary_mask_2 = mask_2 > 0
    # Combine the masks by taking the maximum value at each voxel
    combined_mask = np.maximum(binary_mask_1, binary_mask_2)
    
    # Save the combined mask
    save_tomo(combined_mask, combine_mask_path)
    
def main():
    # Define the paths to the masks
    mask_1_path = '/media/liushuo/data1/data/draw_memb/lzh/pp311/membrane/pp311_membrane_label_1.mrc'
    mask_2_path = '/media/liushuo/data1/data/draw_memb/lzh/pp311/membrane/pp311_membrane_label_2.mrc'
    combine_mask_path = '/media/liushuo/data1/data/draw_memb/lzh/pp311/membrane/pp311_membrane_label.mrc'
    
    # Combine the masks
    combine_masks(mask_1_path, mask_2_path, combine_mask_path)
    
if __name__ == "__main__":
    main()