import numpy as np
from scipy.ndimage import label
from skimage.morphology import ball, cube, dilation, closing
from skimage.measure import regionprops
from mrc.io import get_tomo, save_tomo

def dilate_mito(mito_path, output_path, dilate_type='ball', dilate_size=3):
    """
    Dilate the mitochondria in a 3D MRC file.

    Parameters:
    - mito_path: str
        Path to the mitochondria MRC file.
    - output_path: str
        Path to save the dilated mitochondria MRC file.
    - dilate_type: str
        The type of dilation (default: 'ball'). Currently only supports 'ball'.
    - dilate_size: int
        The size of the dilation. For 'ball', this is the radius of the ball.
    """
    # Load the mitochondria data from the MRC file
    mito = get_tomo(mito_path)

    # Ensure the input labels are integers (not binary)
    unique_labels = np.unique(mito)
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]  # Ignore background (label 0)

    # Create the structuring element based on the dilation type
    if dilate_type == 'ball':
        structuring_element = ball(dilate_size)
    elif dilate_type == 'cube':
        structuring_element = cube(dilate_size)
    else:
        raise ValueError(f"Unsupported dilate_type: {dilate_type}. Currently, only 'ball' is supported.")

    # Initialize the output array
    dilated_mito = np.zeros_like(mito, dtype=mito.dtype)

    # Perform dilation for each label independently
    for label_value in unique_labels:
        print(f"Dilating label {label_value}...")
        mask = (mito == label_value)
        dilated_mask = dilation(mask, structuring_element)
        dilated_mask = closing(dilated_mask, structuring_element)
        dilated_mito[dilated_mask] = label_value

    labeled_mito, num_labels = label(dilated_mito)
    
    props = regionprops(labeled_mito)

    # Create a new array to hold only large objects
    final_mito = np.zeros_like(dilated_mito)

    for prop in props:
        if prop.area >= 1000:
            # If the region size is greater than or equal to min_size, keep it
            print(f"Keeping label {prop.label} with area {prop.area}.")
            final_mito[labeled_mito == prop.label] = prop.label
    
    save_tomo(final_mito, output_path, voxel_size=17.14)
    # Save the dilated mitochondria back to an MRC file
    # save_tomo(dilated_mito, output_path, voxel_size=17.14)

    print(f"Successfully saved the dilated mitochondria to {output_path}.")

mito_path = f'/media/liushuo/data1/data/synapse_seg/p545/mito/p545_mito_dilated.mrc'
output_path = f'/media/liushuo/data1/data/synapse_seg/p545/mito/p545_mito_label.mrc'

if __name__ == "__main__":
    dilate_mito(mito_path, output_path, dilate_type='cube', dilate_size=5)
