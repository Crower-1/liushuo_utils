import numpy as np
import mrcfile
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, cube
from scipy.ndimage import label as scipy_label

def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.

    Parameters:
    - path: str
        Path to the MRC file.

    Returns:
    - data: ndarray
        The 3D data loaded from the MRC file.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data
        data = data
    return data

def save_tomo(data, path, voxel_size=17.14):
    """
    Save a 3D numpy array as an MRC file.

    Parameters:
    - data: ndarray
        The 3D data to save.
    - voxel_size: float
        The voxel size of the data.
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(np.int8)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def process_masks(membrane_mask_path, synapse_mask_path, output_path):
    """
    Process the membrane mask and synapse mask to extract and save the largest instance.

    Parameters:
    - membrane_mask_path: str
        Path to the membrane mask MRC file.
    - synapse_mask_path: str
        Path to the synapse mask MRC file.
    - output_path: str
        Path where the result will be saved.
    """
    # Step 1: Load the masks
    membrane_mask = get_tomo(membrane_mask_path)
    synapse_mask = get_tomo(synapse_mask_path)
    
    # Step 2: Perform morphological operations on synapse mask
    # Define a cube-shaped structure element for dilation and erosion
    selem5 = cube(5)  # You can adjust the size of the cube for dilation/erosion
    selem7 = cube(7)
    # Perform dilation (7 iterations) and erosion (5 iterations)
    synapse_dilated = binary_dilation(synapse_mask, selem7)  # Dilation
    synapse_eroded = binary_erosion(synapse_mask, selem5)   # Erosion

    # Get the boundary mask by subtracting erosion from dilation
    boundary_mask = synapse_dilated & ~synapse_eroded

    # Step 3: Perform AND operation between boundary mask and membrane mask
    combined_mask = boundary_mask & membrane_mask

    # Step 4: Label the connected components in the combined mask
    labeled_mask, num_labels = scipy_label(combined_mask)

    # Step 5: Find the largest two connected components by volume
    component_sizes = np.array([np.sum(labeled_mask == i) for i in range(1, num_labels + 1)])
    largest_two_idx = component_sizes.argsort()[-2:][::-1]  # Indices of the two largest components

    # Step 6: Create masks for the largest two components
    largest_two_masks = np.zeros_like(combined_mask, dtype=int)
    for idx in largest_two_idx:
        largest_two_masks[labeled_mask == (idx + 1)] = 1  # Mark these components as '1'


    # Step 7: Binarize the largest instance mask (value 7 for the largest mask)
    # result_mask = largest_instance_mask.astype(np.int32) * 7

    # Step 8: Perform closing operation with a size of 7
    result_mask = binary_closing(largest_two_masks, selem7)

    # Step 9: Save the result mask as an MRC file
    save_tomo(result_mask.astype(np.int8), output_path)

    print(f"Processed mask saved to: {output_path}")

# # Example usage
membrane_mask_path = '/media/liushuo/data1/data/tcl_demo/pp0039/membrane/pp0039_membrane_label.mrc'
synapse_mask_path = '/media/liushuo/data1/data/tcl_demo/pp0039/membrane/pre_mask.mrc'
output_path = '/media/liushuo/data1/data/tcl_demo/pp0039/membrane/pre_memb.mrc'
process_masks(membrane_mask_path, synapse_mask_path, output_path)
