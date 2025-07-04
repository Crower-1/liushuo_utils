from mrc.io import get_tomo, save_tomo
import numpy as np

def extract_cube_with_center(tomo_path, output_path, center, size=64):
    """Extract a cube centered at a specific point from a tomogram and save it as a new MRC file.

    Args:
        tomo_path (str): Path to the input tomogram.
        output_path (str): Path to the output MRC file.
        center (tuple): A tuple of three integers (x, y, z) representing the center of the cube.
        size (int): The size of the cube to extract. The cube will be size x size x size.
    """
    # Load the tomogram
    tomo = get_tomo(tomo_path)

    # Calculate the half size
    half_size = size // 2

    # Extract the cube
    x_start = max(center[2] - half_size, 0)
    x_end = min(center[2] + half_size + 1, tomo.shape[2])
    
    y_start = max(center[1] - half_size, 0)
    y_end = min(center[1] + half_size + 1, tomo.shape[1])
    
    z_start = max(center[0] - half_size, 0)
    z_end = min(center[0] + half_size + 1, tomo.shape[0])

    extracted_cube = tomo[z_start:z_end, y_start:y_end, x_start:x_end]

    # Save the extracted cube as an MRC file
    save_tomo(extracted_cube, output_path, voxel_size=21.75, datetype=np.float32)

    print(f"Cube centered at {center} with size {size} extracted and saved to {output_path}")
    
if __name__ == "__main__":
    # Example usage
    tomo_path = '/media/liushuo/data1/data/fig_demo_2/pp370/cage.mrc'  # Replace with your MRC file path
    output_path = '/media/liushuo/data1/data/fig_demo_2/pp370/cage_2175.mrc'  # Replace with your desired output path
    center = (124, 661, 604)  # Replace with your desired center coordinates
    size = 64  # Size of the cube to extract

    extract_cube_with_center(tomo_path, output_path, center, size)