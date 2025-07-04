import tifffile as tiff

def process_tiff(input_path, output_path):
    """
    Reads a 3D TIF file, extracts the 0th frame, and saves it as a 2D TIF.
    
    Parameters:
    - input_path: str, path to the input 3D TIF file.
    - output_path: str, path to save the output 2D TIF file.
    """
    try:
        # Read the 3D TIF image
        image_3d = tiff.imread(input_path)
        print(f"Input 3D TIF shape: {image_3d.shape}")

        # Extract the 0th frame (0, y, x)
        frame_0 = image_3d[0]
        print(f"Extracted frame shape: {frame_0.shape}")

        # Save the 2D frame as a new TIF
        tiff.imwrite(output_path, frame_0)
        print(f"2D TIF saved at: {output_path}")
    except Exception as e:
        print(f"Error processing TIF file: {e}")

# Example usage:
process_tiff("/media/liushuo/新加卷1/data/lz/all_frame/tif_crop/images/patch_0000_label.tif", "/media/liushuo/新加卷1/data/lz/all_frame/tif_crop/images/output_2d_image.tif")
