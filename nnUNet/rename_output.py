import os

def rename_mrc_files(folder_path):
    """
    Traverse all *.mrc files in a given folder and rename them to *_0000.mrc format.

    Args:
        folder_path (str): The path to the folder containing .mrc files.
    """
    if not os.path.exists(folder_path):
        print(f"The folder path {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".mrc"):
            file_path = os.path.join(folder_path, filename)
            # Extract the base name without extension
            base_name = os.path.splitext(filename)[0]
            # Create the new name with *_0000.mrc format
            new_name = f"{base_name}_0000.mrc"
            new_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(file_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    folder_path = "/home/liushuo/Documents/data/nnUNet/nnUNet_raw/Dataset001_Synapse/pp3266/output"
    rename_mrc_files(folder_path)