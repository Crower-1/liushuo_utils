import os
import shutil

def organize_mrc_files(directory):
    """
    Organize MRC files into subfolders based on their names.

    Args:
        directory (str): The path to the directory containing MRC files.
    """
    if not os.path.exists(directory):
        print(f"The specified directory does not exist: {directory}")
        return

    # Get all files in the directory
    files = os.listdir(directory)

    # Iterate through the files and organize MRC files
    for file in files:
        if file.endswith('.mrc'):
            tomo_name = os.path.splitext(file)[0]  # Extract the name without extension
            subfolder_path = os.path.join(directory, tomo_name)

            # Create the subfolder if it does not exist
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            # Move the file to the corresponding subfolder
            src_path = os.path.join(directory, file)
            dest_path = os.path.join(subfolder_path, file)
            shutil.move(src_path, dest_path)

            print(f"Moved {file} to {subfolder_path}")

if __name__ == "__main__":
    # directory = input("Enter the directory containing MRC files: ").strip()
    directory = '/media/liushuo/新加卷/data/actin trace/actintomo/20200820/tomo'
    organize_mrc_files(directory)
