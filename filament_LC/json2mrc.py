import os
import json
import numpy as np
import mrcfile as mf
from skimage.morphology import cube, dilation, ball

class common:
    def angle_p(A, B, C):
        # Compute the angle at point B formed by points A, B, C (in degrees)
        BA = np.array([A[0] - B[0], A[1] - B[1], A[2] - B[2]])
        BC = np.array([C[0] - B[0], C[1] - B[1], C[2] - B[2]])
        l_BA = np.sqrt(BA.dot(BA))
        l_BC = np.sqrt(BC.dot(BC))
        dot_value = BA.dot(BC)
        cos_ = dot_value / (l_BA * l_BC)
        angle_in_rad = np.arccos(cos_)
        angle_in_d = angle_in_rad * 180 / np.pi
        return angle_in_d

    def angle_v(u, v):
        # Compute the angle between two vectors (in degrees)
        if u is None:
            u = np.array([1, 0, 0])
        if v is None:
            v = np.array([1, 0, 0])
        u = np.array(u)
        v = np.array(v)
        cos_value = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        cos_value = np.clip(cos_value, -1, 1)  # Ensure cosine value is within valid range
        return np.arccos(cos_value) * 180 / np.pi

    def link(A, B):
        # Generate a list of coordinates linking point A to point B
        points = []
        for t in np.linspace(0, 1, 11):
            points.append([
                round(t * (B[0] - A[0]) + A[0]),
                round(t * (B[1] - A[1]) + A[1]),
                round(t * (B[2] - A[2]) + A[2])
            ])
        return points

def process_actin_to_mask(image_path, json_path):
    """Read JSON and MRC files, generate a corresponding mask file."""
    # 1. Read the JSON file
    with open(json_path, "r") as f:
        actins = json.load(f)

    # Extract tomo_name and MRC file path
    tomo_name = os.path.splitext(os.path.basename(json_path))[0]
    tomo_name = tomo_name.replace("_point", "")
    mrc_path = image_path

    # 2. Read the MRC file to get z, y, x dimensions
    with mf.open(mrc_path, permissive=True) as mrc:
        z, y, x = mrc.data.shape
        voxel_size = mrc.voxel_size

    # 3. Create a zero-filled array of size (z, y, x) in int16 format
    masks = np.zeros((z, y, x), dtype=np.int16)

    # 4. Process each actin, generate local masks, and merge them
    for actin in actins:
        actin_id = actin["id"]
        print(f"Processing actin {actin_id}...")
        seedlist = np.array(actin["points"], dtype=np.float32)  # Now points are (z, y, x)

        # Create a local mask for the current actin
        local_mask = np.zeros((z, y, x), dtype=np.int16)

        # Generate lines connecting seed points and fill the local mask
        for i in range(len(seedlist) - 1):
            points = common.link(seedlist[i], seedlist[i + 1])
            points = np.array(points)
            local_mask[points[:, 0], points[:, 1], points[:, 2]] = 1  # Adjust indexing for (z, y, x)

        # Apply dilation
        selem_cube = cube(3)
        local_mask = dilation(local_mask, selem_cube)

        selem_z = np.zeros((3, 1, 1), dtype=np.int8)  # Affects only the z-axis
        selem_z[:, 0, 0] = 1
        local_mask = dilation(local_mask, selem_z)

        # Merge the local mask into the global mask
        masks[local_mask > 0] = actin_id

    # 5. Save the mask as an MRC file
    output_path = os.path.join(os.path.dirname(json_path), f"{tomo_name}_label.mrc")
    
    with mf.new(output_path, overwrite=True) as mrc:
        data = masks.astype(np.int16)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size  # Set voxel size if required

    print(f"Mask saved to {output_path}")

# Example call
image_path = f'/media/liushuo/data1/data/liucong/TS_009/TS_009_9.30Apx.mrc'
json_path = f"/media/liushuo/data1/data/liucong/TS_009/TS_009_9.30Apx.CorrelationLines.json"
process_actin_to_mask(image_path, json_path)
