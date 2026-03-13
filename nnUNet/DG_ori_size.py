#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import mrcfile

# ---------- IO utils ----------

def load_mrc_with_voxel(path):
    """
    Load a 3D MRC file and return (data, voxel_size_float).
    """
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.ascontiguousarray(mrc.data).astype(np.float32, copy=False)
        vs = mrc.voxel_size
        if hasattr(vs, "x"):
            voxel_size = float(np.mean([vs.x, vs.y, vs.z]))
        else:
            voxel_size = float(vs) if np.isscalar(vs) else 1.0
    return data, voxel_size


def save_mrc(data, path, voxel_size=1.0, dtype=None):
    """
    Save a 3D numpy array as MRC, preserving/forcing dtype if specified.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(data if dtype is None else data.astype(dtype, copy=False))
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = float(voxel_size)


# ---------- preprocessing ----------

def normalize01(img):
    """
    Normalize image to [0,1]. If flat, return zeros.
    """
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


# ---------- splitting logic ----------

VOXELS_THRESHOLD = 256 * 1024 * 1024  # trigger split if total voxels exceed this

def maybe_split_into_quadrants(img, lbl):
    """
    If volume size exceeds threshold, split along Y and X midpoints into 4 tiles.
    Z is unchanged. Return a list of (img_sub, lbl_sub).
    """
    z, y, x = img.shape
    total = z * y * x
    if total <= VOXELS_THRESHOLD:
        return [(img, lbl)]

    y_mid = y // 2
    x_mid = x // 2

    tiles = []
    # Top-Left (y0..y_mid, x0..x_mid)
    tiles.append((img[:, :y_mid, :x_mid], lbl[:, :y_mid, :x_mid]))
    # Top-Right
    tiles.append((img[:, :y_mid, x_mid:], lbl[:, :y_mid, x_mid:]))
    # Bottom-Left
    tiles.append((img[:, y_mid:, :x_mid], lbl[:, y_mid:, :x_mid]))
    # Bottom-Right
    tiles.append((img[:, y_mid:, x_mid:], lbl[:, y_mid:, x_mid:]))

    return tiles


# ---------- dataset writer ----------

def create_dataset_json(output_dir, num_training, file_ending=".mrc"):
    dataset_info = {
        "channel_names": {"0": "cryoET"},
        "labels": {
            "background": 0,
            "ER": [1, 6],
            "mitochondria": [2, 7],
            "MT": [3, 8],
            "vesicle": [4, 9],
            "membrane": [5, 6, 7, 8, 9],
            "ER_memb": 6,
            "mito_memb": 7,
            "MT_memb": 8,
            "vesicle_memb": 9,
            "actin": 10
        },
        "regions_class_order": [1,2,3,4,5,6,7,8,9,10],
        "numTraining": int(num_training),
        "file_ending": file_ending
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_info, f, indent=4)


def generate_image_paths(base_path, base_names):
    """
    Build image paths as: {base_path}/{name}/{name}.mrc
    """
    return [os.path.join(base_path, n, f"{n}.mrc") for n in base_names]


def copy_and_rename_whole_volumes(image_paths, output_dir):
    """
    For each image_path, infer label_path = image_path.replace('.mrc', '_label.mrc').
    Load full volumes, normalize image, optionally split into 4 quadrants (x,y midpoint)
    if volume exceeds threshold, then save to:
      imagesTr/Synapse_XXXX_0000.mrc  (float32, normalized)
      labelsTr/Synapse_XXXX.mrc       (int8)
    Returns number of saved pairs.
    """
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    saved = 0
    for image_path in image_paths:
        label_path = image_path.replace(".mrc", "_label.mrc")
        if not os.path.exists(image_path):
            print(f"[Skip] Image not found: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"[Skip] Label not found for {image_path}: {label_path}")
            continue

        img, img_voxel = load_mrc_with_voxel(image_path)
        lbl, lbl_voxel = load_mrc_with_voxel(label_path)

        if img.shape != lbl.shape:
            print(f"[Skip] Shape mismatch: {os.path.basename(image_path)} {img.shape} vs {lbl.shape}")
            continue

        # normalize image only
        img = normalize01(img)

        # maybe split into 4 quadrants along (y,x)
        pairs = maybe_split_into_quadrants(img, lbl)

        for (img_sub, lbl_sub) in pairs:
            sample_id = f"{saved:04d}"
            img_out = os.path.join(images_dir, f"Synapse_{sample_id}_0000.mrc")
            lbl_out = os.path.join(labels_dir,  f"Synapse_{sample_id}.mrc")

            save_mrc(img_sub, img_out, voxel_size=img_voxel, dtype=np.float32)
            # label saved as int8
            save_mrc(lbl_sub, lbl_out, voxel_size=lbl_voxel, dtype=np.int8)

            print(f"[OK] {os.path.basename(image_path)} -> Synapse_{sample_id}  shape={img_sub.shape}")
            saved += 1

    return saved


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Create nnUNet dataset by copying whole volumes; split into 4 quadrants if too large.")
    p.add_argument("--base_path", type=str, required=False,
                   default="/media/liushuo/data1/data/synapse_seg",
                   help="Base directory containing per-sample folders.")
    p.add_argument("--base_names", type=str, nargs="+", required=False,
                   default=['pp267','pp1776','pp1033','pp4001','pp3266','p545','pp0312','pp366','pp1189','pp387'],
                   help="Sample names (each has {name}/{name}.mrc and {name}/{name}_label.mrc).")
    p.add_argument("--output_dir", type=str, required=False,
                   default="/media/liushuo/data3/nnUNet_dataset/nnUNet_raw/Dataset012_10tomo_no_crop",
                   help="Destination nnUNet dataset root.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = generate_image_paths(args.base_path, args.base_names)
    num = copy_and_rename_whole_volumes(image_paths, args.output_dir)
    if num == 0:
        raise RuntimeError("No samples were written. Please check paths and file names.")

    create_dataset_json(args.output_dir, num_training=num)
    print(f"Done. Wrote {num} samples to {args.output_dir} (labels=int8, split-if-large).")


if __name__ == "__main__":
    main()
