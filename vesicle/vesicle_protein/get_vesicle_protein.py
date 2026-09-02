from mrc.io import get_tomo, save_tomo
import numpy as np
from skimage.filters import frangi, sato, meijering
from skimage.measure import label
from scipy.ndimage import find_objects, distance_transform_edt


def _get_filter_config(method: str):
    if method == 'Frangi':
        sigmas = (0.5, 1.0, 2.0, 3.0)
        return frangi, {
            'sigmas': sigmas,
            'alpha': 0.4,
            'beta': 0.6,
            'gamma': 4,
            'black_ridges': True
        }, max(sigmas)
    if method == 'Sato':
        sigmas = (0.5, 0.8, 1.0, 1.2, 1.5)
        return sato, {
            'sigmas': sigmas,
            'black_ridges': True
        }, max(sigmas)
    if method == 'Meijering':
        sigmas = (0.5, 0.8, 1.0, 1.2, 1.5)
        return meijering, {
            'sigmas': sigmas,
            'black_ridges': True
        }, max(sigmas)
    raise ValueError(f'Unsupported method: {method}')


def extract_all_protein(tomo_data, method='Frangi', region_mask=None) -> np.ndarray:
    filter_func, filter_kwargs, max_sigma = _get_filter_config(method)

    if region_mask is None:
        return filter_func(tomo_data, **filter_kwargs)

    region_mask = region_mask > 0
    protein_pro = np.zeros_like(tomo_data, dtype=np.float32)
    if not np.any(region_mask):
        return protein_pro

    region_label = label(region_mask, connectivity=1)
    region_slices = find_objects(region_label)
    total_regions = sum(1 for s in region_slices if s is not None)
    margin = int(np.ceil(max_sigma * 3))
    shape = tomo_data.shape
    print(f'Filtering {total_regions} region(s) in vesicle_protein_region...')

    processed = 0
    for region_id, region_slice in enumerate(region_slices, start=1):
        if region_slice is None:
            continue

        expanded_slice = []
        for dim, slc in enumerate(region_slice):
            start = max(0, slc.start - margin)
            stop = min(shape[dim], slc.stop + margin)
            expanded_slice.append(slice(start, stop))
        expanded_slice = tuple(expanded_slice)

        tomo_crop = tomo_data[expanded_slice]
        crop_response = filter_func(tomo_crop, **filter_kwargs)
        crop_region_mask = region_label[expanded_slice] == region_id
        protein_pro[expanded_slice][crop_region_mask] = crop_response[crop_region_mask]

        processed += 1
        percent = 100.0 * processed / total_regions if total_regions > 0 else 100.0
        print(
            f'\rFiltering progress: {processed}/{total_regions} ({percent:.1f}%)',
            end='',
            flush=True
        )

    if total_regions > 0:
        print()

    return protein_pro


def keep_instances_intersecting_mask(instance_label: np.ndarray,
                                     target_mask: np.ndarray) -> np.ndarray:
    """
    Keep only labeled instances that intersect with target_mask.
    Relabel kept instances to consecutive IDs starting from 1.
    """
    kept_label = np.zeros_like(instance_label, dtype=np.int32)

    # Find instance IDs that appear in target_mask region
    intersect_ids = np.unique(instance_label[target_mask > 0])

    # Remove background
    intersect_ids = intersect_ids[intersect_ids != 0]

    if len(intersect_ids) == 0:
        return kept_label

    # Create mapping: old_id -> new_id
    new_ids = np.arange(1, len(intersect_ids) + 1, dtype=np.int32)

    # Keep only intersecting instances
    mask_keep = np.isin(instance_label, intersect_ids)
    kept_label[mask_keep] = instance_label[mask_keep]

    # Relabel to consecutive IDs
    id_map = np.zeros(instance_label.max() + 1, dtype=np.int32)
    id_map[intersect_ids] = new_ids
    kept_label = id_map[kept_label]

    return kept_label


def build_vesicle_protein_region(mask_data: np.ndarray,
                                 dilation_radius: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the candidate region for vesicle-associated proteins by selecting all
    voxels within `dilation_radius` of the vesicle membrane, then excluding the
    vesicle lumen/body label.

    This replaces explicit morphological dilation with an EDT-based distance
    threshold, which is usually faster for large 3D volumes.
    """
    vesicle_memb_mask = (mask_data == 9)
    if not np.any(vesicle_memb_mask):
        return vesicle_memb_mask.astype(np.uint8), np.zeros_like(mask_data, dtype=np.uint8)

    # Distance to nearest membrane voxel for all non-membrane voxels.
    dist_to_membrane = distance_transform_edt(~vesicle_memb_mask)
    vesicle_protein_region = dist_to_membrane <= dilation_radius
    vesicle_protein_region[mask_data == 4] = False

    return vesicle_memb_mask.astype(np.uint8), vesicle_protein_region.astype(np.uint8)
    
    
def main():
    tomo_path = '/media/liushuo/data1/data/synapse_seg/pp676/pp676.mrc'
    mask_path = '/media/liushuo/data1/data/synapse_seg/pp676/synapse_seg/pp676_semantic_label.mrc'

    # Output paths
    protein_pro_save_path = '/media/liushuo/data1/data/synapse_seg/pp676/synapse_seg/protein_prob.mrc'
    vesicle_protein_label_save_path = '/media/liushuo/data1/data/synapse_seg/pp676/synapse_seg/vesicle_protein_label.mrc'

    # Load data
    tomo_data = get_tomo(tomo_path)
    mask_data = get_tomo(mask_path)

    # Step 1: Vesicle membrane mask and retained vesicle-protein region
    vesicle_memb_mask, vesicle_protein_region = build_vesicle_protein_region(
        mask_data,
        dilation_radius=7
    )

    # Step 2: Extract or load protein probability
    use_precomputed_protein_pro = False
    if use_precomputed_protein_pro:
        # 2.1 if precomputed protein exist, Load precomputed protein probability map
        protein_pro_path = '/media/liushuo/data1/data/synapse_seg/pp676/synapse_seg/protein_prob.mrc'
        protein_pro = get_tomo(protein_pro_path)
    else:
        protein_pro = extract_all_protein(
            tomo_data,
            method='Frangi',
            region_mask=vesicle_protein_region
        )
        save_tomo(
            protein_pro.astype(np.float32),
            protein_pro_save_path,
            voxel_size=17.14,
            datetype=np.float32
        )

    # Step 3: Threshold to binary protein mask
    protein_mask = np.zeros_like(protein_pro, dtype=np.uint8)
    # frange 0.0008 sato 0.15
    protein_mask[protein_pro > 0.0002] = 1

    # Step 4: Legacy non-vesicle region filtering, kept here only for reference
    # non_vesicle_protein_mask = np.zeros_like(protein_mask, dtype=np.uint8)
    # non_vesicle_protein_mask[(mask_data != 0) & (mask_data != 9) & (mask_data != 4)] = 1
    # non_vesicle_protein_mask = dilation(non_vesicle_protein_mask, cube(5))

    # Step 5: Keep proteins only inside the retained vesicle-protein region
    vesicle_protein_mask = np.zeros_like(protein_mask, dtype=np.uint8)
    vesicle_protein_mask[
        (protein_mask == 1) & (vesicle_protein_region > 0)
    ] = 1

    # Step 6: Label connected components as independent instances
    vesicle_protein_label = label(vesicle_protein_mask, connectivity=2)

    # Step 7: Keep only protein instances intersecting vesicle membrane
    vesicle_protein_label = keep_instances_intersecting_mask(
        vesicle_protein_label,
        vesicle_memb_mask
    )

    # Step 8: Remove protein voxels on vesicle membrane and relabel
    vesicle_protein_label[vesicle_memb_mask != 0] = 0
    vesicle_protein_label = label(vesicle_protein_label > 0, connectivity=1)

    # Step 9: Save labeled protein instances
    save_tomo(
        vesicle_protein_label.astype(np.int16),
        vesicle_protein_label_save_path,
        voxel_size=17.14,
        datetype=np.int16
    )

    print(f'Saved protein probability to: {protein_pro_save_path}')
    print(f'Saved vesicle protein instance label to: {vesicle_protein_label_save_path}')
    print(f'Number of kept protein instances: {vesicle_protein_label.max()}')
    
    
if __name__ == '__main__':
    main()
