from mrc.io import get_tomo, save_tomo
import numpy as np
from skimage.filters import frangi, sato, meijering
from skimage.morphology import dilation, ball
from skimage.measure import label


def extract_all_protein(tomo_data, method='Frangi') -> np.ndarray:
    if method == 'Frangi':
        protein_pro = frangi(
            tomo_data,
            sigmas=(0.5, 0.8, 1.0, 1.2, 1.6, 2.0),
            alpha=0.3,
            beta=0.8,
            gamma=4,
            black_ridges=True
        )
    elif method == 'Sato':
        protein_pro = sato(
            tomo_data,
            sigmas=(0.5,0.8,1.0,1.2,1.5),
            black_ridges=True
        )
    elif method == 'Meijering':
        protein_pro = meijering(
            tomo_data,
            sigmas=(0.5,0.8,1.0,1.2,1.5),
            black_ridges=True
        )
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
                                 dilation_radius: int = 7) -> np.ndarray:
    """
    Build the candidate region for vesicle-associated proteins by dilating the
    vesicle membrane and excluding the vesicle lumen/body label.
    """
    vesicle_memb_mask = np.zeros_like(mask_data, dtype=np.uint8)
    vesicle_memb_mask[mask_data == 9] = 1

    vesicle_protein_region = dilation(vesicle_memb_mask, ball(dilation_radius))
    vesicle_protein_region[mask_data == 4] = 0

    return vesicle_memb_mask, vesicle_protein_region
    
    
def main():
    tomo_path = '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266.mrc'
    mask_path = '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_label.mrc'

    # Output paths
    protein_pro_save_path = '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_protein_prob.mrc'
    vesicle_protein_label_save_path = '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_vesicle_protein_label.mrc'

    # Load data
    tomo_data = get_tomo(tomo_path)
    mask_data = get_tomo(mask_path)

    # Step 1: Extract protein probability
    protein_pro = extract_all_protein(tomo_data, method='Meijering')  # You can choose 'Frangi', 'Sato' or 'Meijering' as well

    # Save probability map if needed
    save_tomo(
        protein_pro.astype(np.float32),
        protein_pro_save_path,
        voxel_size=17.14,
        datetype=np.float32
    )

    # Step 2: Threshold to binary protein mask
    
    
    # # 2.1 if precomputed protein exist, Load precomputed protein probability map
    # protein_pro_path = f'/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_protein_prob.mrc'
    # protein_pro = get_tomo(protein_pro_path)
    
    protein_mask = np.zeros_like(protein_pro, dtype=np.uint8)
    # frange 0.0008 sato 0.15
    protein_mask[protein_pro > 0.15] = 1

    # Step 3: Vesicle membrane mask and retained vesicle-protein region
    vesicle_memb_mask, vesicle_protein_region = build_vesicle_protein_region(
        mask_data,
        dilation_radius=7
    )

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
    vesicle_protein_label = label(vesicle_protein_mask, connectivity=1)

    # Step 7: Keep only protein instances intersecting vesicle membrane
    vesicle_protein_label = keep_instances_intersecting_mask(
        vesicle_protein_label,
        vesicle_memb_mask
    )

    # Step 8: Save labeled protein instances
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
