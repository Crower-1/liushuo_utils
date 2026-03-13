import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
from skimage.morphology import ball, dilation

from mrc.io import get_tomo, save_tomo


_VESICLE_PROTEIN_LABEL = None
_VESICLE_LABEL = None
_STRUCT_ELEM = None


def _collect_intersections_for_vesicle(vesicle_id: int) -> tuple[int, np.ndarray]:
    vesicle_instance_mask = _VESICLE_LABEL == vesicle_id
    dilated_vesicle_mask = dilation(vesicle_instance_mask, _STRUCT_ELEM)

    intersecting_proteins = np.unique(
        _VESICLE_PROTEIN_LABEL[dilated_vesicle_mask]
    )
    intersecting_proteins = intersecting_proteins[intersecting_proteins != 0]

    return vesicle_id, intersecting_proteins.astype(np.int32)


def _print_progress(done: int, total: int) -> None:
    percent = 100.0 * done / total
    print(f'Processed vesicles: {done}/{total} ({percent:.1f}%)')


def find_connector_proteins(vesicle_protein_label: np.ndarray,
                            vesicle_label: np.ndarray,
                            dilation_radius: int = 5,
                            max_workers: int | None = None) -> np.ndarray:
    """
    Keep protein instances that intersect with at least two dilated vesicle
    instances. Protein instance IDs are preserved in the output.
    """
    global _VESICLE_PROTEIN_LABEL, _VESICLE_LABEL, _STRUCT_ELEM

    protein_to_vesicles = defaultdict(set)
    _VESICLE_PROTEIN_LABEL = vesicle_protein_label
    _VESICLE_LABEL = vesicle_label
    _STRUCT_ELEM = ball(dilation_radius)

    vesicle_ids = np.unique(vesicle_label)
    vesicle_ids = vesicle_ids[vesicle_ids != 0]
    total_vesicles = len(vesicle_ids)

    if total_vesicles == 0:
        print('No vesicle instances found in vesicle_label.')
        return np.zeros_like(vesicle_protein_label, dtype=np.int32)

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, min(max_workers, total_vesicles))

    print(f'Found {total_vesicles} vesicle instances.')
    print(
        f'Starting dilation and intersection analysis with {max_workers} worker(s).'
    )

    progress_interval = max(1, total_vesicles // 20)

    if max_workers == 1:
        for done, vesicle_id in enumerate(vesicle_ids, start=1):
            current_vesicle_id, intersecting_proteins = (
                _collect_intersections_for_vesicle(int(vesicle_id))
            )
            for protein_id in intersecting_proteins:
                protein_to_vesicles[int(protein_id)].add(int(current_vesicle_id))

            if done % progress_interval == 0 or done == total_vesicles:
                _print_progress(done, total_vesicles)
    else:
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(_collect_intersections_for_vesicle, int(vesicle_id))
                for vesicle_id in vesicle_ids
            ]

            done = 0
            for future in as_completed(futures):
                current_vesicle_id, intersecting_proteins = future.result()
                for protein_id in intersecting_proteins:
                    protein_to_vesicles[int(protein_id)].add(int(current_vesicle_id))

                done += 1
                if done % progress_interval == 0 or done == total_vesicles:
                    _print_progress(done, total_vesicles)

    connector_ids = np.array(
        [protein_id for protein_id, ids in protein_to_vesicles.items() if len(ids) >= 2],
        dtype=np.int32
    )

    connector_label = np.zeros_like(vesicle_protein_label, dtype=np.int32)
    if connector_ids.size == 0:
        return connector_label

    connector_mask = np.isin(vesicle_protein_label, connector_ids)
    connector_label[connector_mask] = vesicle_protein_label[connector_mask]

    return connector_label


def main():
    vesicle_protein_label_path = (
        '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_vesicle_protein_label.mrc'
    )
    vesicle_label_path = (
        '/media/liushuo/data1/data/synapse_seg/pp3266/vesicle/pp3266_vesicle_label.mrc'
    )
    connector_label_save_path = (
        '/media/liushuo/data1/data/synapse_seg/pp3266/pp3266_connector_label.mrc'
    )

    print(f'Loading vesicle protein label: {vesicle_protein_label_path}')
    vesicle_protein_label = get_tomo(vesicle_protein_label_path)
    print(f'Loading vesicle label: {vesicle_label_path}')
    vesicle_label = get_tomo(vesicle_label_path)

    print('Finding connector proteins...')
    connector_label = find_connector_proteins(
        vesicle_protein_label,
        vesicle_label,
        dilation_radius=4
    )

    print(f'Saving connector label: {connector_label_save_path}')
    save_tomo(
        connector_label.astype(np.int16),
        connector_label_save_path,
        voxel_size=17.14,
        datetype=np.int16
    )

    connector_ids = np.unique(connector_label)
    connector_ids = connector_ids[connector_ids != 0]

    print(f'Saved connector label to: {connector_label_save_path}')
    print(f'Number of connector proteins: {len(connector_ids)}')


if __name__ == '__main__':
    main()
