import dataclasses

import numpy as np
import torch
import gemmi
from scipy.spatial.transform import Rotation as R
import zarr

from torch.utils.data import Dataset

from .base import (
    _load_xmap_from_mtz_path,
    _load_xmap_from_path,
    _sample_xmap_and_scale,
    _get_ligand_mask_float,
    _sample_xmap,
    _get_identity_matrix,
    _get_random_orientation,
    _get_centroid_from_res,
    _get_transform_from_orientation_centroid,
    _get_res_from_structure_chain_res,
    _get_structure_from_path,
    _get_res_from_arrays,
    _get_grid_from_hdf5
)


class EventScoringDataset(Dataset):

    def __init__(self, zarr_path, sample_indexes):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        self.z_map_sample_metadata_table = self.root['z_map_sample_metadata']
        self.z_map_sample_table = self.root['z_map_sample']
        self.pose_table = self.root['known_hit_pose']
        self.ligand_data_table = self.root['ligand_data']
        self.annotation_table = self.root['annotation']

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the z map and pose
        pose_data = self.pose_table[sample_idx[1]]
        event_map_idx = pose_data['event_map_sample_idx']
        mtz_map_data = self.mtz_map_table[event_map_idx]
        annotation = self.annotation_table[event_map_idx]

        #
        z_map = _get_grid_from_hdf5(mtz_map_data)

        # Get the valid data
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]

        # Subsample if training
        if annotation['partition'] == 'train':
            rng = np.random.default_rng()
            num_centres = rng.integers(1, 5)

            # For each centre mask atoms close to it
            total_mask = np.full(valid_elements.size, False)
            for _centre in num_centres:
                selected_atom = rng.integers(0, valid_elements.size)
                poss_distances = valid_poss - valid_poss[selected_atom, :].reshape((1, 3))
                close_mask = poss_distances[np.linalg.norm(poss_distances, axis=1) < 2.5]
                total_mask[close_mask] = True

        else:
            total_mask = np.full(valid_elements.size, True)

        residue = _get_res_from_arrays(valid_poss[total_mask], valid_elements[total_mask])

        # residue = _get_res_from_hdf5(pose_data)

        # Get the event from the database
        # event = self.data[event_map_data['event_idx']]

        # Get sampling transform for the event map
        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )
        orientation = _get_random_orientation()
        centroid = _get_centroid_from_res(residue)
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get the sampling transform for the reference event map
        ligand_sample_array = np.zeros(
            (32, 32, 32),
            dtype=np.float32,
        )
        ligand_orientation = _get_random_orientation()
        transformed_residue = _get_res_from_arrays(
            valid_poss[total_mask],
            valid_elements[total_mask],
        )

        ligand_centroid = _get_centroid_from_res(transformed_residue)
        ligand_map_transform = _get_transform_from_orientation_centroid(
            ligand_orientation,
            ligand_centroid
        )

        # Get sample images
        z_map_sample = _sample_xmap_and_scale(
            z_map,
            transform,
            np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(z_map, transformed_residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            ligand_map_transform,
            np.copy(ligand_sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        # Make the image
        image_density = np.stack(
            [
                z_map_sample,
            ],
            axis=0
        )
        image_density_float = image_density.astype(np.float32)

        image_mol = np.stack(
            [
                z_map_sample,
            ],
            axis=0
        )
        image_mol_float = image_mol.astype(np.float32)

        # Make the annotation
        if annotation['annotation']:
            hit = 1.0
        else:
            hit = 0.0

        label = np.array(hit)
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_density_float), torch.from_numpy(image_mol_float), torch.from_numpy(label_float)
