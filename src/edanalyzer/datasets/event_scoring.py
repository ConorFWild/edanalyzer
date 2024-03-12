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
        self.annotations = {
            self.z_map_sample_metadata_table[_x['event_map_table_idx']]['event_idx']: _x
            for _x
            in self.annotation_table}

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the z map and pose
        z_map_sample_metadata = self.z_map_sample_metadata_table[sample_idx[1]]
        z_map_sample_idx = z_map_sample_metadata['idx']
        # print([sample_idx, z_map_sample_idx])
        assert sample_idx[1] == z_map_sample_idx
        # event_map_idx = pose_data['event_map_sample_idx']

        pose_data_idx = z_map_sample_metadata['pose_data_idx']
        rng = np.random.default_rng()
        random_ligand = True
        if pose_data_idx != -1:
            random_ligand_sample = rng.random()
            if random_ligand_sample > 0.5:
                random_ligand = False
                pose_data = self.pose_table[pose_data_idx]
            else:
                pose_data = self.pose_table[rng.integers(0,len(self.pose_table))]
        else:
            pose_data = self.pose_table[rng.integers(0,len(self.pose_table))]
        z_map_sample_data = self.z_map_sample_table[z_map_sample_idx]
        annotation = self.annotations[z_map_sample_metadata['event_idx']]

        #
        z_map = _get_grid_from_hdf5(z_map_sample_data)

        # Subsample if training
        if annotation['partition'] == 'train':
            translation = 3*(2*(rng.random(3)-0.5))
            centroid = np.array([22.5,22.5,22.5]) + translation


        else:
            centroid = np.array([22.5,22.5,22.5])

        # Get sampling transform for the z map
        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )
        orientation = _get_random_orientation()
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get the sampling transform for the ligand map
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]
        ligand_sample_array = np.zeros(
            (32, 32, 32),
            dtype=np.float32,
        )
        ligand_orientation = _get_random_orientation()
        transformed_residue = _get_res_from_arrays(
            valid_poss,
            valid_elements,
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
                image_ligand_mask,
            ],
            axis=0
        )
        image_mol_float = image_mol.astype(np.float32)

        # Make the annotation
        if (pose_data_idx != -1) & (not random_ligand):
            hit = 1.0
        elif (pose_data_idx != -1) & (random_ligand):
            hit = 0.0
        elif pose_data_idx == -1:
            hit = 0.0
        else:
            raise Exception

        label = np.array(hit)
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_density_float), torch.from_numpy(image_mol_float), torch.from_numpy(
            label_float)
