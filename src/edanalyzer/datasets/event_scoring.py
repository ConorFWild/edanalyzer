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

    def __init__(self, zarr_path, sample_indexes, pos_train_pose_samples):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        # self.z_map_sample_metadata_table = self.root['z_map_sample_metadata']
        # self.xmap_sample_table = self.root['xmap_sample']
        # self.z_map_sample_table = self.root['z_map_sample']
        # self.pose_table = self.root['known_hit_pose']
        # self.ligand_data_table = self.root['ligand_data']
        # self.annotation_table = self.root['annotation']

        self.pandda_2_z_map_sample_metadata_table = self.root['pandda_2']['z_map_sample_metadata']
        self.pandda_2_xmap_sample_table = self.root['pandda_2']['xmap_sample']
        self.pandda_2_z_map_sample_table = self.root['pandda_2']['z_map_sample']
        self.pandda_2_pose_table = self.root['pandda_2']['known_hit_pose']
        self.pandda_2_ligand_data_table = self.root['pandda_2']['ligand_data']
        self.pandda_2_annotation_table = self.root['pandda_2']['annotation']

        # self.annotations = {
        #     self.z_map_sample_metadata_table[_x['event_map_table_idx']]['event_idx']: _x
        #     for _x
        #     in self.annotation_table
        # }
        # self.annotations = {
        #     _x['event_idx']: _x
        #     for _x
        #     in self.annotation_table
        # }
        self.pandda_2_annotations = {
            _x['event_idx']: _x
            for _x
            in self.pandda_2_annotation_table
        }

        self.sample_indexes = sample_indexes

        self.pos_train_pose_samples = pos_train_pose_samples

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the z map and pose
        if sample_idx[0] == 'normal':
            z_map_sample_metadata = self.z_map_sample_metadata_table[sample_idx[1]]
        else:
            z_map_sample_metadata = self.pandda_2_z_map_sample_metadata_table[sample_idx[1]]
        z_map_sample_idx = z_map_sample_metadata['idx']
        # print([sample_idx, z_map_sample_idx])
        assert sample_idx[1] == z_map_sample_idx
        # event_map_idx = pose_data['event_map_sample_idx']

        pose_data_idx = z_map_sample_metadata['pose_data_idx']
        if sample_idx[0] == 'normal':

            xmap_sample_data = self.xmap_sample_table[z_map_sample_idx]
            z_map_sample_data = self.z_map_sample_table[z_map_sample_idx]
            annotation = self.annotations[z_map_sample_metadata['event_idx']]
        else:
            xmap_sample_data = self.pandda_2_xmap_sample_table[z_map_sample_idx]
            z_map_sample_data = self.pandda_2_z_map_sample_table[z_map_sample_idx]
            annotation = self.pandda_2_annotations[z_map_sample_metadata['event_idx']]

        # If training replace with a random ligand
        rng = np.random.default_rng()
        # random_ligand = True
        if pose_data_idx != -1:
            # random_ligand_sample = rng.random()
            # if (random_ligand_sample > 0.5) & (annotation['partition'] == 'train'):
            #     random_ligand = False
            #     pose_data = self.pose_table[pose_data_idx]
            # else:
            #     pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
            if sample_idx[0] == 'normal':

                pose_data = self.pose_table[pose_data_idx]
            else:
                pose_data = self.pandda_2_pose_table[pose_data_idx]
        else:
            if sample_idx[0] == 'normal':

                # pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
                selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
                pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]
            else:
                selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
                pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]



        #
        xmap = _get_grid_from_hdf5(xmap_sample_data)
        z_map = _get_grid_from_hdf5(z_map_sample_data)

        # Subsample if training
        if annotation['partition'] == 'train':
            translation = 3*(2*(rng.random(3)-0.5))
            centroid = np.array([22.5,22.5,22.5]) + translation

        else:
            centroid = np.array([22.5,22.5,22.5])


        # Get sampling transform for the z map
        sample_array = np.zeros(
            (32, 32, 32),
            dtype=np.float32,
        )
        orientation = _get_random_orientation()
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid,
            n=32
        )

        # Get the sampling transform for the ligand map
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
                close_mask = poss_distances[np.linalg.norm(poss_distances, axis=1) < 3.5]
                total_mask[close_mask] = True

        else:
            total_mask = np.full(valid_elements.size, True)

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
            ligand_centroid,
            n=32
        )


        # Get sample images
        # xmap_sample = _sample_xmap_and_scale(
        #     xmap,
        #     transform,
        #     np.copy(sample_array)
        # )
        xmap_sample = np.copy(sample_array)
        z_map_sample = _sample_xmap_and_scale(
            z_map,
            transform,
            np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(
            z_map,
            transformed_residue,
        )
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            ligand_map_transform,
            # transform,
            np.copy(ligand_sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        if pose_data_idx != -1:
            image_decoded_density = _sample_xmap(
                ligand_mask_grid,
                transform,
                np.copy(sample_array)
            )
        elif pose_data_idx == -1:
            image_decoded_density = np.copy(sample_array)
        else:
            raise Exception

        # Make the image
        image_density = np.stack(
            [
                xmap_sample,
            ],
            axis=0
        )
        image_density_float = image_density.astype(np.float32)

        image_z = np.stack(
            [
                z_map_sample,
            ],
            axis=0
        )
        image_z_float = image_z.astype(np.float32)

        image_mol = np.stack(
            [
                image_ligand_mask,
            ],
            axis=0
        )
        image_mol_float = image_mol.astype(np.float32)

        image_decoded_density = np.stack(
            [
                image_decoded_density,
            ],
            axis=0
        )
        image_decoded_density_float = image_decoded_density.astype(np.float32)

        # res mask
        # if annotation['partition'] == 'train':
        #     grid = gemmi.FloatGrid(32, 32, 32)
        #     grid.spacegroup = gemmi.SpaceGroup('P1')
        #     uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
        #     grid.set_unit_cell(uc)
        #     grid_array = np.array(grid, copy=False)
        #     grid_array[:,:,:] = image_z_float[:,:,:]
        #     rsg = gemmi.transform_map_to_f_phi(grid)
        #     dmin = 1 + (1.0*rng.random())
        #     data = rsg.prepare_asu_data(dmin=float(dmin))
        #     grid_low_res = data.transform_f_phi_to_map(exact_size=[32, 32, 32])
        #     grid_low_res_array = np.array(grid_low_res, copy=False)
        #     image_z_float[:, :, :] = grid_low_res_array[:,:,:]

        # Make the annotation
        # if (pose_data_idx != -1) & (not random_ligand):
        #     hit = 1.0
        # elif (pose_data_idx != -1) & (random_ligand):
        #     hit = 0.0
        # if pose_data_idx != -1:
        #     hit = 1.0
        # elif pose_data_idx == -1:
        #     hit = 0.0
        # else:
        #     raise Exception
        if pose_data_idx != -1:
            hit = [0.0, 1.0]
        elif pose_data_idx == -1:
            hit = [1.0, 0.0]
        else:
            raise Exception

        label = np.array(hit)
        label_float = label.astype(np.float32)

        return (
            sample_idx,
            torch.from_numpy(image_density_float),
            torch.from_numpy(image_z_float),
            torch.from_numpy(image_mol_float),
            torch.from_numpy(image_decoded_density_float),
            torch.from_numpy(label_float)
        )
