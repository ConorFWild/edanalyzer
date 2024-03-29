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


@dataclasses.dataclass
class BuildScoringDatasetItem:
    experiment_model_dir: str
    pandda_path: str
    dtag: str
    model_idx: int
    event_idx: int
    known_hit_key: str
    ligand_key: str
    rmsd: float
    score: float
    size: float
    local_strength: float
    rscc: float
    signal: float
    noise: float
    signal_noise: float
    x_ligand: float
    y_ligand: float
    z_ligand: float
    x: float
    y: float
    z: float
    build_path: str
    bdc: float
    xmap_path: str
    mean_map_path: str
    mtz_path: str
    zmap_path: str
    train_test: str


class BuildScoringDataset(Dataset):
    def __init__(
            self,
            data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )

        try:
            structure = _get_structure_from_path(sample.build_path)
            xmap = _load_xmap_from_path(sample.xmap_path)
            mean_map = _load_xmap_from_path(sample.mean_map_path)
            z_map = _load_xmap_from_path(sample.zmap_path)
            raw_xmap = _load_xmap_from_mtz_path(sample.mtz_path)
        except Exception as e:
            print(e)
            image = np.stack(
                [
                    sample_array,
                    sample_array,
                    sample_array,
                    sample_array,
                ],
                axis=0,
            )
            image_float = image.astype(np.float32)

            label = np.array(3.0)
            label_float = label.astype(np.float32)
            return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)

        residue = _get_res_from_structure_chain_res(
            structure,
            0,
            0
        )

        # Get sampling transform
        # orientation = _get_identity_matrix()
        orientation = _get_random_orientation()
        centroid = _get_centroid_from_res(residue)
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get sample image

        # event_map = _load_xmap_from_path(sample.event_map_path)
        # event_map_sample = _sample_xmap_and_scale(
        #     event_map, transform, np.copy(sample_array)
        # )

        xmap_sample = _sample_xmap_and_scale(
            xmap, transform, np.copy(sample_array)
        )

        mean_map_sample = _sample_xmap_and_scale(
            mean_map, transform, np.copy(sample_array)
        )

        z_map_sample = _sample_xmap_and_scale(
            z_map, transform, np.copy(sample_array)
        )

        raw_xmap_sample = _sample_xmap_and_scale(
            raw_xmap, transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(xmap, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )
        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask > 0.9] = 1.0

        # Make the image
        image = np.stack(
            [
                xmap_sample * image_ligand_mask,
                mean_map_sample * image_ligand_mask,
                z_map_sample * image_ligand_mask,
                raw_xmap_sample * image_ligand_mask,
            ],
            axis=0,
        )
        image_float = image.astype(np.float32)

        # Make the annotation
        if sample.rmsd > 3.0:
            rmsd = 3.0
        else:
            rmsd = sample.rmsd
        label = np.array(rmsd)
        label_float = label.astype(np.float32)

        return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)






def _get_res_from_hdf5(pose_data):
    return _get_res_from_arrays(pose_data['positions'], pose_data['elements'])


class BuildScoringDatasetHDF5(Dataset):

    def __init__(self, root, sample_indexes):
        # self.data = data
        self.root = root

        self.event_map_table = root.event_map_sample
        self.pose_table = root.known_hit_pose
        self.delta_table = root.delta
        self.annotation_table = root.annotation

        self.pandda_2_event_map_table = root.pandda_2_event_map_sample
        self.pandda_2_pose_table = root.pandda_2_known_hit_pose
        self.pandda_2_delta_table = root.pandda_2_delta
        self.pandda_2_annotation_table = root.pandda_2_annotation

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the event map and pose
        if sample_idx[0] == 'normal':
            pose_data = self.pose_table[sample_idx[1]]
            event_map_idx = pose_data['event_map_sample_idx']

            event_map_data = self.event_map_table[event_map_idx]
            delta = self.delta_table[sample_idx[1]]
            annotation = self.annotation_table[event_map_idx]
        else:
            pose_data = self.pandda_2_pose_table[sample_idx[1]]
            event_map_idx = pose_data['event_map_sample_idx']

            event_map_data = self.pandda_2_event_map_table[event_map_idx]
            delta = self.pandda_2_delta_table[sample_idx[1]]
            annotation = self.pandda_2_annotation_table[event_map_idx]

        #
        event_map = _get_grid_from_hdf5(event_map_data)

        # Get the valid data
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]
        valid_deltas = delta['delta'][valid_mask]

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

        # Get sampling transform
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

        # Get sample image
        event_map_sample = _sample_xmap_and_scale(
            event_map, transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(event_map, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        # Make the image
        image = np.stack(
            [event_map_sample * image_ligand_mask, ],
            axis=0
        )

        image_float = image.astype(np.float32)

        # Make the annotation
        if sample_idx[0] == 'normal':

            rmsd = np.sqrt(np.mean(np.square(valid_deltas[total_mask])))

            if rmsd > 3.0:
                rmsd = 3.0

        else:
            rmsd = 3.0

        label = np.array(rmsd)
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_float), torch.from_numpy(label_float)


class BuildScoringDatasetCorrelation(Dataset):

    def __init__(self, root, sample_indexes):
        # self.data = data
        self.root = root

        self.event_map_table = root.event_map_sample
        self.pose_table = root.known_hit_pose
        self.delta_table = root.delta
        self.annotation_table = root.annotation

        # self.pandda_2_event_map_table = root.pandda_2_event_map_sample
        # self.pandda_2_pose_table = root.pandda_2_known_hit_pose
        # self.pandda_2_delta_table = root.pandda_2_delta
        # self.pandda_2_annotation_table = root.pandda_2_annotation

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the event map and pose
        # if sample_idx[0] == 'normal':
        pose_data = self.pose_table[sample_idx[1]]
        event_map_idx = pose_data['event_map_sample_idx']

        event_map_data = self.event_map_table[event_map_idx]
        # mtz_map_data = self.
        delta = self.delta_table[sample_idx[1]]
        annotation = self.annotation_table[event_map_idx]
        reference_pose_data = self.pose_table[delta['pose_idx']]
        # else:
        #     pose_data = self.pandda_2_pose_table[sample_idx[1]]
        #     event_map_idx = pose_data['event_map_sample_idx']
        #
        #     event_map_data = self.pandda_2_event_map_table[event_map_idx]
        #     delta = self.pandda_2_delta_table[sample_idx[1]]
        #     annotation = self.pandda_2_annotation_table[event_map_idx]

        #
        reference_event_map = _get_grid_from_hdf5(event_map_data)
        event_map = _get_grid_from_hdf5(event_map_data)
        # mtz_map = _get_grid_from_hdf5(mtz_map_data)

        # Get the valid data
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]
        valid_deltas = delta['delta'][valid_mask]

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
        reference_poss = reference_pose_data['positions'][valid_mask][total_mask]
        ref_res = _get_res_from_arrays(reference_poss, valid_elements[total_mask])
        alignment = R.align_vectors(  # Alignment rotates lig being sampled to overlap lig in ref frame - i.e.
            reference_poss - np.mean(reference_poss, axis=0),  # Align to
            valid_poss[total_mask] - np.mean(valid_poss[total_mask], axis=0),
        )[0].as_matrix()
        ref_orientation = np.matmul(alignment, orientation)
        ref_centroid = _get_centroid_from_res(ref_res)
        ref_transform = _get_transform_from_orientation_centroid(
            ref_orientation,
            ref_centroid
        )

        # Get sample image
        event_map_sample = _sample_xmap_and_scale(
            event_map, transform, np.copy(sample_array)
        )
        ref_event_map_sample = _sample_xmap_and_scale(
            reference_event_map, ref_transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(event_map, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        masked_event_map = event_map_sample * image_ligand_mask
        masked_reference_event_map = ref_event_map_sample * image_ligand_mask

        sample_mat = np.hstack(
            (
                masked_event_map[masked_event_map != 0.0].reshape(-1, 1),
                masked_reference_event_map[masked_event_map != 0.0].reshape(-1, 1)
            )
        )
        # print(sample_mat)

        corrmat = np.corrcoef(
            sample_mat.T
        )
        # print(corrmat)
        corr = corrmat[0, 1]

        # print([corr, sample_idx, delta['pose_idx']])

        # assert (corr == 1.0) | (sample_idx[1] != delta['pose_idx'])

        # Make the image
        image = np.stack(
            [event_map_sample * image_ligand_mask, ],
            axis=0
        )

        image_float = image.astype(np.float32)

        # Make the annotation
        # if sample_idx[0] == 'normal':
        #
        #     rmsd = np.sqrt(np.mean(np.square(valid_deltas[total_mask])))
        #
        #     if rmsd > 3.0:
        #         rmsd = 3.0
        #
        # else:
        #     rmsd = 3.0

        label = np.array(corr)
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_float), torch.from_numpy(label_float)


class BuildScoringDatasetCorrelationZarr(Dataset):

    def __init__(self, zarr_path, sample_indexes):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        self.event_map_table = self.root['event_map_sample']
        self.mtz_map_table = self.root['mtz_sample']
        self.pose_table = self.root['known_hit_pose']
        self.delta_table = self.root['delta']
        self.annotation_table = self.root['annotation']

        # self.pandda_2_event_map_table = root.pandda_2_event_map_sample
        # self.pandda_2_pose_table = root.pandda_2_known_hit_pose
        # self.pandda_2_delta_table = root.pandda_2_delta
        # self.pandda_2_annotation_table = root.pandda_2_annotation

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the event map and pose
        # if sample_idx[0] == 'normal':
        pose_data = self.pose_table[sample_idx[1]]
        event_map_idx = pose_data['event_map_sample_idx']

        event_map_data = self.event_map_table[event_map_idx]
        mtz_map_data = self.mtz_map_table[event_map_idx]
        delta = self.delta_table[sample_idx[1]]
        annotation = self.annotation_table[event_map_idx]
        reference_pose_data = self.pose_table[delta['pose_idx']]
        # else:
        #     pose_data = self.pandda_2_pose_table[sample_idx[1]]
        #     event_map_idx = pose_data['event_map_sample_idx']
        #
        #     event_map_data = self.pandda_2_event_map_table[event_map_idx]
        #     delta = self.pandda_2_delta_table[sample_idx[1]]
        #     annotation = self.pandda_2_annotation_table[event_map_idx]

        #
        reference_event_map = _get_grid_from_hdf5(event_map_data)
        event_map = _get_grid_from_hdf5(event_map_data)
        mtz_map = _get_grid_from_hdf5(mtz_map_data)

        # Get the valid data
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]
        valid_deltas = delta['delta'][valid_mask]

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
        reference_poss = reference_pose_data['positions'][valid_mask][total_mask]
        ref_res = _get_res_from_arrays(reference_poss, valid_elements[total_mask])
        alignment = R.align_vectors(  # Alignment rotates lig being sampled to overlap lig in ref frame - i.e.
            reference_poss - np.mean(reference_poss, axis=0),  # Align to
            valid_poss[total_mask] - np.mean(valid_poss[total_mask], axis=0),
        )[0].as_matrix()
        ref_orientation = np.matmul(alignment, orientation)
        ref_centroid = _get_centroid_from_res(ref_res)
        ref_transform = _get_transform_from_orientation_centroid(
            ref_orientation,
            ref_centroid
        )

        # Get sample image
        event_map_sample = _sample_xmap_and_scale(
            event_map, transform, np.copy(sample_array)
        )
        ref_event_map_sample = _sample_xmap_and_scale(
            reference_event_map, ref_transform, np.copy(sample_array)
        )
        mtz_map_sample = _sample_xmap_and_scale(
            mtz_map, transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(event_map, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        masked_event_map = event_map_sample * image_ligand_mask
        masked_reference_event_map = ref_event_map_sample * image_ligand_mask

        sample_mat = np.hstack(
            (
                masked_event_map[masked_event_map != 0.0].reshape(-1, 1),
                masked_reference_event_map[masked_event_map != 0.0].reshape(-1, 1)
            )
        )
        # print(sample_mat)

        corrmat = np.corrcoef(
            sample_mat.T
        )
        # print(corrmat)
        corr = corrmat[0, 1]

        # print([corr, sample_idx, delta['pose_idx']])

        # assert (corr == 1.0) | (sample_idx[1] != delta['pose_idx'])

        # Make the image
        image = np.stack(
            [
                event_map_sample * image_ligand_mask,
                mtz_map_sample * image_ligand_mask
            ],
            axis=0
        )

        image_float = image.astype(np.float32)

        # Make the annotation
        # if sample_idx[0] == 'normal':
        #
        #     rmsd = np.sqrt(np.mean(np.square(valid_deltas[total_mask])))
        #
        #     if rmsd > 3.0:
        #         rmsd = 3.0
        #
        # else:
        #     rmsd = 3.0

        label = np.array(corr)
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_float), torch.from_numpy(label_float)


def _get_predicted_density_from_res(residue, event_map):
    optimized_structure = gemmi.Structure()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')

    chain.add_residue(residue)
    model.add_chain(chain)
    optimized_structure.add_model(model)

    # Get the electron density of the optimized structure
    optimized_structure.cell = event_map.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = 2.0  #*2
    # dencalc.rate = 1.5
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    # dencalc.initialize_grid_to_size(event_map.nu, event_map.nv, event_map.nw)
    dencalc.put_model_density_on_grid(optimized_structure[0])
    calc_grid = dencalc.grid

    return calc_grid


class BuildScoringDatasetSyntheticCorrelationZarr(Dataset):

    def __init__(self, zarr_path, sample_indexes):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        self.event_map_table = self.root['event_map_sample']
        self.mtz_map_table = self.root['mtz_sample']
        self.pose_table = self.root['known_hit_pose']
        self.delta_table = self.root['delta']
        self.annotation_table = self.root['annotation']

        self.sample_indexes = sample_indexes

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample idx
        sample_idx = self.sample_indexes[idx]

        # Get the event map and pose
        # if sample_idx[0] == 'normal':
        pose_data = self.pose_table[sample_idx[1]]
        event_map_idx = pose_data['event_map_sample_idx']

        event_map_data = self.event_map_table[event_map_idx]
        mtz_map_data = self.mtz_map_table[event_map_idx]
        delta = self.delta_table[sample_idx[1]]
        annotation = self.annotation_table[event_map_idx]
        reference_pose_data = self.pose_table[delta['pose_idx']]

        #
        event_map = _get_grid_from_hdf5(event_map_data)
        mtz_map = _get_grid_from_hdf5(mtz_map_data)

        # Get the valid data
        valid_mask = pose_data['elements'] != 0
        valid_poss = pose_data['positions'][valid_mask]
        valid_elements = pose_data['elements'][valid_mask]
        valid_deltas = delta['delta'][valid_mask]

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
        reference_poss = reference_pose_data['positions'][valid_mask][total_mask]
        ref_res = _get_res_from_arrays(reference_poss, valid_elements[total_mask])
        alignment = R.align_vectors(  # Alignment rotates lig being sampled to overlap lig in ref frame - i.e.
            reference_poss - np.mean(reference_poss, axis=0),  # Align to
            valid_poss[total_mask] - np.mean(valid_poss[total_mask], axis=0),
        )[0].as_matrix()
        ref_orientation = np.matmul(alignment, orientation)
        ref_centroid = _get_centroid_from_res(ref_res)
        ref_transform = _get_transform_from_orientation_centroid(
            ref_orientation,
            ref_centroid
        )

        predicted_map = _get_predicted_density_from_res(residue, event_map)
        reference_predicted_map = _get_predicted_density_from_res(ref_res, event_map)


        # Get sample image
        event_map_sample = _sample_xmap_and_scale(
            event_map, transform, np.copy(sample_array)
        )

        predicted_map_sample = _sample_xmap_and_scale(
            predicted_map, transform, np.copy(sample_array)
        )

        ref_predictited_map_sample = _sample_xmap_and_scale(
            reference_predicted_map, transform, np.copy(sample_array)
        )

        mtz_map_sample = _sample_xmap_and_scale(
            mtz_map, transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(event_map, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )

        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

        masked_event_map = event_map_sample * image_ligand_mask
        masked_predicted_map = predicted_map_sample * image_ligand_mask
        masked_ref_predicted_map = ref_predictited_map_sample * image_ligand_mask

        mask = (masked_predicted_map != 0.0) & (masked_ref_predicted_map != 0.0)
        mask_size = np.sum(mask)
        if mask_size < 10:
            corr = 0.0
            base_corr = 0.0
        else:

            sample_mat = np.hstack(
                (
                    masked_predicted_map[mask].reshape(-1, 1),
                    masked_ref_predicted_map[mask].reshape(-1, 1)
                )
            )
            # print(sample_mat)

            corrmat = np.corrcoef(
                sample_mat.T
            )
            # print(corrmat)
            corr = corrmat[0, 1]

            base_corr = np.corrcoef(
                np.hstack(
                    (
                        masked_event_map[mask].reshape(-1, 1),
                        masked_ref_predicted_map[mask].reshape(-1, 1)
                    )
                ).T
            )[0,1]

        rmsd = np.sqrt(np.mean(np.square(valid_deltas[total_mask])))
        # if (rmsd < 0.1) & (corr <= 0.9):
        #
        #     raise Exception(f"RMSD is {rmsd} but correlation is {corr}")

        # if corr >= 0.9:
        #     print(f"HIGH CORR: RMSD is {round(rmsd, 2)}, mask size is {round(mask_size, 2)} and correlation is {round(corr, 2)} and base correlation is {round(base_corr, 2)}")
        #
        # if corr <= 0.1:
        #     if corr > 0.0:
        #         print(f"LOW CORR: RMSD {round(rmsd, 2)}, mask size: {round(mask_size, 2)}, mask pred sum: {round(np.sum(masked_predicted_map[mask]),2)}, mask ref pred sum: {round(np.sum(masked_ref_predicted_map[mask]),2)}, correlation: {round(corr, 2)}, base correlation: {round(base_corr, 2)}, ")


        # print([corr, sample_idx, delta['pose_idx']])

        # assert (corr == 1.0) | (sample_idx[1] != delta['pose_idx'])

        # Make the image
        image = np.stack(
            [
                event_map_sample * image_ligand_mask,
                mtz_map_sample * image_ligand_mask
            ],
            axis=0
        )

        image_float = image.astype(np.float32)

        # Make the annotation
        if sample_idx[0] == 'normal':

            _rmsd = np.sqrt(np.mean(np.square(valid_deltas[total_mask])))

            if _rmsd > 2.5:
                _rmsd = 2.5

            rmsd = 1.0-((2/2.5)*_rmsd)

        else:
            rmsd = -1.0

        label = np.array([corr, base_corr, rmsd])
        label_float = label.astype(np.float32)

        return sample_idx, torch.from_numpy(image_float), torch.from_numpy(label_float)
