import dataclasses
import itertools

import numpy as np
import torch
import gemmi
from scipy.spatial.transform import Rotation as R
import zarr
from rich import print as rprint
import pandas as pd

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
    _get_grid_from_hdf5,
    _get_ed_mask_float
)


from scipy.ndimage import gaussian_filter

from skimage.segmentation import expand_labels



patch_lower = [x for x in range(8)]
patch_upper = [x for x in range(24,32)]
patch_set = [patch_lower, patch_upper]

# print(patch_set)

patch_arrays = [
    np.array([[x, y, z] for x, y, z in itertools.product(xs, ys, zs)])
    for xs, ys, zs
    in itertools.product(patch_set, patch_set, patch_set)
]

# print(patch_arrays[0])

patches = [
    (patch[:, 0].flatten(), patch[:,1].flatten(), patch[:,2].flatten(), )
    for patch
    in patch_arrays
]
# print(patches[0])


def get_mask_array():
    rng = np.random.default_rng()
    mask_array = np.ones((32, 32, 32), dtype=np.float32)
    for patch in patches:
        if rng.random() > 0.5:
            mask_array[patch] = 0.0

    return mask_array


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
    dencalc.d_min = 0.75  #*2
    # dencalc.rate = 1.5
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    # dencalc.initialize_grid_to_size(event_map.nu, event_map.nv, event_map.nw)
    dencalc.put_model_density_on_grid(optimized_structure[0])
    calc_grid = dencalc.grid

    return calc_grid

def _get_overlap_volume(orientation, centroid, known_hit_pose_residue, decoy_residue):
    transform = _get_transform_from_orientation_centroid(
        orientation,
        centroid,
        n=64,
        sd=0.25
    )

    known_hit_score_mask_grid = _get_ligand_mask_float(
        known_hit_pose_residue,
        radius=2.5,
        n=180,
        r=45.0
    )

    decoy_score_mask_grid = _get_ligand_mask_float(
        decoy_residue,
        radius=2.5,
        n=180,
        r=45.0
    )

    known_hit_predicted_density = _get_predicted_density_from_res(known_hit_pose_residue, known_hit_score_mask_grid)
    decoy_predicted_density = _get_predicted_density_from_res(decoy_residue, decoy_score_mask_grid)

    decoy_score_mask_arr = np.array(decoy_score_mask_grid, copy=False)
    decoy_predicted_density_arr = np.array(decoy_predicted_density, copy=False)
    known_hit_score_mask_arr = np.array(known_hit_score_mask_grid, copy=False)
    known_hit_predicted_density_arr = np.array(known_hit_predicted_density, copy=False)


    sel = decoy_score_mask_arr > 0.0

    decoy_predicted_density_sel = decoy_predicted_density_arr[sel]
    known_hit_predicted_density_sel = known_hit_predicted_density_arr[sel]

    # data = np.hstack(
    #     [
    #         decoy_predicted_density_sel.reshape(-1,1),
    #         known_hit_predicted_density_sel.reshape(-1, 1),
    #
    #     ]
    # )
    # score = np.corrcoef(data.T)[0, 1]

    score = 1- ( np.sum(np.clip(decoy_predicted_density_sel - known_hit_predicted_density_sel, 0.0, None)) / np.sum(decoy_predicted_density_sel))

    return score


class BuildScoringDataset(Dataset):

    def __init__(self, zarr_path, config):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        self.test_train =  config['test_train']

        self.meta_table = self.root['meta_sample']
        self.xmap_table = self.root['xmap_sample']
        self.zmap_table = self.root['z_map_sample']
        self.decoy_table = self.root['decoy_pose_sample']
        self.ligand_data_table = self.root['ligand_data']
        self.known_hit_pose = self.root['known_hit_pose']
        self.delta_table = self.root['delta']

        # self.pandda_2_annotation_table = self.root['annotation']
        # self.pandda_2_frag_table = self.root['ligand_confs']  # ['ligand_fragments']

        # self.pandda_2_annotations = {
        #     _x['event_idx']: _x
        #     for _x
        #     in self.pandda_2_annotation_table
        # }

        self.sample_indexes = config['samples']
        #     pos_sample_indexes = [_v for _v in self.sample_indexes if _v['conf'] == 'High']
        #     self.resampled_indexes = self.sample_indexes + (pos_sample_indexes * config['pos_resample_rate'])
        # else:
        if config['test_train'] == 'train':
            # self.resampled_indexes = self.sample_indexes #+ (self.resampled_indexes * config['pos_resample_rate'])
            self.resampled_indexes = []
            for _sample in self.sample_indexes:
                decoy_table = _sample['meta_to_decoy']
                valid_decoys = decoy_table[decoy_table['rmsd'] < 6.0].reset_index()
                bins = pd.cut(valid_decoys['rmsd'], bins=np.linspace(0.0, 6.0, num=61))
                frequencies = 1 / bins.value_counts()
                valid_decoys['p'] = frequencies[bins].reset_index()['rmsd']
                new_sample = {
                    'meta': _sample['meta'],
                    'meta_to_decoy': valid_decoys,
                }
                self.resampled_indexes.append(new_sample)

        elif config['test_train'] == 'test':
            self.resampled_indexes = []
            for _sample in self.sample_indexes:
                for _idx, _row in _sample['meta_to_decoy'].iterrows():
                    new_sample = {
                        'meta': _sample['meta'],
                        'decoy_idx': int(_row['idx']),
                    }
                    self.resampled_indexes.append(new_sample)


        # self.pos_train_pose_samples = pos_train_pose_samples

        # self.fraction_background_replace = config['fraction_background_replace']
        # self.xmap_radius = config['xmap_radius']
        self.max_x_blur = config['max_x_blur']
        self.max_z_blur = config['max_z_blur']
        # self.drop_atom_rate = config['drop_atom_rate']
        self.max_pos_atom_mask_radius = config['max_pos_atom_mask_radius']
        # self.max_translate = config['max_translate']
        self.max_x_noise = config['max_x_noise']
        self.max_z_noise = config['max_z_noise']
        self.p_flip = config['p_flip']
        self.z_mask_radius = config['z_mask_radius']
        self.z_cutoff = config['z_cutoff']

    def __len__(self):
        return len(self.resampled_indexes)

    def __getitem__(self, idx: int):
        # Get the sample data

        sample_data = self.resampled_indexes[idx]

        # Get the metadata, decoy pose and embedding
        # _meta_idx, _decoy_idx, _embedding_idx, _train = sample_data['meta'], int(sample_data['decoy']), sample_data['embedding'], sample_data['train']
        _meta_idx = sample_data['meta']
        # try:
        if self.test_train == 'train':
            _decoy_idx = int(sample_data['meta_to_decoy'].sample(weights=sample_data['meta_to_decoy']['p']).iloc[0]['idx'])
        elif self.test_train == 'test':
            _decoy_idx = sample_data['decoy_idx']
        # except:
        #     print('meta to decoy')
        #     print(sample_data['meta_to_decoy'])

        # rprint(
        #     [
        #         _meta_idx,
        #         _decoy_idx,
        #         # _embedding_idx,
        #         # _train,
        #     ]
        # )
        _meta = self.meta_table[_meta_idx]
        _decoy = self.decoy_table[_decoy_idx]
        # _embedding = self.decoy_table[_embedding_idx]

        # Get rng
        rng = np.random.default_rng()

        # Get the decoy
        # if self.test_train == 'train':
        valid_mask = _decoy['elements'] != 0
        valid_indicies = np.nonzero(valid_mask)
        random_drop_index = rng.integers(0, len(valid_indicies[0]))
        drop_index = valid_indicies[0][random_drop_index]
        valid_poss = _decoy['positions'][(drop_index,),]
        valid_elements = _decoy['elements'][(drop_index,),]

        centroid = np.mean(valid_poss, axis=0)

        sample_array = np.zeros(
            (32, 32, 32),
            dtype=np.float32,
        )

        if self.test_train == 'train':
            orientation = _get_random_orientation()
        else:
            orientation = np.eye(3)
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid,
            n=32
        )

        decoy_residue = _get_res_from_arrays(
            valid_poss,
            valid_elements,
        )

        # decoy_mask_grid = _get_ligand_mask_float(
        #     decoy_residue,
        #     radius=2.0,
        #     n=90,
        #     r=45.0
        # )
        # image_decoy_sample = _sample_xmap(
        #     decoy_mask_grid,
        #     transform,
        #     np.copy(sample_array)
        # )
        #
        # image_decoy_mask = np.copy(sample_array)
        # image_decoy_mask[image_decoy_sample > 0.0] = 1.0
        # image_decoy_mask[image_decoy_sample <= 0.0] = 0.0

        # xmap_mask_float = _get_ed_mask_float()

        selected_atom_mask_grid = _get_ligand_mask_float(
            decoy_residue,
            radius=self.max_pos_atom_mask_radius,
            n=90,
            r=45.0
        )
        image_selected_atom_mask = _sample_xmap(
            selected_atom_mask_grid,
            transform,
            np.copy(sample_array)
        )
        image_selected_atom_mask[image_selected_atom_mask < 0.5] = 0.0
        image_selected_atom_mask[image_selected_atom_mask >= 0.5] = 1.0


        # Get the ligand mask
        valid_mask = _decoy['elements'] != 0
        valid_poss = _decoy['positions'][valid_mask]
        valid_elements = _decoy['elements'][valid_mask]
        decoy_residue = _get_res_from_arrays(
            valid_poss,
            valid_elements,
        )

        decoy_score_mask_grid = _get_ligand_mask_float(
            decoy_residue,
            radius=1.5,
            n=90,
            r=45.0
        )
        image_score_decoy_mask = _sample_xmap(
            decoy_score_mask_grid,
            transform,
            np.copy(sample_array)
        )

        # Get mask of hit for score calculation
        score = _decoy['overlap_score']

        # Get maps
        xmap_sample_data = self.xmap_table[_meta['idx']]['sample']
        z_map_sample_data = self.zmap_table[_meta['idx']]['sample']

        # selected_atom_mask_array = np.array(selected_atom_mask_grid)

        # mask
        selected_atom_mask_array = np.array(selected_atom_mask_grid)
        # xmap_sample_data = xmap_sample_data * selected_atom_mask_array
        # z_map_sample_data = z_map_sample_data * selected_atom_mask_array

        if self.test_train == 'train':
            u_s = rng.uniform(0.0, self.max_x_blur)
            xmap_sample_data = gaussian_filter(xmap_sample_data, sigma=u_s)

            u_s = rng.uniform(0.0, self.max_z_blur)
            z_map_sample_data = gaussian_filter(z_map_sample_data, sigma=u_s)

        xmap = _get_grid_from_hdf5(xmap_sample_data)
        zmap = _get_grid_from_hdf5(z_map_sample_data)

        xmap_sample = _sample_xmap(
            xmap,
            transform,
            np.copy(sample_array)
        )
        z_map_sample = _sample_xmap(
            zmap,
            transform,
            np.copy(sample_array)
        )

        if self.test_train == 'train':
            u_s = rng.uniform(0.0, self.max_x_noise)
            noise = rng.normal(size=(32,32,32)) * u_s
            z_map_sample += noise.astype(np.float32)

            u_s = rng.uniform(0.0, self.max_z_noise)
            noise = rng.normal(size=(32,32,32)) * u_s
            xmap_sample += noise.astype(np.float32)

        # xmap_sample = xmap_sample
        # z_map_sample = z_map_sample
        #

        if (self.test_train == 'train') & (rng.uniform(0.0, 1.0) > self.p_flip):
            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 0)
                z_map_sample = np.flip(z_map_sample, 0)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 0)
                image_selected_atom_mask = np.flip(image_selected_atom_mask, 0)

            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 1)
                z_map_sample = np.flip(z_map_sample, 1)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 1)
                image_selected_atom_mask = np.flip(image_selected_atom_mask, 1)

            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 2)
                z_map_sample = np.flip(z_map_sample, 2)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 2)
                image_selected_atom_mask = np.flip(image_selected_atom_mask, 2)


        # high_z_mask = (z_map_sample > self.z_cutoff).astype(int)
        # high_z_mask_expanded = expand_labels(high_z_mask, distance=self.z_mask_radius / 0.5)
        # high_z_mask_expanded[high_z_mask_expanded != 1] = 0

        # rmsd = _decoy['rmsd']
        deltas = self.delta_table[_decoy_idx]
        delta = deltas['delta'][drop_index]
        rmsd = np.clip(delta / self.max_pos_atom_mask_radius, a_min = 0.0, a_max=1.0)

        # if self.test:
        if self.test_train == 'train':
            if rmsd < 1.5:
                hit = [0.025, 0.975]
            elif rmsd >= 1.5:
                hit = [0.975, 0.025]
            else:
                raise Exception
        else:
            if rmsd < 1.5:
                hit = [0.0, 1.0]
            elif rmsd >= 1.5:
                hit = [1.0, 0.0]
            else:
                raise Exception


        # Return data
        return (
            [
                _meta['idx'],
                _decoy['idx'],
                0,
                str(_meta['system']),
                str(_meta['dtag']),
                int(_meta['event_num']),
            ],
            torch.from_numpy(
                np.stack(
                    [
                        z_map_sample * image_selected_atom_mask,# * image_score_decoy_mask,# * image_decoy_mask,
                        xmap_sample * image_selected_atom_mask ,# * image_score_decoy_mask,
                        image_score_decoy_mask
                        # image_score_decoy_mask
                    ],
                    axis=0,
                    dtype=np.float32
                )),
            torch.from_numpy(
                np.stack(
                    [image_score_decoy_mask],
                    axis=0,
                    dtype=np.float32
                )),
            torch.from_numpy(np.array(rmsd, dtype=np.float32)),
            torch.from_numpy(np.array(score, dtype=np.float32)),
            torch.from_numpy(np.array(hit, dtype=np.float32))
        )
