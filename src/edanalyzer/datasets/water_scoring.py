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

def get_structure(path):
    return gemmi.read_structure(str(path))

# def get_xmap(path):
#     m = gemmi.read_ccp4_map(str(path), setup=True)
#     return m.grid

def get_xmap(path):
    m = gemmi.read_mtz_file(str(path), )
    grid = m.transform_f_phi_to_map('FWT', 'PHWT', sample_rate=3)
    return grid


def get_water(st, water):
    return st[0][str(water[0])][str(water[1])][0]

def get_structure_masks(st, water_atom, template, radius=1.5):
    ns = gemmi.NeighborSearch(st[0], st.cell, 5)
    for n_chain, chain in enumerate(st[0]):
        for n_res, res in enumerate(chain):
            if res.name in ['HOH', 'LIG', 'XXX']:
                continue
            for n_atom, atom in enumerate(res):
                ns.add_atom(atom, n_chain, n_res, n_atom)

    marks = ns.find_neighbors(water_atom, min_dist=0.1, max_dist=6)
    rprint(f'Got {len(marks)} marks in cell {ns}!')

    # Setup the grids
    mask_carbon = gemmi.FloatGrid(template.nu, template.nv, template.nw)
    mask_carbon.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    mask_carbon.set_unit_cell(st.cell)
    mask_sulfur = gemmi.FloatGrid(template.nu, template.nv, template.nw)
    mask_sulfur.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    mask_sulfur.set_unit_cell(st.cell)
    mask_nitrogen = gemmi.FloatGrid(template.nu, template.nv, template.nw)
    mask_nitrogen.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    mask_nitrogen.set_unit_cell(st.cell)
    mask_oxygen = gemmi.FloatGrid(template.nu, template.nv, template.nw)
    mask_oxygen.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    mask_oxygen.set_unit_cell(st.cell)

    # Go through marks
    for mark in marks:
        rprint(mark)
        pos = mark.pos()
        rprint([pos.x, pos.y, pos.z])
        if mark.element.name == 'C':
            mask_carbon.set_points_around(pos, radius, 1.0)
        elif mark.element.name == 'O':
            mask_oxygen.set_points_around(pos, radius, 1.0)
        elif mark.element.name == 'N':
            mask_nitrogen.set_points_around(pos, radius, 1.0)
        elif mark.element.name == 'S':
            mask_sulfur.set_points_around(pos, radius, 1.0)
        else:
            continue

    # Symetrize grids on max
    mask_carbon.symmetrize_max()
    mask_oxygen.symmetrize_max()
    mask_nitrogen.symmetrize_max()
    mask_sulfur.symmetrize_max()

    rprint(f'Total carbon volume: {np.array(mask_carbon).sum()}')
    rprint(f'Total oxygen volume: {np.array(mask_oxygen).sum()}')
    rprint(f'Total nitrogen volume: {np.array(mask_nitrogen).sum()}')
    rprint(f'Total sulfur volume: {np.array(mask_sulfur).sum()}')
    
    # Return grids
    return mask_carbon, mask_oxygen, mask_nitrogen, mask_sulfur 

class WaterScoringDataset(Dataset):

    def __init__(self, config):
        # self.data = data
        self.test_train =  config['test_train']
        self.data = config['data']
        self.data_idx_mapping = {j: data_idx for j, data_idx in enumerate(self.data)}

        self.max_x_blur = config['max_x_blur']
        self.max_z_blur = config['max_z_blur']
        self.max_pos_atom_mask_radius = config['max_pos_atom_mask_radius']
        self.max_x_noise = config['max_x_noise']
        self.max_z_noise = config['max_z_noise']
        self.p_flip = config['p_flip']
        self.z_mask_radius = config['z_mask_radius']
        self.z_cutoff = config['z_cutoff']

        self.grid_sampling = int(config['grid_sampling']) #32
        self.grid_length = float(config['grid_length']) / self.grid_sampling
        self.grid_step = self.grid_length / self.grid_sampling
        self.sample_array = np.zeros(
            (self.grid_sampling, self.grid_sampling, self.grid_sampling),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng()


    def __len__(self):
        return len(self.data_idx_mapping)

    def __getitem__(self, idx: int):
        # Get the data
        data_idx = self.data_idx_mapping[idx]
        sample_data = self.data[data_idx]

        # Get the structure
        st = get_structure(sample_data['pdb'])

        # Get the xmap
        xmap = get_xmap(sample_data['xmap'])

        # Get the relevant water
        water_residue = get_water(st, sample_data['landmark'])
        water_atom = water_residue[0]
       
        # Get the cartesian centroid of the water
        pos = water_atom.pos
        centroid = np.array([pos.x, pos.y, pos.z])

        # Get A random orientation around the water
        if self.test_train == 'train':
            orientation = _get_random_orientation()
        else:
            orientation = np.eye(3)

        # Get the transform to the sample frame
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid,
            n=self.grid_sampling,
            sd=self.grid_step
        )

        # Get the xmap sample
        xmap_sample_data = _sample_xmap(
            xmap,
            transform,
            np.copy(self.sample_array)
        )

        # Get the xmap mask
        decoy_score_mask_grid = _get_ligand_mask_float(
            water_residue,
            radius=self.max_pos_atom_mask_radius,
            n=180,
            r=45.0
        )
        image_score_decoy_mask = _sample_xmap(
            decoy_score_mask_grid,
            transform,
            np.copy(self.sample_array)
        )
        image_score_decoy_mask[image_score_decoy_mask < 0.5] = 0.0
        image_score_decoy_mask[image_score_decoy_mask >= 0.5] = 1.0

        # Get the structure mask
        structure_masks = get_structure_masks(st, water_atom, xmap)

        # Sample the structure masks
        structure_mask_samples = []
        for structure_mask in structure_masks:
            rprint(structure_mask)
            rprint(structure_mask.spacegroup)
            rprint(structure_mask.unit_cell)
            structure_mask_samples.append(
                _sample_xmap(
                structure_mask, 
                transform, 
                np.copy(self.sample_array),
                ) 
            )
       
        # Gaussian filter the map
        if self.test_train == 'train':
            u_s = self.rng.uniform(0.0, self.max_x_blur)
            xmap_sample_data = gaussian_filter(xmap_sample_data, sigma=u_s)

        # Sample the xmap
        xmap_sample = _sample_xmap(
            xmap,
            transform,
            np.copy(self.sample_array)
        )

        # Noise the map
        if self.test_train == 'train':
            u_s = self.rng.uniform(0.0, self.max_z_noise)
            noise = self.rng.normal(size=(self.grid_sampling,self.grid_sampling,self.grid_sampling)) * u_s
            xmap_sample += noise.astype(np.float32)

        # Potentially flip the maps
        if (self.test_train == 'train') & (self.rng.uniform(0.0, 1.0) > self.p_flip):
            if self.rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 0)
                z_map_sample = np.flip(z_map_sample, 0)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 0)
                # image_selected_atom_mask = np.flip(image_selected_atom_mask, 0)

            if self.rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 1)
                z_map_sample = np.flip(z_map_sample, 1)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 1)
                # image_selected_atom_mask = np.flip(image_selected_atom_mask, 1)

            if self.rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample = np.flip(xmap_sample, 2)
                z_map_sample = np.flip(z_map_sample, 2)
                image_score_decoy_mask = np.flip(image_score_decoy_mask, 2)
                # image_selected_atom_mask = np.flip(image_selected_atom_mask, 2)


        # Get the score tensor
        if self.test_train == 'train':
            if sample_data['annotation'] == 'truePositive':
                hit = [0.025, 0.975]
            elif sample_data['annotation'] == 'falsePositive':
                hit = [0.975, 0.025]
            else:
                raise Exception
        else:
            if sample_data['annotation'] == 'truePositive':
                hit = [0.0, 1.0]
            elif sample_data['annotation'] == 'falsePositive':
                hit = [1.0, 0.0]
            else:
                raise Exception


        # Return data
        return (
            [
                idx,
                data_idx[0],
                data_idx[1],
                str(sample_data['dtag']),
                sample_data['landmark'][0],
                sample_data['landmark'][1],
            ],
            torch.from_numpy(
                np.stack(
                    [
                        xmap_sample * image_score_decoy_mask ,# * image_score_decoy_mask,
                        structure_mask_samples[0] * image_score_decoy_mask,
                        structure_mask_samples[1] * image_score_decoy_mask,
                        structure_mask_samples[2] * image_score_decoy_mask,
                        structure_mask_samples[3] * image_score_decoy_mask,
                        image_score_decoy_mask
                    ],
                    axis=0,
                    dtype=np.float32
                )),
            torch.from_numpy(np.array(hit, dtype=np.float32)),
            centroid
        )
