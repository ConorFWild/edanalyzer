import dataclasses
import itertools

import numpy as np
import torch
import gemmi
from scipy.spatial.transform import Rotation as R
import zarr
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

from scipy.ndimage import gaussian_filter

from torch.utils.data import Dataset

from skimage.segmentation import expand_labels

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

Z_LEVELS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, ]
X_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, ]

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

def truncate(xmap, res):
    sf = gemmi.transform_map_to_f_phi(xmap)
    data = sf.prepare_asu_data(dmin=res)
    grid = data.get_f_phi_on_grid([90,90,90])
    return grid

# class DepEventScoringDataset(Dataset):
#
#     def __init__(self, zarr_path, sample_indexes, pos_train_pose_samples):
#         # self.data = data
#         self.root = zarr.open(zarr_path, mode='r')
#
#         # self.z_map_sample_metadata_table = self.root['z_map_sample_metadata']
#         # self.xmap_sample_table = self.root['xmap_sample']
#         # self.z_map_sample_table = self.root['z_map_sample']
#         # self.pose_table = self.root['known_hit_pose']
#         # self.ligand_data_table = self.root['ligand_data']
#         # self.annotation_table = self.root['annotation']
#
#         self.pandda_2_z_map_sample_metadata_table = self.root['pandda_2']['z_map_sample_metadata']
#         self.pandda_2_xmap_sample_table = self.root['pandda_2']['xmap_sample']
#         self.pandda_2_z_map_sample_table = self.root['pandda_2']['z_map_sample']
#         self.pandda_2_pose_table = self.root['pandda_2']['known_hit_pose']
#         self.pandda_2_ligand_data_table = self.root['pandda_2']['ligand_data']
#         self.pandda_2_annotation_table = self.root['pandda_2']['annotation']
#         self.pandda_2_frag_table = self.root['pandda_2']['ligand_confs']  # ['ligand_fragments']
#
#         # self.annotations = {
#         #     self.z_map_sample_metadata_table[_x['event_map_table_idx']]['event_idx']: _x
#         #     for _x
#         #     in self.annotation_table
#         # }
#         # self.annotations = {
#         #     _x['event_idx']: _x
#         #     for _x
#         #     in self.annotation_table
#         # }
#         self.pandda_2_annotations = {
#             _x['event_idx']: _x
#             for _x
#             in self.pandda_2_annotation_table
#         }
#
#         self.sample_indexes = sample_indexes
#
#         self.pos_train_pose_samples = pos_train_pose_samples
#
#     def __len__(self):
#         return len(self.sample_indexes)
#
#     def __getitem__(self, idx: int):
#         # Get the sample idx
#         sample_data = self.sample_indexes[idx]
#         _table, _z, _f, _t = sample_data['table'], sample_data['z'], sample_data['f'], sample_data['t']
#
#         # Get the z map and pose
#         if _table == 'normal':
#             z_map_sample_metadata = self.z_map_sample_metadata_table[_z]
#         else:
#             z_map_sample_metadata = self.pandda_2_z_map_sample_metadata_table[_z]
#         z_map_sample_idx = z_map_sample_metadata['idx']
#         # print([sample_idx, z_map_sample_idx])
#         assert _z == z_map_sample_idx
#         # event_map_idx = pose_data['event_map_sample_idx']
#         ligand_data_idx = z_map_sample_metadata['ligand_data_idx']
#         ligand_data = self.pandda_2_ligand_data_table[ligand_data_idx]
#
#         pose_data_idx = z_map_sample_metadata['pose_data_idx']
#         if _table == 'normal':
#
#             xmap_sample_data = self.xmap_sample_table[z_map_sample_idx]
#             z_map_sample_data = self.z_map_sample_table[z_map_sample_idx]
#             annotation = self.annotations[z_map_sample_metadata['event_idx']]
#         else:
#             xmap_sample_data = self.pandda_2_xmap_sample_table[z_map_sample_idx]
#             z_map_sample_data = self.pandda_2_z_map_sample_table[z_map_sample_idx]
#             annotation = self.pandda_2_annotations[z_map_sample_metadata['event_idx']]
#
#         # If training replace with a random ligand
#         rng = np.random.default_rng()
#         # random_ligand = True
#         # if pose_data_idx != -1:
#         #     # random_ligand_sample = rng.random()
#         #     # if (random_ligand_sample > 0.5) & (annotation['partition'] == 'train'):
#         #     #     random_ligand = False
#         #     #     pose_data = self.pose_table[pose_data_idx]
#         #     # else:
#         #     #     pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
#         #     if sample_idx[0] == 'normal':
#         #
#         #         pose_data = self.pose_table[pose_data_idx]
#         #     else:
#         #         pose_data = self.pandda_2_pose_table[pose_data_idx]
#         # else:
#         #     if sample_idx[0] == 'normal':
#         #
#         #         # pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
#         #         selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
#         #         pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]
#         #     else:
#         #         selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
#         #         pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]
#         # if pose_data_idx != -1:
#         #     frag_data = self.pandda_2_frag_table[ligand_data['idx']]
#
#         # else:
#         frag_data = self.pandda_2_frag_table[_f]
#
#         #
#         if annotation['partition'] == 'train':
#             u_s = rng.uniform(0.0, 1.5)
#             xmap_sample_data = gaussian_filter(xmap_sample_data, sigma=u_s)
#
#             u_s = rng.uniform(0.0, 1.5)
#             z_map_sample_data = gaussian_filter(z_map_sample_data, sigma=u_s)
#
#
#         xmap = _get_grid_from_hdf5(xmap_sample_data)
#         z_map = _get_grid_from_hdf5(z_map_sample_data)
#
#         # Subsample if training
#         if annotation['partition'] == 'train':
#             translation = 3*(2*(rng.random(3)-0.5))
#             centroid = np.array([22.5,22.5,22.5]) + translation
#
#         else:
#             centroid = np.array([22.5,22.5,22.5])
#
#
#         # Get sampling transform for the z map
#         sample_array = np.zeros(
#             (32, 32, 32),
#             dtype=np.float32,
#         )
#         if annotation['partition'] == 'train':
#
#             orientation = _get_random_orientation()
#         else:
#             orientation = np.eye(3)
#         transform = _get_transform_from_orientation_centroid(
#             orientation,
#             centroid,
#             n=32
#         )
#
#         # Get the sampling transform for the ligand map
#         # valid_mask = pose_data['elements'] != 0
#         # valid_poss = pose_data['positions'][valid_mask]
#         # valid_elements = pose_data['elements'][valid_mask]
#         valid_mask = frag_data['elements'] > 1
#         if annotation['partition'] == 'train':
#             do_drop = rng.random()
#             if do_drop > 0.5:
#                 valid_indicies = np.nonzero(valid_mask)
#                 random_drop_index = rng.integers(0, len(valid_indicies))
#                 drop_index = valid_indicies[random_drop_index]
#                 valid_mask[drop_index] = False
#         valid_poss = frag_data['positions'][valid_mask]
#         valid_elements = frag_data['elements'][valid_mask]
#
#         valid_poss = (valid_poss - np.mean(valid_poss, axis=0)) + np.array([22.5, 22.5, 22.5])
#
#         # # Subsample if training
#         # if annotation['partition'] == 'train':
#         #     rng = np.random.default_rng()
#         #     num_centres = rng.integers(1, 5)
#         #
#         #     # For each centre mask atoms close to it
#         #     total_mask = np.full(valid_elements.size, False)
#         #     for _centre in num_centres:
#         #         selected_atom = rng.integers(0, valid_elements.size)
#         #         poss_distances = valid_poss - valid_poss[selected_atom, :].reshape((1, 3))
#         #         close_mask = poss_distances[np.linalg.norm(poss_distances, axis=1) < 3.5]
#         #         total_mask[close_mask] = True
#         #
#         # else:
#         #     total_mask = np.full(valid_elements.size, True)
#
#         ligand_sample_array = np.zeros(
#             (32, 32, 32),
#             dtype=np.float32,
#         )
#         ligand_orientation = _get_random_orientation()
#         # transformed_residue = _get_res_from_arrays(
#         #     valid_poss[total_mask],
#         #     valid_elements[total_mask],
#         # )
#         transformed_residue = _get_res_from_arrays(
#             valid_poss,
#             valid_elements,
#         )
#
#         ligand_centroid = _get_centroid_from_res(transformed_residue)
#         ligand_map_transform = _get_transform_from_orientation_centroid(
#             ligand_orientation,
#             ligand_centroid,
#             n=32
#         )
#
#
#         if annotation['partition'] == 'train':
#             # mask = get_mask_array()
#             mask = np.ones((32,32,32), dtype=np.float32)
#
#         else:
#             mask = np.ones((32,32,32), dtype=np.float32)
#
#         xmap_mask_float = _get_ed_mask_float()
#
#         # Get sample images
#         xmap_sample = _sample_xmap_and_scale(
#             xmap,
#             transform,
#             np.copy(sample_array)
#         )
#         # xmap_sample = np.copy(sample_array)
#         z_map_sample = _sample_xmap_and_scale(
#             z_map,
#             transform,
#             np.copy(sample_array)
#         )
#         if annotation['partition'] == 'train':
#             u_s = rng.uniform(0.0, 0.5)
#             noise = rng.normal(size=(32,32,32)) * u_s
#             z_map_sample += noise.astype(np.float32)
#
#             u_s = rng.uniform(0.0, 0.5)
#             noise = rng.normal(size=(32,32,32)) * u_s
#             xmap_sample += noise.astype(np.float32)
#
#         ligand_mask_grid = _get_ligand_mask_float(
#             transformed_residue,
#         )
#         image_ligand_mask = _sample_xmap(
#             ligand_mask_grid,
#             ligand_map_transform,
#             # transform,
#             np.copy(ligand_sample_array)
#         )
#
#         # image_ligand_mask[image_ligand_mask < 0.9] = 0.0
#         # image_ligand_mask[image_ligand_mask >= 0.9] = 1.0
#         # image_ligand_mask = np.copy(sample_array)
#
#         # if pose_data_idx != -1:
#         #     image_decoded_density = _sample_xmap(
#         #         ligand_mask_grid,
#         #         transform,
#         #         np.copy(sample_array)
#         #     )
#         # elif pose_data_idx == -1:
#         #     image_decoded_density = np.copy(sample_array)
#         # else:
#         #     raise Exception
#         image_decoded_density = np.copy(sample_array)
#
#         # Make the image
#         image_density = np.stack(
#             [
#                 xmap_sample,
#             ],
#             axis=0
#         )
#         image_density_float = image_density.astype(np.float32) * mask
#
#         image_z = np.stack(
#             [
#                 z_map_sample,
#                 xmap_sample * xmap_mask_float
#             ],
#             axis=0
#         )
#         image_z_float = image_z.astype(np.float32)  * mask
#
#         image_mol = np.stack(
#             [
#                 image_ligand_mask,
#             ],
#             axis=0
#         )
#         image_mol_float = image_mol.astype(np.float32)
#
#         image_decoded_density = np.stack(
#             [
#                 image_decoded_density,
#             ],
#             axis=0
#         )
#         image_decoded_density_float = image_decoded_density.astype(np.float32) * mask
#
#         # res mask
#         # if annotation['partition'] == 'train':
#         #     grid = gemmi.FloatGrid(32, 32, 32)
#         #     grid.spacegroup = gemmi.SpaceGroup('P1')
#         #     uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
#         #     grid.set_unit_cell(uc)
#         #     grid_array = np.array(grid, copy=False)
#         #     grid_array[:,:,:] = image_z_float[:,:,:]
#         #     rsg = gemmi.transform_map_to_f_phi(grid)
#         #     dmin = 1 + (1.0*rng.random())
#         #     data = rsg.prepare_asu_data(dmin=float(dmin))
#         #     grid_low_res = data.transform_f_phi_to_map(exact_size=[32, 32, 32])
#         #     grid_low_res_array = np.array(grid_low_res, copy=False)
#         #     image_z_float[:, :, :] = grid_low_res_array[:,:,:]
#
#         # Make the annotation
#         # if (pose_data_idx != -1) & (not random_ligand):
#         #     hit = 1.0
#         # elif (pose_data_idx != -1) & (random_ligand):
#         #     hit = 0.0
#         # if pose_data_idx != -1:
#         #     hit = 1.0
#         # elif pose_data_idx == -1:
#         #     hit = 0.0
#         # else:
#         #     raise Exception
#         # if pose_data_idx != -1:
#         #     hit = [0.0, 1.0]
#         # elif pose_data_idx == -1:
#         #     hit = [1.0, 0.0]
#         # else:
#         #     raise Exception
#
#         if annotation['partition'] == 'train':
#             if _t == 'High':
#                 hit = [0.025, 0.975]
#             elif _t == 'Medium':
#                 hit = [0.5, 0.5]
#             elif _t == 'Low':
#                 hit = [0.975, 0.025]
#             else:
#                 raise Exception
#         else:
#             if _t == 'High':
#                 hit = [0.0, 1.0]
#             elif _t == 'Medium':
#                 hit = [0.5, 0.5]
#             elif _t == 'Low':
#                 hit = [1.0, 0.0]
#             else:
#                 raise Exception
#
#         label = np.array(hit)
#         label_float = label.astype(np.float32)
#
#         return (
#             [
#                 _table,
#                 _z,
#                 _f,
#                 str( z_map_sample_metadata['system']),
#                 str( z_map_sample_metadata['dtag']),
#                 int( z_map_sample_metadata['event_num']),
#              ],
#             torch.from_numpy(image_density_float),
#             torch.from_numpy(image_z_float),
#             torch.from_numpy(image_mol_float),
#             torch.from_numpy(image_decoded_density_float),
#             torch.from_numpy(label_float)
#         )

class EventScoringDataset(Dataset):

    def __init__(self, config):
        # self.data = data
        self.test_train =  config['test_train']

        zarr_path = config['zarr_path']
        self.root = zarr.open(zarr_path, mode='r')

        self.pandda_2_z_map_sample_metadata_table = self.root['pandda_2']['z_map_sample_metadata']
        self.pandda_2_xmap_sample_table = self.root['pandda_2']['xmap_sample']
        self.pandda_2_z_map_sample_table = self.root['pandda_2']['z_map_sample']
        self.pandda_2_pose_table = self.root['pandda_2']['known_hit_pose']
        self.pandda_2_ligand_data_table = self.root['pandda_2']['ligand_data']
        self.pandda_2_annotation_table = self.root['pandda_2']['annotation']
        self.pandda_2_frag_table = self.root['pandda_2']['ligand_confs']

        self.pandda_2_annotations = config['pandda_2_annotations']
        self.sample_indexes = config['indexes']
        if config['test_train'] == 'train':
            pos_sample_indexes = [_v for _v in self.sample_indexes if _v['conf'] == 'High']
            self.resampled_indexes = self.sample_indexes + (pos_sample_indexes * config['pos_resample_rate'])
        else:
            self.resampled_indexes = self.sample_indexes
        # self.pos_train_pose_samples = configp'pos_train_pose_samples

        # ligand_data_df = pd.DataFrame(
        #     self.pandda_2_ligand_data_table.get_basic_selection(slice(None), fields=['idx', 'canonical_smiles',]))

        self.unique_smiles = config['unique_smiles']

        self.metadata_table = config['metadata_table']
        self.sampled_metadata_table = config['sampled_metadata_table']
        self.metadata_table_high_conf = config['metadata_table_high_conf']
        self.metadata_table_med_conf = config['metadata_table_med_conf']
        self.metadata_table_low_conf = config['metadata_table_low_conf']

        self.unique_smiles = config['unique_smiles']
        self.unique_smiles_frequencies = config['unique_smiles_frequencies']
        # print(self.unique_smiles)
        # print(self.unique_smiles_frequencies)

        self.fraction_background_replace = config['fraction_background_replace']
        self.xmap_radius = config['xmap_radius']
        self.max_x_blur = config['max_x_blur']
        self.max_z_blur = config['max_z_blur']
        self.drop_atom_rate = config['drop_atom_rate']
        self.max_pos_atom_mask_radius = config['max_pos_atom_mask_radius']
        self.max_translate = config['max_translate']
        self.max_x_noise = config['max_x_noise']
        self.max_z_noise = config['max_z_noise']
        self.p_flip = config['p_flip']
        self.z_mask_radius = config['z_mask_radius']
        self.z_cutoff = config['z_cutoff']

        self.ligand = config['ligand']

    def __len__(self):
        return len(self.resampled_indexes)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng()

        # Get the sample idx
        sample_data = self.resampled_indexes[idx]
        _z= sample_data['z']

        # Get the z map and pose
        z_map_sample_metadata = self.pandda_2_z_map_sample_metadata_table[_z]
        z_map_sample_idx = z_map_sample_metadata['idx']
        conf = z_map_sample_metadata['Confidence']
        # try:
        res = z_map_sample_metadata['res']
        # except:
            # res = 0.590471872361221 + 0.01
        assert _z == z_map_sample_idx
        ligand_data_idx = z_map_sample_metadata['ligand_data_idx']
        xmap_sample_data = self.pandda_2_xmap_sample_table[z_map_sample_idx]['sample']
        z_map_sample_data = self.pandda_2_z_map_sample_table[z_map_sample_idx]['sample']
        annotation = self.pandda_2_annotations[z_map_sample_metadata['event_idx']]




        # If training replace positives with negatives
        # if (annotation['partition'] == 'train') & (rng.uniform(0.0, 1.0) > 0.5) & (conf == 'High'):
        #     conf = 'Low'
        #     low_conf_sample = self.metadata_table_low_conf.sample().iloc[0]
        #     z_map_sample_data = self.pandda_2_z_map_sample_table[low_conf_sample['idx']]

        #
        if (rng.uniform(0.0, 1.0) > self.p_flip) & (self.test_train == 'train'):
            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample_data = np.flip(xmap_sample_data, 0)
                z_map_sample_data = np.flip(z_map_sample_data, 0)
            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample_data = np.flip(xmap_sample_data, 1)
                z_map_sample_data = np.flip(z_map_sample_data, 1)
            if rng.uniform(0.0, 1.0) > 0.5:
                xmap_sample_data = np.flip(xmap_sample_data, 2)
                z_map_sample_data = np.flip(z_map_sample_data, 2)

        # If training
        pose_data_idx = z_map_sample_metadata['pose_data_idx']
        if (rng.uniform(0.0, 1.0) > self.fraction_background_replace) & (self.test_train == 'train'):
            if pose_data_idx != -1:  # High confidence sample: chop in low confidence background
                pose_data = self.pandda_2_pose_table[pose_data_idx]
            else:  # Low confidence sample: chop in low confidence background
                high_conf_sample = self.metadata_table_high_conf.sample().iloc[0]
                pose_data_idx = high_conf_sample['pose_data_idx']
                pose_data = self.pandda_2_pose_table[pose_data_idx]

            # Select new background
            low_conf_sample = self.metadata_table_low_conf.sample().iloc[0]
            low_conf_z_map_sample_data = self.pandda_2_z_map_sample_table[low_conf_sample['idx']]['sample']
            low_conf_x_map_sample_data = self.pandda_2_xmap_sample_table[low_conf_sample['idx']]['sample']

            # Mask around ligand and paste in new background
            _valid_mask = pose_data['elements'] > 1
            if (self.test_train == 'train') & (rng.random() > self.drop_atom_rate):
                _valid_indicies = np.nonzero(_valid_mask)
                num_valid = len(_valid_indicies[0])
                for _j in range(rng.integers(0, max(num_valid - 5, 1))):
                    _valid_indicies = np.nonzero(_valid_mask)
                    _random_drop_index = rng.integers(0, len(_valid_indicies[0]))
                    drop_index = _valid_indicies[0][_random_drop_index]
                    _valid_mask[drop_index] = False
            _valid_poss = pose_data['positions'][_valid_mask]
            _valid_elements = pose_data['elements'][_valid_mask]
            _transformed_residue = _get_res_from_arrays(
                _valid_poss,
                _valid_elements,
            )
            _radius = rng.uniform(1.0, self.max_pos_atom_mask_radius)
            _ligand_mask_grid = _get_ligand_mask_float(
                _transformed_residue,
                _radius,
                90,
                45.0
            )
            _ligand_mask_array = np.array(_ligand_mask_grid) > 0

            z_map_sample_data[~_ligand_mask_array] = low_conf_z_map_sample_data[~_ligand_mask_array]
            xmap_sample_data[~_ligand_mask_array] = low_conf_x_map_sample_data[~_ligand_mask_array]


        # If training replace with a random ligand
        if (self.test_train == 'train') & (conf in ['Low', 'Medium']):
            # smiles = self.unique_smiles[rng.integers(0, len(self.unique_smiles))]
            smiles = self.unique_smiles.sample(weights=self.unique_smiles_frequencies).iloc[0]
        else:
            ligand_data = self.pandda_2_ligand_data_table[ligand_data_idx]
            smiles = ligand_data['canonical_smiles']

        # Get the molecule
        try:
            m = Chem.MolFromSmiles(smiles)
            m2 = Chem.AddHs(m)
            cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
            m3 = Chem.RemoveHs(m2)
            embedding = [_conf.GetPositions() for _conf in m3.GetConformers()][0]
        except Exception as e:
            print(f'Got exception in getting conf: {e}')
            hit = [1.0, 0.0]
            label = np.array(hit)
            label_float = label.astype(np.float32)
            sample_array_z = np.zeros(
                (2, 32, 32, 32),
                dtype=np.float32,
            )
            sample_array_mol = np.zeros(
                (1, 32, 32, 32),
                dtype=np.float32,
            )
            return (
                [
                    'pandda_2',
                    _z,
                    -1,
                    str(z_map_sample_metadata['system']),
                    str(z_map_sample_metadata['dtag']),
                    int(z_map_sample_metadata['event_num']),
                ],
                0,
                torch.from_numpy(np.copy(sample_array_z)),
                torch.from_numpy(np.copy(sample_array_mol)),
                0,
                torch.from_numpy(label_float)
            )

        #
        if self.test_train == 'train':
            u_s = rng.uniform(0.0, self.max_x_blur)
            xmap_sample_data = gaussian_filter(xmap_sample_data, sigma=u_s)

            u_s = rng.uniform(0.0, self.max_z_blur)
            z_map_sample_data = gaussian_filter(z_map_sample_data, sigma=u_s)

        # if (annotation['partition'] == 'train') & (rng.uniform(0.0, 1.0) > 0.5):
        #     xmap_sample_data[:,:,:] = 0.0
        #     ...

        xmap = _get_grid_from_hdf5(xmap_sample_data)
        z_map = _get_grid_from_hdf5(z_map_sample_data)

        # if annotation['partition'] == 'train':
        #
        #     if res > 2.5:
        #         truncation_res = res +0.001
        #     else:
        #         truncation_res = rng.uniform(res+0.001, 2.5)
        #
        #     xmap = truncate(xmap, truncation_res)
        #     z_map = truncate(z_map, truncation_res)

        # Subsample if training
        if self.test_train == 'train':
            translation = self.max_translate*(2*(rng.random(3)-0.5))
            centroid = np.array([22.5,22.5,22.5]) + translation

        else:
            centroid = np.array([22.5,22.5,22.5])

        # Get sampling transform for the z map
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

        # Get the ligand
        valid_poss = (embedding - np.mean(embedding, axis=0)) + np.array([8.0,8.0,8.0])
        valid_elements = np.array(
                [m3.GetAtomWithIdx(_atom_idx).GetAtomicNum() for _atom_idx in [a.GetIdx() for a in m3.GetAtoms()]])
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
            ligand_centroid,
            n=32
        )
        ligand_mask_grid = _get_ligand_mask_float(
            transformed_residue,
        )

        if self.test_train == 'train':
            mask = np.ones((32,32,32), dtype=np.float32)

        else:
            mask = np.ones((32,32,32), dtype=np.float32)

        # Get sample images
        xmap_sample = _sample_xmap(
            xmap,
            transform,
            np.copy(sample_array)
        )
        # xmap_sample = np.copy(sample_array)
        z_map_sample = _sample_xmap(
            z_map,
            transform,
            np.copy(sample_array)
        )
        # if annotation['partition'] == 'train':
        #     translate = rng.uniform(-0.5, 0.5)
        #     scale = rng.uniform(0.9, 1.1)
        #     z_map_sample = (z_map_sample * scale) + translate
        #
        #     translate = rng.uniform(-0.5, 0.5)
        #     scale = rng.uniform(0.9, 1.1)
        #     xmap_sample = (xmap_sample * scale) + translate

        if self.test_train == 'train':
            u_s = rng.uniform(0.0, self.max_x_noise)
            noise = rng.normal(size=(32,32,32)) * u_s
            z_map_sample += noise.astype(np.float32)

            u_s = rng.uniform(0.0, self.max_z_noise)
            noise = rng.normal(size=(32,32,32)) * u_s
            xmap_sample += noise.astype(np.float32)

        # Make the image
        if self.ligand:
            image_ligand_mask = _sample_xmap(
                ligand_mask_grid,
                ligand_map_transform,
                np.copy(ligand_sample_array)
            )
        else:
            image_ligand_mask = np.copy(sample_array)

        if self.ligand:
            _density_mask = (z_map_sample > self.z_cutoff).astype(int)
            # high_z_mask[high_z_mask == 0] = -1
            density_mask = expand_labels(_density_mask, distance=self.z_mask_radius / 0.5)
            density_mask[density_mask != 1] = 0
        else:
            density_mask = _get_ed_mask_float(radius=self.xmap_radius)

        image_z = np.stack(
            [
                z_map_sample * density_mask,
                xmap_sample * density_mask
            ],
            axis=0
        )
        image_z_float = image_z.astype(np.float32) # * mask

        # image_z = np.stack(
        #     [
        #         z_map_sample > _cutoff
        #         for _cutoff
        #         in Z_LEVELS
        #
        #     ] + [
        #         (xmap_sample > _cutoff) * xmap_mask_float
        #         for _cutoff
        #         in X_LEVELS
        #     ],
        #     axis=0
        # )
        # image_z_float = image_z.astype(np.float32) # * mask

        image_mol = np.stack(
            [
                image_ligand_mask,
            ],
            axis=0
        )
        image_mol_float = image_mol.astype(np.float32)


        # if self.test_train == 'train':
        #     # if conf == 'High':
        #     #     hit = [self.label_noise, 1-self.label_noise]
        #     # elif conf == 'Medium':
        #     #     hit = [0.5, 0.5]
        #     # elif conf == 'Low':
        #     #     hit = [1-self.label_noise, self.label_noise]
        #     if conf == 'High':
        #         hit = [0.0, 1.0]
        #     elif conf == 'Medium':
        #         hit = [0.5, 0.5]
        #     elif conf == 'Low':
        #         hit = [1.0, 0.0]
        #     else:
        #         raise Exception
        # else:
        #     if conf == 'High':
        #         hit = [0.0, 1.0]
        #     elif conf == 'Medium':
        #         hit = [0.5, 0.5]
        #     elif conf == 'Low':
        #         hit = [1.0, 0.0]
        #     else:
        #         raise Exception


        if conf == 'High':
            hit = [0.0, 0.0, 1.0]
        elif conf == 'Medium':
            hit = [0.0, 1.0, 0.0]
        elif conf == 'Low':
            hit = [1.0, 0.0, 0.0]
        else:
            raise Exception

        # if conf == 'High':
        #     hit = [0.0, 1.0]
        # elif conf == 'Medium':
        #     hit = [0.5, 0.5]
        # elif conf == 'Low':
        #     hit = [1.0, 0.0, 0.0]
        # else:
        #     raise Exception


        label = np.array(hit)
        label_float = label.astype(np.float32)

        return (
            [
                'pandda_2',
                _z,
                ligand_data_idx,
                str( z_map_sample_metadata['system']),
                str( z_map_sample_metadata['dtag']),
                int( z_map_sample_metadata['event_num']),
                str(conf)
             ],
            0,
            torch.from_numpy(image_z_float),
            torch.from_numpy(image_mol_float),
            0,
            torch.from_numpy(label_float)
        )
