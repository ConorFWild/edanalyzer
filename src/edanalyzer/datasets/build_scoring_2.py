import dataclasses
import itertools

import numpy as np
import torch
import gemmi
from scipy.spatial.transform import Rotation as R
import zarr
from rich import print as rprint


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


def _get_overlap_volume(orientation, centroid, known_hit_pose_residue, decoy_residue):
    transform = _get_transform_from_orientation_centroid(
        orientation,
        centroid,
        n=64,
        sd=0.25
    )

    known_hit_score_mask_grid = _get_ligand_mask_float(
        known_hit_pose_residue,
        radius=1.0,
        n=180,
        r=45.0
    )

    decoy_score_mask_grid = _get_ligand_mask_float(
        decoy_residue,
        radius=1.0,
        n=180,
        r=45.0
    )

    known_hit_score_sample = _sample_xmap(
        known_hit_score_mask_grid,
        transform,
        np.zeros([64, 64, 64], dtype=np.float32)
    )
    initial_known_hit_sum = np.sum(known_hit_score_sample)
    # print(known_hit_score_sample)
    known_hit_score_sample[known_hit_score_sample >= 0.1] = 1.0
    known_hit_score_sample[known_hit_score_sample < 0.1] = 0.0

    decoy_score_sample = _sample_xmap(
        decoy_score_mask_grid,
        transform,
        np.zeros([64, 64, 64], dtype=np.float32),
    )
    initial_decoy_sum = np.sum(decoy_score_sample)

    decoy_score_sample[decoy_score_sample >= 0.1] = 1.0
    decoy_score_sample[decoy_score_sample < 0.1] = 0.0

    # print(
    #     {
    #         'initial_known_hit_sum': initial_known_hit_sum,
    #         'initial_decoy_sum': initial_decoy_sum,
    #         'known hit sample sum': np.sum(known_hit_score_sample),
    #         'decoy sum': np.sum(decoy_score_sample),
    #         'n': np.power(64, 3)
    #     }
    # )


    score = np.sum(known_hit_score_sample * decoy_score_sample) / np.sum(decoy_score_sample)


    return score



class BuildScoringDataset(Dataset):

    def __init__(self, zarr_path, sample_indexes, pos_train_pose_samples):
        # self.data = data
        self.root = zarr.open(zarr_path, mode='r')

        self.meta_table = self.root['meta_sample']
        self.xmap_table = self.root['xmap_sample']
        self.zmap_table = self.root['z_map_sample']
        self.decoy_table = self.root['decoy_pose_sample']
        self.ligand_data_table = self.root['ligand_data']
        self.known_hit_pose = self.root['known_hit_pose']
        # self.pandda_2_annotation_table = self.root['annotation']
        # self.pandda_2_frag_table = self.root['ligand_confs']  # ['ligand_fragments']

        # self.pandda_2_annotations = {
        #     _x['event_idx']: _x
        #     for _x
        #     in self.pandda_2_annotation_table
        # }

        self.sample_indexes = sample_indexes

        self.pos_train_pose_samples = pos_train_pose_samples

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, idx: int):
        # Get the sample data

        sample_data = self.sample_indexes[idx]

        # Get the metadata, decoy pose and embedding
        _meta_idx, _decoy_idx, _embedding_idx, _train = sample_data['meta'], sample_data['decoy'], sample_data['embedding'], sample_data['train']
        _meta = self.meta_table[_meta_idx]
        _decoy = self.decoy_table[_decoy_idx]
        _embedding = self.decoy_table[_embedding_idx]

        # Get rng
        rng = np.random.default_rng()

        # Get the decoy
        valid_mask = _decoy['elements'] != 0
        # rprint(f'Initial valid mask sum: {valid_mask.sum()}')
        if _train:
            do_drop = rng.random()
            if do_drop > 0.5:
                valid_indicies = np.nonzero(valid_mask)
                print(valid_indicies)
                print(len(valid_indicies))
                random_drop_index = rng.integers(0, high=len(valid_indicies), size=3)
                print(random_drop_index)
                print(type(random_drop_index))
                print(random_drop_index.dtype)
                drop_index = valid_indicies[random_drop_index]
                valid_mask[drop_index] = False
        valid_poss = _decoy['positions'][valid_mask]
        valid_elements = _decoy['elements'][valid_mask]

        centroid = np.mean(valid_poss, axis=0)
        # print(centroid)
        # rprint(_decoy)
        # rprint(f'Sampling ligand centroid at: {centroid} from array of shape {valid_poss.shape} from {valid_mask.sum()}')

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

        decoy_residue = _get_res_from_arrays(
            valid_poss,
            valid_elements,
        )

        decoy_mask_grid = _get_ligand_mask_float(
            decoy_residue,
            radius=2.0,
            n=90,
            r=45.0
        )
        image_decoy_sample = _sample_xmap(
            decoy_mask_grid,
            transform,
            np.copy(sample_array)
        )

        image_decoy_mask = np.copy(sample_array)
        image_decoy_mask[image_decoy_sample > 0.0] = 1.0
        image_decoy_mask[image_decoy_sample <= 0.0] = 0.0


        decoy_score_mask_grid = _get_ligand_mask_float(
            decoy_residue,
            radius=1.0,
            n=90,
            r=45.0
        )
        image_score_decoy_mask = _sample_xmap(
            decoy_score_mask_grid,
            transform,
            np.copy(sample_array)
        )
        # image_score_decoy_mask[image_score_decoy_mask > 0.0] = 1.0

        # valid_poss = (valid_poss - np.mean(valid_poss, axis=0)) + np.array([22.5, 22.5, 22.5])

        # Get mask of hit for score calculation
        known_hit_pose = self.known_hit_pose[_meta['idx']]
        known_hit_pose_valid_mask = known_hit_pose['elements'] != 0
        known_hit_pose_valid_poss = known_hit_pose['positions'][known_hit_pose_valid_mask]
        known_hit_pose_valid_elements = known_hit_pose['elements'][known_hit_pose_valid_mask]
        known_hit_pose_residue = _get_res_from_arrays(
            known_hit_pose_valid_poss,
            known_hit_pose_valid_elements,
        )

        known_hit_pose_mask_grid = _get_ligand_mask_float(
            known_hit_pose_residue,
            radius=1.0,
            n=90,
            r=45.0
        )
        image_known_hit_pose_mask = _sample_xmap(
            known_hit_pose_mask_grid,
            transform,
            np.copy(sample_array)
        )
        # image_known_hit_pose_mask[image_known_hit_pose_mask >0.0] = 1.0

        # score = np.sum(image_score_decoy_mask * image_known_hit_pose_mask) / np.sum(image_score_decoy_mask)
        # data = np.hstack(
        #     [
        #         image_score_decoy_mask[image_score_decoy_mask > 0].reshape(-1,1),
        #         image_known_hit_pose_mask[image_score_decoy_mask > 0].reshape(-1, 1),
        #
        #     ]
        # )
        # score = np.corrcoef(data.T)[0, 1]
        score = _get_overlap_volume(
            orientation,
            centroid,
            known_hit_pose_residue,
            decoy_residue,
        )

        # Get maps
        xmap_data = self.xmap_table[_meta['idx']]
        zmap_data = self.zmap_table[_meta['idx']]

        xmap = _get_grid_from_hdf5(xmap_data)
        zmap = _get_grid_from_hdf5(zmap_data)
        # xmap_array = np.array(xmap, copy=False)
        # xmap_array[:,:,:] = (xmap_array - np.mean(xmap_array)) / np.std(xmap_array)
        # zmap_array = np.array(zmap, copy=False)
        # xmap_array[:,:,:] = (zmap_array - np.mean(zmap_array)) / np.std(zmap_array)

        # xmap_sample = _sample_xmap_and_scale(
        #     xmap,
        #     transform,
        #     np.copy(sample_array)
        # )
        # z_map_sample = _sample_xmap_and_scale(
        #     zmap,
        #     transform,
        #     np.copy(sample_array)
        # )
        xmap_sample = _sample_xmap_and_scale(
            xmap,
            transform,
            np.copy(sample_array)
        )
        z_map_sample = _sample_xmap_and_scale(
            zmap,
            transform,
            np.copy(sample_array)
        )

        if _train:
            u_s = rng.uniform(0.0, 0.25)
            noise = rng.normal(size=(32,32,32)) * u_s
            z_map_sample += noise.astype(np.float32)

            u_s = rng.uniform(0.0, 0.25)
            noise = rng.normal(size=(32,32,32)) * u_s
            xmap_sample += noise.astype(np.float32)


        # Get decoy for mol embedding
        # embedding = self.decoy_table[_embedding['idx']]
        # embedding_valid_mask = _embedding['elements'] != 0
        # if _train:
        #     do_drop = rng.random()
        #     if do_drop > 0.5:
        #         embedding_valid_indicies = np.nonzero(embedding_valid_mask)
        #         embedding_random_drop_index = rng.integers(0, len(embedding_valid_indicies))
        #         embedding_drop_index = embedding_valid_indicies[embedding_random_drop_index]
        #         embedding_valid_mask[embedding_drop_index] = False
        # embedding_valid_poss = _embedding['positions'][embedding_valid_mask]
        # embedding_valid_elements = _embedding['elements'][embedding_valid_mask]
        #
        # embedding_valid_poss = (embedding_valid_poss - np.mean(embedding_valid_poss, axis=0)) + np.array([22.5, 22.5, 22.5])
        #
        # embedding_orientation = _get_random_orientation()
        # embedding_transform = _get_transform_from_orientation_centroid(
        #     embedding_orientation,
        #     np.array([22.5,22.5,22.5]),
        #     n=32
        # )
        #
        # embedding_residue = _get_res_from_arrays(
        #     embedding_valid_poss,
        #     embedding_valid_elements,
        # )
        #
        # embedding_mask_grid = _get_ligand_mask_float(
        #     embedding_residue,
        #     radius=1.0
        # )
        # image_embedding_mask = _sample_xmap(
        #     embedding_mask_grid,
        #     embedding_transform,
        #     np.copy(sample_array)
        # )
        image_embedding_mask = np.zeros([32,32,32], dtype=np.float32)
        # score_his = np.zeros(10, dtype=np.float32)
        # score_his[int(10*score)] = 1.0

        rmsd = _decoy['rmsd']
        if _train:
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
                _embedding['idx'],
                str(_meta['system']),
                str(_meta['dtag']),
                int(_meta['event_num']),
            ],
            torch.from_numpy(
                np.stack(
                    [
                        z_map_sample,# * image_decoy_mask,
                        xmap_sample * image_decoy_mask,
                        image_score_decoy_mask
                    ],
                    axis=0,
                    dtype=np.float32
                )),
            torch.from_numpy(
                np.stack(
                    [image_decoy_mask],
                    axis=0,
                    dtype=np.float32
                )),
            torch.from_numpy(np.array(rmsd, dtype=np.float32)),
            torch.from_numpy(np.array(score, dtype=np.float32)),
            torch.from_numpy(np.array(hit, dtype=np.float32))
        )

        # # Get the z map and pose
        # if _table == 'normal':
        #     z_map_sample_metadata = self.z_map_sample_metadata_table[_z]
        # else:
        #     z_map_sample_metadata = self.pandda_2_z_map_sample_metadata_table[_z]
        # z_map_sample_idx = z_map_sample_metadata['idx']
        # # print([sample_idx, z_map_sample_idx])
        # assert _z == z_map_sample_idx
        # # event_map_idx = pose_data['event_map_sample_idx']
        # ligand_data_idx = z_map_sample_metadata['ligand_data_idx']
        # ligand_data = self.pandda_2_ligand_data_table[ligand_data_idx]
        #
        # pose_data_idx = z_map_sample_metadata['pose_data_idx']
        # if _table == 'normal':
        #
        #     xmap_sample_data = self.xmap_sample_table[z_map_sample_idx]
        #     z_map_sample_data = self.z_map_sample_table[z_map_sample_idx]
        #     annotation = self.annotations[z_map_sample_metadata['event_idx']]
        # else:
        #     xmap_sample_data = self.pandda_2_xmap_sample_table[z_map_sample_idx]
        #     z_map_sample_data = self.pandda_2_z_map_sample_table[z_map_sample_idx]
        #     annotation = self.pandda_2_annotations[z_map_sample_metadata['event_idx']]
        #
        # # If training replace with a random ligand
        # rng = np.random.default_rng()
        # # random_ligand = True
        # # if pose_data_idx != -1:
        # #     # random_ligand_sample = rng.random()
        # #     # if (random_ligand_sample > 0.5) & (annotation['partition'] == 'train'):
        # #     #     random_ligand = False
        # #     #     pose_data = self.pose_table[pose_data_idx]
        # #     # else:
        # #     #     pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
        # #     if sample_idx[0] == 'normal':
        # #
        # #         pose_data = self.pose_table[pose_data_idx]
        # #     else:
        # #         pose_data = self.pandda_2_pose_table[pose_data_idx]
        # # else:
        # #     if sample_idx[0] == 'normal':
        # #
        # #         # pose_data = self.pose_table[rng.integers(0, len(self.pose_table))]
        # #         selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
        # #         pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]
        # #     else:
        # #         selected_pose_idx = rng.integers(0, len(self.pos_train_pose_samples))
        # #         pose_data = self.pandda_2_pose_table[self.pos_train_pose_samples[selected_pose_idx]]
        # # if pose_data_idx != -1:
        # #     frag_data = self.pandda_2_frag_table[ligand_data['idx']]
        #
        # # else:
        # frag_data = self.pandda_2_frag_table[_f]
        #
        # #
        #
        #
        # # Subsample if training
        # if annotation['partition'] == 'train':
        #     translation = 3*(2*(rng.random(3)-0.5))
        #     centroid = np.array([22.5,22.5,22.5]) + translation
        #
        # else:
        #     centroid = np.array([22.5,22.5,22.5])
        #
        #
        # # Get sampling transform for the z map
        # sample_array = np.zeros(
        #     (32, 32, 32),
        #     dtype=np.float32,
        # )
        # orientation = _get_random_orientation()
        # transform = _get_transform_from_orientation_centroid(
        #     orientation,
        #     centroid,
        #     n=32
        # )
        #
        # # Get the sampling transform for the ligand map
        # # valid_mask = pose_data['elements'] != 0
        # # valid_poss = pose_data['positions'][valid_mask]
        # # valid_elements = pose_data['elements'][valid_mask]
        # valid_mask = frag_data['elements'] != 0
        # if annotation['partition'] == 'train':
        #     do_drop = rng.random()
        #     if do_drop > 0.5:
        #         valid_indicies = np.nonzero(valid_mask)
        #         random_drop_index = rng.integers(0, len(valid_indicies))
        #         drop_index = valid_indicies[random_drop_index]
        #         valid_mask[drop_index] = False
        # valid_poss = frag_data['positions'][valid_mask]
        # valid_elements = frag_data['elements'][valid_mask]
        #
        # valid_poss = (valid_poss - np.mean(valid_poss, axis=0)) + np.array([22.5, 22.5, 22.5])
        #
        # # # Subsample if training
        # # if annotation['partition'] == 'train':
        # #     rng = np.random.default_rng()
        # #     num_centres = rng.integers(1, 5)
        # #
        # #     # For each centre mask atoms close to it
        # #     total_mask = np.full(valid_elements.size, False)
        # #     for _centre in num_centres:
        # #         selected_atom = rng.integers(0, valid_elements.size)
        # #         poss_distances = valid_poss - valid_poss[selected_atom, :].reshape((1, 3))
        # #         close_mask = poss_distances[np.linalg.norm(poss_distances, axis=1) < 3.5]
        # #         total_mask[close_mask] = True
        # #
        # # else:
        # #     total_mask = np.full(valid_elements.size, True)
        #
        # ligand_sample_array = np.zeros(
        #     (32, 32, 32),
        #     dtype=np.float32,
        # )
        # ligand_orientation = _get_random_orientation()
        # # transformed_residue = _get_res_from_arrays(
        # #     valid_poss[total_mask],
        # #     valid_elements[total_mask],
        # # )
        # transformed_residue = _get_res_from_arrays(
        #     valid_poss,
        #     valid_elements,
        # )
        #
        # ligand_centroid = _get_centroid_from_res(transformed_residue)
        # ligand_map_transform = _get_transform_from_orientation_centroid(
        #     ligand_orientation,
        #     ligand_centroid,
        #     n=32
        # )
        #
        #
        # if annotation['partition'] == 'train':
        #     mask = get_mask_array()
        # else:
        #     mask = np.ones((32,32,32), dtype=np.float32)
        #
        # xmap_mask_float = _get_ed_mask_float()
        #
        # # Get sample images
        #
        #
        #
        # ligand_mask_grid = _get_ligand_mask_float(
        #     z_map,
        #     transformed_residue,
        # )
        # image_ligand_mask = _sample_xmap(
        #     ligand_mask_grid,
        #     ligand_map_transform,
        #     # transform,
        #     np.copy(ligand_sample_array)
        # )
        #
        # # image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        # # image_ligand_mask[image_ligand_mask >= 0.9] = 1.0
        # # image_ligand_mask = np.copy(sample_array)
        #
        # # if pose_data_idx != -1:
        # #     image_decoded_density = _sample_xmap(
        # #         ligand_mask_grid,
        # #         transform,
        # #         np.copy(sample_array)
        # #     )
        # # elif pose_data_idx == -1:
        # #     image_decoded_density = np.copy(sample_array)
        # # else:
        # #     raise Exception
        # image_decoded_density = np.copy(sample_array)
        #
        # # Make the image
        # image_density = np.stack(
        #     [
        #         xmap_sample,
        #     ],
        #     axis=0
        # )
        # image_density_float = image_density.astype(np.float32) * mask
        #
        # image_z = np.stack(
        #     [
        #         z_map_sample,
        #         xmap_sample * xmap_mask_float
        #     ],
        #     axis=0
        # )
        # image_z_float = image_z.astype(np.float32)  * mask
        #
        # image_mol = np.stack(
        #     [
        #         image_ligand_mask,
        #     ],
        #     axis=0
        # )
        # image_mol_float = image_mol.astype(np.float32)
        #
        # image_decoded_density = np.stack(
        #     [
        #         image_decoded_density,
        #     ],
        #     axis=0
        # )
        # image_decoded_density_float = image_decoded_density.astype(np.float32) * mask
        #
        # # res mask
        # # if annotation['partition'] == 'train':
        # #     grid = gemmi.FloatGrid(32, 32, 32)
        # #     grid.spacegroup = gemmi.SpaceGroup('P1')
        # #     uc = gemmi.UnitCell(16.0, 16.0, 16.0, 90.0, 90.0, 90.0)
        # #     grid.set_unit_cell(uc)
        # #     grid_array = np.array(grid, copy=False)
        # #     grid_array[:,:,:] = image_z_float[:,:,:]
        # #     rsg = gemmi.transform_map_to_f_phi(grid)
        # #     dmin = 1 + (1.0*rng.random())
        # #     data = rsg.prepare_asu_data(dmin=float(dmin))
        # #     grid_low_res = data.transform_f_phi_to_map(exact_size=[32, 32, 32])
        # #     grid_low_res_array = np.array(grid_low_res, copy=False)
        # #     image_z_float[:, :, :] = grid_low_res_array[:,:,:]
        #
        # # Make the annotation
        # # if (pose_data_idx != -1) & (not random_ligand):
        # #     hit = 1.0
        # # elif (pose_data_idx != -1) & (random_ligand):
        # #     hit = 0.0
        # # if pose_data_idx != -1:
        # #     hit = 1.0
        # # elif pose_data_idx == -1:
        # #     hit = 0.0
        # # else:
        # #     raise Exception
        # # if pose_data_idx != -1:
        # #     hit = [0.0, 1.0]
        # # elif pose_data_idx == -1:
        # #     hit = [1.0, 0.0]
        # # else:
        # #     raise Exception
        #
        # if annotation['partition'] == 'train':
        #     if _t == 'High':
        #         hit = [0.025, 0.975]
        #     elif _t == 'Medium':
        #         hit = [0.5, 0.5]
        #     elif _t == 'Low':
        #         hit = [0.975, 0.025]
        #     else:
        #         raise Exception
        # else:
        #     if _t == 'High':
        #         hit = [0.0, 1.0]
        #     elif _t == 'Medium':
        #         hit = [0.5, 0.5]
        #     elif _t == 'Low':
        #         hit = [1.0, 0.0]
        #     else:
        #         raise Exception
        #
        # label = np.array(hit)
        # label_float = label.astype(np.float32)
        #
        # return (
        #     [
        #         _table,
        #         _z,
        #         _f,
        #         str( z_map_sample_metadata['system']),
        #         str( z_map_sample_metadata['dtag']),
        #         int( z_map_sample_metadata['event_num']),
        #      ],
        #     torch.from_numpy(image_density_float),
        #     torch.from_numpy(image_z_float),
        #     torch.from_numpy(image_decoy_mask),
        #     torch.from_numpy(image_decoded_density_float),
        #     torch.from_numpy(label_float)
        # )
