import os
from pathlib import Path

import logging
import sys

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony
import numpy as np
import tables
import zarr
import pandas as pd
import torch

from edanalyzer.datasets.event_scoring import EventScoringDataset
from edanalyzer.models.event_scoring import LitEventScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks import EarlyStopping

# import ray
# from ray.train.lightning import (
#     RayDDPStrategy,
#     RayLightningEnvironment,
#     RayTrainReportCallback,
#     prepare_trainer,
# )
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer
# from ray.tune.search.bayesopt import BayesOptSearch
# from ray.tune.search.ax import AxSearch
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.basic_variant import BasicVariantGenerator

from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
import optuna


def sample(iterable, num, replace, weights):
    df = pd.Series(iterable)
    return [_x for _x in df.sample(num, replace=replace, weights=weights)]


# def _get_train_test_idxs(root):
#     # for each z map sample
#     # 1. Get the corresponding ligand data
#     # 2. Get the corresponding fragments
#     # 3. Sample the map n times positively (random corresponding fragment, random conformation)
#     # 4. Sample the the map n times negatively (weighted random fragment of different size, random conformation)
#     # 5. Sample the fragments n times negatively (random corresponding fragment, random conformation, random negative map)
#     # Sample distribution is now 1/3 positive to negative, and every positive map appears the same number of
#     # times negatively, and every positive fragment appears appears twice negatively
#     rng = np.random.default_rng()
#
#     table_type = 'pandda_2'
#
#     metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
#     table_annotation = root[table_type]['annotation']
#     annotation_df = pd.DataFrame(table_annotation[:])
#     ligand_idx_smiles_df = pd.DataFrame(
#         root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
#     fragment_df = pd.DataFrame(
#         root[table_type]['ligand_fragments'].get_basic_selection(slice(None), fields=['idx', 'num_heavy_atoms',
#                                                                                       'fragment_canonical_smiles',
#                                                                                       'ligand_canonical_smiles']))
#
#     train_samples = metadata_table[annotation_df['partition'] == b'train']
#     test_samples = metadata_table[annotation_df['partition'] == b'test']
#
#
#     frag_smiles_to_conf = {
#         _fragment: fragment_df[fragment_df['fragment_canonical_smiles'] == _fragment]
#         for _fragment
#         in fragment_df['fragment_canonical_smiles'].unique()
#     }
#
#     fragment_to_sizes = {}
#     # size_to_fragments = {}
#     for smiles in fragment_df['fragment_canonical_smiles'].unique():
#         corresponding_sizes = fragment_df[fragment_df['fragment_canonical_smiles'] == smiles][
#             'num_heavy_atoms'].unique()
#         fragment_to_sizes[smiles] = [_x for _x in corresponding_sizes][0]
#
#     size_to_fragments = {}
#     for size in fragment_df['num_heavy_atoms'].unique():
#         corresponding_fragments = fragment_df[fragment_df['num_heavy_atoms'] == size][
#             'fragment_canonical_smiles'].unique()
#         size_to_fragments[size] = [_x for _x in corresponding_fragments]
#
#     fragments_to_ligands = {}
#     for smiles in fragment_df['fragment_canonical_smiles'].unique():
#         corresponding_ligands = fragment_df[fragment_df['fragment_canonical_smiles'] == smiles][
#             'ligand_canonical_smiles'].unique()
#         fragments_to_ligands[smiles] = [x for x in corresponding_ligands]
#
#     ligand_to_fragments = {}
#     for smiles in fragment_df['ligand_canonical_smiles'].unique():
#         corresponding_fragments = fragment_df[fragment_df['ligand_canonical_smiles'] == smiles][
#             'fragment_canonical_smiles'].unique()
#         ligand_to_fragments[smiles] = [x for x in corresponding_fragments]
#
#     smiles_to_conf = {
#         _fragment: fragment_df[fragment_df['fragment_canonical_smiles'] == _fragment]
#         for _fragment
#         in fragment_df['fragment_canonical_smiles'].unique()
#     }
#
#     pos_z_samples = []
#     neg_z_samples = []
#     pos_fragment_samples = []
#     neg_fragment_samples = []
#     positive_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#     negative_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#
#     pos_frag_size_samples = {_j: 0 for _j in size_to_fragments}
#     neg_frag_size_samples = {_j: 0 for _j in size_to_fragments}
#
#     # Loop over the z samples adding positive samples for each
#     for _idx, z in train_samples.iterrows():
#         ligand_data_idx = z['ligand_data_idx']
#         if ligand_data_idx == -1:
#             continue
#         ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#         ligand_canonical_smiles = ligand_data['canonical_smiles']
#         if ligand_canonical_smiles not in ligand_to_fragments:
#             continue
#         fragments = ligand_to_fragments[ligand_canonical_smiles]
#         if len(fragments) == 0:
#             continue
#
#         # Pos samples
#         fragment_smiles_samples = sample(fragments, 10, replace=True, weights=None)
#         fragment_conf_samples = []
#         for _fragment in fragment_smiles_samples:
#             positive_fragment_sample_distribution[_fragment] += 1
#             fragment_conf_samples.append(frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0])
#             pos_frag_size_samples[fragment_to_sizes[_fragment]] += 1
#         pos_fragment_samples += fragment_conf_samples
#         pos_z_samples += [z['idx'] for _j in range(10)]
#
#     print(f'Got {len(pos_fragment_samples)} pos samples!')
#
#     # Loop over the z samples adding the inherent negative samples
#     for _idx, z in train_samples[train_samples['ligand_data_idx'] == -1].sample(len(pos_fragment_samples),
#                                                                                 replace=True).iterrows():
#         ligand_data_idx = z['ligand_data_idx']
#         if ligand_data_idx != -1:
#             continue
#
#         # Select a uniform random fragment
#         fragment_freq = {
#             k: v / positive_fragment_sample_distribution[k]
#             for k, v
#             in negative_fragment_sample_distribution.items()
#             if positive_fragment_sample_distribution[k] > 0
#         }
#
#         fragment = min(fragment_freq, key=lambda _k: fragment_freq[_k])
#         negative_fragment_sample_distribution[fragment] += 1
#         neg_frag_size_samples[fragment_to_sizes[fragment]] += 1
#
#         fragment_conf_sample = frag_smiles_to_conf[fragment].sample(1)['idx'].iloc[0]
#
#         neg_fragment_samples += [fragment_conf_sample, ]
#         neg_z_samples += [z['idx'], ]
#
#     print(f'Got {len(pos_fragment_samples)} neg decoy samples!')
#
#     # pos_z_samples = []
#     # neg_z_samples = []
#     # pos_fragment_samples = []
#     # neg_fragment_samples = []
#     # positive_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#     # negative_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#     #
#     # pos_frag_size_samples = {_j: 0 for _j in size_to_fragments}
#     # neg_frag_size_samples = {_j: 0 for _j in size_to_fragments}
#     #
#     # # Loop over the z samples adding positive samples for each
#     # for _idx, z in train_samples.iterrows():
#     #     ligand_data_idx = z['ligand_data_idx']
#     #     if ligand_data_idx == -1:
#     #         continue
#     #     ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#     #     ligand_canonical_smiles = ligand_data['canonical_smiles']
#     #     if ligand_canonical_smiles not in ligand_to_fragments:
#     #         continue
#     #     fragments = ligand_to_fragments[ligand_canonical_smiles]
#     #     if len(fragments) == 0:
#     #         continue
#     #
#     #     # Pos samples
#     #     fragment_smiles_samples = sample(fragments, 10, replace=True, weights=None)
#     #     fragment_conf_samples = []
#     #     for _fragment in fragment_smiles_samples:
#     #         positive_fragment_sample_distribution[_fragment] += 1
#     #         fragment_conf_samples.append(frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0])
#     #         pos_frag_size_samples[fragment_to_sizes[_fragment]] += 1
#     #     pos_fragment_samples += fragment_conf_samples
#     #     pos_z_samples += [z['idx'] for _j in range(10)]
#     #
#     # print(f'Got {len(pos_fragment_samples)} pos samples!')
#     #
#     # # Loop over the z samples adding negative samples for each z map
#     # for _idx, z in train_samples.iterrows():
#     #     ligand_data_idx = z['ligand_data_idx']
#     #     if ligand_data_idx == -1:
#     #         continue
#     #     ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#     #     ligand_canonical_smiles = ligand_data['canonical_smiles']
#     #     if ligand_canonical_smiles not in ligand_to_fragments:
#     #         continue
#     #     fragments = ligand_to_fragments[ligand_canonical_smiles]
#     #     if len(fragments) == 0:
#     #         continue
#     #
#     #     # Neg samples
#     #     fragment_sizes = [fragment_to_sizes[_fragment] for _fragment in fragments]
#     #     legal_sizes = [j for j in size_to_fragments if all([j not in [y, ] for y in fragment_sizes])]
#     #
#     #     if len(legal_sizes) == 0:
#     #         continue
#     #
#     #     size_freq = {
#     #         k: v / pos_frag_size_samples[k]
#     #         for k, v
#     #         in neg_frag_size_samples.items()
#     #         if k in legal_sizes
#     #     }
#     #
#     #     size_to_sample = min(size_freq, key=lambda _k: size_freq[_k])
#     #
#     #     possible_fragments = [_f for _f in size_to_fragments[size_to_sample] if
#     #                           positive_fragment_sample_distribution[_f] > 0]
#     #     possible_fragment_counts = {
#     #         _fragment: positive_fragment_sample_distribution[_fragment]
#     #         for _fragment
#     #         in possible_fragments
#     #         if positive_fragment_sample_distribution[_fragment] > 0
#     #     }
#     #
#     #     if len(possible_fragment_counts) == 0:
#     #         continue
#     #
#     #     num_neg_samples = 0
#     #     fragment_smiles_samples = []
#     #     while num_neg_samples < 11:
#     #         fragment_of_other_sizes_counts = {
#     #             k: v / positive_fragment_sample_distribution[k]
#     #             for k, v
#     #             in negative_fragment_sample_distribution.items()
#     #             if k in possible_fragments
#     #         }
#     #         next_sample = min(fragment_of_other_sizes_counts, key=lambda _x: fragment_of_other_sizes_counts[_x])
#     #         fragment_smiles_samples.append(next_sample)
#     #
#     #         negative_fragment_sample_distribution[next_sample] += 1
#     #
#     #         num_neg_samples += 1
#     #
#     #     fragment_conf_samples = []
#     #     for _fragment in fragment_smiles_samples:
#     #         fragment_conf_samples.append(frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0])
#     #         neg_frag_size_samples[fragment_to_sizes[_fragment]] += 1
#     #
#     #     neg_fragment_samples += fragment_conf_samples
#     #     neg_z_samples += [z['idx'] for _j in range(10)]
#     #
#     # print(f'Got {len(pos_fragment_samples)} neg decoy samples!')
#
#     #
#     # pos_z_samples = []
#     # neg_z_samples = []
#     # pos_fragment_samples = []
#     # neg_fragment_samples = []
#     # positive_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#     # negative_fragment_sample_distribution = {_fragment: 0 for _fragment in fragment_to_sizes}
#     #
#     #
#     # # Loop over the z samples adding positive samples for each
#     # for _idx, z in train_samples.iterrows():
#     #     ligand_data_idx = z['ligand_data_idx']
#     #     if ligand_data_idx == -1:
#     #         continue
#     #     ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#     #     ligand_canonical_smiles = ligand_data['canonical_smiles']
#     #     if ligand_canonical_smiles not in ligand_to_fragments:
#     #         continue
#     #     fragments = ligand_to_fragments[ligand_canonical_smiles]
#     #     if len(fragments) == 0:
#     #         continue
#     #
#     #     # fragment_sizes = [fragment_to_sizes[_fragment] for _fragment in fragments] + [
#     #     #     fragment_to_sizes[_fragment] - 1 for _fragment in fragments] + [fragment_to_sizes[_fragment] + 1 for
#     #     #                                                                     _fragment in fragments] + [
#     #     #                      fragment_to_sizes[_fragment] - 2 for _fragment in fragments] + [
#     #     #                      fragment_to_sizes[_fragment] + 2 for _fragment in fragments]
#     #     # fragments_of_other_sizes = []
#     #     # for _size, _fragments_of_size in size_to_fragments.items():
#     #     #     if _size in fragment_sizes:
#     #     #         continue
#     #     #
#     #     #     for _fragment in _fragments_of_size:
#     #     #         if positive_fragment_sample_distribution[_fragment] > 0:
#     #     #             fragments_of_other_sizes.append(_fragment)
#     #     #
#     #     # other_possible_fragments = {k: v for k, v in negative_fragment_sample_distribution.items() if
#     #     #                             k in fragments_of_other_sizes}
#     #     # if len(other_possible_fragments) == 0:
#     #     #     continue
#     #
#     #     # Pos samples
#     #     fragment_smiles_samples = sample(fragments, 10, replace=True, weights=None)
#     #     fragment_conf_samples = []
#     #     for _fragment in fragment_smiles_samples:
#     #         positive_fragment_sample_distribution[_fragment] += 1
#     #         fragment_conf_samples.append(frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0])
#     #     pos_fragment_samples += fragment_conf_samples
#     #     pos_z_samples += [z['idx'] for _j in range(10)]
#     #
#     # print(f'Got {len(pos_fragment_samples)} pos samples!')
#     #
#     # # Loop over the z samples adding negative samples for each z map
#     # for _idx, z in train_samples.iterrows():
#     #     ligand_data_idx = z['ligand_data_idx']
#     #     if ligand_data_idx == -1:
#     #         continue
#     #     ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#     #     ligand_canonical_smiles = ligand_data['canonical_smiles']
#     #     if ligand_canonical_smiles not in ligand_to_fragments:
#     #         continue
#     #     fragments = ligand_to_fragments[ligand_canonical_smiles]
#     #     if len(fragments) == 0:
#     #         continue
#     #
#     #     # Neg samples
#     #     fragment_sizes = [fragment_to_sizes[_fragment] for _fragment in fragments] + [fragment_to_sizes[_fragment]-1 for _fragment in fragments] + [fragment_to_sizes[_fragment]+1 for _fragment in fragments]+ [fragment_to_sizes[_fragment]-2 for _fragment in fragments] + [fragment_to_sizes[_fragment]+2 for _fragment in fragments]
#     #     fragments_of_other_sizes = []
#     #     for _size, _fragments_of_size in size_to_fragments.items():
#     #         if _size in fragment_sizes:
#     #             continue
#     #
#     #         for _fragment in _fragments_of_size:
#     #             if positive_fragment_sample_distribution[_fragment] > 0:
#     #                 fragments_of_other_sizes.append(_fragment)
#     #
#     #     other_possible_fragments = {k: v for k, v in negative_fragment_sample_distribution.items() if k in fragments_of_other_sizes}
#     #     if len(other_possible_fragments) == 0:
#     #         continue
#     #
#     #     num_neg_samples = 0
#     #     fragment_smiles_samples = []
#     #     while num_neg_samples < 11:
#     #         fragment_of_other_sizes_counts = {
#     #             k: v / positive_fragment_sample_distribution[k]
#     #             for k, v
#     #             in negative_fragment_sample_distribution.items()
#     #             if k in fragments_of_other_sizes
#     #         }
#     #         next_sample = min(fragment_of_other_sizes_counts, key=lambda _x: fragment_of_other_sizes_counts[_x])
#     #         fragment_smiles_samples.append(next_sample)
#     #
#     #         negative_fragment_sample_distribution[next_sample] += 1
#     #
#     #         num_neg_samples += 1
#     #
#     #     fragment_conf_samples = []
#     #     for _fragment in fragment_smiles_samples:
#     #         fragment_conf_samples.append(frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0])
#     #
#     #     neg_fragment_samples += fragment_conf_samples
#     #     neg_z_samples += [z['idx'] for _j in range(10)]
#     #
#     # print(f'Got {len(pos_fragment_samples)} neg decoy samples!')
#
#     # # Loop over the z samples adding the inherent negative samples
#     # for _idx, z in train_samples[train_samples['ligand_data_idx'] == -1].sample(len(pos_fragment_samples),
#     #                                                                             replace=True).iterrows():
#     #     ligand_data_idx = z['ligand_data_idx']
#     #     if ligand_data_idx != -1:
#     #         continue
#     #
#     #     # Select a uniform random fragment
#     #     fragment = \
#     #     sample([_x for _x in fragment_to_sizes if positive_fragment_sample_distribution[_x] > 0], 1, False, None)[0]
#     #
#     #     fragment_conf_sample = frag_smiles_to_conf[fragment].sample(1)['idx'].iloc[0]
#     #
#     #     neg_fragment_samples += [fragment_conf_sample, ]
#     #     neg_z_samples += [z['idx'], ]
#
#     test_pos_z_samples = []
#     test_neg_z_samples = []
#     test_pos_fragment_samples = []
#     test_neg_fragment_samples = []
#
#     # Loop over the z samples adding the test samples
#     for _idx, z in test_samples.iterrows():
#         #
#         ligand_data_idx = z['ligand_data_idx']
#         if ligand_data_idx != -1:
#             ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
#             ligand_canonical_smiles = ligand_data['canonical_smiles']
#             if ligand_canonical_smiles not in ligand_to_fragments:
#                 continue
#
#             fragments = ligand_to_fragments[ligand_canonical_smiles]
#             if len(fragments) == 0:
#                 continue
#
#             for _fragment in fragments:
#                 fragment_conf_sample = frag_smiles_to_conf[_fragment].sample(1)['idx'].iloc[0]
#                 test_pos_fragment_samples.append(fragment_conf_sample)
#                 test_pos_z_samples.append(z['idx'])
#
#
#         else:
#             fragment = \
#             sample([_x for _x in fragment_to_sizes if positive_fragment_sample_distribution[_x] > 0], 1, False, None)[0]
#             fragment_conf_sample = frag_smiles_to_conf[fragment].sample(1)['idx'].iloc[0]
#
#             test_neg_fragment_samples.append(fragment_conf_sample)
#             test_neg_z_samples.append(z['idx'])
#
#
#     train_idxs = [{'table': table_type, 'z': z, 'f': f, 't': t} for z, f, t in
#                   zip(pos_z_samples + neg_z_samples, pos_fragment_samples + neg_fragment_samples,
#                       ([True] * len(pos_z_samples)) + ([False] * len(neg_z_samples)))]
#     test_idxs = [{'table': table_type, 'z': z, 'f': f, 't': t} for z, f, t
#                  in zip(
#             test_pos_z_samples + test_neg_z_samples,
#             test_pos_fragment_samples + test_neg_fragment_samples,
#             ([True] * len(test_pos_fragment_samples)) + ([False] * len(test_neg_fragment_samples))
#         )
#                  ]
#
#     return train_idxs, test_idxs

def _dep_get_train_test_idxs_full_conf(root):
    # for each z map sample
    # 1. Get the corresponding ligand data
    # 2. Get the corresponding fragments
    # 3. Sample the map n times positively (random corresponding fragment, random conformation)
    # 4. Sample the the map n times negatively (weighted random fragment of different size, random conformation)
    # 5. Sample the fragments n times negatively (random corresponding fragment, random conformation, random negative map)
    # Sample distribution is now 1/3 positive to negative, and every positive map appears the same number of
    # times negatively, and every positive fragment appears appears twice negatively
    rng = np.random.default_rng()

    table_type = 'pandda_2'

    print(f'Loading metadata table')
    metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
    table_annotation = root[table_type]['annotation']
    print(f'Loading annotation table')
    annotation_df = pd.DataFrame(table_annotation[:])
    print(f'Loading ligand table')
    ligand_idx_smiles_df = pd.DataFrame(
        root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
    print(f'Loading conf table')
    ligand_conf_df = pd.DataFrame(
        root[table_type]['ligand_confs'].get_basic_selection(slice(None), fields=['idx', 'num_heavy_atoms',
                                                                                  'fragment_canonical_smiles',
                                                                                  'ligand_canonical_smiles']))

    train_samples = metadata_table[annotation_df['partition'] == b'train']
    test_samples = metadata_table[annotation_df['partition'] == b'test']

    print(f'Getting ligand smiles to conf mapping')
    ligand_smiles_to_conf = {
        _smiles: ligand_conf_df[ligand_conf_df['ligand_canonical_smiles'] == _smiles]
        for _smiles
        in ligand_conf_df['ligand_canonical_smiles'].unique()
    }

    pos_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []
    positive_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    negative_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    train_pos_conf = []
    train_neg_conf = []

    # Loop over the z samples adding positive samples for each
    print(f'Getting positive train samples')
    for _idx, z in train_samples.iterrows():
        ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx == -1:
        #     continue
        if z['Confidence'] != 'High':
            continue
        ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
        ligand_canonical_smiles = ligand_data['canonical_smiles']
        if ligand_canonical_smiles not in ligand_smiles_to_conf:
            continue
        confs = ligand_smiles_to_conf[ligand_canonical_smiles]
        if len(confs) == 0:
            continue

        # Pos samples
        ligand_conf_samples = []
        for x in range(10):
            positive_ligand_sample_distribution[ligand_canonical_smiles] += 1
            ligand_conf_samples.append(ligand_smiles_to_conf[ligand_canonical_smiles].sample(1)['idx'].iloc[0])
        pos_conf_samples += ligand_conf_samples
        pos_z_samples += [z['idx'] for _j in range(10)]
        train_pos_conf += [z['Confidence'] for _j in range(10)]

    print(f'Got {len(pos_conf_samples)} pos samples!')

    # Loop over the z samples adding the inherent negative samples
    print(f'Getting negative train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'Low'].sample(len(pos_conf_samples),
                                                                              replace=True).iterrows():
        # ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx != -1:
        #     continue
        if z['Confidence'] != 'Low':
            continue

        # Select a uniform random fragment
        ligand_freq = {
            k: v / positive_ligand_sample_distribution[k]
            for k, v
            in negative_ligand_sample_distribution.items()
            if positive_ligand_sample_distribution[k] > 0
        }

        ligand_smiles = min(ligand_freq, key=lambda _k: ligand_freq[_k])
        negative_ligand_sample_distribution[ligand_smiles] += 1

        lig_conf_sample = ligand_smiles_to_conf[ligand_smiles].sample(1)['idx'].iloc[0]

        neg_conf_samples += [lig_conf_sample, ]
        neg_z_samples += [z['idx'], ]
        train_neg_conf += [z['Confidence'], ]

    print(f'Got {len(neg_conf_samples)} neg decoy samples!')

    test_pos_z_samples = []
    test_neg_z_samples = []
    test_pos_conf_samples = []
    test_neg_conf_samples = []
    test_pos_conf = []
    test_neg_conf = []

    # Loop over the z samples adding the test samples
    print(f'Getting test samples')
    for _idx, z in test_samples.iterrows():
        #
        ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx != -1:
        if z['Confidence'] == 'High':
            ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
            ligand_canonical_smiles = ligand_data['canonical_smiles']
            if ligand_canonical_smiles not in ligand_smiles_to_conf:
                continue

            lig_conf_sample = ligand_smiles_to_conf[ligand_canonical_smiles].sample(1)['idx'].iloc[0]

            test_pos_conf_samples.append(lig_conf_sample)
            test_pos_z_samples.append(z['idx'])
            test_pos_conf.append(z['Confidence'])

        # else:
        elif z['Confidence'] == 'Low':
            fragment = \
                sample(
                    [_x for _x in positive_ligand_sample_distribution if positive_ligand_sample_distribution[_x] > 0],
                    1, False,
                    None)[0]
            lig_conf_sample = ligand_smiles_to_conf[fragment].sample(1)['idx'].iloc[0]

            test_neg_conf_samples.append(lig_conf_sample)
            test_neg_z_samples.append(z['idx'])
            test_neg_conf.append(z['Confidence'])

    rprint({
        'pos_z_samples len': len(pos_z_samples),
        'neg_z_samples len': len(neg_z_samples),
        'pos_conf_samples len': len(pos_conf_samples),
        'neg_conf_samples len': len(neg_conf_samples),
        'train_pos_conf len': len(train_pos_conf),
        'train_neg_conf len': len(train_neg_conf),

    })
    train_idxs = [
        {'table': table_type, 'z': z, 'f': f, 't': t}
        for z, f, t
        in zip(
            pos_z_samples + neg_z_samples,
            pos_conf_samples + neg_conf_samples,
            train_pos_conf + train_neg_conf
        )
        # ([True] * len(pos_z_samples)) + ([False] * len(neg_z_samples)))]
    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'train_pos_conf len': len(test_pos_conf),
        'train_neg_conf len': len(test_neg_conf),
    })
    test_idxs = [{'table': table_type, 'z': z, 'f': f, 't': t} for z, f, t
                 in zip(
            test_pos_z_samples + test_neg_z_samples,
            test_pos_conf_samples + test_neg_conf_samples,
            # ([True] * len(test_pos_conf_samples)) + ([False] * len(test_neg_conf_samples))
            test_pos_conf + test_neg_conf
        )
                 ]
    return train_idxs, test_idxs


def _get_train_test_idxs_full_conf(root):
    # for each z map sample
    # 1. Get the corresponding ligand data
    # 2. Get the corresponding fragments
    # 3. Sample the map n times positively (random corresponding fragment, random conformation)
    # 4. Sample the the map n times negatively (weighted random fragment of different size, random conformation)
    # 5. Sample the fragments n times negatively (random corresponding fragment, random conformation, random negative map)
    # Sample distribution is now 1/3 positive to negative, and every positive map appears the same number of
    # times negatively, and every positive fragment appears appears twice negatively
    rng = np.random.default_rng()

    table_type = 'pandda_2'

    print(f'Loading metadata table')
    metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
    table_annotation = root[table_type]['annotation']
    print(f'Loading annotation table')
    annotation_df = pd.DataFrame(table_annotation[:])
    print(f'Loading ligand table')
    ligand_idx_smiles_df = pd.DataFrame(
        root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
    print(f'Getting valid smiles...')
    valid_smiles_df = pd.DataFrame(root[table_type]['valid_smiles'][:])
    # print(f'Loading conf table')
    # ligand_conf_df = pd.DataFrame(
    #     root[table_type]['ligand_confs'].get_basic_selection(slice(None), fields=['idx', 'num_heavy_atoms',
    #                                                                               'fragment_canonical_smiles',
    #                                                                               'ligand_canonical_smiles']))
    valid_smiles_mask = valid_smiles_df.iloc[ligand_idx_smiles_df.iloc[metadata_table['ligand_data_idx']]['idx']][
        'valid']
    print(valid_smiles_mask)
    train_samples = metadata_table[(annotation_df['partition'] == b'train') & valid_smiles_mask]
    test_samples = metadata_table[(annotation_df['partition'] == b'test') & valid_smiles_mask]

    # print(f'Getting ligand smiles to conf mapping')
    # ligand_smiles_to_conf = {
    #     _smiles: ligand_conf_df[ligand_conf_df['ligand_canonical_smiles'] == _smiles]
    #     for _smiles
    #     in ligand_conf_df['ligand_canonical_smiles'].unique()
    # }

    pos_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []
    # positive_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    # negative_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    train_pos_conf = []
    train_neg_conf = []

    # Loop over the z samples adding the inherent negative samples
    print(f'Getting negative train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'Low'].iterrows():
        neg_z_samples += [z['idx'], ]
        train_neg_conf.append('Low')

    # Loop over the z samples adding positive samples for each
    print(f'Getting positive train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'High'].iterrows():  # .sample(len(neg_z_samples),
        #     replace=True).iterrows():
        pos_z_samples += [z['idx'], ]
        train_pos_conf.append('High')

    print(f'Got {len(pos_z_samples)} pos samples!')

    print(f'Got {len(neg_conf_samples)} neg decoy samples!')

    test_pos_z_samples = []
    test_neg_z_samples = []
    test_pos_conf_samples = []
    test_neg_conf_samples = []
    test_pos_conf = []
    test_neg_conf = []

    test_med_z_samples = []
    test_med_conf = []
    # Loop over the z samples adding the test samples
    print(f'Getting test samples')
    for _idx, z in test_samples.iterrows():
        if z['Confidence'] == 'High':
            test_pos_z_samples.append(z['idx'])
            test_pos_conf.append('High')

        elif z['Confidence'] == 'Medium':
            test_med_z_samples.append(z['idx'])
            test_med_conf.append('Medium')


        elif z['Confidence'] == 'Low':
            test_neg_z_samples.append(z['idx'])
            test_neg_conf.append('Low')

    rprint({
        'pos_z_samples len': len(pos_z_samples),
        'neg_z_samples len': len(neg_z_samples),
        'pos_conf_samples len': len(pos_conf_samples),
        'neg_conf_samples len': len(neg_conf_samples),
        'train_pos_conf len': len(train_pos_conf),
        'train_neg_conf len': len(train_neg_conf),

    })
    train_idxs = [
        {'z': z, 'conf': conf}
        for z, conf
        in zip(
            pos_z_samples + neg_z_samples,
            train_pos_conf + train_neg_conf
        )

    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'test_pos_conf len': len(test_pos_conf),
        'test_neg_conf len': len(test_neg_conf),

    })
    test_idxs = [{'z': z, 'conf': conf} for z, conf
                 in zip(
            test_pos_z_samples + test_neg_z_samples + test_med_z_samples,
            test_pos_conf + test_neg_conf + test_med_conf
        )
                 ]

    res = {}
    for train_test, sample_indexes in zip(['train', 'test'], [train_idxs, test_idxs]):
        res[train_test] = {}
        res[train_test]['indexes'] = sample_indexes

        res[train_test]['pandda_2_annotations'] = {
            _x['event_idx']: _x
            for _x
            in root['pandda_2']['annotation']
        }

        res[train_test]['ligand_data_df'] = ligand_idx_smiles_df

        res[train_test]['metadata_table'] = metadata_table
        sampled_metadata_table = metadata_table.iloc[[x['z'] for x in sample_indexes]]
        res[train_test]['sampled_metadata_table'] = sampled_metadata_table
        res[train_test]['metadata_table_high_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'High']
        res[train_test]['metadata_table_low_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'Low']

        selected_pos_samples = metadata_table.iloc[[x['z'] for x in sample_indexes]]
        selected_smiles = ligand_idx_smiles_df.iloc[selected_pos_samples['ligand_data_idx']]['canonical_smiles']
        print(selected_smiles)
        unique_smiles, smiles_counts = np.unique(selected_smiles, return_counts=True)

        res[train_test]['unique_smiles'] = pd.Series(unique_smiles)
        res[train_test]['unique_smiles_frequencies'] = pd.Series(smiles_counts.astype(float) / np.sum(smiles_counts))
    return res['train'], res['test']


def main(config_path, batch_size=12, num_workers=None):
    rprint(f'Running train event scoring from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the Database
    database_path = Path(config['working_directory']) / "database.db"
    rprint(f'loading database from: {database_path}')

    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    output_dir = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output')

    zarr_path = output_dir / 'event_data_3.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    rprint(f'Getting train/test data...')
    train_config, test_config = _get_train_test_idxs_full_conf(root)

    rprint(f'Constructing train and test dataloaders...')

    # Get the model
    rprint('Constructing model...')
    study_name = 'event_scoring_test'
    output = output_dir / study_name
    if not output.exists():
        os.mkdir(output)

    # Train
    rprint('Constructing trainer...')

    rprint(f'Training...')

    _config = {
        'lr': 0.00027608304667883787, 'wd': 0.004428399357109647, 'fraction_background_replace': 0.9977586581425819,
        'xmap_radius': 5.679885665547368, 'max_x_blur': 0.47417970205607624, 'max_z_blur': 0.6342802782754948,
        'drop_rate': 0.34074819841381004, 'planes_1': 16, 'drop_1': 0.04973841976629942, 'planes_2': 32,
        'drop_2': 0.4773445563051335, 'planes_3': 64, 'drop_3': 0.7220893799410683, 'planes_4': 32,
        'drop_4': 0.42842911756667934,
        'planes_5': 256, 'drop_5': 0.8164371048868642, 'drop_atom_rate': 0.6835634852890703,
        'max_pos_atom_mask_radius': 3.5505538045507197, 'max_translate': 1.504396022687739,
        'max_x_noise': 0.6267668374814633,
        'max_z_noise': 0.9535320617031404, 'pos_resample_rate': 10, 'p_flip': 0.3678479092419647,
        'z_mask_radius': 2.659870974428465, 'z_cutoff': 1.892141827304312, 'combo_layer': 64, 'blocks_1': 2,
        'blocks_2': 2,
        'blocks_3': 1, 'blocks_4': 2, 'grad_clip': 0.0004551500618521706, 'batch_size': 32
    }

    _test_config = {
        'zarr_path': zarr_path,
    }
    _test_config.update(test_config)
    _test_config.update(_config)
    _test_config.update({'test_train': 'test'})

    model = LitEventScoring.load_from_checkpoint(
        output_dir / 'event_scoring_prod_50/351/sample-mnist-epoch=27-medianfpr99=0.07.ckpt')
    model.eval()
    model.output = output

    dataset_test = DataLoader(
        EventScoringDataset(
            _test_config
        ),
        batch_size=1,
        num_workers=19,
        drop_last=False
    )

    results = []
    for test_batch in dataset_test:
        idx, x, z, m, d, y = test_batch
        y = y.view(y.size(0), -1)
        y_hat = model(None, z, m, None)
        for _j, _ in enumerate(y_hat):
            res = {
                "system": str(idx[3][_j]),
                "dtag": str(idx[4][_j]),
                "event_num": int(idx[5][_j]),
                'y': y[_j],
                'y_hat': y_hat[_j],
            }
            rprint(res)
            results.append(res)
    df = pd.DataFrame(results)
    df.to_csv(output / 'results.csv')


if __name__ == "__main__":
    fire.Fire(main)
