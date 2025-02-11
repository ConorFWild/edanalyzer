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
    med_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []
    # positive_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    # negative_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    train_pos_conf = []
    train_med_conf = []
    train_neg_conf = []

    # Loop over the z samples adding the inherent negative samples
    print(f'Getting negative train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'Low'].iterrows():
        neg_z_samples += [z['idx'], ]
        train_neg_conf.append('Low')

    for _idx, z in train_samples[train_samples['Confidence'] == 'Medium'].iterrows():
        med_z_samples.append(z['idx'])
        train_med_conf.append('Medium')

    # Loop over the z samples adding positive samples for each
    print(f'Getting positive train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'High'].iterrows():  # .sample(len(neg_z_samples),
        #     replace=True).iterrows():
        pos_z_samples += [z['idx'], ]
        train_pos_conf.append('High')

    print(f'Got {len(pos_z_samples)} pos samples!')

    print(f'Got {len(neg_conf_samples)} neg decoy samples!')

    test_pos_z_samples = []
    test_med_z_samples = []
    test_neg_z_samples = []
    test_pos_conf_samples = []
    test_neg_conf_samples = []
    test_pos_conf = []
    test_med_conf = []
    test_neg_conf = []

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
        'med_z_samples len': len(med_z_samples),
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
            pos_z_samples + med_z_samples + neg_z_samples,
            train_pos_conf + train_med_conf + train_neg_conf
        )

    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_med_z_samples len': len(test_med_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'test_pos_conf len': len(test_pos_conf),
        'test_neg_conf len': len(test_neg_conf),

    })
    test_idxs = [{'z': z, 'conf': conf} for z, conf
                 in zip(
            test_pos_z_samples + test_med_z_samples + test_neg_z_samples,
            test_pos_conf + test_med_conf + test_neg_conf
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

        res[train_test]['metadata_table_med_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'Medium']

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

    # zarr_path = output_dir / 'event_data_with_mtzs_3.zarr'
    zarr_path = output_dir / 'event_data_3.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    # Get the HDF5 root group
    # root = fileh.root
    # all_train_pose_idxs = []
    # all_test_pose_idxs = []

    # table_type = 'pandda_2'

    # metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
    # table_annotation = root[table_type]['annotation']
    # annotation_df = pd.DataFrame(table_annotation[:])
    # train_annotations = annotation_df[annotation_df['partition'] == b'train']
    # test_annotations = annotation_df[annotation_df['partition'] == b'test']

    # ligand_idx_smiles_df = pd.DataFrame(
    #     root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))

    # fragment_df = pd.DataFrame(
    #     root[table_type]['ligand_fragments'].get_basic_selection(slice(None), fields=['idx', 'fragment_canonical_smiles']))

    # negative_train_samples = metadata_table[
    #     (metadata_table['pose_data_idx'] == -1) & (annotation_df['partition'] == b'train')]
    # print(f'Got {len(negative_train_samples)} negative train samples')
    # negative_test_samples = metadata_table[
    # (metadata_table['pose_data_idx'] == -1) & (annotation_df['partition'] == b'test')]
    # print(f'Got {len(negative_test_samples)} negative test samples')

    # unique_smiles = ligand_idx_smiles_df['canonical_smiles'].unique()
    # print(f'Number of unique smiles: {len(unique_smiles)}')

    # unique_fragments = fragment_df['fragment_canonical_smiles'].unique()

    # pos_train_pose_samples = []

    # for smiles in unique_smiles:
    #     print(f'{smiles}')
    #     ligand_idx_df = ligand_idx_smiles_df[ligand_idx_smiles_df['canonical_smiles'] == smiles]
    #     corresponding_samples = metadata_table[metadata_table['ligand_data_idx'].isin(ligand_idx_df['idx'])]
    #
    #     # Get the train events for this ligand
    #     corresponding_train_event_annotations = train_annotations[
    #         train_annotations['idx'].isin(corresponding_samples['idx'])]
    #     pos_train_annotations = corresponding_train_event_annotations[
    #         corresponding_train_event_annotations['annotation'] == True]
    #     print(
    #         f'Got {len(corresponding_train_event_annotations)} train annotations, of which {len(pos_train_annotations)} positive')
    #
    #     # Get the test events for this ligand
    #     corresponding_test_event_annotations = test_annotations[
    #         test_annotations['idx'].isin(corresponding_samples['idx'])]
    #     pos_test_annotations = corresponding_test_event_annotations[
    #         corresponding_test_event_annotations['annotation'] == True]
    #     print(
    #         f'Got {len(corresponding_test_event_annotations)} test annotations, of which {len(pos_test_annotations)} positive')
    #
    #     # Get the pos and neg train samples
    #     if len(pos_train_annotations) > 0:
    #         pos_train_samples = pos_train_annotations.sample(50, replace=True)
    #         neg_train_samples = negative_train_samples.sample(50)
    #         all_train_pose_idxs += [(table_type, x) for x in pos_train_samples['idx']]
    #         all_train_pose_idxs += [(table_type, x) for x in neg_train_samples['idx']]
    #         pos_train_pose_samples += [x for x in corresponding_samples[corresponding_samples['idx'].isin(pos_train_annotations['idx'])].sample(50, replace=True)['pose_data_idx']]
    #
    #     # Get the pos and neg test samples
    #     if len(pos_test_annotations) > 0:
    #         pos_test_samples = pos_test_annotations
    #         # neg_test_samples = negative_test_samples
    #         all_test_pose_idxs += [(table_type, x) for x in pos_test_samples['idx']]
    #
    # all_test_pose_idxs += [(table_type, x) for x in negative_test_samples['idx']]

    # for fragment_smiles in unique_fragments:
    #     print(f'{fragment_smiles}')
    #     fragment_idx_df = fragment_df[fragment_df['fragment_canonical_smiles'] == fragment_smiles]
    #     corresponding_ligands = ligand_idx_smiles_df[ligand_idx_smiles_df['canonical_smiles'].isin(fragment_idx_df['ligand_canonical_smiles'])]
    #     corresponding_samples = metadata_table[metadata_table['ligand_data_idx'].isin(corresponding_ligands['idx'])]
    #
    #     # Get the train events for this ligand
    #     corresponding_train_event_annotations = train_annotations[
    #         train_annotations['idx'].isin(corresponding_samples['idx'])]
    #     pos_train_annotations = corresponding_train_event_annotations[
    #         corresponding_train_event_annotations['annotation'] == True]
    #     print(
    #         f'Got {len(corresponding_train_event_annotations)} train annotations, of which {len(pos_train_annotations)} positive')
    #
    #     # Get the test events for this ligand
    #     corresponding_test_event_annotations = test_annotations[
    #         test_annotations['idx'].isin(corresponding_samples['idx'])]
    #     pos_test_annotations = corresponding_test_event_annotations[
    #         corresponding_test_event_annotations['annotation'] == True]
    #     print(
    #         f'Got {len(corresponding_test_event_annotations)} test annotations, of which {len(pos_test_annotations)} positive')
    #
    #     # Get the pos and neg train samples
    #     if len(pos_train_annotations) > 0:
    #         pos_train_samples = pos_train_annotations.sample(50, replace=True)
    #         neg_train_samples = negative_train_samples.sample(50)
    #         all_train_pose_idxs += [(table_type, x) for x in pos_train_samples['idx']]
    #         all_train_pose_idxs += [(table_type, x) for x in neg_train_samples['idx']]
    #         pos_train_pose_samples += [x for x in corresponding_samples[corresponding_samples['idx'].isin(pos_train_annotations['idx'])].sample(50, replace=True)['pose_data_idx']]
    #
    #     # Get the pos and neg test samples
    #     if len(pos_test_annotations) > 0:
    #         pos_test_samples = pos_test_annotations
    #         # neg_test_samples = negative_test_samples
    #         all_test_pose_idxs += [(table_type, x) for x in pos_test_samples['idx']]
    #
    # all_test_pose_idxs += [(table_type, x) for x in negative_test_samples['idx']]

    # for table_type in ['normal', 'pandda_2']:
    # # for table_type in ['pandda_2']:
    #
    #     train_pose_idxs = []
    #     test_pose_idxs = []
    #     if table_type == 'normal':
    #         table_annotation = root['annotation']
    #         table_poses = root['known_hit_pose']
    #         table_z_map_sample_metadata = root['z_map_sample_metadata']
    #
    #
    #         # table_poses = root.pandda_2_known_hit_pose
    #
    #         # Get train and test event idxs
    #         rprint(f'Getting idxs of valid train event maps...')
    #         # train_event_table_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(
    #         #         (table_annotation['partition'] == b'train') & (table_annotation['annotation']))
    #         # ])
    #         # train_database_event_idxs = set(
    #         #     [
    #         #         table_z_map_sample_metadata[_x]['event_idx']
    #         #         for _x
    #         #         in train_event_table_idxs
    #         #     ]
    #         # )
    #         # rprint(f'Getting idxs of valid test event maps...')
    #         # test_event_table_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(
    #         #         (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
    #         # ])
    #         # test_database_event_idxs = set(
    #         #     [
    #         #         table_z_map_sample_metadata[_x]['event_idx']
    #         #         for _x
    #         #         in test_event_table_idxs
    #         #     ]
    #         # )
    #         # exclude_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(~table_annotation['annotation'])
    #         # ])
    #         train_database_event_idxs = set([
    #             _x['event_idx']
    #             for _x
    #             in table_annotation.get_mask_selection(
    #                 (table_annotation['partition'] == b'train') & (table_annotation['annotation']))
    #         ])
    #         rprint(f'Getting idxs of valid test event maps...')
    #
    #         test_database_event_idxs = set([
    #             _x['event_idx']
    #             for _x
    #             in table_annotation.get_mask_selection(
    #                 (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
    #         ])
    #         #
    #
    #         rprint(f"Filtering poses to those matching valid event maps...")
    #         for row in table_z_map_sample_metadata[:]:
    #             # z_map_sample_metadata_idx = row['event_map_sample_idx']
    #             database_event_idx = row['event_idx']
    #             if database_event_idx in train_database_event_idxs:
    #                 train_pose_idxs.append((table_type, row['idx']))
    #             elif database_event_idx in test_database_event_idxs:
    #                 test_pose_idxs.append((table_type, row['idx']))
    #     else:
    #         try:
    #             table_annotation = root['pandda_2']['annotation']
    #             table_z_map_sample_metadata = root['pandda_2']['z_map_sample_metadata']
    #             for row in table_z_map_sample_metadata[:]:
    #                 annotation = table_annotation[row['idx']]
    #
    #                 if annotation['partition'] == b"train":
    #                     train_pose_idxs.append((table_type, row['idx']))
    #                 elif annotation['partition'] == b"test":
    #                     test_pose_idxs.append((table_type, row['idx']))
    #         except:
    #             continue
    #
    #     rprint(f"\tGot {len(train_pose_idxs)} train samples")
    #     rprint(f"\tGot {len(test_pose_idxs)} test samples")
    #
    #     positive_train_pose_idxs = [_x for _x in train_pose_idxs if
    #                                 table_z_map_sample_metadata[_x[1]]['pose_data_idx'] != -1]
    #     negative_train_pose_idxs = [_x for _x in train_pose_idxs if
    #                                 table_z_map_sample_metadata[_x[1]]['pose_data_idx'] == -1]
    #
    #     all_train_pose_idxs += negative_train_pose_idxs + (
    #                 positive_train_pose_idxs * (int(len(negative_train_pose_idxs) / len(positive_train_pose_idxs))))
    #     all_test_pose_idxs += test_pose_idxs
    #     rprint(f"Got {len(positive_train_pose_idxs)} postivie train samples")
    #     rprint(f"Got {len(negative_train_pose_idxs)} negative test samples")

    rprint(f'Getting train/test data...')
    # all_train_pose_idxs, all_test_pose_idxs = _get_train_test_idxs(root)
    train_config, test_config = _get_train_test_idxs_full_conf(root)

    # rprint(f"Got {len(all_train_pose_idxs)} train samples")
    # rprint(f"Got {len(all_test_pose_idxs)} test samples")

    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    rprint(f'Constructing train and test dataloaders...')

    # Get the model
    rprint('Constructing model...')
    study_name = 'event_scoring_prod_51'
    output = output_dir / study_name
    if not output.exists():
        os.mkdir(output)

    # Train
    rprint('Constructing trainer...')

    # trainer = lt.Trainer(accelerator='gpu', logger=logger,
    #                      callbacks=[
    #                          checkpoint_callback,
    #                          checkpoint_callback_best_10,
    #                          checkpoint_callback_best_99,
    #                          checkpoint_callback_best_95,
    #                          # StochasticWeightAveraging(swa_lrs=1e-3,
    #                          #                           # swa_epoch_start=0.75,
    #                          #                           swa_epoch_start=0.5,
    #                          #                           )
    #                      ],
    #                      enable_progress_bar=False,
    #                      gradient_clip_val=1.5,
    #                      max_epochs=50
    #                      )
    rprint(f'Training...')

    # trainer.fit(model, dataset_train, dataset_test)

    # def train_func(_config):
    #     print(f'Compiling!')
    #     model = LitEventScoring(output, _config)
    #     # model = torch.compile(LitEventScoring(output, _config))
    #     print('Compiled!')
    #
    #     trainer = lt.Trainer(
    #         # devices="auto",
    #         accelerator="gpu",
    #         gradient_clip_val=1.5,
    #         strategy=RayDDPStrategy(),
    #         logger=logger,
    #         callbacks=[
    #             checkpoint_callback,
    #             checkpoint_callback_best_10,
    #             checkpoint_callback_best_99,
    #             checkpoint_callback_best_95,
    #             RayTrainReportCallback(),
    #         ],
    #         plugins=[RayLightningEnvironment()],
    #         enable_progress_bar=False,
    #     )
    #     trainer = prepare_trainer(trainer)
    #
    #     _train_config = {
    #         'zarr_path': zarr_path,
    #     }
    #     _train_config.update(train_config)
    #     _train_config.update(_config)
    #     dataset_train = DataLoader(
    #         EventScoringDataset(
    #             _train_config
    #         ),
    #         batch_size=128,  # batch_size,
    #         shuffle=True,
    #         num_workers=19,
    #         drop_last=True
    #     )
    #     rprint(f"Got {len(dataset_train)} training samples")
    #
    #     _test_config = {
    #         'zarr_path': zarr_path,
    #     }
    #     _test_config.update(test_config)
    #     _test_config.update(_config)
    #     dataset_test = DataLoader(
    #         EventScoringDataset(
    #             _test_config
    #         ),
    #         batch_size=batch_size,
    #         num_workers=19,
    #         drop_last=True
    #     )
    #     rprint(f"Got {len(dataset_test)} test samples")
    #
    #     trainer.fit(model, dataset_train, dataset_test)
    #
    # search_space = {
    #     "lr": tune.loguniform(1e-4, 1e0),
    #     "wd": tune.loguniform(1e-4, 1e0),
    #     'fraction_background_replace': tune.loguniform(1e-2, 5e-1),
    #     'xmap_radius': tune.uniform(3.0, 7.0),
    #     'max_x_blur': tune.uniform(0.0, 3.0),
    #     'max_z_blur': tune.uniform(0.0, 3.0),
    #     'drop_rate': tune.uniform(0.0, 1.0)
    #     # "batch_size": tune.choice([32, 64]),
    # }
    #
    # scaling_config = ScalingConfig(
    #     num_workers=1, use_gpu=True, resources_per_worker={"CPU": 19, "GPU": 1}
    # )
    #
    # run_config = RunConfig(
    #     storage_path=output_dir,
    #     checkpoint_config=CheckpointConfig(
    #         num_to_keep=2,
    #         checkpoint_score_attribute="fpr99",
    #         checkpoint_score_order="min",
    #     ),
    # )
    #
    # # Define a TorchTrainer without hyper-parameters for Tuner
    # ray_trainer = TorchTrainer(
    #     train_func,
    #     scaling_config=scaling_config,
    #     run_config=run_config,
    # )
    #
    # num_samples = 200
    # scheduler = ASHAScheduler(max_t=20, grace_period=2, reduction_factor=2)
    # # algo = BayesOptSearch(metric="fpr99", mode="min")
    # # algo = TuneBOHB(metric="fpr99", mode="min")
    # # algo =  AxSearch()
    # # algo = HyperOptSearch(
    # #
    # #     metric="fpr99", mode="min",
    # #     points_to_evaluate=[
    # #         # {
    # #         #     'lr': 0.03503427000766074,
    # #         #     'wd': 0.0033389364254906707,
    # #         #     'fraction_background_replace': 0.4240318020166584,
    # #         #     'xmap_radius': 6.187276156207498,
    # #         #     'max_x_blur': 0.3479295147607111,
    # #         #     'max_z_blur': 0.3479295147607111,
    # #         #     'drop_rate': 0.5
    # #         # },
    # #         {'lr': 0.0035822737616734305,
    # #          'wd': 0.0002263704977559898,
    # #          'fraction_background_replace': 0.15345573507886795,
    # #          'xmap_radius': 5.396850198944849,
    # #          'max_x_blur': 1.1240403061803592,
    # #          'max_z_blur': 0.15653895006777918,
    # #          'drop_rate': 0.21255856328991862,
    # #          }
    # #     ],
    # #     n_initial_points=5
    # # )
    # # algo=BasicVariantGenerator(
    # #             points_to_evaluate=[
    # #                     {
    # #                         'lr': 0.03503427000766074,
    # #                         'wd': 0.0033389364254906707,
    # #                         'fraction_background_replace': 0.4240318020166584,
    # #                         'xmap_radius': 6.187276156207498,
    # #                         'max_x_blur': 0.3479295147607111,
    # #                         'max_z_blur': 0.3479295147607111,
    # #                         'drop_rate': 0.5
    # #                     },
    # #                     {'lr': 0.0035822737616734305,
    # #                      'wd': 0.0002263704977559898,
    # #                      'fraction_background_replace': 0.15345573507886795,
    # #                      'xmap_radius': 5.396850198944849,
    # #                      'max_x_blur': 1.1240403061803592,
    # #                      'max_z_blur': 0.15653895006777918,
    # #                      'drop_rate': 0.21255856328991862,
    # #                      }
    # #                 ],
    # #         )
    # ray.init()
    #
    # tuner = tune.Tuner(
    #     ray_trainer,
    #     param_space={'train_loop_config': search_space},  # Unpacked as arguments to TorchTrainer - Goes to train_func as config dict
    #     tune_config=tune.TuneConfig(
    #         # search_alg=algo,
    #         metric="fpr99",
    #         mode="min",
    #         num_samples=num_samples,
    #         scheduler=scheduler,
    #     ), )
    #
    # print(tuner.fit())

    def objective(trial):
        # Suggest hyperparameters
        _config = {
            "lr": trial.suggest_loguniform('lr', 1e-4, 1e0),
            "wd": trial.suggest_loguniform('wd', 1e-4, 1e0),
            'fraction_background_replace': trial.suggest_uniform('fraction_background_replace', 0.0, 1e0),
            'xmap_radius': trial.suggest_uniform('xmap_radius', 3.0, 7.0),
            'max_x_blur': trial.suggest_uniform('max_x_blur', 0.0, 3.0),
            'max_z_blur': trial.suggest_uniform('max_z_blur', 0.0, 3.0),
            'drop_rate': trial.suggest_uniform('drop_rate', 0.0, 1.0),
            'planes_1': trial.suggest_categorical('planes_1', [2, 4, 8, 16, ]),
            'drop_1': trial.suggest_uniform('drop_1', 0.0, 1.0),
            'planes_2': trial.suggest_categorical('planes_2', [4, 8, 16, 32, ]),
            'drop_2': trial.suggest_uniform('drop_2', 0.0, 1.0),
            'planes_3': trial.suggest_categorical('planes_3', [8, 16, 32, 64, ]),
            'drop_3': trial.suggest_uniform('drop_3', 0.0, 1.0),
            'planes_4': trial.suggest_categorical('planes_4', [16, 32, 64, 128, ]),
            'drop_4': trial.suggest_uniform('drop_4', 0.0, 1.0),
            'planes_5': trial.suggest_categorical('planes_5', [32, 64, 128, 256, ]),
            'drop_5': trial.suggest_uniform('drop_5', 0.0, 1.0),
            'drop_atom_rate': trial.suggest_uniform('drop_atom_rate', 0.0, 1.0),
            'max_pos_atom_mask_radius': trial.suggest_uniform('max_pos_atom_mask_radius', 1.01, 4.0),
            'max_translate': trial.suggest_uniform('max_translate', 0.0, 5.0),
            'max_x_noise': trial.suggest_uniform('max_x_noise', 0.0, 2.0),
            'max_z_noise': trial.suggest_uniform('max_z_noise', 0.0, 2.0),
            'pos_resample_rate': trial.suggest_int('pos_resample_rate', 0, 10),
            'p_flip': trial.suggest_uniform('p_flip', 0.0, 1.0),
            'z_mask_radius': trial.suggest_uniform('z_mask_radius', 1.0, 3.5),
            'z_cutoff': trial.suggest_uniform('z_cutoff', 1.5, 3.0),
            'combo_layer': trial.suggest_categorical('combo_layer', [8, 16, 32, 64, ]),
            'blocks_1': trial.suggest_categorical('blocks_1', [1, 2, ]),
            'blocks_2': trial.suggest_categorical('blocks_2', [1, 2, ]),
            'blocks_3': trial.suggest_categorical('blocks_3', [1, 2, ]),
            'blocks_4': trial.suggest_categorical('blocks_4', [1, 2, ]),
            'grad_clip': trial.suggest_loguniform('grad_clip', 1e-4, 1e1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, ]),
            # "batch_size": tune.choice([32, 64]),
        }
        print(f'Running trial with config:')
        rprint(_config)

        trial_output_dir = output / f'{trial.number}'
        if not trial_output_dir.exists():
            os.mkdir(trial_output_dir)

        checkpoint_callback = ModelCheckpoint(dirpath=str(trial_output_dir))
        checkpoint_callback_best_95 = ModelCheckpoint(
            monitor='fpr95',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr95:.2f}'
        )
        checkpoint_callback_best_99 = ModelCheckpoint(
            monitor='fpr99',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr99:.2f}'
        )
        checkpoint_callback_best_10 = ModelCheckpoint(
            monitor='fpr10',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr10:.2f}'
        )
        checkpoint_callback_best_median99 = ModelCheckpoint(
            monitor='medianfpr99',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{medianfpr99:.2f}'
        )
        checkpoint_callback_best_best_scorer_hit = ModelCheckpoint(
            monitor='best_scorer_hit',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{best_scorer_hit:.2f}'
        )

        logger = CSVLogger(str(trial_output_dir / 'logs'))

        print(f'Compiling!')
        model = LitEventScoring(trial_output_dir, _config)
        # model = torch.compile(LitEventScoring(output, _config))
        print('Compiled!')

        trainer = lt.Trainer(
            # devices="auto",
            accelerator="gpu",
            gradient_clip_val=_config['grad_clip'],
            logger=logger,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_best_10,
                checkpoint_callback_best_99,
                checkpoint_callback_best_95,
                checkpoint_callback_best_median99,
                checkpoint_callback_best_best_scorer_hit,
                PyTorchLightningPruningCallback(trial, monitor='best_scorer_hit'),
                EarlyStopping('best_scorer_hit', patience=5)
            ],
            enable_progress_bar=False,
            max_epochs=60
        )

        _train_config = {
            'zarr_path': zarr_path,
        }
        _train_config.update(train_config)
        _train_config.update(_config)
        _train_config.update({'test_train': 'train'})
        dataset_train = DataLoader(
            EventScoringDataset(
                _train_config
            ),
            batch_size=_config['batch_size'],  # batch_size,
            shuffle=True,
            num_workers=19,
            drop_last=True
        )
        rprint(f"Got {len(dataset_train)} training samples")

        _test_config = {
            'zarr_path': zarr_path,
        }
        _test_config.update(test_config)
        _test_config.update(_config)
        _test_config.update({'test_train': 'test'})
        dataset_test = DataLoader(
            EventScoringDataset(
                _test_config
            ),
            batch_size=batch_size,
            num_workers=19,
            drop_last=True
        )
        rprint(f"Got {len(dataset_test)} test samples")

        trainer.fit(model, dataset_train, dataset_test, )
        return trainer.callback_metrics['best_scorer_hit'].item()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # Unique identifier of the study.
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=60,
    )
    if output_dir.exists():
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            # direction='minimize',
            direction='maximize',
            load_if_exists=True,
            pruner=pruner,
            sampler=TPESampler(constant_liar=True)
        )
        study.enqueue_trial(
            {
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

            },
            skip_if_exists=True
        )
    else:
        print(f'Loading study!')
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name,
            sampler=TPESampler(constant_liar=True),
            pruner=pruner,
        )
    study.optimize(objective, n_trials=300)


if __name__ == "__main__":
    fire.Fire(main)
