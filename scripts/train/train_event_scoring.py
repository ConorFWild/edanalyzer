from pathlib import Path

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

from edanalyzer.datasets.event_scoring import EventScoringDataset
from edanalyzer.models.event_scoring import LitEventScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging


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
    valid_smiles_mask = valid_smiles_df.iloc[ligand_idx_smiles_df.iloc[metadata_table['ligand_data_idx']]['idx']]['valid']
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

    # Loop over the z samples adding positive samples for each
    print(f'Getting positive train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'High'].iterrows():# .sample(len(neg_z_samples),
                                                                         #     replace=True).iterrows():
        pos_z_samples += [z['idx'],]

    print(f'Got {len(pos_z_samples)} pos samples!')

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
        if z['Confidence'] == 'High':
            test_pos_z_samples.append(z['idx'])

        elif z['Confidence'] == 'Low':
            test_neg_z_samples.append(z['idx'])

    rprint({
        'pos_z_samples len': len(pos_z_samples),
        'neg_z_samples len': len(neg_z_samples),
        'pos_conf_samples len': len(pos_conf_samples),
        'neg_conf_samples len': len(neg_conf_samples),
        'train_pos_conf len': len(train_pos_conf),
        'train_neg_conf len': len(train_neg_conf),

    })
    train_idxs = [
        {'z': z, }
        for z
        in pos_z_samples + neg_z_samples

    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'train_pos_conf len': len(test_pos_conf),
        'train_neg_conf len': len(test_neg_conf),
    })
    test_idxs = [{'z': z, } for z
                 in  test_pos_z_samples + test_neg_z_samples
                 ]
    return train_idxs, test_idxs


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
    zarr_path = output_dir / 'event_data_2.zarr'

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
    all_train_pose_idxs, all_test_pose_idxs = _get_train_test_idxs_full_conf(root)

    rprint(f"Got {len(all_train_pose_idxs)} train samples")
    rprint(f"Got {len(all_test_pose_idxs)} test samples")

    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    rprint(f'Constructing train and test dataloaders...')
    dataset_train = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_train_pose_idxs,
            # pos_train_pose_samples
            None
        ),
        batch_size=128,#batch_size,
        shuffle=True,
        num_workers=19,
        drop_last=True
    )
    rprint(f"Got {len(dataset_train)} training samples")
    dataset_test = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_test_pose_idxs,
            # pos_train_pose_samples
            None
        ),
        batch_size=batch_size,
        num_workers=19,
        drop_last=True
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Get the model
    rprint('Constructing model...')
    output = output_dir / 'event_scoring_prod_quantized'
    model = LitEventScoring(output)



    # Train
    rprint('Constructing trainer...')
    checkpoint_callback = ModelCheckpoint(dirpath=str(output ))
    checkpoint_callback_best_95 = ModelCheckpoint(
        monitor='fpr95',
        dirpath=str(output),
        filename='sample-mnist-{epoch:02d}-{fpr95:.2f}'
    )
    checkpoint_callback_best_99 = ModelCheckpoint(
        monitor='fpr99',
        dirpath=str(output),
        filename='sample-mnist-{epoch:02d}-{fpr99:.2f}'
    )
    checkpoint_callback_best_10 = ModelCheckpoint(
        monitor='fpr10',
        dirpath=str(output),
        filename='sample-mnist-{epoch:02d}-{fpr10:.2f}'
    )
    logger = CSVLogger(str( output / 'logs'))
    trainer = lt.Trainer(accelerator='gpu', logger=logger,
                         callbacks=[
                             checkpoint_callback,
                             checkpoint_callback_best_10,
                             checkpoint_callback_best_99,
                             checkpoint_callback_best_95,
                             # StochasticWeightAveraging(swa_lrs=1e-3,
                             #                           # swa_epoch_start=0.75,
                             #                           swa_epoch_start=0.5,
                             #                           )
                         ],
                         enable_progress_bar=False,
                         gradient_clip_val=1.5,
                         max_epochs=50
                         )
    rprint(f'Training...')
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
