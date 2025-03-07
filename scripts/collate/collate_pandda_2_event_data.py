import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import joblib
import pickle
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import tables
import zarr
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem import AllChem

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample
from edanalyzer.data.event_data import (
    comment_dtype,
    _make_z_map_sample_metadata_table,
    _make_z_map_sample_table,
    _make_ligand_data_table,
    _make_known_hit_pose_table,
    _make_annotation_table,
    _make_comment_table,
    _get_closest_res_from_dataset_dir,
    _get_z_map_sample_from_dataset_dir,
    _get_pose_sample_from_dataset_dir,
    _get_ligand_data_sample_from_dataset_dir,
    _get_annotation_sample_from_dataset_dir,
    _get_z_map_metadata_sample_from_dataset_dir,
    _make_xmap_sample_table,
    _get_xmap_sample_from_dataset_dir
)


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    #
    pandda_key = config['panddas']['pandda_key'],
    test_systems = config['test']['test_systems']

    #
    # Open a file in "w"rite mode
    # zarr_path = 'output/event_data_with_mtzs_2.zarr'
    zarr_path = '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/event_data_4.zarr'

    root = zarr.open(zarr_path, mode='a')

    #### COMMENTS BEGIN HERE



    try:
        del root['pandda_2']
    except:
        rprint(f'No PanDDA 2 group!')
    pandda_2_group = root.create_group('pandda_2')

    # Create 2 new tables in group1
    z_map_sample_metadata_table = _make_z_map_sample_metadata_table(pandda_2_group)
    z_map_sample_table = _make_z_map_sample_table(pandda_2_group)
    xmap_sample_table = _make_xmap_sample_table(pandda_2_group)
    ligand_data_table = _make_ligand_data_table(pandda_2_group)
    known_hit_pose_table = _make_known_hit_pose_table(pandda_2_group)
    annotation_table = _make_annotation_table(pandda_2_group)
    comment_table = _make_comment_table(pandda_2_group)

    # PanDDA 2 events

    #
    rprint(f"Querying events...")
    with pony.orm.db_session:

        # Loop over PanDDA directories
        z_map_sample_metadata_idx = 0
        idx_pose = 0
        idx_ligand_data = 0
        annotation_idx = 0
        for pandda_dir in Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score/').glob('*'):
            if pandda_dir.name == 'TcCS':
                continue

            # if pandda_dir.name != 'PTP1B':
            #     continue

            # Skip if not a directory
            if not pandda_dir.is_dir():
                continue
            rprint(f'PanDDA dir is: {pandda_dir}')

            # Skip if no inspect table
            pandda_inspect_table = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
            if not pandda_inspect_table.exists():
                rprint(f'\tNO INSPECT TABLE! SKIPPING!')

                continue

            # Get the inspect table
            inspect_table = pd.read_csv(pandda_inspect_table)

            # if not inspect_table[constants.PANDDA_INSPECT_VIEWED].all():
            #     rprint(f'\tNOT FINISHED INSPECTING TABLE! SKIPPING!')
            #
            #     continue

            # Iterate the inspect table
            for _idx, _row in inspect_table.iterrows():
                # Unpack the row information
                dtag, event_idx, bdc, conf, viewed, size = _row['dtag'], _row['event_idx'], _row['1-BDC'], _row[
                    constants.PANDDA_INSPECT_HIT_CONDFIDENCE], _row[constants.PANDDA_INSPECT_VIEWED], _row[constants.PANDDA_INSPECT_CLUSTER_SIZE]

                system = _get_system_from_dtag(dtag)

                rprint(f'\tProcessing event: {dtag} {event_idx} {conf}')

                if not viewed:
                    rprint('\t\tNot Viewed! Skipping!')
                    continue

                # if conf == 'Medium':
                #     rprint(f'\t\tAmbiguous event! Skipping!')
                #     continue

                initial_x, initial_y, initial_z = _row['x'], _row['y'], _row['z']

                dataset_dir = pandda_dir / 'processed_datasets' / dtag

                processed_dataset_yaml = dataset_dir / 'processed_dataset.yaml'
                with open(processed_dataset_yaml, 'r') as f:
                    processed_dataset = yaml.safe_load(f)

                selected_model = processed_dataset['Summary']['Selected Model']

                event_distances = {}
                event_centroids = {}
                for event_num, event in processed_dataset['Models'][selected_model]['Events'].items():
                    event_centroid = event['Centroid']
                    distance = np.linalg.norm(np.array(event_centroid) - np.array([initial_x, initial_y, initial_z]))
                    event_distances[event_num] = distance
                    event_centroids[event_num] = event_centroid

                if len(event_distances) > 0:
                    closest_event_id = min(event_centroids, key=lambda _event_num: event_distances[_event_num])
                    x, y, z = event_centroids[closest_event_id]
                    rprint(f'Closest event distance is {event_distances[closest_event_id]}')

                else:
                    rprint(
                        f'Could not match high confidence ligand {dtag} {event_idx} to an initial event!\n'
                        f'Check model in {dataset_dir} is appropriate!\n'
                        'SKIPPING!'
                    )
                    continue

                model_dir = dataset_dir / 'modelled_structures'

                # Get the corresponding residue
                try:
                    resid, res, dist = _get_closest_res_from_dataset_dir(
                        dataset_dir,
                        x, y, z
                    )
                except Exception as e:
                    print(e)
                    print('Couldn\'t match res! Skipping!')
                    continue
                if (conf == 'High') & (dist > 6.0):
                    rprint(
                        f'Could not match high confidence ligand {dtag} {event_idx} to a build!\n'
                        f'Check model in {dataset_dir} is appropriate!\n'
                        'SKIPPING!'
                    )
                    # raise Exception
                    continue

                # Get the z map sample
                z_map_sample = _get_z_map_sample_from_dataset_dir(
                    dataset_dir,
                    x, y, z,
                    z_map_sample_metadata_idx,
                )
                xmap_sample = _get_xmap_sample_from_dataset_dir(
                    dataset_dir,
                    x, y, z,
                    z_map_sample_metadata_idx,
                )
                if conf == 'High':
                    # Get the pose sample
                    pose_sample = _get_pose_sample_from_dataset_dir(
                        model_dir,
                        res,
                        x, y, z,
                        idx_pose
                    )

                # Get the ligand data
                ligand_data_sample = _get_ligand_data_sample_from_dataset_dir(
                    dataset_dir,
                    res,
                    idx_ligand_data,
                )
                if not ligand_data_sample:
                    rprint(f'\t\tNO LIGAND DATA! SKIPPING!')
                    continue

                # Get the annotation data
                annotation_sample = _get_annotation_sample_from_dataset_dir(
                    dataset_dir,
                    conf,
                    test_systems,
                    annotation_idx,
                    z_map_sample_metadata_idx
                )

                # Get the z map metadata sample
                if conf == 'High':
                    tmp_idx_pose = idx_pose
                else:
                    # tmp_idx_ligand_data = -1
                    tmp_idx_pose = -1
                tmp_idx_ligand_data = idx_ligand_data

                z_map_metadata_sample = _get_z_map_metadata_sample_from_dataset_dir(
                    z_map_sample_metadata_idx,
                    event_idx,
                    resid,
                    tmp_idx_ligand_data,
                    tmp_idx_pose,
                    system,
                    dtag,
                    event_idx,
                    conf,
                    size,
                    x,
                    y,
                    z,
                    _row['high_resolution']
                )

                z_map_sample_metadata_table.append(z_map_metadata_sample)
                z_map_sample_table.append(z_map_sample)
                xmap_sample_table.append(xmap_sample)
                ligand_data_table.append(ligand_data_sample)

                comment_sample = np.array(
                [
                    (
                        z_map_sample_metadata_idx,
                        z_map_sample_metadata_idx,
                        _row['Comment']
                    )
                ],
                dtype=comment_dtype
            )
                comment_table.append(comment_sample)

                if conf == 'High':
                    known_hit_pose_table.append(pose_sample)
                    idx_pose += 1
                idx_ligand_data += 1
                annotation_table.append(annotation_sample)
                z_map_sample_metadata_idx += 1
                annotation_idx += 1


                ...
            ...

    #### COMMENTS END HERE

    # # Generate fragment embeddings
    # # 1. Create a zarr group for fragment embeddings
    # # 1. Iterate over collected canonical smiles,
    # # 2. a. creating the assoicated molecular graph, and
    # # 2. b. Then collect the rotatable bonds and
    # # 2. c. producing n embeddings of it.
    # # 2. d. fragment the molecular graph on them.
    # # 2. e. for each fragment
    # # 2. e. 1. collect the associated embeddings of each fragment > size 3
    # # 2. e. 2. for each embedding
    # # 2. e. 2. a. create a record for the embedding
    # # 2. e. 2. b. save a record in the zarr group for them
    #
    # # 1.
    # # del root['pandda_2']['ligand_fragments']
    # ligand_fragment_dtype = [
    #     ('idx', 'i8'),
    #     ('ligand_data_idx', 'i8'),
    #     ('num_heavy_atoms', 'i8'),
    #     ('fragment_canonical_smiles', '<U300'),
    #     ('ligand_canonical_smiles', '<U300'),
    #     ('positions', '<f4', (30, 3)),
    #     ('elements', '<i4', (30,))]
    # mol_frag_group = root['pandda_2'].create_dataset(
    #     'ligand_fragments',
    #     shape=(0,),
    #     chunks=(1,),
    #     dtype=ligand_fragment_dtype
    # )
    #
    # # 2.
    # mol_frag_idx = 0
    #
    # for _ligand_data in ligand_data_table:
    #
    #     m = Chem.MolFromSmiles(_ligand_data['canonical_smiles'])
    #
    #     # 2.b.
    #     rot_bonds = rot_atom_pairs = m.GetSubstructMatches(RotatableBondSmarts)
    #     # print(rot_bonds)
    #     rot_bond_set = set([m.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    #     # print(rot_bond_set)
    #
    #     # 2.c.
    #     m2 = Chem.AddHs(m)
    #     cids = AllChem.EmbedMultipleConfs(m2, numConfs=50)
    #     # print(f'Got {len(cids)} embeddings')
    #
    #     if len(rot_bond_set) == 0:
    #         fragment_atom_idx_sets = [[_atom.GetIdx() for _atom in m.GetAtoms()], ]
    #
    #         frags = [m, ]
    #
    #     else:
    #
    #         # 2.d.
    #         fragment_atom_idx_sets = []
    #         frags = Chem.GetMolFrags(
    #             Chem.FragmentOnBonds(m, rot_bond_set),
    #             asMols=True,
    #             fragsMolAtomMapping=fragment_atom_idx_sets,
    #         )
    #         # print(f'Got {len(frags)} fragments!')
    #         # print(fragment_atom_idx_sets)
    #
    #         # 2.e.
    #         for _frag, _fragment_atom_idx_set in zip(frags, fragment_atom_idx_sets):
    #             # print(f'Fragment atom idxs: {_fragment_atom_idx_set}')
    #
    #             # 2.e.1.
    #             real_atoms_idx_set = tuple(_y for _y in _fragment_atom_idx_set if _y < m.GetNumAtoms())
    #             heavy_atoms = [_x for _x in real_atoms_idx_set if m.GetAtomWithIdx(_x).GetAtomicNum() != 1]
    #             # print(f'Fragment Num Heavy Atoms: {len(heavy_atoms)}')
    #             if len(heavy_atoms) <= 3:
    #                 # print(f'Fragment too small at {len(heavy_atoms)}! Skipping!')
    #                 continue
    #
    #             frag_periphery = tuple(_atom.GetIsotope() for _atom in _frag.GetAtoms() if _atom.GetIsotope() != 0)
    #             # print(frag_periphery)
    #
    #             #
    #             for _atom in _frag.GetAtoms():
    #                 if _atom.GetIsotope() != 0:
    #                     _atom.SetIsotope(0)
    #
    #             # 2.e.2
    #             for embedding in [_conf.GetPositions()[real_atoms_idx_set + frag_periphery, :] for _conf in
    #                               m2.GetConformers()]:
    #                 # 2.e.2.a.
    #                 fragment_smiles = Chem.MolToSmiles(_frag)
    #                 # print(f'Fragment Smiles: {fragment_smiles}')
    #
    #                 #
    #                 # print(f'Embedding Size: {embedding.shape[0]}')
    #                 poss = np.zeros((30, 3))
    #                 poss[:embedding.shape[0], :] = embedding[:, :]
    #                 mol_els = np.array([m.GetAtomWithIdx(_atom_idx).GetAtomicNum() for _atom_idx in
    #                                     real_atoms_idx_set + frag_periphery])
    #                 els = np.zeros(30)
    #                 els[:len(mol_els)] = mol_els[:]
    #
    #                 # 2. e. 2. b.
    #                 record = np.array(
    #                     [
    #                         (
    #                     mol_frag_idx,
    #                     _ligand_data['idx'],
    #                     len(heavy_atoms),
    #                     fragment_smiles,
    #                     _ligand_data['canonical_smiles'],
    #                     poss,
    #                     els
    #                 )]
    #                     ,
    #                     dtype=ligand_fragment_dtype)
    #                 # print(record)
    #
    #                 #
    #                 mol_frag_group.append(record)
    #                 mol_frag_idx += 1

    valid_smiles_dtype = [
        ('idx', 'i8'),
        ('valid', '?'),
    ]
    try:
        del root['pandda_2']['ligand_confs']
    except Exception as e:
        print(e)
    valid_smiles_group = root['pandda_2'].create_dataset(
        'valid_smiles',
        shape=(0,),
        chunks=(1,),
        dtype=valid_smiles_dtype
    )

    df = pd.DataFrame(
        root['pandda_2']['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles', ]))

    unique_smiles_series = df['canonical_smiles'].unique()

    smiles_validity = {}
    for idx, smiles in enumerate(unique_smiles_series):
        # idx = _row['idx']
        # smiles = _row['canonical_smiles']
        print(f'{idx}/{len(unique_smiles_series)} : {smiles}')
        try:
            m = Chem.MolFromSmiles(smiles)
            m2 = Chem.AddHs(m)
            cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
            m3 = Chem.RemoveHs(m2)
            embedding = [_conf.GetPositions() for _conf in m3.GetConformers()][0]

            smiles_validity[smiles] = True
        except Exception as e:
            print(e)
            smiles_validity[smiles] = False

    for _idx, _row in df.iterrows():
        smiles = _row['canonical_smiles']
        if smiles_validity[smiles]:
            valid_smiles_group.append(
                np.array(
                    [(_idx, True)],
                    dtype=valid_smiles_dtype
                )
            )
        else:
            valid_smiles_group.append(
                np.array(
                    [(_idx, False)],
                    dtype=valid_smiles_dtype
                )
            )

    print(f'Generating ligand confs...')
    ligand_data_table = root['pandda_2']['ligand_data']
    try:
        del root['pandda_2']['ligand_confs']
    except Exception as e:
        print(e)
    ligand_conf_dtype = [
        ('idx', 'i8'),
        ('ligand_data_idx', 'i8'),
        ('num_heavy_atoms', 'i8'),
        ('fragment_canonical_smiles', '<U300'),
        ('ligand_canonical_smiles', '<U300'),
        ('positions', '<f4', (150, 3)),
        ('elements', '<i4', (150,))]
    mol_conf_group = root['pandda_2'].create_dataset(
            'ligand_confs',
            shape=(0,),
            chunks=(1,),
            dtype=ligand_conf_dtype
        )
    #
    # mol_conf_idx = 0
    #
    # # for _ligand_data in ligand_data_table:
    # z_map_sample_metadata_table = root['pandda_2']['z_map_sample_metadata']
    #
    # high_conf = z_map_sample_metadata_table.get_mask_selection(z_map_sample_metadata_table['Confidence'] == ['High'])
    # num = len(high_conf)
    # for j, _z_map_sample_metadata in enumerate(high_conf):
    #     print(f'{j} / {num}')
    #     if _z_map_sample_metadata['Confidence'] != 'High':
    #         continue
    #
    #     _ligand_data = ligand_data_table[_z_map_sample_metadata['ligand_data_idx']]
    #
    #     m = Chem.MolFromSmiles(_ligand_data['canonical_smiles'])
    #
    #     # 2.c.
    #     m2 = Chem.AddHs(m)
    #     cids = AllChem.EmbedMultipleConfs(m2, numConfs=50)
    #     # print(f'Got {len(cids)} embeddings')
    #
    #     # 2.e.2
    #     for embedding in [_conf.GetPositions() for _conf in m3.GetConformers()]:
    #         # 2.e.2.a.
    #         # fragment_smiles = Chem.MolToSmiles(_frag)
    #         # print(f'Fragment Smiles: {fragment_smiles}')
    #
    #         #
    #         # print(f'Embedding Size: {embedding.shape[0]}')
    #         poss = np.zeros((150, 3))
    #         poss[:embedding.shape[0], :] = embedding[:, :]
    #         mol_els = np.array(
    #             [m2.GetAtomWithIdx(_atom_idx).GetAtomicNum() for _atom_idx in [a.GetIdx() for a in m2.GetAtoms()]])
    #         els = np.zeros(150)
    #         els[:len(mol_els)] = mol_els[:]
    #
    #         # 2. e. 2. b.
    #         record = np.array([(
    #             mol_conf_idx,
    #             _ligand_data['idx'],
    #             len(mol_els),
    #             _ligand_data['canonical_smiles'],
    #             _ligand_data['canonical_smiles'],
    #             poss,
    #             els
    #         )],
    #             dtype=ligand_conf_dtype)
    #         # print(record)
    #
    #         #
    #         mol_conf_group.append(record)
    #         mol_conf_idx += 1


if __name__ == "__main__":
    fire.Fire(main)
