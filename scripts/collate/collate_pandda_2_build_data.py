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
    _make_z_map_sample_metadata_table,
    _make_z_map_sample_table,
    _make_ligand_data_table,
    _make_known_hit_pose_table,
    _make_annotation_table,
    _get_closest_res_from_dataset_dir,
    _get_z_map_sample_from_dataset_dir,
    _get_pose_sample_from_dataset_dir,
    _get_ligand_data_sample_from_dataset_dir,
    _get_annotation_sample_from_dataset_dir,
    _get_z_map_metadata_sample_from_dataset_dir,
    _make_xmap_sample_table,
    _get_xmap_sample_from_dataset_dir
)

meta_sample_dtype = [
    ('idx', '<i4'),
    # ('event_idx', '<i4'),
    ('res_id', '<U32'),
    ('system', '<U32'),
    ('dtag', '<U32'),
    ('event_num', 'i8'),
    ('Confidence', '<U32'),
    ('size', '<f4')
]
mtz_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]
xmap_sample_dtype = [('idx', '<i4'),  ('sample', '<f4', (90, 90, 90))]
z_map_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]
known_hit_pose_sample_dtype = [
    ('idx', '<i8'),
    ('positions', '<f4', (150, 3)),
    ('atoms', '<U5', (150,)),
    ('elements', '<i4', (150,)),
]
decoy_pose_sample_dtype = [
    ('idx', '<i4'),
    # ('database_event_idx', '<i4'),
    ('meta_idx', '<i4'),
    # ('mtz_sample_idx', '<i4'),
    ('positions', '<f4', (150, 3)),
    ('atoms', '<U5', (150,)),
    ('elements', '<i4', (150,)),
    ('rmsd', '<f4')]
delta_dtype = [
    ('idx', '<i4'),
    ('pose_idx', '<i4'),
    ('delta', '<f4', (150,)),
    ('delta_vec', '<f4', (150, 3)),
]
annotation_dtype = [

    ('idx', '<i4'),
    ('event_map_table_idx', '<i4'),
    ('annotation', '?'),
    ('partition', 'S32')]
ligand_data_dtype = [
    ('idx', 'i8'),
    ('num_heavy_atoms', 'i8'),
    ('canonical_smiles', '<U300'),
    ('atom_ids', '<U5', (150,)),
    ('connectivity', '?', (150, 150,))
]


def _get_event_autobuilds_paths(dataset_dir, event_idx):
    autobuild_dir = dataset_dir / 'autobuild'
    autobuild_paths = [x for x in autobuild_dir.glob('*')]
    return autobuild_paths


def _get_build_data(build_path, pose_sample):
    st = gemmi.read_structure(str(build_path))
    poss, atom, elements = _res_to_array(st[0][0][0], )
    rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(pose_sample - poss, axis=1))) / poss.shape[0])
    return poss, atom, elements, rmsd


def _get_pose_sample_from_res(
        model_dir,
        res,
        x, y, z,
        idx_pose
):
    centroid = np.array([x, y, z])
    poss, atom, elements = _res_to_array(res, )
    com = np.mean(poss, axis=0).reshape((1, 3))
    event_to_lig_com = com - centroid.reshape((1, 3))
    _poss_centered = poss - com
    _rmsd_target = np.copy(_poss_centered) + np.array([22.5, 22.5, 22.5]).reshape(
        (1, 3)) + event_to_lig_com
    size = min(150, _rmsd_target.shape[0])
    atom_array = np.zeros(150, dtype='<U5')
    elements_array = np.zeros(150, dtype=np.int32)
    pose_array = np.zeros((150, 3))
    pose_array[:size, :] = _rmsd_target[:size, :]
    atom_array[:size] = atom[:size]
    elements_array[:size] = elements[:size]

    known_hit_pos_sample = np.array([(
        idx_pose,
        pose_array,
        atom_array,
        elements_array,
    )],
        dtype=known_hit_pose_sample_dtype
    )
    return known_hit_pos_sample

def setup_store(zarr_path):
    root = zarr.open(zarr_path, mode='w')

    table_meta_sample = root.create_dataset(
        'meta_sample',
        shape=(0,),
        chunks=(1,),
        dtype=meta_sample_dtype
    )
    table_mtz_sample = root.create_dataset(
        'mtz_sample',
        shape=(0,),
        chunks=(1,),
        dtype=mtz_sample_dtype
    )
    table_xmap_sample = root.create_dataset(
        'xmap_sample',
        shape=(0,),
        chunks=(1,),
        dtype=xmap_sample_dtype
    )
    table_z_map_sample = root.create_dataset(
        'z_map_sample',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_dtype
    )

    table_known_hit_pos_sample = root.create_dataset(
        'known_hit_pose',
        shape=(0,),
        chunks=(1,),
        dtype=known_hit_pose_sample_dtype
    )

    table_decoy_pose_sample = root.create_dataset(
        'decoy_pose_sample',
        shape=(0,),
        chunks=(1,),
        dtype=decoy_pose_sample_dtype
    )

    delta_table = root.create_dataset(
        'delta',
        shape=(0,),
        chunks=(1,),
        dtype=delta_dtype
    )

    annotation_table = root.create_dataset(
        'annotation',
        shape=(0,),
        chunks=(1,),
        dtype=annotation_dtype

    )

    ligand_data_table = root.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(1,),
        dtype=ligand_data_dtype
    )

    return root


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
    zarr_path = '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/build_data.zarr'

    root = setup_store(zarr_path)
    table_meta_sample = root['meta_sample']
    table_xmap_sample = root['xmap_sample']
    table_z_map_sample = root['z_map_sample']
    table_known_hit_pos_sample = root['known_hit_pose']
    table_decoy_pose_sample = root['decoy_pose_sample']
    delta_table = root['delta']
    annotation_table = root['annotation']
    ligand_data_table = root['ligand_data']

    # Iterate over annotated pandda 2 datasets
    meta_idx = 0
    pose_idx = 0
    # tmp_pose_idx = 0
    for pandda_dir in Path(
            '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score/').glob(
        '*'):
        rprint(f'########### Processing system: {pandda_dir.name} ###############')
        if pandda_dir.name == 'TcCS':
            continue

        # Skip if not a directory
        if not pandda_dir.is_dir():
            rprint(f'Skipping Directory!')
            continue
        rprint(f'PanDDA dir is: {pandda_dir}')

        # Skip if no inspect table
        pandda_inspect_table = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not pandda_inspect_table.exists():
            rprint(f'\tNO INSPECT TABLE! SKIPPING!')
            continue

        # Get the inspect table
        inspect_table = pd.read_csv(pandda_inspect_table)

        # Iterate the inspect table
        for _idx, _row in inspect_table.iterrows():
            # Unpack the row information
            dtag, event_idx, bdc, conf, viewed, size = _row['dtag'], _row['event_idx'], _row['1-BDC'], _row[
                constants.PANDDA_INSPECT_HIT_CONDFIDENCE], _row[constants.PANDDA_INSPECT_VIEWED], _row[
                constants.PANDDA_INSPECT_CLUSTER_SIZE]

            system = _get_system_from_dtag(dtag)

            x, y, z = _row['x'], _row['y'], _row['z']
            dataset_dir = pandda_dir / 'processed_datasets' / dtag

            model_dir = dataset_dir / 'modelled_structures'

            # Get the corresponding residue
            resid, res, dist = _get_closest_res_from_dataset_dir(
                dataset_dir,
                x, y, z
            )

            # Process hits only
            if not ((conf == 'High') & (dist < 6.0)):
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
                meta_idx,
            )
            xmap_sample = _get_xmap_sample_from_dataset_dir(
                dataset_dir,
                x, y, z,
                meta_idx,
            )
            # Get the modelled pose sample
            pose_sample = _get_pose_sample_from_res(
                model_dir,
                res,
                x, y, z,
                meta_idx
            )
            pose_elements = pose_sample["elements"][pose_sample["elements"] != 0]

            # Get the ligand data
            ligand_data_sample = _get_ligand_data_sample_from_dataset_dir(
                dataset_dir,
                res,
                meta_idx,
            )
            if not ligand_data_sample:
                rprint(f'\t\tNO LIGAND DATA! SKIPPING!')
                continue

            # Get the autobuild pose samples
            # Generate the decoy/rmsd pairs
            # poses, atoms, elements, rmsds = _get_known_hit_poses(
            #     known_hits[known_hit_dataset][known_hit_residue],
            #     event_to_lig_com
            # )
            decoy_pose_samples = []
            delta_samples = []
            tmp_pose_idx = pose_idx
            for build_path in _get_event_autobuilds_paths(dataset_dir, event_idx):
                pose, atom, element, rmsd = _get_build_data(
                    build_path,
                    pose_sample['positions'][pose_sample['elements'] != 0],
                )

                if not np.array_equal(element, pose_elements):
                    rprint(f'Ligand doesn\'t match! Skipping! {element} vs {pose_elements}')
                    continue

                atom_array = np.zeros(150, dtype='<U5')
                elements_array = np.zeros(150, dtype=np.int32)
                pose_array = np.zeros((150, 3))
                num_atoms = element.size
                pose_array[:num_atoms, :] = pose[:num_atoms, :]
                atom_array[:num_atoms] = atom[:num_atoms]
                elements_array[:num_atoms] = element[:num_atoms]
                if rmsd > 15:
                    continue
                known_hit_pos_sample = np.array(
                    [
                        (
                            pose_idx,
                            meta_idx,
                            pose_array,
                            atom_array,
                            elements_array,
                            rmsd
                        )
                    ],
                    dtype=decoy_pose_sample_dtype
                )
                decoy_pose_samples.append(known_hit_pos_sample)

                _delta_vecs = pose_sample['positions'][pose_sample['elements'] != 0] - pose
                _delta = np.linalg.norm(_delta_vecs, axis=1)
                delta_sample = np.array([(
                    pose_idx,
                    # idx_pose,
                    meta_idx,
                    _delta,
                    _delta_vecs,
                )], dtype=delta_dtype
                )
                delta_samples.append(delta_sample)
                tmp_pose_idx += 1

            # Make the metadata sample
            meta_sample = np.array(
                [(
                    meta_idx,
                    # event_idx,
                    resid,
                    system,
                    dtag,
                    event_idx,
                    conf,
                    size
                )],
                dtype=meta_sample_dtype
            )
            rprint(meta_sample)

            # Add items to store
            table_meta_sample.append(meta_sample)
            table_xmap_sample.append(xmap_sample)
            table_z_map_sample.append(z_map_sample)
            table_known_hit_pos_sample.append(pose_sample)
            # annotation_table.append()
            ligand_data_table.append(ligand_data_sample)

            for decoy_pose_sample, delta_sample in zip(decoy_pose_samples, delta_samples):
                table_decoy_pose_sample.append(
                    decoy_pose_sample
                )
                delta_table.append(
                    delta_sample
                )

            meta_idx += 1
            pose_idx = tmp_pose_idx


if __name__ == "__main__":
    fire.Fire(main)
