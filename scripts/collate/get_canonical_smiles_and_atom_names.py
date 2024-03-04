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

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample




def main(config_path):
    # Get the database
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the Database
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Open the store
    zarr_path = 'output/build_data_correlation.zarr'
    root = zarr.open(zarr_path, mode='a')

    # Open the event map store
    event_map_table = root['event_map_sample']
    pose_table = root['known_hit_pose']


    # Try to create the atom name/canonical smiles array
    try:
        root.create_group(
            'ligand_data',
            dtype=[
                ('canonical_smiles', '<U200'), ('atom_ids', '<U5', (60,))
            ]
        )
    except:
        rprint(f"Already created ligand data table!")

    pose_table_idxs = pose_table['idx']
    pose_table_database_events = pose_table['database_event_idx']

    with pony.orm.db_session:

        # Iterate over event maps
        for _record in event_map_table:


            # Get corresponding event
            database_event_idx = _record['event_idx']
            database_event = EventORM[database_event_idx]

            # Get a pose with the event
            rprint(f"Getting matched poses...")
            matching_pose_mask = pose_table_database_events == database_event_idx
            matching_pose_table_idxs = pose_table_idxs[matching_pose_mask]
            rprint(f'Got {len(matching_pose_table_idxs)} matching poses')
            matched_pose = pose_table[matching_pose_table_idxs[0]]
            matched_pose_atom_array = matched_pose['elements'][matched_pose['elements'] != 0]

            # Get event cif
            dtag_dir = Path(database_event.pandda.path) / 'processed_datasets' / database_event.dtag / 'ligand_files'
            smiles = [x for x in dtag_dir.glob('*.smiles')]
            cif_paths = [x for x in dtag_dir.glob('*.cif') if x.stem not in constants.LIGAND_IGNORE_REGEXES]
            rprint(f'{database_event.dtag}: {len(smiles)} smiles : {len(cif_paths)} cifs')

            # See which cif matches the element sequence
            for _cif_path in cif_paths:
                cif = gemmi.cif.read(str(_cif_path))

                key = "comp_LIG"
                try:
                    cif['comp_LIG']
                except:
                    key = "data_comp_XXX"

                # Find the relevant atoms loop
                atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
                atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
                atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
                if not atom_charge_loop:
                    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
                    if not atom_charge_loop:
                        atom_charge_loop = [0] * len(atom_id_loop)

                aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
                if not aromatic_atom_loop:
                    aromatic_atom_loop = [None] * len(atom_id_loop)

                # Get the mapping
                id_to_idx = {}
                for j, atom_id in enumerate(atom_id_loop):
                    id_to_idx[atom_id] = j

                atom_ints = np.array(
                    [gemmi.Element(_atom_name).atomic_number for _atom_name in atom_type_loop if gemmi.Element(_atom_name).atomic_number != 1]
                )
                rprint(atom_ints)
                rprint(matched_pose_atom_array)
                if np.array_equal(atom_ints, matched_pose_atom_array):
                    rprint(f'MATCHED!')
                else:
                    rprint(f"MATCH FAILED!")



            # Make atom name array

            # Get Mol

            # Get canoncial smiles

            # Store canonical smiles

    ...

if __name__ == "__main__":
    fire.Fire(main)