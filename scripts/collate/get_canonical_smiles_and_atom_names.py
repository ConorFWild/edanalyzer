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

    with pony.orm.db_session:

        # Iterate over event maps
        for _record in event_map_table:

            # Get corresponding event
            database_event_idx = _record['event_idx']
            database_event = EventORM[database_event_idx]

            # Get event cif
            dtag_dir = Path(database_event.pandda.path) / 'processed_datasets' / database_event.dtag / 'ligand_files'
            smiles = [x for x in dtag_dir.glob('*.smiles')]
            cifs = [x for x in dtag_dir.glob('*.cif')]
            rprint(f'{database_event.dtag}: {len(smiles)} smiles : {len(cifs)} cifs')

            # Make atom name array

            # Get Mol

            # Get canoncial smiles

            # Store canonical smiles

    ...

if __name__ == "__main__":
    fire.Fire(main)