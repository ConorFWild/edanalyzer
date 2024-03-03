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
    zarr_path = 'output/build_data_correlation.zarr'
    root = zarr.open(zarr_path, mode='r')

    # Create 2 new tables in group1
    rprint(f"Getting or creating table")

    table_mtz_sample = root['mtz_sample']
    table_event_map_sample = root['event_map_sample']
    table_known_hit_pos_sample = root['known_hit_pose']

    # PanDDA 2 events


    #
    rprint(f"Querying events...")
    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query_events = [_x for _x in pony.orm.select(
            (event, event.annotations, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)]

        for _record in table_event_map_sample:
            database_event_idx = _record['event_idx']
            # database_event = query_events[database_event_idx][0]
            database_event = EventORM[database_event_idx]
            assert database_event.id == database_event_idx

            dtag = database_event.dtag

            rprint(f"Database event idx: {database_event_idx}. Dtag: {dtag}")


            # rprint(f"Matched to event: {database_event}")

            exit()


if __name__ == "__main__":
    fire.Fire(main)
