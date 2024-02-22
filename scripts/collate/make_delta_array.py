import os
import pickle
import time
from pathlib import Path

import yaml
import fire
import pony
from rich import print as rprint

import pandas as pd
import gemmi
import numpy as np

import tables

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer import constants
from edanalyzer.data.build_data import Delta


def _make_test_dataset_psuedo_pandda(
        config_path
):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Open a file in "w"rite mode
    fileh = tables.open_file("output/build_data_v2.h5", mode="r+")
    print(fileh)

    # Get the HDF5 root group
    root = fileh.root
    table_known_hit_pos_sample = root.known_hit_pose
    try:
        table_delta = root.delta
    except:
        table_delta = fileh.create_table(root, "delta", Delta, )

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Get the idxs of annotated
    # annotated_idxs = table_annotation.cols.event_map_table_idx[:]

    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        # query = [_x for _x in pony.orm.select(_event for _event in EventORM)]

        # Get the closest pose for each event map
        close_poses = {}
        begin_get_close_poses = time.time()
        for _x in table_known_hit_pos_sample.where("""rmsd == 0.0"""):
            y = _x.fetch_all_fields()
            close_poses[y['event_map_sample_idx']] = y
        finish_get_close_poses = time.time()
        rprint(f'Got {len(close_poses)} close poses in: {finish_get_close_poses - begin_get_close_poses}')

        # For each pose, get the delta squared
        delta_row = table_delta.row
        for _x in table_known_hit_pos_sample.iterrows():
            _event_sample_idx = _x['event_map_sample_idx']
            _ref_pose = close_poses[_event_sample_idx]['positions']
            _pose = _x['positions']
            _delta_vecs = _ref_pose - _pose
            _delta = np.linalg.norm(_delta_vecs, axis=1)

            delta_row['idx'] = _x['idx']
            delta_row['pose_idx'] = _x['idx']
            delta_row['delta'] = _delta
            delta_row['delta_vecs'] = _delta_vecs
            delta_row.append()

    fileh.close()


if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)
