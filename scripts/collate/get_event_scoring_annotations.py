import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import pickle
import zarr
import tables
import numpy as np

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import BuildAnnotation


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'rb') as f:
        custom_annotations = pickle.load(f)

    #
    working_dir = Path(config['working_directory'])
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
    root = zarr.open('output/event_data.zarr', 'a')


    annotation_dtype = [
        ('idx', '<i4'),
        ('event_map_table_idx', '<i4'),
        ('annotation', '?'),
        ('partition', 'S32')]
    try:
        annotation_table = root.create_dataset(
            'annotation',
            shape=(0,),
            chunks=(1,),
            dtype=annotation_dtype
        )
    except:
        annotation_table = root['annotation']
    z_map_sample_metadata_table = root['z_map_sample_metadata']
    train_valid = [x['idx'] for x in
                   annotation_table.get_mask_selection((annotation_table['partition'] == 'train') & annotation_table['annotation'])]
    test_valid = [
        x['idx']
        for x
        in annotation_table.get_mask_selection((annotation_table['partition'] == 'test') & annotation_table['annotation'])]
    rprint(f"Got {len(train_valid)} train datasets and {len(test_valid)} test datasets!")

    #
    fileh = tables.open_file("output/build_data_correlation.h5", mode="r")

    # Get the HDF5 root group
    root = fileh.root
    table_event_sample = root.event_map_sample
    # table_annotations = root.annotations

    # Get the PanDDA inspect table
    inspect_tables = [
        pd.read_csv(working_dir / 'build_annotation_pandda_2' / 'analyses' / 'pandda_inspect_events.csv'),
        pd.read_csv(working_dir / 'build_annotation_pandda' / 'analyses' / 'pandda_inspect_events.csv')
    ]

    combined_inspect_table = pd.concat(inspect_tables, ignore_index=True)

    # Get the mapping from inspect table rows to database events
    database_event_idx_to_annotation = {}
    for _row in combined_inspect_table.iterrows():
        event_sample_idx = _row['dtag']
        database_event_idx = table_event_sample[event_sample_idx]['event_idx']
        database_event_idx_to_annotation[database_event_idx] = _row['Ligand Confidence']

    rprint(database_event_idx_to_annotation)

    #
    with pony.orm.db_session:
        # query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
        annotation_idx = 0
        for z_map_sample_metadata in z_map_sample_metadata_table:
            if z_map_sample_metadata['pose_data_idx'] == -1:
                rprint(f'Negative sample! Skipping!')
                continue

            # Get the corresponding database event
            database_event_idx = z_map_sample_metadata['event_idx']

            # Get the
            try:
                annotation = database_event_idx_to_annotation[database_event_idx]
            except:
                rprint(f'{database_event_idx} not in annotations!')
                annotation ='Low'

            # Get the annotation
            if annotation == "High":
                annotation_bool = True
            else:
                annotation_bool = False

            # Get the partition
            event = EventORM[database_event_idx]
            if annotation_bool:
                rprint(event.pandda.system.name)
            if event.pandda.system.name in test_systems:
                partition = 'test'
            else:
                partition = 'train'

            # Update
            annotation_sample = np.array(
                (
                    annotation_idx,
                    z_map_sample_metadata['idx'],
                    annotation_bool,
                    partition
                ),
                dtype=annotation_dtype
            )
            annotation_table.append(annotation_sample)

            annotation_idx += 1

        train_valid = [x['idx'] for x in annotation_table[(annotation_table['partition'] == 'train') & annotation_table['annotation']]]
        test_valid = [x['idx'] for x in annotation_table[(annotation_table['partition'] == 'test') & annotation_table['annotation']]]
        rprint(f"Got {len(train_valid)} train datasets and {len(test_valid)} test datasets!")

    fileh.close()


if __name__ == "__main__":
    fire.Fire(main)
