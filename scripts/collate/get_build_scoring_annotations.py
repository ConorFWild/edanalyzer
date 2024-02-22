import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import pickle
import tables

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
    fileh = tables.open_file("output/build_data_v2.h5", mode="r+")

    # Get the HDF5 root group
    root = fileh.root
    table_event_sample = root.event_map_sample

    # Create 2 new tables in group1
    rprint(f"Creating table")
    # try:
    #     root.annotation.remove()
    # except Exception as e:
    #     rprint(e)
    try:
        table_annotation =
    except:
        table_annotation = fileh.create_table(root, "annotation", BuildAnnotation, )
    train_valid = [x['idx'] for x in table_annotation.where("""(partition == b'train') & (annotation)""")]
    test_valid = [x['idx'] for x in table_annotation.where("""(partition == b'test') & (annotation)""")]
    rprint(f"Got {len(train_valid)} train datasets and {len(test_valid)} test datasets!")

    # Get the PanDDA inspect table
    inspect_table = pd.read_csv(working_dir / 'build_annotation_pandda' / 'analyses' / 'pandda_inspect_events.csv')

    #
    with pony.orm.db_session:
        query = [_x for _x in pony.orm.select(_y for _y in EventORM)]

        #
        table_annotation_row = table_annotation.row
        # annotation_idx = 0
        idx_col = table_annotation.cols.idx[:]
        if idx_col.size == 0:
            annotation_idx = 0
        else:
            annotation_idx = int(idx_col.max()) + 1

        for _idx, row in inspect_table.iterrows():
            event_table_idx = row['dtag']

            # Get the annotation
            annotation = row['Ligand Confidence']
            if annotation == "High":
                annotation_bool = True
            else:
                annotation_bool = False

            # Get the partition
            event = query[table_event_sample[event_table_idx]['event_idx']]
            if annotation_bool:
                rprint(event.pandda.system.name)
            if event.pandda.system.name in test_systems:
                partition = 'test'
            else:
                partition = 'train'

            # Update
            table_annotation_row['idx'] = annotation_idx
            table_annotation_row['event_map_table_idx'] = event_table_idx
            table_annotation_row['annotation'] = annotation_bool
            table_annotation_row['partition'] = partition
            table_annotation_row.append()

            annotation_idx += 1

        table_annotation.flush()

        train_valid = [x['idx'] for x in table_annotation.where("""(partition == b'train') & (annotation)""")]
        test_valid = [x['idx'] for x in table_annotation.where("""(partition == b'test') & (annotation)""")]
        rprint(f"Got {len(train_valid)} train datasets and {len(test_valid)} test datasets!")

    fileh.close()


if __name__ == "__main__":
    fire.Fire(main)
