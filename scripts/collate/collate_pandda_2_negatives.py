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
    # custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    # with open(custom_annotations_path, 'rb') as f:
    #     custom_annotations = pickle.load(f)

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

    result_dir = Path('output') / 'panddas'

    # Loop over panddas

    for pandda_dir in result_dir.glob('*'):
        inspect_table_path = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not inspect_table_path.exists():
            continue

        # Get the inspect table
        inspect_table = pd.read_csv(inspect_table_path)

        # Get the high ranking low conf
        high_rank_low_conf = inspect_table[(inspect_table['Ligand Confidence'] == "Low") & (inspect_table['z_peak'] > 0.5)]

        rprint(f'Got {len(high_rank_low_conf)} high ranking, low confidence events')


if __name__ == "__main__":
    fire.Fire(main)