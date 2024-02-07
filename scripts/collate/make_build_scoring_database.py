from pathlib import Path

import fire
from rich import print as rprint
import yaml

from edanalyzer import constants
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load database
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Define train/test split


    # Make dataset


    # Pickle


if __name__ == "__main__":
    fire.Fire(main)
