import pickle
from pathlib import Path

import fire
import yaml
from rich import print as rprint

from edanalyzer.data.annotations import (
    _get_custom_annotations_from_database,
    _get_custom_annotations_from_pandda
)


def main(config_path):
    rprint(f'Running collate_annotations from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    custom_annotations: dict[tuple[str, str, int], bool] = {}
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"

    rprint(f"Getting custom annotations...")

    # Parse old databases
    for database_path_pattern in config['custom_annotations.databases']:

        for path in Path('/').glob(database_path_pattern[1:]):
            rprint(f"Getting annotations from: {path}")
            _custom_annotations: dict[tuple[str, str, int], bool] = _get_custom_annotations_from_database(path)
            rprint(f"\tGot {len(_custom_annotations)} annotations!")
            custom_annotations.update(_custom_annotations)

    # Parse custom panddas
    for custom_pandda_path_pattern in config['custom_annotations']['panddas']:
        for path in Path('/').glob(custom_pandda_path_pattern[1:]):
            rprint(f"Getting annotations from: {path}")
            _custom_annotations: dict[tuple[str, str, int], bool] = _get_custom_annotations_from_pandda(path)
            rprint(f"\tGot {len(_custom_annotations)} annotations!")
            custom_annotations.update(_custom_annotations)

    with open(custom_annotations_path, "wb") as f:
        pickle.dump(custom_annotations, f)
    rprint(f"\tGot {len(custom_annotations)} custom annotations!")


if __name__ == "__name__":
    fire.Fire(main)
