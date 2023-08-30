import dataclasses
import os
import pickle
import pathlib
from pathlib import Path

import yaml
import fire
import pony
import rich
from rich import print as rprint
import pandas as pd

from edanalyzer.database_pony import db, EventORM  # import *


# from pony.orm import *

@dataclasses.dataclass
class ConfigTrain:
    max_epochs: int
    # def __init__(self, dic):
    #     self.max_epochs = dic['max_epochs']


@dataclasses.dataclass
class ConfigTest:
    test_interval: int
    test_convergence_interval: int
    # def __init__(self, dic):
    #     self.test_interval = dic['test_interval']
    #     self.test_convergence_interval = dic['test_convergence_interval']


@dataclasses.dataclass
class Config:
    name: str
    steps: list[str]
    working_directory: Path
    datasets: list[str]
    exclude: list[str]
    train: ConfigTrain
    test: ConfigTest
    custom_annotations: Path
    cpus: int

    # def __init__(self, dic):
    #     self.name = dic["name"]
    #     self.steps = dic['steps']
    #     self.working_directory = Path(dic['working_directory'])
    #     self.datasets = [x for x in dic['datasets']]
    #     self.exclude = [x for x in dic['exclude']]
    #     self.train = ConfigTrain(dic['train'])
    #     self.test = ConfigTest(dic['test'])
    #     self.custom_annotations = dic['custom_annotations']
    #     self.cpus = dic['cpus']


def _get_custom_annotations(path):
    # get the events
    db.bind(provider='sqlite', filename=f"{path}")
    db.generate_mapping()

    with pony.orm.db_session:
        events = pony.orm.select((event, event.partitions, event.annotations, event.ligand, event.pandda,
                                  event.pandda.system, event.pandda.experiment) for event in EventORM)[:]

        custom_annotations = {}
        for event_info in events:
            event = event_info[0]
            if event.pandda:
                event_id = (
                    str(event.pandda.path),
                    str(event.dtag),
                    int(event.event_idx)
                )
                if event.annotations:
                    annotations = {_a.source: _a.annotation for _a in event.annotations}

                    if "manual" in annotations:
                        annotation = annotations["manual"]
                    # else:
                    #     annotation = annotations["auto"]
                        custom_annotations[event_id] = annotation

    db.disconnect()
    return custom_annotations


def _make_database(
        name,
        working_directory,
        datasets,
        exclude,
        cpus,
        custom_annotations
):
    database_path = working_directory / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Get the possible pandda paths
    possible_pandda_paths = [
        path
        for dataset_pattern
        in datasets
        for path
        in Path('/').glob(dataset_pattern[1:])
        if not any([path.match(exclude_pattern) for exclude_pattern in exclude])

    ]
    rprint(f"Got {len(possible_pandda_paths)} pandda paths!")
    rprint(possible_pandda_paths)

    # Get the pandda event tables
    inspect_tables = {}
    for possible_pandda_path in possible_pandda_paths:
        analyse_table_path = possible_pandda_path / "analyses" / "pandda_inspect_events.csv"
        if analyse_table_path.exists():
            try:
                analyse_table = pd.read_csv(analyse_table_path)
                inspect_tables[possible_pandda_path] = analyse_table
            except Exception as e:
                print(f"\tERROR READING INSPECT TABLE: {analyse_table_path} : {e}")
    rprint(f"Got {len(inspect_tables)} pandda inspect tables!")

    with pony.orm.db_session:
        # Multiprocess PanDDAs, returning valid events for addition to the
            # Skip if no valid inspect file


        ...


def __main__(config_yaml="config.yaml"):
    # Initialize the config
    with open(config_yaml, "r") as f:
        dic = yaml.safe_load(f)

        config = Config(
            name=dic["name"],
            steps=dic['steps'],
            working_directory=Path(dic['working_directory']),
            datasets=[x for x in dic['datasets']],
            exclude=[x for x in dic['exclude']],
            train=ConfigTrain(
                dic['train']['max_epochs'],
            ),
            test=ConfigTest(
                dic['test']['test_interval'],
                dic['test']['test_convergence_interval'],
            ),
            custom_annotations=dic['custom_annotations'],
            cpus=dic['cpus']
        )
        rprint(config)

    if not config.working_directory.exists():
        os.mkdir(config.working_directory)

    # Parse custom annotations
    if "Annotations" in config.steps:
        custom_annotations_path = config.working_directory / "custom_annotations.pickle"
        if custom_annotations_path.exists():
            with open(custom_annotations_path, 'rb') as f:
                custom_annotations = pickle.load(f)
        else:
            custom_annotations: dict[tuple[str, str, int], bool] = _get_custom_annotations(config.custom_annotations)
            with open(custom_annotations_path, "wb") as f:
                pickle.dump(custom_annotations, f)
        rprint(f"\tGot {len(custom_annotations)} custom annotations!")

    # Construct the dataset
    if "Collate" in config.steps:
        _make_database(
            config.name,
            config.working_directory,
            config.datasets,
            config.exclude,
            config.cpus,
            custom_annotations
        )

    # Run training/testing
    if 'Train+Test' in config.steps:
        ...
    # Summarize train/test results
    if 'Summarize' in config.steps:
        ...


if __name__ == "__main__":
    fire.Fire(__main__)
