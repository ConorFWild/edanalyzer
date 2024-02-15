from pathlib import Path

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony

from edanalyzer.datasets.event_scoring import EventScoringDataset, EventScoringDatasetItem
from edanalyzer.models.event_scoring import LitEventScoring
from edanalyzer.data.database_schema import db, EventORM, SystemORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def main(config_path, batch_size=12, num_workers=20):
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

    # Get the dataset
    with pony.orm.db_session:
        query = [(_x, _x.pandda, _x.pandda.system, _x.annotations) for _x in pony.orm.select(_y for _y in EventORM)]
        rprint(f"Got {len(query)} events to train and test with.")
        dataset_train = DataLoader(
            EventScoringDataset(
                [
                    EventScoringDatasetItem(annotation=_event[3].annotation, **_event[0].to_dict(
                        exclude=[
                            'id', 'ligand', 'dataset', 'pandda', 'annotations', 'partitions'],
                    ))
                    for _event
                    in query
                    if _event[2].name not in config['test']['test_systems']
                ]
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        rprint(f"Got {len(dataset_train)} training datasets")
        dataset_test = DataLoader(
            EventScoringDataset(
                [
                    EventScoringDatasetItem(annotation=_event[3].annotation, **_event[0].to_dict(
                        exclude=['id', 'ligand', 'dataset', 'pandda', 'annotations', 'partitions'], ))
                    for _event
                    in query
                    if _event[2].name in config['test']['test_systems']
                ]
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        rprint(f"Got {len(dataset_test)} test datasets")

    # Get the model
    model = LitEventScoring()

    # Train
    checkpoint_callback = ModelCheckpoint(dirpath='output/event_scoring_6')
    logger = CSVLogger("output/event_scoring_6/logs")
    trainer = lt.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], enable_progress_bar=False)
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
