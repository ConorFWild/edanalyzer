from pathlib import Path

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony

from edanalyzer.datasets.build_scoring import BuildScoringDataset, BuildScoringDatasetItem
from edanalyzer.models.build_scoring import LitBuildScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

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
        query = [_x for _x in pony.orm.select(_y for _y in AutobuildORM)]
        dataset_train = DataLoader(
            BuildScoringDataset(
                [
                    BuildScoringDatasetItem(**_event.to_dict(exclude='id'))
                    for _event
                    in query
                    if _event.train_test == "Train"
                ]
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        dataset_test = DataLoader(
            BuildScoringDataset(
                [
                    BuildScoringDatasetItem(**_event.to_dict(exclude='id'))
                    for _event
                    in query
                    if _event.train_test == "Test"
                ]
            ),
            batch_size=batch_size,
            # shuffle=True,
            num_workers=num_workers,
        )

    # Get the model
    model = LitBuildScoring()

    # Train
    checkpoint_callback = ModelCheckpoint(dirpath='output/build_scoring_3')
    logger = CSVLogger("output/build_scoring_3/logs")
    trainer = lt.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], enable_progress_bar=False)
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
