from pathlib import Path

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony
import numpy as np
import tables
import zarr

from edanalyzer.datasets.build_scoring import BuildScoringDatasetSyntheticCorrelationZarr
from edanalyzer.models.build_scoring import LitBuildScoringCorrelation
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def main(config_path, batch_size=12, num_workers=None):
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

    # fileh = tables.open_file("output/build_data_correlation.h5", mode="r")

    zarr_path = 'output/build_data_correlation.zarr'
    root = zarr.open(zarr_path, mode='r')

    # Get the HDF5 root group
    # root = fileh.root
    train_pose_idxs = []
    test_pose_idxs = []
    for table_type in ['normal', ]:
        if table_type == 'normal':
            table_annotation = root['annotation']
            table_poses = root['known_hit_pose']
        else:
            table_annotation = root.pandda_2_annotation
            table_poses = root.pandda_2_known_hit_pose

        # Get train and test event idxs
        rprint(f'Getting idxs of valid train event maps...')
        train_event_table_idxs = set([
            _x['event_map_table_idx']
            for _x
            in table_annotation.get_mask_selection(
                (table_annotation['partition'] == b'train') & (table_annotation['annotation'])) #.where("""(partition == b'train') & (annotation)""")
        ])
        rprint(f'Getting idxs of valid test event maps...')
        test_event_table_idxs = set([
            _x['event_map_table_idx']
            for _x
            in table_annotation.get_mask_selection(
                (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
        ])

        #

        rprint(f"Filtering poses to those matching valid event maps...")
        for row in table_poses:
            pose_event_table_idx = row['event_map_sample_idx']
            if pose_event_table_idx in train_event_table_idxs:
                train_pose_idxs.append((table_type, row['idx']))
            elif pose_event_table_idx in test_event_table_idxs:
                test_pose_idxs.append((table_type, row['idx']))
        rprint(f"\tGot {len(train_pose_idxs)} train samples")
        rprint(f"\tGot {len(test_pose_idxs)} test samples")
    rprint(f"Got {len(train_pose_idxs)} train samples")
    rprint(f"Got {len(test_pose_idxs)} test samples")


    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    dataset_train = DataLoader(
        BuildScoringDatasetSyntheticCorrelationZarr(
            zarr_path,
            train_pose_idxs
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_train)} training samples")
    dataset_test = DataLoader(
        BuildScoringDatasetSyntheticCorrelationZarr(
            zarr_path,
            test_pose_idxs
        ),
        batch_size=batch_size,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Get the model
    model = LitBuildScoringCorrelation()

    # Train
    checkpoint_callback = ModelCheckpoint(dirpath='output/build_scoring_pred_both')
    logger = CSVLogger("output/build_scoring_pred_both/logs")
    trainer = lt.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback],
                         enable_progress_bar=False
                         )
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
