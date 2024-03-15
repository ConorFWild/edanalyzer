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

from edanalyzer.datasets.event_scoring import EventScoringDataset
from edanalyzer.models.event_scoring import LitEventScoring
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

    zarr_path = 'output/event_data.zarr'
    root = zarr.open(zarr_path, mode='r')

    # Get the HDF5 root group
    # root = fileh.root
    all_train_pose_idxs = []
    all_test_pose_idxs = []

    for table_type in ['normal', 'pandda_2']:
        train_pose_idxs = []
        test_pose_idxs = []
        if table_type == 'normal':
            table_annotation = root['annotation']
            table_poses = root['known_hit_pose']
            table_z_map_sample_metadata = root['z_map_sample_metadata']


            # table_poses = root.pandda_2_known_hit_pose

            # Get train and test event idxs
            rprint(f'Getting idxs of valid train event maps...')
            train_event_table_idxs = set([
                _x['event_map_table_idx']
                for _x
                in table_annotation.get_mask_selection(
                    (table_annotation['partition'] == b'train') & (table_annotation['annotation']))
            ])
            train_database_event_idxs = set(
                [
                    table_z_map_sample_metadata[_x]['event_idx']
                    for _x
                    in train_event_table_idxs
                ]
            )
            rprint(f'Getting idxs of valid test event maps...')
            test_event_table_idxs = set([
                _x['event_map_table_idx']
                for _x
                in table_annotation.get_mask_selection(
                    (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
            ])
            test_database_event_idxs = set(
                [
                    table_z_map_sample_metadata[_x]['event_idx']
                    for _x
                    in test_event_table_idxs
                ]
            )
            exclude_idxs = set([
                _x['event_map_table_idx']
                for _x
                in table_annotation.get_mask_selection(~table_annotation['annotation'])
            ])

            #

            rprint(f"Filtering poses to those matching valid event maps...")
            for row in table_z_map_sample_metadata:
                # z_map_sample_metadata_idx = row['event_map_sample_idx']
                database_event_idx = row['event_idx']
                if database_event_idx in train_database_event_idxs:
                    train_pose_idxs.append((table_type, row['idx']))
                elif database_event_idx in test_database_event_idxs:
                    test_pose_idxs.append((table_type, row['idx']))
        else:
            table_annotation = root['pandda_2']['annotation']
            table_z_map_sample_metadata = root['pandda_2']['z_map_sample_metadata']
            for row in table_z_map_sample_metadata:
                annotation = table_annotation[row['idx']]

                if annotation['partition'] == b"train":
                    train_pose_idxs.append((table_type, row['idx']))
                elif annotation['partition'] == b"test":
                    test_pose_idxs.append((table_type, row['idx']))

        rprint(f"\tGot {len(train_pose_idxs)} train samples")
        rprint(f"\tGot {len(test_pose_idxs)} test samples")

        positive_train_pose_idxs = [_x for _x in train_pose_idxs if
                                    table_z_map_sample_metadata[_x[1]]['pose_data_idx'] != -1]
        negative_train_pose_idxs = [_x for _x in train_pose_idxs if
                                    table_z_map_sample_metadata[_x[1]]['pose_data_idx'] == -1]

        all_train_pose_idxs += negative_train_pose_idxs + (
                    positive_train_pose_idxs * (int(len(negative_train_pose_idxs) / len(positive_train_pose_idxs))))
        all_test_pose_idxs += test_pose_idxs
        rprint(f"Got {len(positive_train_pose_idxs)} postivie train samples")
        rprint(f"Got {len(negative_train_pose_idxs)} negative test samples")
    rprint(f"Got {len(all_train_pose_idxs)} train samples")
    rprint(f"Got {len(all_test_pose_idxs)} test samples")




    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    dataset_train = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_train_pose_idxs
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_train)} training samples")
    dataset_test = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_test_pose_idxs
        ),
        batch_size=batch_size,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Get the model
    model = LitEventScoring()

    # Train
    checkpoint_callback = ModelCheckpoint(dirpath='output/event_scoring_pandda_2')
    logger = CSVLogger("output/event_scoring_pandda_2/logs")
    trainer = lt.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback],
                         enable_progress_bar=False
                         )
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
