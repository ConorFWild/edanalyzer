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
import pandas as pd

from edanalyzer.datasets.event_scoring import EventScoringDataset
from edanalyzer.models.event_scoring import LitEventScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging


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

    zarr_path = 'output/event_data_with_mtzs_2.zarr'
    root = zarr.open(zarr_path, mode='r')

    # Get the HDF5 root group
    # root = fileh.root
    all_train_pose_idxs = []
    all_test_pose_idxs = []

    table_type = 'pandda_2'

    metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
    table_annotation = root[table_type]['annotation']
    annotation_df = pd.DataFrame(table_annotation[:])
    train_annotations = annotation_df[annotation_df['partition'] == b'train']
    test_annotations = annotation_df[annotation_df['partition'] == b'test']

    ligand_idx_smiles_df = pd.DataFrame(
        root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))

    negative_train_samples = metadata_table[
        (metadata_table['pose_data_idx'] == -1) & (annotation_df['partition'] == b'train')]
    print(f'Got {len(negative_train_samples)} negative train samples')
    negative_test_samples = metadata_table[
        (metadata_table['pose_data_idx'] == -1) & (annotation_df['partition'] == b'test')]
    print(f'Got {len(negative_test_samples)} negative test samples')

    unique_smiles = ligand_idx_smiles_df['canonical_smiles'].unique()
    print(f'Number of unique smiles: {len(unique_smiles)}')

    pos_train_pose_samples = []

    for smiles in unique_smiles:
        print(f'{smiles}')
        ligand_idx_df = ligand_idx_smiles_df[ligand_idx_smiles_df['canonical_smiles'] == smiles]
        corresponding_samples = metadata_table[metadata_table['ligand_data_idx'].isin(ligand_idx_df['idx'])]

        # Get the train events for this ligand
        corresponding_train_event_annotations = train_annotations[
            train_annotations['idx'].isin(corresponding_samples['idx'])]
        pos_train_annotations = corresponding_train_event_annotations[
            corresponding_train_event_annotations['annotation'] == True]
        print(
            f'Got {len(corresponding_train_event_annotations)} train annotations, of which {len(pos_train_annotations)} positive')

        # Get the test events for this ligand
        corresponding_test_event_annotations = test_annotations[
            test_annotations['idx'].isin(corresponding_samples['idx'])]
        pos_test_annotations = corresponding_test_event_annotations[
            corresponding_test_event_annotations['annotation'] == True]
        print(
            f'Got {len(corresponding_test_event_annotations)} test annotations, of which {len(pos_test_annotations)} positive')

        # Get the pos and neg train samples
        if len(pos_train_annotations) > 0:
            pos_train_samples = pos_train_annotations.sample(50, replace=True)
            neg_train_samples = negative_train_samples.sample(50)
            all_train_pose_idxs += [(table_type, x) for x in pos_train_samples['idx']]
            all_train_pose_idxs += [(table_type, x) for x in neg_train_samples['idx']]
            pos_train_pose_samples += [x for x in corresponding_samples[corresponding_samples['idx'].isin(pos_train_annotations['idx'])].sample(50, replace=True)['pose_data_idx']]

        # Get the pos and neg test samples
        if len(pos_test_annotations) > 0:
            pos_test_samples = pos_test_annotations
            # neg_test_samples = negative_test_samples
            all_test_pose_idxs += [(table_type, x) for x in pos_test_samples['idx']]

    all_test_pose_idxs += [(table_type, x) for x in negative_test_samples['idx']]

    # for table_type in ['normal', 'pandda_2']:
    # # for table_type in ['pandda_2']:
    #
    #     train_pose_idxs = []
    #     test_pose_idxs = []
    #     if table_type == 'normal':
    #         table_annotation = root['annotation']
    #         table_poses = root['known_hit_pose']
    #         table_z_map_sample_metadata = root['z_map_sample_metadata']
    #
    #
    #         # table_poses = root.pandda_2_known_hit_pose
    #
    #         # Get train and test event idxs
    #         rprint(f'Getting idxs of valid train event maps...')
    #         # train_event_table_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(
    #         #         (table_annotation['partition'] == b'train') & (table_annotation['annotation']))
    #         # ])
    #         # train_database_event_idxs = set(
    #         #     [
    #         #         table_z_map_sample_metadata[_x]['event_idx']
    #         #         for _x
    #         #         in train_event_table_idxs
    #         #     ]
    #         # )
    #         # rprint(f'Getting idxs of valid test event maps...')
    #         # test_event_table_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(
    #         #         (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
    #         # ])
    #         # test_database_event_idxs = set(
    #         #     [
    #         #         table_z_map_sample_metadata[_x]['event_idx']
    #         #         for _x
    #         #         in test_event_table_idxs
    #         #     ]
    #         # )
    #         # exclude_idxs = set([
    #         #     _x['event_map_table_idx']
    #         #     for _x
    #         #     in table_annotation.get_mask_selection(~table_annotation['annotation'])
    #         # ])
    #         train_database_event_idxs = set([
    #             _x['event_idx']
    #             for _x
    #             in table_annotation.get_mask_selection(
    #                 (table_annotation['partition'] == b'train') & (table_annotation['annotation']))
    #         ])
    #         rprint(f'Getting idxs of valid test event maps...')
    #
    #         test_database_event_idxs = set([
    #             _x['event_idx']
    #             for _x
    #             in table_annotation.get_mask_selection(
    #                 (table_annotation['partition'] == b'test') & (table_annotation['annotation']))
    #         ])
    #         #
    #
    #         rprint(f"Filtering poses to those matching valid event maps...")
    #         for row in table_z_map_sample_metadata[:]:
    #             # z_map_sample_metadata_idx = row['event_map_sample_idx']
    #             database_event_idx = row['event_idx']
    #             if database_event_idx in train_database_event_idxs:
    #                 train_pose_idxs.append((table_type, row['idx']))
    #             elif database_event_idx in test_database_event_idxs:
    #                 test_pose_idxs.append((table_type, row['idx']))
    #     else:
    #         try:
    #             table_annotation = root['pandda_2']['annotation']
    #             table_z_map_sample_metadata = root['pandda_2']['z_map_sample_metadata']
    #             for row in table_z_map_sample_metadata[:]:
    #                 annotation = table_annotation[row['idx']]
    #
    #                 if annotation['partition'] == b"train":
    #                     train_pose_idxs.append((table_type, row['idx']))
    #                 elif annotation['partition'] == b"test":
    #                     test_pose_idxs.append((table_type, row['idx']))
    #         except:
    #             continue
    #
    #     rprint(f"\tGot {len(train_pose_idxs)} train samples")
    #     rprint(f"\tGot {len(test_pose_idxs)} test samples")
    #
    #     positive_train_pose_idxs = [_x for _x in train_pose_idxs if
    #                                 table_z_map_sample_metadata[_x[1]]['pose_data_idx'] != -1]
    #     negative_train_pose_idxs = [_x for _x in train_pose_idxs if
    #                                 table_z_map_sample_metadata[_x[1]]['pose_data_idx'] == -1]
    #
    #     all_train_pose_idxs += negative_train_pose_idxs + (
    #                 positive_train_pose_idxs * (int(len(negative_train_pose_idxs) / len(positive_train_pose_idxs))))
    #     all_test_pose_idxs += test_pose_idxs
    #     rprint(f"Got {len(positive_train_pose_idxs)} postivie train samples")
    #     rprint(f"Got {len(negative_train_pose_idxs)} negative test samples")
    rprint(f"Got {len(all_train_pose_idxs)} train samples")
    rprint(f"Got {len(all_test_pose_idxs)} test samples")




    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    dataset_train = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_train_pose_idxs,
            pos_train_pose_samples
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_train)} training samples")
    dataset_test = DataLoader(
        EventScoringDataset(
            zarr_path,
            all_test_pose_idxs,
            pos_train_pose_samples
        ),
        batch_size=batch_size,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Get the model
    model = LitEventScoring()

    # Train
    checkpoint_callback = ModelCheckpoint(dirpath='output/event_scoring_balanced_2')
    logger = CSVLogger("output/event_scoring_balanced_2/logs")
    trainer = lt.Trainer(accelerator='gpu', logger=logger,
                         callbacks=[
                             checkpoint_callback,
                             StochasticWeightAveraging(swa_lrs=1e-4)
                         ],
                         enable_progress_bar=False,
                         gradient_clip_val=0.1,

                         )
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
