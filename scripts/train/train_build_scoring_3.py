import shutil
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

from edanalyzer.datasets.build_scoring_3 import BuildScoringDataset
from edanalyzer.models.build_scoring_2 import LitBuildScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging


def sample(iterable, num, replace, weights):
    df = pd.Series(iterable)
    return [_x for _x in df.sample(num, replace=replace, weights=weights)]


def _get_train_test_idxs_full_conf(root, config):
    # for each z map sample
    # 1. Get the corresponding ligand data
    # 2. Get the corresponding fragments
    # 3. Sample the map n times positively (random corresponding fragment, random conformation)
    # 4. Sample the the map n times negatively (weighted random fragment of different size, random conformation)
    # 5. Sample the fragments n times negatively (random corresponding fragment, random conformation, random negative map)
    # Sample distribution is now 1/3 positive to negative, and every positive map appears the same number of
    # times negatively, and every positive fragment appears appears twice negatively
    rng = np.random.default_rng()

    # table_type = 'pandda_2'

    metadata_table = pd.DataFrame(root['meta_sample'][:])
    # table_annotation = root[table_type]['annotation']
    # annotation_df = pd.DataFrame(table_annotation[:])
    ligand_idx_smiles_df = pd.DataFrame(
        root['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
    decoy_table = pd.DataFrame(
        root['decoy_pose_sample'].get_basic_selection(slice(None), fields=['idx', 'meta_idx', 'rmsd']))

    train_samples = metadata_table[~metadata_table['system'].isin(config['test']['test_systems'])]
    test_samples = metadata_table[metadata_table['system'].isin(config['test']['test_systems'])]

    meta_to_decoy = {
        idx: decoy_table[decoy_table['meta_idx'] == idx]
        for idx
        in metadata_table['idx']
    }

    # ligand_smiles_to_conf = {
    #     _smiles: ligand_conf_df[ligand_conf_df['ligand_canonical_smiles'] == _smiles]
    #     for _smiles
    #     in ligand_conf_df['ligand_canonical_smiles'].unique()
    # }

    pos_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []
    # positive_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    # negative_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    train_pos_conf = []
    train_neg_conf = []

    # Add the train samples
    train_idxs = []
    for _idx, _meta in train_samples.iterrows():
        decoys = meta_to_decoy[_meta["idx"]]
        # close_samples = decoys[decoys['rmsd'] < 1.5]
        # far_samples = decoys[decoys['rmsd'] >= 1.5]

        # num_samples = min([len(close_samples), len(far_samples)])
        for _decoy_idx, _decoy in decoys.iterrows():
            train_idxs.append(
                {
                    'meta': _meta['idx'],
                    'decoy': _decoy['idx'],
                    'embedding': decoys.sample(1)['idx'].iloc[0],
                    'train': True
                }
            )

    # Get the test samples
    test_idxs = []
    for _idx, _meta in test_samples.iterrows():
        decoys = meta_to_decoy[_meta["idx"]]
        for _decoy_idx, _decoy in decoys.iterrows():
            test_idxs.append(
                {
                    'meta': _meta['idx'],
                    'decoy': _decoy['idx'],
                    'embedding': decoys.sample(1)['idx'].iloc[0],
                    'train': False
                }
            )

    return train_idxs, test_idxs


def main(config_path, batch_size=12, num_workers=None):
    rprint(f'Running train event scoring from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the Database
    database_path = Path(config['working_directory']) / "database.db"
    rprint(f'loading database from: {database_path}')

    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    output_dir = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output')

    # zarr_path = output_dir / 'event_data_with_mtzs_3.zarr'
    zarr_path = output_dir / 'build_data_augmented.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    rprint(f'Getting train/test data...')
    all_train_pose_idxs, all_test_pose_idxs = _get_train_test_idxs_full_conf(root, config)

    rprint(f"Got {len(all_train_pose_idxs)} train samples")
    rprint(f"Got {len(all_test_pose_idxs)} test samples")

    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    rprint(f'Constructing train and test dataloaders...')
    dataset_train = DataLoader(
        BuildScoringDataset(
            zarr_path,
            all_train_pose_idxs,
            # pos_train_pose_samples
            None
        ),
        batch_size=128,  # batch_size,
        shuffle=True,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_train)} training samples")
    dataset_test = DataLoader(
        BuildScoringDataset(
            zarr_path,
            all_test_pose_idxs,
            # pos_train_pose_samples
            None
        ),
        batch_size=batch_size,
        num_workers=19,
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Get the model
    rprint('Constructing model...')
    output = output_dir / 'build_scoring_nsys=87_opt=adamw_ls=2.5e-2_bs=128_lr=e-2_wd=e-4_sch=pl_cd=10_wn=0.25_msk=0.35'
    try:
        shutil.rmtree(output)
    except Exception as e:
        print(e)
    model = LitBuildScoring(output)

    # Train
    rprint('Constructing trainer...')
    checkpoint_callback = ModelCheckpoint(dirpath=str(output))
    logger = CSVLogger(str(output / 'logs'))
    trainer = lt.Trainer(accelerator='gpu', logger=logger,
                         callbacks=[
                             checkpoint_callback,
                             StochasticWeightAveraging(swa_lrs=1e-3,
                                                       # swa_epoch_start=0.75,
                                                       swa_epoch_start=0.5,
                                                       )
                         ],
                         enable_progress_bar=False,
                         gradient_clip_val=1.5,
                         max_epochs=400
                         )
    rprint(f'Training...')
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)
