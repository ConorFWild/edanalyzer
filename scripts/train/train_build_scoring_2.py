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


def sample(iterable, num, replace, weights):
    df = pd.Series(iterable)
    return [_x for _x in df.sample(num, replace=replace, weights=weights)]




def _get_train_test_idxs_full_conf(root):
    # for each z map sample
    # 1. Get the corresponding ligand data
    # 2. Get the corresponding fragments
    # 3. Sample the map n times positively (random corresponding fragment, random conformation)
    # 4. Sample the the map n times negatively (weighted random fragment of different size, random conformation)
    # 5. Sample the fragments n times negatively (random corresponding fragment, random conformation, random negative map)
    # Sample distribution is now 1/3 positive to negative, and every positive map appears the same number of
    # times negatively, and every positive fragment appears appears twice negatively
    rng = np.random.default_rng()

    table_type = 'pandda_2'

    metadata_table = pd.DataFrame(root['meta_sample'][:])
    # table_annotation = root[table_type]['annotation']
    annotation_df = pd.DataFrame(table_annotation[:])
    ligand_idx_smiles_df = pd.DataFrame(
        root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
    ligand_conf_df = pd.DataFrame(
        root[table_type]['ligand_confs'].get_basic_selection(slice(None), fields=['idx', 'num_heavy_atoms',
                                                                                  'fragment_canonical_smiles',
                                                                                  'ligand_canonical_smiles']))

    train_samples = metadata_table[annotation_df['partition'] == b'train']
    test_samples = metadata_table[annotation_df['partition'] == b'test']

    ligand_smiles_to_conf = {
        _smiles: ligand_conf_df[ligand_conf_df['ligand_canonical_smiles'] == _smiles]
        for _smiles
        in ligand_conf_df['ligand_canonical_smiles'].unique()
    }

    pos_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []
    # positive_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    # negative_ligand_sample_distribution = {_ligand: 0 for _ligand in ligand_smiles_to_conf}
    train_pos_conf = []
    train_neg_conf = []

    # Loop over the z samples adding positive samples for each
    for _idx, z in train_samples.iterrows():
        ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx == -1:
        #     continue
        if z['Confidence'] != 'High':
            continue
        ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
        ligand_canonical_smiles = ligand_data['canonical_smiles']
        if ligand_canonical_smiles not in ligand_smiles_to_conf:
            continue
        confs = ligand_smiles_to_conf[ligand_canonical_smiles]
        if len(confs) == 0:
            continue

        # Pos samples
        ligand_conf_samples = []
        for x in range(10):
            positive_ligand_sample_distribution[ligand_canonical_smiles] += 1
            ligand_conf_samples.append(ligand_smiles_to_conf[ligand_canonical_smiles].sample(1)['idx'].iloc[0])
        pos_conf_samples += ligand_conf_samples
        pos_z_samples += [z['idx'] for _j in range(10)]
        train_pos_conf += [z['Confidence'] for _j in range(10)]

    print(f'Got {len(pos_conf_samples)} pos samples!')

    # Loop over the z samples adding the inherent negative samples
    for _idx, z in train_samples[train_samples['Confidence'] == 'Low'].sample(len(pos_conf_samples),
                                                                              replace=True).iterrows():
        # ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx != -1:
        #     continue
        if z['Confidence'] != 'Low':
            continue

        # Select a uniform random fragment
        ligand_freq = {
            k: v / positive_ligand_sample_distribution[k]
            for k, v
            in negative_ligand_sample_distribution.items()
            if positive_ligand_sample_distribution[k] > 0
        }

        ligand_smiles = min(ligand_freq, key=lambda _k: ligand_freq[_k])
        negative_ligand_sample_distribution[ligand_smiles] += 1

        lig_conf_sample = ligand_smiles_to_conf[ligand_smiles].sample(1)['idx'].iloc[0]

        neg_conf_samples += [lig_conf_sample, ]
        neg_z_samples += [z['idx'], ]
        train_neg_conf += [z['Confidence'], ]

    print(f'Got {len(neg_conf_samples)} neg decoy samples!')

    test_pos_z_samples = []
    test_neg_z_samples = []
    test_pos_conf_samples = []
    test_neg_conf_samples = []
    test_pos_conf = []
    test_neg_conf = []

    # Loop over the z samples adding the test samples
    for _idx, z in test_samples.iterrows():
        #
        ligand_data_idx = z['ligand_data_idx']
        # if ligand_data_idx != -1:
        if z['Confidence'] == 'High':
            ligand_data = root[table_type]['ligand_data'][ligand_data_idx]
            ligand_canonical_smiles = ligand_data['canonical_smiles']
            if ligand_canonical_smiles not in ligand_smiles_to_conf:
                continue

            lig_conf_sample = ligand_smiles_to_conf[ligand_canonical_smiles].sample(1)['idx'].iloc[0]

            test_pos_conf_samples.append(lig_conf_sample)
            test_pos_z_samples.append(z['idx'])
            test_pos_conf.append(z['Confidence'])

        else:
            fragment = \
                sample(
                    [_x for _x in positive_ligand_sample_distribution if positive_ligand_sample_distribution[_x] > 0],
                    1, False,
                    None)[0]
            lig_conf_sample = ligand_smiles_to_conf[fragment].sample(1)['idx'].iloc[0]

            test_neg_conf_samples.append(lig_conf_sample)
            test_neg_z_samples.append(z['idx'])
            test_neg_conf.append(z['Confidence'])



    rprint({
        'pos_z_samples len': len(pos_z_samples),
        'neg_z_samples len': len(neg_z_samples),
        'pos_conf_samples len': len(pos_conf_samples),
        'neg_conf_samples len': len(neg_conf_samples),
        'train_pos_conf len': len(train_pos_conf),
        'train_neg_conf len': len(train_neg_conf),

    })
    train_idxs = [
        {'table': table_type, 'z': z, 'f': f, 't': t}
        for z, f, t
        in zip(
            pos_z_samples + neg_z_samples,
            pos_conf_samples + neg_conf_samples,
            train_pos_conf + train_neg_conf
        )
        # ([True] * len(pos_z_samples)) + ([False] * len(neg_z_samples)))]
    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'train_pos_conf len': len(test_pos_conf),
        'train_neg_conf len': len(test_neg_conf),
    })
    test_idxs = [{'table': table_type, 'z': z, 'f': f, 't': t} for z, f, t
                 in zip(
            test_pos_z_samples + test_neg_z_samples,
            test_pos_conf_samples + test_neg_conf_samples,
            # ([True] * len(test_pos_conf_samples)) + ([False] * len(test_neg_conf_samples))
            test_pos_conf + test_neg_conf
        )
                 ]
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
    zarr_path = output_dir / 'build_data.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    rprint(f'Getting train/test data...')
    all_train_pose_idxs, all_test_pose_idxs = _get_train_test_idxs_full_conf(root)

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
        batch_size=128,#batch_size,
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
    output = output_dir / 'build_scoring_nsys=87_opt=adamw_ls=2.5e-2_bs=128_lr=e-2_wd=e-1_sch=pl_cd=10_wn=0.5_r=5.5'
    model = LitEventScoring(output)

    # Train
    rprint('Constructing trainer...')
    checkpoint_callback = ModelCheckpoint(dirpath=str(output ))
    logger = CSVLogger(str( output / 'logs'))
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
