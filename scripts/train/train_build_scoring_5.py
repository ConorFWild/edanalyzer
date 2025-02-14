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

from edanalyzer.datasets.build_scoring_5 import BuildScoringDataset
from edanalyzer.models.build_scoring_2 import LitBuildScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
import optuna
from lightning.pytorch.callbacks import EarlyStopping


import logging
import sys
import os

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

    decoy_table['idx'] = np.arange(len(decoy_table))
    rprint(f'Len of decoy table: {len(decoy_table)}')

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
        if len(decoys) == 0:
            print(f'No decoys for {_meta}!')
            continue
        # close_samples = decoys[decoys['rmsd'] < 1.5]
        # far_samples = decoys[decoys['rmsd'] >= 1.5]

        # num_samples = min([len(close_samples), len(far_samples)])
        # for _decoy_idx, _decoy in decoys.iterrows():
        train_idxs.append(
            {
                'meta': _meta['idx'],
                'meta_to_decoy': decoys,
                # 'decoy': _decoy['idx'],
                # 'embedding': decoys.sample(1)['idx'].iloc[0],
                # 'train': True
            }
        )

    # Get the test samples
    test_idxs = []
    for _idx, _meta in test_samples.iterrows():
        decoys = meta_to_decoy[_meta["idx"]]
        if len(decoys) == 0:
            print(f'No decoys for {_meta}!')
            continue
        # for _decoy_idx, _decoy in decoys.iterrows():
        test_idxs.append(
            {
                'meta': _meta['idx'],
                'meta_to_decoy': decoys,
                # 'decoy': _decoy['idx'],
                # 'embedding': decoys.sample(1)['idx'].iloc[0],
                # 'train': False
            }
        )

    train_config = {
        'test_train': 'train',
        'samples': train_idxs,

    }

    test_config = {
        'test_train': 'test',
        'samples': test_idxs,

    }

    return train_config, test_config


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
    zarr_path = output_dir / 'build_data_augmented_2.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    rprint(f'Getting train/test data...')
    train_config, test_config = _get_train_test_idxs_full_conf(root, config)

    study_name = 'build_scoring_prod_2'
    output = output_dir / study_name
    if not output.exists():
        os.mkdir(output)

    def objective(trial):
        # Suggest hyperparameters
        _config = {
            "lr": trial.suggest_loguniform('lr', 1e-4, 1e0),
            "wd": trial.suggest_loguniform('wd', 1e-4, 1e0),
            'fraction_background_replace': trial.suggest_uniform('fraction_background_replace', 0.0, 1e0),
            'xmap_radius': trial.suggest_uniform('xmap_radius', 3.0, 7.0),
            'max_x_blur': trial.suggest_uniform('max_x_blur', 0.0, 3.0),
            'max_z_blur': trial.suggest_uniform('max_z_blur', 0.0, 3.0),
            'drop_rate': trial.suggest_uniform('drop_rate', 0.0, 1.0),
            'planes_1': trial.suggest_categorical('planes_1', [2, 4, 8, 16, ]),
            'drop_1': trial.suggest_uniform('drop_1', 0.0, 1.0),
            'planes_2': trial.suggest_categorical('planes_2', [4, 8, 16, 32, ]),
            'drop_2': trial.suggest_uniform('drop_2', 0.0, 1.0),
            'planes_3': trial.suggest_categorical('planes_3', [8, 16, 32, 64, ]),
            'drop_3': trial.suggest_uniform('drop_3', 0.0, 1.0),
            'planes_4': trial.suggest_categorical('planes_4', [16, 32, 64, 128, ]),
            'drop_4': trial.suggest_uniform('drop_4', 0.0, 1.0),
            'planes_5': trial.suggest_categorical('planes_5', [32, 64, 128, 256, ]),
            'drop_5': trial.suggest_uniform('drop_5', 0.0, 1.0),
            'drop_atom_rate': trial.suggest_uniform('drop_atom_rate', 0.0, 1.0),
            'max_pos_atom_mask_radius': trial.suggest_uniform('max_pos_atom_mask_radius', 1.01, 4.0),
            'max_translate': trial.suggest_uniform('max_translate', 0.0, 5.0),
            'max_x_noise': trial.suggest_uniform('max_x_noise', 0.0, 2.0),
            'max_z_noise': trial.suggest_uniform('max_z_noise', 0.0, 2.0),
            'pos_resample_rate': trial.suggest_int('pos_resample_rate', 0, 10),
            'p_flip': trial.suggest_uniform('p_flip', 0.0, 1.0),
            'z_mask_radius': trial.suggest_uniform('z_mask_radius', 1.0, 3.5),
            'z_cutoff': trial.suggest_uniform('z_cutoff', 1.5, 3.0),
            'blocks_1': trial.suggest_categorical('blocks_1', [1, 2, ]),
            'blocks_2': trial.suggest_categorical('blocks_2', [1, 2, ]),
            'blocks_3': trial.suggest_categorical('blocks_3', [1, 2, ]),
            'blocks_4': trial.suggest_categorical('blocks_4', [1, 2, ]),
            'grad_clip': trial.suggest_loguniform('grad_clip', 1e-4, 1e1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, ]),
            # "batch_size": tune.choice([32, 64]),
        }
        print(f'Running trial with config:')
        rprint(_config)

        trial_output_dir = output / f'{trial.number}'
        if not trial_output_dir.exists():
            os.mkdir(trial_output_dir)

        checkpoint_callback = ModelCheckpoint(dirpath=str(trial_output_dir))
        rprint('Constructing trainer...')
        checkpoint_callback_best = ModelCheckpoint(
            monitor='test_loss',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{test_loss:.2f}'
        )
        checkpoint_callback_best_rmsd = ModelCheckpoint(
            monitor='rmsd',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{rmsd:.2f}'
        )
        logger = CSVLogger(str(trial_output_dir / 'logs'))

        print(f'Compiling!')
        model = LitBuildScoring(trial_output_dir, _config)
        # model = torch.compile(LitEventScoring(output, _config))
        print('Compiled!')

        trainer = lt.Trainer(
            # devices="auto",
            accelerator="gpu",
            gradient_clip_val=_config['grad_clip'],
            logger=logger,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_best,
                checkpoint_callback_best_rmsd,
                PyTorchLightningPruningCallback(trial, monitor='rmsd'),
                EarlyStopping('rmsd', patience=5)
            ],
            enable_progress_bar=False,
            max_epochs=60
        )

        _train_config = {
            'zarr_path': zarr_path,
        }
        _train_config.update(train_config)
        _train_config.update(_config)
        _train_config.update({'test_train': 'train'})
        dataset_train = DataLoader(
            BuildScoringDataset(
                zarr_path,
                _train_config
            ),
            batch_size=_config['batch_size'],  # batch_size,
            shuffle=True,
            num_workers=19,
            drop_last=True
        )
        rprint(f"Got {len(dataset_train)} training samples")

        _test_config = {
            'zarr_path': zarr_path,
        }
        _test_config.update(test_config)
        _test_config.update(_config)
        _test_config.update({'test_train': 'test'})
        dataset_test = DataLoader(
            BuildScoringDataset(
                zarr_path,
                _test_config
            ),
            batch_size=batch_size,
            num_workers=19,
            drop_last=True
        )
        rprint(f"Got {len(dataset_test)} test samples")

        trainer.fit(model, dataset_train, dataset_test, )
        return trainer.callback_metrics['rmsd'].item()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # Unique identifier of the study.
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=60,
    )
    if output_dir.exists():
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='minimize',
            load_if_exists=True,
            pruner=pruner,
            sampler=TPESampler(constant_liar=True)
        )
        study.enqueue_trial(
            {
                'lr': 1e-2,
                'wd': 1e-4,
                'fraction_background_replace': 0.6603017628273233,
                'xmap_radius': 3.4196300755415834,
                'max_x_blur': 0.509006239862467,
                'max_z_blur': 0.5404658825227494,
                'drop_rate': 0.2740422474916663,
                'planes_1': 8,
                'drop_1': 0.5857699954775857,
                'planes_2': 16,
                'drop_2': 0.55942742430428558,
                'planes_3': 32,
                'drop_3': 0.57119079747990835,
                'planes_4': 32,
                'drop_4': 0.5503045317689869,
                'planes_5': 32,
                'drop_5': 0.53070556902853285,
                'drop_atom_rate': 0.9882224813506627,
                'max_pos_atom_mask_radius': 3.8879501374819583,
                'max_translate': 4.228243978932568,
                'max_x_noise': 1.3199960910884487,
                'max_z_noise': 1.2818494818488133,
                'pos_resample_rate': 1,
                'p_flip': 0.1435919086355283,
                'z_mask_radius': 2.6144817327593346,
                'z_cutoff': 2.324217445782787,
                'combo_layer': 8,
                'blocks_1': 1,
                'blocks_2': 1,
                'blocks_3': 1,
                'blocks_4': 1,
                'grad_clip': 1.5,
                'batch_size': 128,

            },
            skip_if_exists=True
        )
    else:
        print(f'Loading study!')
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name,
            sampler=TPESampler(constant_liar=True),
            pruner=pruner,
        )
    study.optimize(objective, n_trials=300)


if __name__ == "__main__":
    fire.Fire(main)
