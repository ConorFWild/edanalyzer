import os
from pathlib import Path

import logging
import sys

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
import torch

from edanalyzer.datasets.event_scoring import EventScoringDataset
from edanalyzer.models.event_scoring import LitEventScoring
from edanalyzer.data.database_schema import db, EventORM, AutobuildORM

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks import EarlyStopping

# import ray
# from ray.train.lightning import (
#     RayDDPStrategy,
#     RayLightningEnvironment,
#     RayTrainReportCallback,
#     prepare_trainer,
# )
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer
# from ray.tune.search.bayesopt import BayesOptSearch
# from ray.tune.search.ax import AxSearch
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.basic_variant import BasicVariantGenerator

from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
import optuna


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

    print(f'Loading metadata table')
    metadata_table = pd.DataFrame(root[table_type]['z_map_sample_metadata'][:])
    table_annotation = root[table_type]['annotation']
    print(f'Loading annotation table')
    annotation_df = pd.DataFrame(table_annotation[:])
    print(f'Loading ligand table')
    ligand_idx_smiles_df = pd.DataFrame(
        root[table_type]['ligand_data'].get_basic_selection(slice(None), fields=['idx', 'canonical_smiles']))
    print(f'Getting valid smiles...')
    valid_smiles_df = pd.DataFrame(root[table_type]['valid_smiles'][:])
    # print(f'Loading conf table')
    # ligand_conf_df = pd.DataFrame(
    #     root[table_type]['ligand_confs'].get_basic_selection(slice(None), fields=['idx', 'num_heavy_atoms',
    #                                                                               'fragment_canonical_smiles',
    #                                                                               'ligand_canonical_smiles']))
    valid_smiles_mask = valid_smiles_df.iloc[ligand_idx_smiles_df.iloc[metadata_table['ligand_data_idx']]['idx']][
        'valid']

    comments_df = pd.DataFrame(root[table_type]['comments'][:])
    comments = np.array([''] * len(metadata_table), dtype='U250')
    comments[comments_df['meta_idx']] = comments_df['comment']


    print(valid_smiles_mask)
    filter_mask = ~pd.Series(comments).isin(['other', 'sym'])
    train_samples = metadata_table[(annotation_df['partition'] == b'train') & valid_smiles_mask & (filter_mask)]
    test_samples = metadata_table[(annotation_df['partition'] == b'test') & valid_smiles_mask & (filter_mask)]



    pos_z_samples = []
    med_z_samples = []
    neg_z_samples = []
    pos_conf_samples = []
    neg_conf_samples = []

    train_pos_conf = []
    train_med_conf = []
    train_neg_conf = []

    # Loop over the z samples adding the inherent negative samples
    print(f'Getting negative train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'Low'].iterrows():
        neg_z_samples += [z['idx'], ]
        train_neg_conf.append('Low')

    # Loop over the z samples adding positive samples for each
    print(f'Getting positive train samples')
    for _idx, z in train_samples[train_samples['Confidence'] == 'High'].iterrows():  # .sample(len(neg_z_samples),
        #     replace=True).iterrows():
        pos_z_samples += [z['idx'], ]
        train_pos_conf.append('High')

    print(f'Got {len(pos_z_samples)} pos samples!')

    print(f'Got {len(neg_conf_samples)} neg decoy samples!')

    test_pos_z_samples = []
    test_med_z_samples = []
    test_neg_z_samples = []
    test_pos_conf_samples = []
    test_neg_conf_samples = []
    test_pos_conf = []
    test_med_conf = []
    test_neg_conf = []

    # Loop over the z samples adding the test samples
    print(f'Getting test samples')
    for _idx, z in test_samples.iterrows():
        if z['Confidence'] == 'High':
            test_pos_z_samples.append(z['idx'])
            test_pos_conf.append('High')

        elif z['Confidence'] == 'Medium':
            test_med_z_samples.append(z['idx'])
            test_med_conf.append('Medium')

        elif z['Confidence'] == 'Low':
            test_neg_z_samples.append(z['idx'])
            test_neg_conf.append('Low')

    rprint({
        'pos_z_samples len': len(pos_z_samples),
        'med_z_samples len': len(med_z_samples),
        'neg_z_samples len': len(neg_z_samples),
        'pos_conf_samples len': len(pos_conf_samples),
        'neg_conf_samples len': len(neg_conf_samples),
        'train_pos_conf len': len(train_pos_conf),
        'train_neg_conf len': len(train_neg_conf),

    })
    train_idxs = [
        {'z': z, 'conf': conf}
        for z, conf
        in zip(
            pos_z_samples + med_z_samples + neg_z_samples,
            train_pos_conf + train_med_conf + train_neg_conf
        )

    ]
    rprint({
        'test_pos_z_samples len': len(test_pos_z_samples),
        'test_med_z_samples len': len(test_med_z_samples),
        'test_neg_z_samples len': len(test_neg_z_samples),
        'test_pos_conf_samples len': len(test_pos_conf_samples),
        'test_neg_conf_samples len': len(test_neg_conf_samples),
        'test_pos_conf len': len(test_pos_conf),
        'test_neg_conf len': len(test_neg_conf),

    })
    test_idxs = [{'z': z, 'conf': conf} for z, conf
                 in zip(
            test_pos_z_samples + test_med_z_samples + test_neg_z_samples,
            test_pos_conf + test_med_conf + test_neg_conf
        )
                 ]

    res = {}
    for train_test, sample_indexes in zip(['train', 'test'], [train_idxs, test_idxs]):
        res[train_test] = {}
        res[train_test]['indexes'] = sample_indexes

        res[train_test]['pandda_2_annotations'] = {
            _x['event_idx']: _x
            for _x
            in root['pandda_2']['annotation']
        }

        res[train_test]['ligand_data_df'] = ligand_idx_smiles_df

        res[train_test]['metadata_table'] = metadata_table
        sampled_metadata_table = metadata_table.iloc[[x['z'] for x in sample_indexes]]
        res[train_test]['sampled_metadata_table'] = sampled_metadata_table
        res[train_test]['metadata_table_high_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'High']

        res[train_test]['metadata_table_med_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'Medium']

        res[train_test]['metadata_table_low_conf'] = sampled_metadata_table[
            sampled_metadata_table['Confidence'] == 'Low']

        selected_pos_samples = metadata_table.iloc[[x['z'] for x in sample_indexes]]
        selected_smiles = ligand_idx_smiles_df.iloc[selected_pos_samples['ligand_data_idx']]['canonical_smiles']
        print(selected_smiles)
        unique_smiles, smiles_counts = np.unique(selected_smiles, return_counts=True)

        res[train_test]['unique_smiles'] = pd.Series(unique_smiles)
        res[train_test]['unique_smiles_frequencies'] = pd.Series(smiles_counts.astype(float) / np.sum(smiles_counts))
    return res['train'], res['test']


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

    zarr_path = output_dir / 'event_data_4.zarr'

    root = zarr.open(str(zarr_path), mode='r')

    rprint(f'Getting train/test data...')
    train_config, test_config = _get_train_test_idxs_full_conf(root)

    # Get the dataset

    # with pony.orm.db_session:
    #     query = [_x for _x in pony.orm.select(_y for _y in EventORM)]
    rprint(f'Constructing train and test dataloaders...')

    # Get the model
    rprint('Constructing model...')
    study_name = 'event_scoring_no_ligand_prod_0'
    output = output_dir / study_name
    if not output.exists():
        os.mkdir(output)

    # Train
    rprint('Constructing trainer...')

    rprint(f'Training...')

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
            'combo_layer': trial.suggest_categorical('combo_layer', [8, 16, 32, 64, ]),
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
        checkpoint_callback_best_95 = ModelCheckpoint(
            monitor='fpr95',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr95:.2f}'
        )
        checkpoint_callback_best_99 = ModelCheckpoint(
            monitor='fpr99',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr99:.2f}'
        )
        checkpoint_callback_best_10 = ModelCheckpoint(
            monitor='fpr10',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{fpr10:.2f}'
        )
        checkpoint_callback_best_median99 = ModelCheckpoint(
            monitor='medianfpr99',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{medianfpr99:.2f}'
        )
        checkpoint_callback_best_best_scorer_hit = ModelCheckpoint(
            monitor='best_scorer_hit',
            dirpath=str(trial_output_dir),
            filename='sample-mnist-{epoch:02d}-{best_scorer_hit:.2f}'
        )

        logger = CSVLogger(str(trial_output_dir / 'logs'))

        print(f'Compiling!')
        _config.update({'ligand': False})
        model = LitEventScoring(trial_output_dir, _config)
        print('Compiled!')

        trainer = lt.Trainer(
            accelerator="gpu",
            gradient_clip_val=_config['grad_clip'],
            logger=logger,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_best_10,
                checkpoint_callback_best_99,
                checkpoint_callback_best_95,
                checkpoint_callback_best_median99,
                checkpoint_callback_best_best_scorer_hit,
                EarlyStopping('best_scorer_hit', patience=10, mode='max'),
                EarlyStopping('medianfpr99', patience=10, mode='min')
            ],
            enable_progress_bar=False,
            max_epochs=60
        )

        _train_config = {
            'zarr_path': zarr_path,
        }
        _train_config.update(train_config)
        _train_config.update(_config)
        _train_config.update({'test_train': 'train', })
        dataset_train = DataLoader(
            EventScoringDataset(
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
        _test_config.update({'test_train': 'test', })
        dataset_test = DataLoader(
            EventScoringDataset(
                _test_config
            ),
            batch_size=batch_size,
            num_workers=19,
            drop_last=True
        )
        rprint(f"Got {len(dataset_test)} test samples")

        trainer.fit(model, dataset_train, dataset_test, )
        return trainer.callback_metrics['medianfpr99'].item(), trainer.callback_metrics['best_scorer_hit'].item()

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
            # direction='minimize',
            directions=['minimize', 'maximize'],
            load_if_exists=True,
            # pruner=pruner,
            sampler=TPESampler(constant_liar=True)
        )
        study.enqueue_trial(
            {
                'lr': 0.00027608304667883787, 'wd': 0.004428399357109647,
                'fraction_background_replace': 0.9977586581425819,
                'xmap_radius': 5.679885665547368, 'max_x_blur': 0.47417970205607624, 'max_z_blur': 0.6342802782754948,
                'drop_rate': 0.34074819841381004, 'planes_1': 16, 'drop_1': 0.04973841976629942, 'planes_2': 32,
                'drop_2': 0.4773445563051335, 'planes_3': 64, 'drop_3': 0.7220893799410683, 'planes_4': 32,
                'drop_4': 0.42842911756667934,
                'planes_5': 256, 'drop_5': 0.8164371048868642, 'drop_atom_rate': 0.6835634852890703,
                'max_pos_atom_mask_radius': 3.5505538045507197, 'max_translate': 1.504396022687739,
                'max_x_noise': 0.6267668374814633,
                'max_z_noise': 0.9535320617031404, 'pos_resample_rate': 10, 'p_flip': 0.3678479092419647,
                'z_mask_radius': 2.659870974428465, 'z_cutoff': 1.892141827304312, 'combo_layer': 64, 'blocks_1': 2,
                'blocks_2': 2,
                'blocks_3': 1, 'blocks_4': 2, 'grad_clip': 0.0004551500618521706, 'batch_size': 32

            },
            skip_if_exists=True
        )
    else:
        print(f'Loading study!')
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name,
            sampler=TPESampler(constant_liar=True),
            # pruner=pruner,
        )
    study.optimize(objective, n_trials=300)


if __name__ == "__main__":
    fire.Fire(main)
