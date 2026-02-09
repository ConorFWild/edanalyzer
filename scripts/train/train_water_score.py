import shutil
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony
from pony.orm import db_session, show, select
import numpy as np
import pandas as pd

from edanalyzer.datasets.water_scoring import WaterScoringDataset
from edanalyzer.models.water_scoring import LitWaterScoring
from edanalyzer.data.water_scoring import db, WaterAnnotation

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


def _get_train_config(config, input_data, ):
    # rprint(db)
    # show(WaterAnnotation)
    with db_session:
        WaterAnnotation.select().show()
        test_data_idxs = tuple(x for x in config['test_data_idxs'])
        rprint(test_data_idxs)
        query = select(
            (c.dataIdx, c.landmarkIdx, c.annotation) 
            for c in WaterAnnotation 
            )
        query = select(
            (c.dataIdx, c.landmarkIdx, c.annotation) 
            for c in WaterAnnotation 
            if c.dataIdx not in test_data_idxs
            )
        
        annotation_data = {(x[0], x[1]): x[2] for x in query}

    rprint('annotation data')
    rprint(annotation_data)

    train_data = {}
    for data_idx, data in input_data.items():
        for landmark_idx in data['landmarks']:
            idx = (int(data_idx), int(landmark_idx))
            # rprint(idx)
            if idx in annotation_data:
                train_data[idx] = data
                train_data[idx]['annotation'] = annotation_data[idx]

    rprint('Train Data')
    rprint(train_data)

    return train_data


def _get_test_config(config, input_data, ):
    # Get datasets from the database
    with db_session:
        WaterAnnotation.select().show()
        test_data_idxs = tuple(x for x in config['test_data_idxs'])
        rprint(test_data_idxs)
        query = select(
            (c.dataIdx, c.landmarkIdx, c.annotation) 
            for c in WaterAnnotation 
            )
        query = select(
            (c.dataIdx, c.landmarkIdx, c.annotation) 
            for c in WaterAnnotation 
            if c.dataIdx in test_data_idxs
            )
        
        annotation_data = {(x[0], x[1]): x[2] for x in query}

    rprint('annotation data')
    rprint(annotation_data)

    test_data = {}
    for data_idx, data in input_data.items():
        for landmark_idx in data['landmarks']:
            idx = (int(data_idx), int(landmark_idx))
            # rprint(idx)
            if idx in annotation_data:
                test_data[idx] = data
                test_data[idx]['annotation'] = annotation_data[idx]

    rprint('Test Data')
    rprint(test_data)
    
    return test_data
    

def objective(trial, output=None, train_config=None, test_config=None):
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
        'pos_resample_rate': trial.suggest_int('pos_resample_rate', 0, 50),
        'p_flip': trial.suggest_uniform('p_flip', 0.0, 1.0),
        'z_mask_radius': trial.suggest_uniform('z_mask_radius', 1.0, 3.5),
        'z_cutoff': trial.suggest_uniform('z_cutoff', 1.5, 3.0),
        'blocks_1': trial.suggest_categorical('blocks_1', [1, 2, ]),
        'blocks_2': trial.suggest_categorical('blocks_2', [1, 2, ]),
        'blocks_3': trial.suggest_categorical('blocks_3', [1, 2, ]),
        'blocks_4': trial.suggest_categorical('blocks_4', [1, 2, ]),
        'grad_clip': trial.suggest_loguniform('grad_clip', 1e-4, 1e1),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, ]),
    }

    rprint(f'Running trial with config:')
    rprint(_config)

    # Setup trial output
    trial_output_dir = output / f'{trial.number}'
    if not trial_output_dir.exists():
        os.mkdir(trial_output_dir)
    rprint(f'Outputing trial to:')

    # Setup checkpointing
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

    # Setup csv logger
    logger = CSVLogger(str(trial_output_dir / 'logs'))

    # Create the model
    rprint(f'Compiling model!')
    model = LitWaterScoring(trial_output_dir, _config)
    rprint('Compiled!')

    # Create the trainer
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
            EarlyStopping('rmsd', patience=10)
        ],
        enable_progress_bar=False,
        max_epochs=3000,
        check_val_every_n_epoch=50
    )

    # Create the training dataset
    _train_config = {
    }
    _train_config.update(train_config)
    _train_config.update(_config)
    _train_config.update({'test_train': 'train'})
    dataset_train = DataLoader(
        WaterScoringDataset(
            _train_config
        ),
        batch_size=_train_config['batch_size'],  # batch_size,
        shuffle=True,
        num_workers=_train_config['num_workers'],
        drop_last=True
    )
    rprint(f"Got {len(dataset_train)} training samples")

    # Create the testing Dataset
    _test_config = {
    }
    _test_config.update(test_config)
    _test_config.update(_config)
    _test_config.update({'test_train': 'test'})
    dataset_test = DataLoader(
        WaterScoringDataset(
            _test_config
        ),
        batch_size=_test_config['batch_size'],
        num_workers=_test_config['num_workers'],
        drop_last=True
    )
    rprint(f"Got {len(dataset_test)} test samples")

    # Train the model
    trainer.fit(model, dataset_train, dataset_test, )
    return trainer.callback_metrics['rmsd'].item()


def main(config_path, batch_size=12, num_workers=None):
    rprint(f'Running train training from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the Database
    database_path = Path(config['database_path']) 
    rprint(f'loading database from: {database_path}')
    try:
        db.bind(
            provider='sqlite', 
            filename=str(database_path), 
            create_db=True
            )
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Set output dir
    output_dir = Path(config['output_path'])
    rprint(f'Output dir is: {output_dir}')

    # Load the input data
    import json

    with open(config['input_data_path'], 'r') as f:
        input_data = json.load(f)
        
    rprint(input_data)

    # Get the training and test data
    rprint(f'Getting train/test data...')
    train_config = _get_train_config(config, input_data, )
    test_config = _get_test_config(config, input_data, )

    # Setup study logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Define the study
    study_name = str(config['study_name'])
    output = output_dir / study_name
    if not output.exists():
        os.mkdir(output)

    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2,
        max_resource=3000,
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
                'lr': 0.00011438351645738927, 'wd': 0.004660900954194315,
                'fraction_background_replace': 0.8456736753906298, 'xmap_radius': 4.1308903965340065,
                'max_x_blur': 2.820896567200296, 'max_z_blur': 0.6891269656644818, 'drop_rate': 0.7688958132818271,
                'planes_1': 16, 'drop_1': 0.01720841750409113, 'planes_2': 32, 'drop_2': 0.05160404771554821,
                'planes_3': 64, 'drop_3': 0.003800989203779803, 'planes_4': 128, 'drop_4': 0.05806019760101572,
                'planes_5': 256, 'drop_5': 0.6291779900450685, 'drop_atom_rate': 0.5348397956124168,
                'max_pos_atom_mask_radius': 2.693877997882939, 'max_translate': 3.960735194773154,
                'max_x_noise': 0.276422691457226, 'max_z_noise': 0.0685760724760862, 'pos_resample_rate': 23,
                'p_flip': 0.6278772700882774, 'z_mask_radius': 2.005754066726283, 'z_cutoff': 1.7837797684839956,
                'blocks_1': 2, 'blocks_2': 2, 'blocks_3': 2, 'blocks_4': 1, 'grad_clip': 0.0177383347423051,
                'batch_size': 32

            },
            skip_if_exists=True
        )
        study.enqueue_trial(
            {
                'lr': 0.005983342798442442,
                'wd': 0.012964214431742897,
                'fraction_background_replace': 0.3328440837507466,
                'xmap_radius': 6.508258832375548,
                'max_x_blur': 0.6510617344141543,
                'max_z_blur': 2.7672927212022063, 'drop_rate': 0.18061013518045688,
                'planes_1': 16, 'drop_1': 0.33640773416503167, 'planes_2': 8, 'drop_2': 0.30832338312607055,
                'planes_3': 32, 'drop_3': 0.20010006341557052, 'planes_4': 32, 'drop_4': 0.26025619822558266,
                'planes_5': 64, 'drop_5': 0.37172076026315093, 'drop_atom_rate': 0.10299066057162098,
                'max_pos_atom_mask_radius': 3.0, 'max_translate': 1.5441743449960699,
                'max_x_noise': 0.568124255139961,
                'max_z_noise': 0.5330294541618563, 'pos_resample_rate': 10,
                'p_flip': 0.9358861301233835, 'z_mask_radius': 1.5663307317186794,
                'z_cutoff': 2.3401003871737664, 'blocks_1': 2, 'blocks_2': 1, 'blocks_3': 1, 'blocks_4': 2,
                'grad_clip': 0.03176254272214753, 'batch_size': 64

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

    exit()
    study.optimize(
        lambda _x: objective(
            _x, 
            output=output_dir, 
            train_config=train_config, 
            test_config=test_config,
            ), 
        n_trials=300,
        )


if __name__ == "__main__":
    fire.Fire(main)
