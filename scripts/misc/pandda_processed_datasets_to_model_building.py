import os
import re
import shutil
from pathlib import Path

import numpy as np
import fire
from rich import print as rprint

OLD_SYSTEMS_PATH = Path(
    '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/analysis/pandda_2/pandda_autobuilding')

DATA_DIRS = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/data')

CURRENT_SYSTEMS = [
    'A71EV2A',
    'AA_VNAR_XF01',
    'BAZ2BA',
    'BRD1A',
    'CD44MMA',
    'CDH11',
    'CLIC',
    'D68EV3CPROB',
    'DNV2_NS5A',
    'EBNA1',
    'HPrP',
    'JMJD2DA',
    'KLHL14',
    'LC_TbrB1',
    'MID2A',
    'MUREECA',
    'Mac1',
    'Mpro',
    'NISYPROA',
    'NUDT7A',
    'PHIPA',
    'PTG',
    'PTP1B',
    'PlPro',
    'RECQL5A',
    'SARS2_MproA',
    'SARS2_Nprot',
    'SETDB1',
    'SOCS2A',
    'TARG1',
    'TRF1_2',
    'TbrB1',
    'TcHRS',
    'Tif6',
    'VP40',
    'Zika_NS5A',
    'SP100A',
    'FALZA'
]

FORBIDDEN = [
    'KLHL7B'
]


def process_dataset(system_name, experiment_dir):
    # Make output data dir
    data_dir = DATA_DIRS / system_name
    if not data_dir.exists():
        rprint(f'Making data dir: {data_dir}')
        try:
            os.mkdir(data_dir)
        except Exception as e:
            print(e)
    else:
        rprint(f'Already made: {data_dir}')

    # Iterate pandda processed datasets
    processed_datasets = experiment_dir / 'processed_datasets'
    for dataset_dir in processed_datasets.glob('*'):
        # Get dtag
        dtag = dataset_dir.name

        # Get the output data dataset dir
        output_dataset_dir = data_dir / dtag
        if not output_dataset_dir.exists():
            rprint(f'Making data dataset dir: {output_dataset_dir}')
            try:
                os.mkdir(output_dataset_dir)
            except Exception as e:
                print(e)
        else:
            rprint(f'Already made dataset dir: {output_dataset_dir}')

        # Find and copy pdb
        pandda_pdb = dataset_dir / f'{dtag}-pandda-input.pdb'
        data_pdb = output_dataset_dir / 'dimple.pdb'
        if (not data_pdb.exists()) & (pandda_pdb.exists()):
            rprint(f'Copying pandda pdb {pandda_pdb} -> {data_pdb}')
            try:
                shutil.copy(pandda_pdb, data_pdb, follow_symlinks=True)
            except Exception as e:
                print(e)
        else:
            rprint(f'Already copied {data_pdb}')

        # Find and copy mtz
        pandda_mtz = dataset_dir / f'{dtag}-pandda-input.mtz'
        data_mtz = output_dataset_dir / 'dimple.mtz'
        rprint(f'Copying pandda mtz {pandda_mtz} -> {data_mtz}')
        if (not data_mtz.exists()) & (pandda_mtz.exists()):
            try:
                shutil.copy(pandda_mtz, data_mtz, follow_symlinks=True)
            except Exception as e:
                print(e)
        else:
            rprint(f'Already copied {data_mtz}')

        # Find and copy ligand files
        pandda_compound_dir = dataset_dir / f'ligand_files'
        data_compound_dir = output_dataset_dir / 'compound'
        if (not data_compound_dir.exists()) & (pandda_compound_dir.exists()):
            rprint(f'Copying pandda compound dir {pandda_compound_dir} -> {data_compound_dir}')
            try:
                shutil.copytree(pandda_compound_dir, data_compound_dir, symlinks=False)
            except Exception as e:
                print(e)
        else:
            rprint(f'Already copied {data_mtz}')


def main():
    new_systems = {}
    datasets_to_process = []
    for experiment_dir in OLD_SYSTEMS_PATH.glob('*'):
        if not experiment_dir.is_dir():
            continue

        match = re.match('system_(.*)_project_(.*)', experiment_dir.name)
        if not match:
            continue

        system_name = match.group(1)
        project = match.group(2)
        new_systems[project] = system_name
        if system_name == 'refmac-from-coot-refmac-for':
            continue
        if system_name in CURRENT_SYSTEMS:
            continue

        if system_name in FORBIDDEN:
            continue

        datasets_to_process.append([system_name, experiment_dir])

        # new_systems.append(system_name)

    from joblib import Parallel, delayed
    Parallel(n_jobs=20, verbose=10)(
        delayed(process_dataset)(system_name, experiment_dir) for system_name, experiment_dir in datasets_to_process)

    # unique_new_systems = [system for system in np.unique(new_systems) if system not in CURRENT_SYSTEMS]
    # print(unique_new_systems)
    # print(len(unique_new_systems))
    # for system in unique_new_systems:
    #     print(system)
    # for project, system in new_systems.items():
    #     print(f'{project} : {system}')


if __name__ == '__main__':
    fire.Fire(main)
