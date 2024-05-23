import re
from pathlib import Path

import numpy as np
import fire
from rich import print as rprint

OLD_SYSTEMS_PATH = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/analysis/pandda_2/pandda_autobuilding')

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

def main():
    new_systems = []
    for experiment_dir in OLD_SYSTEMS_PATH.glob('*'):
        if not experiment_dir.is_dir():
            continue

        match = re.match('system_(.*)_project', experiment_dir.name)
        if not match:
            continue

        system_name = match.group(1)
        if system_name == 'refmac-from-coot-refmac-for':
            continue
        if system_name in CURRENT_SYSTEMS:
            continue

        # Make output data dir
        data_dir = DATA_DIRS / system_name
        rprint(f'Would make data dir: {data_dir}')

        # Iterate pandda processed datasets
        processed_datasets = experiment_dir / 'processed_datasets'
        for dataset_dir in processed_datasets.glob('*'):
            # Get dtag
            dtag = dataset_dir.name

            # Get the output data dataset dir
            output_dataset_dir = data_dir / dtag
            rprint(f'Would make data dataset dir: {output_dataset_dir}')

            # Find and copy pdb
            pandda_pdb = dataset_dir / f'{dtag}-pandda-input.pdb'
            data_pdb = output_dataset_dir / 'dimple.pdb'
            rprint(f'Would copy pandda pdb {pandda_pdb} -> {data_pdb}')

            # Find and copy mtz
            pandda_mtz = dataset_dir / f'{dtag}-pandda-input.mtz'
            data_mtz = output_dataset_dir / 'dimple.mtz'
            rprint(f'Would copy pandda mtz {pandda_mtz} -> {data_mtz}')

            # Find and copy ligand files
            pandda_compound_dir = dataset_dir / f'ligand_files'
            data_compound_dir =output_dataset_dir / 'compound'
            rprint(f'Would copy pandda compound dir {pandda_compound_dir} -> {data_compound_dir}')

        # new_systems.append(system_name)


    # unique_new_systems = [system for system in np.unique(new_systems) if system not in CURRENT_SYSTEMS]
    # print(unique_new_systems)
    # print(len(unique_new_systems))
    # for system in unique_new_systems:
    #     print(system)

if __name__ == '__main__':
    fire.Fire(main)
