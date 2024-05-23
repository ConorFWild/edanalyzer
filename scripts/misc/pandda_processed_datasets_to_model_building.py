import re
from pathlib import Path

import numpy as np
import fire

OLD_SYSTEMS_PATH = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/analysis/pandda_2/pandda_autobuilding')

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
        new_systems.append(system_name)


    print([system for system in np.unique(new_systems) if system not in CURRENT_SYSTEMS])

if __name__ == '__main__':
    fire.Fire(main)
