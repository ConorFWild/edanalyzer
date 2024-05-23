import os
import shutil
from pathlib import Path

import fire


def main():
    path = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/downloads/SP100A/data')

    for dataset_dir in path.glob('*'):
        compound_dir = dataset_dir / 'compound'
        try:
            os.mkdir(compound_dir)
        except Exception as e:
            print(e)
        try:
            cif_src = dataset_dir / 'ligand.cif'
            cif_dst = compound_dir / 'ligand.cif'
            shutil.move(cif_src, cif_dst)
        except Exception as e:
            print(e)
        try:
            pdb_src = dataset_dir / 'ligand.pdb'
            pdb_dst = compound_dir / 'ligand.pdb'
            shutil.move(pdb_src, pdb_dst)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    fire.Fire(main)
