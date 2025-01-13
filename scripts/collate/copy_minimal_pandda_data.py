import os
import shutil

import fire
from pathlib import Path



DATA_DIRS = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/data')


SYSTEMS_TO_COPY =  {'ActA': '/dls/labxchem/data/lb25897/lb25897-22/processing/analysis/model_building',
 'LM02': '/dls/labxchem/data/lb29209/lb29209-7/processing/analysis/model_building',
 'TRIM16A': '/dls/labxchem/data/lb29658/lb29658-48/processing/analysis/model_building',
 'p53': '/dls/labxchem/data/lb29658/lb29658-51/processing/analysis/model_building',
 'TRIM2A': '/dls/labxchem/data/lb29658/lb29658-52/processing/analysis/model_building',
 'TRIM21': '/dls/labxchem/data/lb29658/lb29658-57/processing/analysis/model_building',
 'SHOC2': '/dls/labxchem/data/lb37031/lb37031-2/processing/analysis/model_building',
 'INPP5DA': '/dls/labxchem/data/lb30602/lb30602-56/processing/analysis/model_building',
 'KLHL17': '/dls/labxchem/data/lb30602/lb30602-58/processing/analysis/model_building',
 'CAU-1': '/dls/labxchem/data/lb31306/lb31306-16/processing/analysis/model_building',
 'SOS2SOS3': '/dls/labxchem/data/lb31306/lb31306-44/processing/analysis/model_building',
 'BpoC': '/dls/labxchem/data/lb32958/lb32958-4/processing/analysis/model_building',
 'PrpB': '/dls/labxchem/data/lb34721/lb34721-1/processing/analysis/model_building',
 'LYSRSCPZ': '/dls/labxchem/data/lb36049/lb36049-7/processing/analysis/model_building',
 'MmaA1-3': '/dls/labxchem/data/lb36493/lb36493-1/processing/analysis/model_building',
 'LmUbC4': '/dls/labxchem/data/lb36570/lb36570-1/processing/analysis/model_building',
 'OLF': '/dls/labxchem/data/lb36923/lb36923-1/processing/analysis/model_building',
 'PCNA': '/dls/labxchem/data/lb38129/lb38129-1/processing/analysis/model_building'}

def main():
    for system, model_building_dir in SYSTEMS_TO_COPY.items():
        print(f'Copying system: {system}')
        output_dir = DATA_DIRS / system

        if not output_dir.exists():
            os.mkdir(output_dir)
        else:
            continue

        for dtag_dir in Path(model_building_dir).glob('*'):
            if not dtag_dir.is_dir():
                print(f'Not a dir! Skipping!')
                continue

            if not (dtag_dir / 'dimple.pdb').exists():
                print(f'No data! Skipping!')
                continue

            dtag = dtag_dir.name
            print(f'Copying dataset: {dtag}')
            dtag_output_dir = output_dir / dtag

            if not dtag_output_dir.exists():
                os.mkdir(dtag_output_dir)
            else:
                continue

            # Copy pdb, mtz and ligand files
            pdb_file_name = 'dimple.pdb'
            shutil.copyfile(
                dtag_dir / pdb_file_name,
                dtag_output_dir / pdb_file_name
            )

            mtz_file_name = 'dimple.mtz'
            shutil.copyfile(
                dtag_dir / mtz_file_name,
                dtag_output_dir / mtz_file_name
            )

            compound_dir_name = 'compound'
            shutil.copytree(
                dtag_dir / compound_dir_name,
                dtag_output_dir / compound_dir_name
            )


if __name__ == "__main__":
    fire.Fire(main)