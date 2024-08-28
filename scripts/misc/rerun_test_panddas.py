import os
import subprocess
from pathlib import Path

import fire
from rich import print as rprint
import yaml
import pandas as pd

TEST_SYSTEMS = [
    "BAZ2BA"
    "MRE11AA"
    "Zika_NS3A"
    "DCLRE1CA",
    "GluN1N2A",
    "Tif6",
    "AURA",
    "A71EV2A",
    "SETDB1",
    "JMJD2DA",
    "PTP1B",
    "BRD1A",
    "PP1",
    "PWWP_C64S",
    "NS3Hel",
    "NSP16",
]
# DATA_DIRS = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/data')
DATA_DIRS = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score')
OUTPUT_DIR = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/test_systems_panddas')
JOB_SCRIPT_TEMPLATE = (
    '#!/bin/bash\n'
    '#SBATCH --nodes=1\n'
    '#SBATCH --cpus-per-task=20\n'
    '#SBATCH --mem-per-cpu=5120\n'
    '#SBATCH --output={output}\n'
    '#SBATCH --error={error}\n'
    '#SBATCH --partition=cs05r\n'

    # 'source act_con\n'
    'source /dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/act\n'
    'conda activate pandda2_ray\n' 
    'python -u /dls/science/groups/i04-1/conor_dev/pandda_2_gemmi/scripts/pandda.py --local_cpus=20 --data_dirs={data_dirs} --out_dir={out_dir} --only_datasets={only_datasets}'
)


def _get_high_conf_datasets(pandda_dir):
    inspect_table = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
    df = pd.read_csv(inspect_table)
    high_conf_datasets = df[df['Ligand Confidence'] == "High"]
    return [x for x in high_conf_datasets['dtag']]

def _get_data_dir(pandda_dir):
    with open(pandda_dir / 'input.yaml', 'r') as f:
        dic = yaml.safe_load(f)

    datasets = dic['Datasets']
    for key, data in datasets.items():
        pdb = data['Files']['PDB']
        if pdb is not None:
            pdb_path = Path(pdb)
            data_dir = pdb_path.parent.parent
            return data_dir

def _get_data_dir_stats(data_dir):
    dataset_paths = [x for x in data_dir.glob('*')]

    return {
        'num_datasets': len(dataset_paths)
    }


def _make_job_script(data_dir, high_conf_datasets):
    job_script = JOB_SCRIPT_TEMPLATE.format(
        output=OUTPUT_DIR / f'{data_dir.name}.o',
        error=OUTPUT_DIR / f'{data_dir.name}.e',
        data_dirs=data_dir,
        out_dir=OUTPUT_DIR / data_dir.name,
        only_datasets=high_conf_datasets
    )
    return job_script


def _save_job_script(job_script, name):
    job_script_path = OUTPUT_DIR / f'{name}.slurm'
    with open(job_script_path, 'w') as f:
        f.write(job_script)

    return job_script_path


def _chmod_job_script(job_script_path):
    os.chmod(job_script_path, 0o755)


def _submit_job_script(job_script_path):
    p = subprocess.Popen(
        f'sbatch {job_script_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = p.communicate()
    return str(stdout), str(stderr)

def main():
    # Iterate over data dirs
    job_script_paths = []
    for pandda_dir in sorted(DATA_DIRS.glob('*')):
        rprint(f'PanDDA dir is: {pandda_dir.name}')

        if pandda_dir.name not in TEST_SYSTEMS:
            rprint(f'{pandda_dir.name} not a test system! Skipping!')

        # Get the high confidence datasets
        high_conf_datasets = _get_high_conf_datasets(pandda_dir)

        # Get the corresponding data dir
        data_dir = _get_data_dir(pandda_dir)
        rprint(f'Dataset dir is: {data_dir.name}')

        # Check Integrity
        data_dir_stats = _get_data_dir_stats(data_dir)
        rprint(f'Dataset stats are: {data_dir_stats}')

        # Skip if too few datasets
        if data_dir_stats['num_datasets'] < 60:
            rprint(f'### Too few datasets! Skipping!')
            continue

        if (OUTPUT_DIR / data_dir.name).exists():
            rprint(f'### Already processed! Skipping!')
            continue

        # Generate a job script
        job_script = _make_job_script(data_dir, high_conf_datasets)
        rprint(f'Jobscript is:\n')
        rprint(job_script)
        raise Exception

        # Save the job script
        job_script_path = _save_job_script(job_script, data_dir.name)
        rprint(f'Job script path is: {job_script_path}')
        job_script_paths.append(job_script_path)

        # chmod the job script
        _chmod_job_script(job_script_path)

        ...
    # Submit the job scripts
    for job_script_path in job_script_paths:
        stdout, stderr = _submit_job_script(job_script_path)
        rprint(stdout)
        rprint(stderr)
        ...
    ...


if __name__ == '__main__':
    fire.Fire(main)
