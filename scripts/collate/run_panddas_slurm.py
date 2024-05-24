import os
import subprocess
from pathlib import Path

import fire
from rich import print as rprint

DATA_DIRS = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/data')
OUTPUT_DIR = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score')
JOB_SCRIPT_TEMPLATE = (
    '#!/bin/bash\n'
    '#SBATCH --nodes=1\n'
    '#SBATCH --cpus-per-task=20\n'
    '#SBATCH --mem-per-cpu=5120\n'
    '#SBATCH --output=SP100A.o\n'
    '#SBATCH --error=SP100A.e\n'
    '#SBATCH --partition=cs04r\n'

    'source act_conconda activate pandda2_ray python -u /dls/science/groups/i04-1/conor_dev/pandda_2_gemmi/scripts/pandda.py --local_cpus=20 --data_dirs={data_dirs} --out_dir={out_dir}'
)


def _get_data_dir_stats(data_dir):
    dataset_paths = [x for x in data_dir.glob('*')]

    return {
        'num_datasets': len(dataset_paths)
    }


def _make_job_script(data_dir, ):
    job_script = JOB_SCRIPT_TEMPLATE.format(
        data_dirs=data_dir,
        out_dir=OUTPUT_DIR / data_dir.name
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
    for data_dir in DATA_DIRS.glob('*'):
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
        job_script = _make_job_script(data_dir, )
        rprint(f'Jobscript is:\n')
        rprint(job_script)

        # Save the job script
        job_script_path = _save_job_script(job_script, data_dir.name)
        rprint(f'Job script path is: {job_script_path}')
        job_script_paths.append(job_script_path)

        # chmod the job script
        _chmod_job_script(job_script_path)

        ...
    # Submit the job scripts
    # for job_script_path in job_script_paths:
    #     _submit_job_script(job_script_path)

        ...
    ...


if __name__ == '__main__':
    fire.Fire(main)
