import subprocess

PANDDA_JOB_SCRIPT = (
    '#!/bin/sh \n'
    'module load buster \n'
    'export PYTHONPATH="" \n'
    'source act_con \n'
    'conda activate pandda2_ray \n'
    'python -u /dls/science/groups/i04-1/conor_dev/pandda_2_gemmi/scripts/pandda.py '
    '--local_cpus={num_cpus} ' 
    '--data_dirs={data_dirs} ' 
    '--out_dir={out_dir} '
    # '--only_datasets="{only_datasets}"'
)

PANDDA_SUBMIT_COMMAND = 'module load global/cluster; qsub -pe smp {num_cpus} -l m_mem_free={m_mem_free}G -o {out_path} -e {err_path} {script_path}'


CHMOD_COMMAND = 'chmod 777 {path}'
def chmod(path):
    p = subprocess.Popen(
        CHMOD_COMMAND.format(path=path),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    p.communicate()

def submit_script(
        script,
        submit_dir,
        script_name="run",
        num_cpus=36,
        m_mem_free=5,
        ):
    job_script_path = submit_dir / f'{script_name}.sh'

    with open(job_script_path, 'w') as f:
        f.write(script)

    chmod(job_script_path)

    submit_command = PANDDA_SUBMIT_COMMAND.format(
        num_cpus=num_cpus,
        m_mem_free=m_mem_free,
        out_path=submit_dir / f"{script_name}.out",
        err_path=submit_dir / f"{script_name}.err",
        script_path=job_script_path
    )

    p = subprocess.Popen(
        submit_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    p.communicate()