from pathlib import Path
import shutil
import time

import fire
from rich import print as rprint
import pandas as pd
import pony
import yaml

from edanalyzer.format import indent_text
from edanalyzer.utils import try_make, try_link
from edanalyzer.shell import PANDDA_JOB_SCRIPT, PANDDA_SUBMIT_COMMAND, submit_script
from edanalyzer import constants
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM


def _run_panddas(config_path, num_cpus=36):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    working_directory = Path(config['working_directory'])
    database_path = working_directory / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}")
        db.generate_mapping()
    except Exception as e:
        print(e)

    n_submitted = 0
    with pony.orm.db_session:
        query_events = pony.orm.select(
            (event, event.annotations, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)
        query = {_x.path: _x for _x in pony.orm.select(
            experiment for experiment in ExperimentORM
        )}

        # Group experiments by system, with their count
        system_experiments = {}
        for _experiment in query.values():
            _experiment_system_name = _experiment.system.name


            try:
                num_datasets = len(
                [x for x in Path(_experiment.model_dir).glob("*")])
            except Exception as e:
                rprint(f'Dataset {_experiment.model_dir} no longer available! Skipping!')
                rprint(e)
                continue
            if not _experiment_system_name in system_experiments:
                system_experiments[_experiment_system_name] = {}
            if num_datasets > 0:
                system_experiments[_experiment_system_name][_experiment.path] = num_datasets

        rprint(f"System Experiments:")
        rprint(system_experiments)

        # Filter to largest experiment of each dataset
        largest_system_experiments = {}
        for _system_name, _system_experiments in system_experiments.items():
            if len(_system_experiments) > 0:
                _largest_experiment_path = max(
                    _system_experiments,
                    key=lambda _x: _system_experiments[_x],
                )
                largest_system_experiments[_system_name] = {
                    "path": _largest_experiment_path,
                    "size": _system_experiments[_largest_experiment_path]
                }
        rprint(f"Largest System Experiments:")
        rprint(largest_system_experiments)

        # Order systems from least datasets of their largest experiment to most for fast results
        sorted_systems = {
            _x: query[largest_system_experiments[_x]['path']]
            for _x
            in sorted(
                largest_system_experiments,
                key=lambda _system: largest_system_experiments[_system]['size'])
        }
        rprint(f"Sorted Systems:")
        rprint(sorted_systems)

        # exit()

        for system_name, experiment in sorted_systems.items():

            if n_submitted

            rprint(f"{system_name}: {experiment.system.name} : {experiment.path}")
            # continue

            model_building_dir = Path(experiment.model_dir)
            result_dir = Path('output') / 'panddas_new_score'
            pandda_dir = result_dir / f"{system_name}"

            # Setup output directories
            try_make(result_dir)
            if not result_dir.exists():
                continue

            if pandda_dir.exists():
                if (pandda_dir / "analyses" / "pandda_analyse_events.csv").exists():
                    continue
                else:
                    shutil.rmtree(pandda_dir)

            # Create the job script
            job_script = PANDDA_JOB_SCRIPT.format(
                num_cpus=num_cpus,
                data_dirs=model_building_dir,
                out_dir=pandda_dir,
                # only_datasets=",".join(experiment_hit_datasets)
            )
            rprint(indent_text(f"Job Script"))
            rprint(indent_text(job_script))
            # exit()

            # Create the submission command
            # submit_script(
            #     job_script,
            #     result_dir,
            #     script_name=f"{system_name}",
            # )

            # Submit the job

            time.sleep(10)


if __name__ == "__main__":
    fire.Fire(_run_panddas)
