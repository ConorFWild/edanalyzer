from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import pickle
import tables

import numpy as np
import gemmi

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.build_data import BuildAnnotation


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    # custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    # with open(custom_annotations_path, 'rb') as f:
    #     custom_annotations = pickle.load(f)

    #
    working_dir = Path(config['working_directory'])
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")



    #
    pandda_key = config['panddas']['pandda_key'],
    test_systems = config['test']['test_systems']

    result_dir = Path('output') / 'panddas'

    # Loop over panddas
    all_pandda_builds = []
    for pandda_dir in result_dir.glob('*'):
        inspect_table_path = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not inspect_table_path.exists():
            continue

        # Get the inspect table
        inspect_table = pd.read_csv(inspect_table_path)

        # Get the high ranking low conf
        high_rank_low_conf = inspect_table[(inspect_table['Ligand Confidence'] == "Low") & (inspect_table['z_peak'] > 0.5)]

        rprint(f'Got {len(high_rank_low_conf)} high ranking, low confidence events')

        all_builds = []
        for _idx, _row in high_rank_low_conf.iterrows():
            x, y, z = _row['x'], _row['y'], _row['z']

            dtag_dir = pandda_dir / 'processed_datasets' / _row['dtag']
            autobuild_dir = dtag_dir / 'autobuild'
            # rprint(autobuild_dir)

            builds = {}
            for _autobuild_path in autobuild_dir.glob('*'):
                st = gemmi.read_structure(str(_autobuild_path))
                centroid = np.mean(_res_to_array(st[0][0][0])[0], axis=0)
                distance = np.linalg.norm(centroid.flatten() - np.array([x,y,z]))
                # rprint(distance)
                builds[_autobuild_path] = distance


            closest_build_key = min(builds, key=lambda _x: builds[_x])
            if builds[closest_build_key] < 3.0:
                all_builds.append((str(pandda_dir), str(closest_build_key)))

        rprint(f'Got {len(all_builds)} builds for high ranking, low confidence events')
        all_pandda_builds += all_builds

    rprint(len(all_pandda_builds))



if __name__ == "__main__":
    fire.Fire(main)