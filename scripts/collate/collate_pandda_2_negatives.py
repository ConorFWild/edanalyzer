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

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array, _get_known_hit_poses
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample, BuildAnnotation


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
    fileh = tables.open_file("output/build_data_v2.h5", mode="a")

    # Get the HDF5 root group
    root = fileh.root

    # Create 2 new tables in group1
    rprint(f"Getting or creating table")
    try:
        table_mtz_sample = root.mtz_sample
    except:
        table_mtz_sample = fileh.create_table(root, "pandda_2_mtz_sample", MTZSample, )
    try:
        table_event_map_sample = root.event_map_sample
    except:
        table_event_map_sample = fileh.create_table(root, "pandda_2_mtz_sample", EventMapSample)
    try:
        table_known_hit_pos_sample = root.known_hit_pose
    except:
        table_known_hit_pos_sample = fileh.create_table(root, "pandda_2_mtz_sample", PoseSample, )
    try:
        table_annotation = root.annotation
    except:
        table_annotation = fileh.create_table(root, "pandda_2_annotation", BuildAnnotation, )

    #
    rprint(f"Querying processed events...")
    idx_col_pose = table_known_hit_pos_sample.cols.idx[:]
    if idx_col_pose.size == 0:
        idx_pose = 0
    else:
        idx_pose = int(idx_col_pose.max()) + 1

    idx_col_event = table_event_map_sample.cols.idx[:]
    if idx_col_event.size == 0:
        idx_event = 0
    else:
        idx_event = int(idx_col_event.max()) + 1

    idx_col_annotation = table_annotation.cols.idx[:]
    if idx_col_annotation.size == 0:
        annotation_idx = 0
    else:
        annotation_idx = int(idx_col_annotation.max()) + 1

    #
    mtz_sample = table_mtz_sample.row
    event_map_sample = table_event_map_sample.row
    known_hit_pos_sample = table_known_hit_pos_sample.row
    table_annotation_row = table_annotation.row

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
        high_rank_low_conf = inspect_table[
            (inspect_table['Ligand Confidence'] == "Low") & (inspect_table['z_peak'] > 0.5)]

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
                distance = np.linalg.norm(centroid.flatten() - np.array([x, y, z]))
                # rprint(distance)
                builds[_autobuild_path] = distance

            closest_build_key = min(builds, key=lambda _x: builds[_x])
            if builds[closest_build_key] < 3.0:
                build = gemmi.read_structure(str(closest_build_key))

                #
                known_hit_residue = 'X_0'
                event_idx = idx_event

                # Get the sample transform
                centroid = np.array([x, y, z])
                transform = gemmi.Transform()
                transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

                # Record the 2fofc map sample
                mtz_path = dtag_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=_row['dtag'])
                mtz_grid = _load_xmap_from_mtz_path(mtz_path)
                mtz_sample_array = _sample_xmap_and_scale(mtz_grid, transform,
                                                          np.zeros((90, 90, 90), dtype=np.float32))
                mtz_sample['idx'] = idx_event
                mtz_sample['event_idx'] = event_idx
                mtz_sample['res_id'] = known_hit_residue
                mtz_sample['sample'] = mtz_sample_array
                mtz_sample.append()

                # Record the event map sample
                event_map_path = dtag_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
                    dtag=_row['dtag'],
                    event_idx=_row['event_idx'],
                    bdc=_row['1-BDC']
                )
                event_map_grid = _load_xmap_from_path(event_map_path)
                event_map_sample_array = _sample_xmap_and_scale(event_map_grid, transform,
                                                                np.zeros((90, 90, 90), dtype=np.float32))
                event_map_sample['idx'] = idx_event
                event_map_sample['event_idx'] = event_idx
                event_map_sample['res_id'] = known_hit_residue
                event_map_sample['sample'] = event_map_sample_array
                event_map_sample.append()

                # Get the base event
                poss, elements = _res_to_array(build[0][0][0], )
                com = np.mean(poss, axis=0).reshape((1, 3))
                event_to_lig_com = com - centroid.reshape((1, 3))
                _poss_centered = poss - com
                _rmsd_target = np.copy(_poss_centered) + np.array([22.5, 22.5, 22.5]).reshape(
                    (1, 3)) + event_to_lig_com
                size = min(60, _rmsd_target.shape[0])
                elements_array = np.zeros(60, dtype=np.int32)
                pose_array = np.zeros((60, 3))
                pose_array[:size, :] = _rmsd_target[:size, :]
                elements_array[:size] = elements[:size]
                known_hit_pos_sample['idx'] = idx_pose
                known_hit_pos_sample['database_event_idx'] = event_idx
                known_hit_pos_sample['event_map_sample_idx'] = idx_event
                known_hit_pos_sample['mtz_sample_idx'] = idx_event
                known_hit_pos_sample['positions'] = pose_array
                known_hit_pos_sample['elements'] = elements_array
                known_hit_pos_sample['rmsd'] = 0.0
                known_hit_pos_sample.append()
                idx_pose += 1

                # Generate the decoy/rmsd pairs
                poses, elements, rmsds = _get_known_hit_poses(
                    build[0][0][0],
                    event_to_lig_com
                )

                #
                system = _get_system_from_dtag(_row['dtag'])
                if system in test_systems:
                    partition = 'test'
                else:
                    partition = 'train'

                table_annotation_row['idx'] = annotation_idx
                table_annotation_row['event_map_table_idx'] = idx_event
                table_annotation_row['annotation'] = True
                table_annotation_row['partition'] = partition
                table_annotation_row.append()

                annotation_idx += 1

                # Record the decoy/rmsd pairs with their event map
                for pose, element, rmsd in zip(poses, elements, rmsds):
                    known_hit_pos_sample['idx'] = idx_pose

                    # Record the event key
                    known_hit_pos_sample['database_event_idx'] = event_idx

                    known_hit_pos_sample['event_map_sample_idx'] = idx_event
                    known_hit_pos_sample['mtz_sample_idx'] = idx_event

                    # Record the data
                    known_hit_pos_sample['positions'] = pose
                    known_hit_pos_sample['elements'] = element
                    known_hit_pos_sample['rmsd'] = rmsd
                    known_hit_pos_sample.append()
                    idx_pose += 1

                idx_event += 1

        rprint(f'Got {len(all_builds)} builds for high ranking, low confidence events')
        all_pandda_builds += all_builds

    # rprint(len(all_pandda_builds))

    fileh.close()



if __name__ == "__main__":
    fire.Fire(main)
