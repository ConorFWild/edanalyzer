import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import joblib
import pickle
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import tables
from scipy.spatial.transform import Rotation as R

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample

rng = np.random.default_rng()

rprint(f'Generating small rotations')
time_begin_gen = time.time()
small_rotations = []
identity = np.eye(3)
for j in range(20):
    rotations = R.random(100000)
    rotmat = rotations.as_matrix()
    mask = (rotmat > (0.9 * np.eye(3)))
    diag = mask[:, np.array([0, 1, 2]), np.array([0, 1, 2])]
    rot_mask = diag.sum(axis=1)
    valid_rots = rotmat[rot_mask == 3, :, :]
    rots = [x for x in valid_rots]
    small_rotations += rots
# while len(small_rotations) < 10000:
#     rot = R.random()
#     if np.allclose(rot.as_matrix(), identity, atol=0.1, rtol=0.0):
#         small_rotations.append(rot)
#         rprint(len(small_rotations))
time_finish_gen = time.time()
rprint(f"Generated small rotations in: {round(time_finish_gen - time_begin_gen, 2)}")


def _get_known_hit_poses(
        res,event_to_lig_com,
        centroid=np.array([22.5, 22.5, 22.5]).reshape((1, 3)),
        translation=10,
        num_poses=50
):
    # Get pos array
    poss, elements = _res_to_array(res)

    size = min(60, poss.shape[0])

    elements_array = np.zeros(60, dtype=np.int32)
    elements_array[:size] = elements[:size]

    # Iterate over poses
    poses = []
    rmsds = []
    for cutoff in [0.25, 0.5, 1.0, 2.0, 3.0, 10.0]:
        num_sampled = 0
        translation = cutoff
        rprint(f"Cutoff: {cutoff}")
        while True:
            # Copy the pos array
            _poss = np.copy(poss)

            # Get rotation and translation
            if cutoff <= 0.5:
                rot = R.from_matrix(small_rotations[rng.integers(0, len(small_rotations))])
            else:
                rot = R.random()

            _translation = rng.uniform(-translation / 3, translation / 3, size=3).reshape((1, 3))

            # Cetner
            com = np.mean(_poss, axis=0).reshape((1, 3))
            _poss_centered = _poss - com

            # Get target
            _rmsd_target = np.copy(_poss_centered) + centroid + event_to_lig_com

            # Randomly perturb and reorient

            _rotated_poss = rot.apply(_poss_centered)
            new_com = _translation + centroid + event_to_lig_com
            _new_poss = _rotated_poss + new_com

            # Get RMSD to original
            rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(_rmsd_target - _new_poss, axis=1))) / _new_poss.shape[0])

            if rmsd < cutoff:
                num_sampled += 1
            else:
                continue

            rmsds.append(rmsd)

            # Pad the poss to a uniform size
            pose_array = np.zeros((60, 3))
            pose_array[:size, :] = _new_poss[:size, :]
            poses.append(pose_array)

            if num_sampled >= num_poses:
                break

    return poses, [elements_array] * 6 * num_poses, rmsds


def _get_close_distances(known_hit_centroid,
                         experiment_hit_results):
    distances = {}
    for j, res in enumerate(experiment_hit_results):
        if not [x for x in res[0].annotations][0]:
            continue
        centroid = np.array([res[0].x, res[0].y, res[0].z])

        distance = np.linalg.norm(centroid - known_hit_centroid)
        distances[j] = distance
    return distances


def _get_close_events(
        known_hit_centroid,
        experiment_hit_results,
        delta=5.0
):
    distances = _get_close_distances(known_hit_centroid, experiment_hit_results)

    return [res for res, dis in zip(experiment_hit_results, distances) if dis < delta]


def _get_closest_event(
        known_hit_centroid,
        experiment_hit_results
):
    distances = _get_close_distances(known_hit_centroid, experiment_hit_results)

    return experiment_hit_results[min(distances, key=lambda _j: distances[_j])]


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'rb') as f:
        custom_annotations = pickle.load(f)

    #
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    #
    pandda_key = config['panddas']['pandda_key'],
    test_systems = config['test']['test_systems']

    #
    # Open a file in "w"rite mode
    fileh = tables.open_file("output/build_data_v2.h5", mode="a")

    # Get the HDF5 root group
    root = fileh.root

    # Create 2 new tables in group1
    rprint(f"Getting or creating table")
    try:
        table_mtz_sample = root.mtz_sample
    except:
        table_mtz_sample = fileh.create_table(root, "mtz_sample", MTZSample, )
    try:
        table_event_map_sample = root.event_map_sample
    except:
        table_event_map_sample = fileh.create_table(root, "event_map_sample", EventMapSample)
    try:
        table_known_hit_pos_sample = root.known_hit_pose
    except:
        table_known_hit_pos_sample = fileh.create_table(root, "known_hit_pose", PoseSample, )

    #
    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query_events = pony.orm.select(
            (event, event.annotations, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)
        query = pony.orm.select(
            experiment for experiment in ExperimentORM
        )

        # Order experiments from least datasets to most for fast results
        experiment_num_datasets = {
            _experiment.path: len([x for x in Path(_experiment.model_dir).glob("*")])
            for _experiment
            in query
        }
        sorted_experiments = sorted(query, key=lambda _experiment: experiment_num_datasets[_experiment.path])

        idx_pose = 0
        idx_event = 0

        for experiment in sorted_experiments:
            experiment_hit_results = [res for res in query_events if
                                      ([x for x in res[0].annotations][0].annotation) & (
                                              experiment.path == res[3].path)]
            experiment_hit_datasets = set(
                [
                    experiment_hit_result[0].dtag
                    for experiment_hit_result
                    in experiment_hit_results
                    if
                    (Path(experiment_hit_result[3].model_dir) / experiment_hit_result[0].dtag / 'refine.pdb').exists()
                ]
            )

            if len(experiment_hit_datasets) == 0:
                print(f"No experiment hit results for {experiment.path}. Skipping!")
                continue

            rprint(f"{experiment.system.name} : {experiment.path} : {experiment_num_datasets[experiment.path]}")
            # continue

            model_building_dir = Path(experiment.model_dir)
            # result_dir = model_building_dir / f"../{pandda_key}"
            # pandda_dir = result_dir / "pandda"

            # if not (pandda_dir / constants.PANDDA_ANALYSIS_DIR / 'pandda_analyse_events.csv').exists():
            #     print(f"PanDDA either not finished or errored! Skipping!")
            #     continue

            # Get the known hits structures
            known_hit_structures = _get_known_hit_structures(
                experiment.model_dir,
                experiment_hit_datasets
            )
            print(f"Got {len(known_hit_structures)} known hit structures")

            # Get the known hits
            known_hits = _get_known_hits(known_hit_structures)
            print(f"Got {len(known_hits)} known hits")

            # Get the known hit centroids
            known_hit_centroids: dict[str, dict[str, np.ndarray]] = _get_known_hit_centroids(known_hits)

            #
            mtz_sample = table_mtz_sample.row
            event_map_sample = table_event_map_sample.row
            known_hit_pos_sample = table_known_hit_pos_sample.row

            # if idx_event > 100:
            #     break

            # Get the closest annotated event to the known hit
            for known_hit_dataset in known_hits:
                rprint(f"Got {len(known_hits[known_hit_dataset])} hits in dataset!")
                for known_hit_residue in known_hits[known_hit_dataset]:

                    # Get the associated event
                    # closest_event = _get_closest_event(
                    #     known_hit_centroids[known_hit_dataset][known_hit_residue],
                    #     [x for x in experiment_hit_results if x[0].dtag == known_hit_dataset]
                    # )
                    close_events = _get_close_events(
                        known_hit_centroids[known_hit_dataset][known_hit_residue],
                        [x for x in experiment_hit_results if x[0].dtag == known_hit_dataset],
                    )
                    close_event_ids = [res[0].id for res in close_events]
                    rprint(f"Got {len(close_events)} close events: {close_event_ids}")

                    for _event in close_events:

                        # Get the sample transform
                        centroid = np.array([_event[0].x, _event[0].y, _event[0].z])
                        transform = gemmi.Transform()
                        transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                        transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

                        # Record the 2fofc map sample
                        mtz_grid = _load_xmap_from_mtz_path(_event[0].initial_reflections)
                        mtz_sample_array = _sample_xmap_and_scale(mtz_grid, transform,
                                                                  np.zeros((90, 90, 90), dtype=np.float32))
                        mtz_sample['idx'] = idx_event
                        mtz_sample['event_idx'] = _event[0].id
                        mtz_sample['res_id'] = known_hit_residue
                        mtz_sample['sample'] = mtz_sample_array
                        mtz_sample.append()

                        # Record the event map sample
                        event_map_grid = _load_xmap_from_path(_event[0].event_map)
                        event_map_sample_array = _sample_xmap_and_scale(event_map_grid, transform,
                                                                        np.zeros((90, 90, 90), dtype=np.float32))
                        event_map_sample['idx'] = idx_event
                        event_map_sample['event_idx'] = _event[0].id
                        event_map_sample['res_id'] = known_hit_residue
                        event_map_sample['sample'] = event_map_sample_array
                        event_map_sample.append()

                        # Get the base event
                        poss, elements = _res_to_array(known_hits[known_hit_dataset][known_hit_residue], )
                        com = np.mean(poss, axis=0).reshape((1, 3))
                        event_to_lig_com = com-centroid.reshape((1,3))
                        _poss_centered = poss - com
                        _rmsd_target = np.copy(_poss_centered) + np.array([22.5, 22.5, 22.5]).reshape((1, 3)) + event_to_lig_com
                        size = min(60, _rmsd_target.shape[0])
                        elements_array = np.zeros(60, dtype=np.int32)
                        pose_array = np.zeros((60, 3))
                        pose_array[:size, :] = _rmsd_target[:size, :]
                        elements_array[:size] = elements[:size]
                        known_hit_pos_sample['idx'] = idx_pose
                        known_hit_pos_sample['database_event_idx'] = _event[0].id
                        known_hit_pos_sample['event_map_sample_idx'] = idx_event
                        known_hit_pos_sample['mtz_sample_idx'] = idx_event
                        known_hit_pos_sample['positions'] = pose_array
                        known_hit_pos_sample['elements'] = elements_array
                        known_hit_pos_sample['rmsd'] = 0.0
                        known_hit_pos_sample.append()
                        idx_pose += 1

                        # Generate the decoy/rmsd pairs
                        poses, elements, rmsds = _get_known_hit_poses(
                            known_hits[known_hit_dataset][known_hit_residue],
                            event_to_lig_com
                        )

                        # Record the decoy/rmsd pairs with their event map
                        for pose, element, rmsd in zip(poses, elements, rmsds):
                            known_hit_pos_sample['idx'] = idx_pose

                            # Record the event key
                            known_hit_pos_sample['database_event_idx'] = _event[0].id

                            known_hit_pos_sample['event_map_sample_idx'] = idx_event
                            known_hit_pos_sample['mtz_sample_idx'] = idx_event

                            # Record the data
                            known_hit_pos_sample['positions'] = pose
                            known_hit_pos_sample['elements'] = element
                            known_hit_pos_sample['rmsd'] = rmsd
                            known_hit_pos_sample.append()
                            idx_pose += 1

                        idx_event += 1



    fileh.close()


if __name__ == "__main__":
    fire.Fire(main)
