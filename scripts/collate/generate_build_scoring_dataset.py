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
import zarr

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array, _get_known_hit_poses, _get_matched_cifs, _get_smiles, \
    _get_atom_ids, _get_connectivity
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample


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

    working_dir = Path(config['working_directory'])
    inspect_table = pd.read_csv(working_dir / 'build_annotation_pandda_2' / 'analyses' / 'pandda_inspect_events.csv')

    #
    pandda_key = config['panddas']['pandda_key'],
    test_systems = config['test']['test_systems']

    #
    # # Open a file in "w"rite mode
    # fileh = tables.open_file("output/build_data_v2.h5", mode="a")
    #
    # # Get the HDF5 root group
    # root = fileh.root
    #
    # # Create 2 new tables in group1
    # rprint(f"Getting or creating table")
    # try:
    #     table_mtz_sample = root.mtz_sample
    # except:
    #     table_mtz_sample = fileh.create_table(root, "mtz_sample", MTZSample, )
    # try:
    #     table_event_map_sample = root.event_map_sample
    # except:
    #     table_event_map_sample = fileh.create_table(root, "event_map_sample", EventMapSample)
    # try:
    #     table_known_hit_pos_sample = root.known_hit_pose
    # except:
    #     table_known_hit_pos_sample = fileh.create_table(root, "known_hit_pose", PoseSample, )

    root = zarr.open('output/build_data_v3.zarr', 'w')
    mtz_sample_dtype = [('idx', '<i4'), ('event_idx', '<i4'), ('res_id', 'S32'), ('sample', '<f4', (90, 90, 90))]
    table_mtz_sample = root.create_dataset(
        'mtz_sample',
        shape=(0,),
        chunks=(20,),
        dtype=mtz_sample_dtype
    )
    event_map_sample_dtype = [('idx', '<i4'), ('event_idx', '<i4'), ('res_id', 'S32'), ('sample', '<f4', (90, 90, 90))]
    table_event_map_sample = root.create_dataset(
        'event_map_sample',
        shape=(0,),
        chunks=(20,),
        dtype=event_map_sample_dtype
    )
    known_hit_pos_sample_dtype = [
        ('idx', '<i4'),
        ('database_event_idx', '<i4'),
        ('event_map_sample_idx', '<i4'),
        ('mtz_sample_idx', '<i4'),
        ('positions', '<f4', (60, 3)),
        ('atoms', '<U5', (60,)),
        ('elements', '<i4', (60,)),
        ('rmsd', '<f4')]
    table_known_hit_pos_sample = root.create_dataset(
        'known_hit_pose',
        shape=(0,),
        chunks=(20,),
        dtype=known_hit_pos_sample_dtype
    )
    delta_dtype = [

        ('idx', '<i4'),
        ('pose_idx', '<i4'),
        ('delta', '<f4', (60,)),
        ('delta_vec', '<f4', (60, 3)),

    ]
    delta_table = root.create_dataset(
        'delta',
        shape=(0,),
        chunks=(20,),
        dtype=delta_dtype
    )
    annotation_dtype = [

        ('idx', '<i4'),
        ('event_map_table_idx', '<i4'),
        ('annotation', '?'),
        ('partition', 'S32')]
    annotation_table = root.create_dataset(
        'annotation',
        shape=(0,),
        chunks=(20,),
        dtype=annotation_dtype

    )
    ligand_data_dtype = [
        ('idx', 'i8'), ('canonical_smiles', '<U300'), ('atom_ids', '<U5', (60,)), ('connectivity', '?', (60, 60,))
    ]
    ligand_data_table = root.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(20,),
        dtype=ligand_data_dtype
    )

    #
    rprint(f"Querying events...")
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

        rprint(f"Querying processed events...")
        # idx_col_pose = table_known_hit_pos_sample.cols.idx[:]
        # idx_pose = int(idx_col_pose.max()) + 1
        # idx_col_event = table_event_map_sample.cols.idx[:]
        # idx_event = int(idx_col_event.max()) + 1
        # processed_event_idxs = table_event_map_sample.cols.event_idx[:]
        idx_pose = 0
        idx_event = 0
        processed_event_idxs = []

        for experiment in sorted_experiments:
            rprint(f"Processing experiment: {experiment.path}")
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
            # mtz_sample = table_mtz_sample.row
            # event_map_sample = table_event_map_sample.row
            # known_hit_pos_sample = table_known_hit_pos_sample.row

            # if idx_event > 100:
            #     break

            # Get the closest annotated event to the known hit
            for known_hit_dataset in known_hits:
                rprint(f"Got {len(known_hits[known_hit_dataset])} hits in dataset {known_hit_dataset}!")
                for known_hit_residue in known_hits[known_hit_dataset]:

                    # Get the associated event
                    close_events = _get_close_events(
                        known_hit_centroids[known_hit_dataset][known_hit_residue],
                        [x for x in experiment_hit_results if x[0].dtag == known_hit_dataset],
                    )
                    close_event_ids = [res[0].id for res in close_events]
                    rprint(f"Got {len(close_events)} close events: {close_event_ids}")

                    for _event in close_events:
                        if _event[0].id in processed_event_idxs:
                            rprint(f"Already generated poses for: {_event[0].id}! Skipping!")
                            continue

                        # Get the associated ligand data
                        matched_cifs = _get_matched_cifs(
                            known_hits[known_hit_dataset][known_hit_residue],
                            _event[0],
                        )
                        if len(matched_cifs) == 0:
                            rprint(f'NO MATCHED LIGAND DATA!!!!!!')
                            continue

                        matched_cif = matched_cifs[0]

                        smiles = _get_smiles(matched_cif)
                        atom_ids_array = np.zeros((60,), dtype='<U5')
                        atom_ids = _get_atom_ids(matched_cif)
                        atom_ids_array[:len(atom_ids)] = atom_ids[:len(atom_ids)]
                        connectivity_array = np.zeros((60, 60), dtype='?')
                        connectivity = _get_connectivity(matched_cif)
                        connectivity_array[:connectivity.shape[0], :connectivity.shape[1]] = connectivity[
                                                                                             :connectivity.shape[0],
                                                                                             :connectivity.shape[1]]

                        ligand_data_sample = np.array([(
                            idx_event,
                            smiles,
                            atom_ids_array,
                            connectivity_array
                        )], dtype=ligand_data_dtype)
                        # rprint(ligand_data_sample)
                        ligand_data_table.append(
                            ligand_data_sample
                        )

                        # Check for an annotation
                        # event_annotation_table = inspect_table[inspect_table['dtag'] == _event[0].id]
                        # if len(event_annotation_table) == 0:
                        #     annotation_data = ()
                        # else:
                        #     annotation_row = event_annotation_table.iloc[0]
                        #     annotation = annotation_row['Ligand Confidence']
                        #     if annotation == "High":
                        #         annotation_bool = True
                        #     else:
                        #         annotation_bool = False
                        #     if _event[0].pandda.system.name in test_systems:
                        #         partition = 'test'
                        #     else:
                        #         partition = 'train'
                        #     annotation_data = (
                        #         idx_event,
                        #         idx_event,
                        #         annotation_bool,
                        #         partition
                        #     )
                        # rprint(annotation_data)
                        # annotation_table.append(
                        #     annotation_data
                        # )

                        # Get the sample transform
                        centroid = np.array([_event[0].x, _event[0].y, _event[0].z])
                        transform = gemmi.Transform()
                        transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                        transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

                        # Record the 2fofc map sample
                        mtz_grid = _load_xmap_from_mtz_path(_event[0].initial_reflections)
                        mtz_sample_array = _sample_xmap_and_scale(mtz_grid, transform,
                                                                  np.zeros((90, 90, 90), dtype=np.float32))
                        # mtz_sample['idx'] = idx_event
                        # mtz_sample['event_idx'] = _event[0].id
                        # mtz_sample['res_id'] = known_hit_residue
                        # mtz_sample['sample'] = mtz_sample_array
                        mtz_sample = np.array(
                            [(
                                idx_event,
                                _event[0].id,
                                known_hit_residue,
                                mtz_sample_array
                            )],
                            dtype=mtz_sample_dtype
                        )
                        # rprint(table_mtz_sample)
                        # rprint(mtz_sample)
                        table_mtz_sample.append(
                            mtz_sample
                        )

                        # Record the event map sample
                        event_map_grid = _load_xmap_from_path(_event[0].event_map)
                        event_map_sample_array = _sample_xmap_and_scale(event_map_grid, transform,
                                                                        np.zeros((90, 90, 90), dtype=np.float32))
                        # event_map_sample['idx'] = idx_event
                        # event_map_sample['event_idx'] = _event[0].id
                        # event_map_sample['res_id'] = known_hit_residue
                        # event_map_sample['sample'] = event_map_sample_array
                        event_map_sample = np.array(
                            [(
                                idx_event,
                                _event[0].id,
                                known_hit_residue,
                                event_map_sample_array
                            )], dtype=event_map_sample_dtype
                        )
                        # rprint(event_map_sample)
                        table_event_map_sample.append(
                            event_map_sample
                        )

                        # Get the base event
                        poss, atom, elements = _res_to_array(known_hits[known_hit_dataset][known_hit_residue], )
                        com = np.mean(poss, axis=0).reshape((1, 3))
                        event_to_lig_com = com - centroid.reshape((1, 3))
                        _poss_centered = poss - com
                        _rmsd_target = np.copy(_poss_centered) + np.array([22.5, 22.5, 22.5]).reshape(
                            (1, 3)) + event_to_lig_com
                        size = min(60, _rmsd_target.shape[0])
                        atom_array = np.zeros(60, dtype='<U5')
                        elements_array = np.zeros(60, dtype=np.int32)
                        pose_array = np.zeros((60, 3))
                        pose_array[:size, :] = _rmsd_target[:size, :]
                        atom_array[:size] = atom[:size]
                        elements_array[:size] = elements[:size]
                        # known_hit_pos_sample['idx'] = idx_pose
                        # known_hit_pos_sample['database_event_idx'] = _event[0].id
                        # known_hit_pos_sample['event_map_sample_idx'] = idx_event
                        # known_hit_pos_sample['mtz_sample_idx'] = idx_event
                        # known_hit_pos_sample['positions'] = pose_array
                        # known_hit_pos_sample['elements'] = elements_array
                        # known_hit_pos_sample['rmsd'] = 0.0
                        known_hit_pos_sample = np.array([(
                            idx_pose,
                            _event[0].id,
                            idx_event,
                            idx_event,
                            pose_array,
                            atom_array,
                            elements_array,
                            0.0
                        )], dtype=known_hit_pos_sample_dtype
                        )
                        # rprint(known_hit_pos_sample)
                        table_known_hit_pos_sample.append(
                            known_hit_pos_sample
                        )
                        idx_pose += 1

                        # Generate the decoy/rmsd pairs
                        poses, atoms, elements, rmsds = _get_known_hit_poses(
                            known_hits[known_hit_dataset][known_hit_residue],
                            event_to_lig_com
                        )

                        # Record the decoy/rmsd pairs with their event map
                        for pose, atom, element, rmsd in zip(poses, atom, elements, rmsds):
                            # known_hit_pos_sample['idx'] = idx_pose

                            # Record the event key
                            # known_hit_pos_sample['database_event_idx'] = _event[0].id
                            #
                            # known_hit_pos_sample['event_map_sample_idx'] = idx_event
                            # known_hit_pos_sample['mtz_sample_idx'] = idx_event
                            #
                            # # Record the data
                            # known_hit_pos_sample['positions'] = pose
                            #
                            # known_hit_pos_sample['atoms'] = atom
                            # known_hit_pos_sample['elements'] = element
                            # known_hit_pos_sample['rmsd'] = rmsd
                            # rprint((idx_pose,
                            #             _event[0].id,
                            #             idx_event,
                            #             idx_event,
                            #             pose,
                            #             atom,
                            #             element,
                            #             rmsd))
                            known_hit_pos_sample = np.array(
                                [
                                    (
                                        idx_pose,
                                        _event[0].id,
                                        idx_event,
                                        idx_event,
                                        pose,
                                        atom,
                                        element,
                                        rmsd
                                    )
                                ],
                                dtype=known_hit_pos_sample_dtype
                            )
                            # rprint(known_hit_pos_sample)
                            table_known_hit_pos_sample.append(
                                known_hit_pos_sample
                            )

                            _delta_vecs = pose_array - pose
                            _delta = np.linalg.norm(_delta_vecs, axis=1)
                            delta_sample = np.array([(
                                idx_pose,
                                idx_pose,
                                _delta,
                                _delta_vecs,
                            )], dtype=delta_dtype
                            )
                            # rprint(delta_sample)
                            delta_table.append(
                                delta_sample
                            )

                            # known_hit_pos_sample.append()
                            idx_pose += 1
                            # exit()

                        idx_event += 1


if __name__ == "__main__":
    fire.Fire(main)
