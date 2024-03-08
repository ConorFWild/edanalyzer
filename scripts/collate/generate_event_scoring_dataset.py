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
from scipy.ndimage import map_coordinates

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array, _get_known_hit_poses, _get_matched_cifs, _get_smiles, \
    _get_atom_ids, _get_connectivity
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM

from pandda_gemmi.event_model.cluster import ClusterDensityDBSCAN
from pandda_gemmi.alignment import Alignment, DFrame
from pandda_gemmi.dataset import XRayDataset, StructureArray, Structure


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
    # inspect_table = pd.read_csv(working_dir / 'build_annotation_pandda_2' / 'analyses' / 'pandda_inspect_events.csv')

    #
    pandda_key = config['panddas']['pandda_key'],
    test_systems = config['test']['test_systems']

    # Construct the data store
    root = zarr.open('output/build_data_v3.zarr', 'w')

    z_map_sample_dtype = [('idx', '<i4'), ('event_idx', '<i4'), ('res_id', 'S32'), ('sample', '<f4', (90, 90, 90))]
    table_event_map_sample = root.create_dataset(
        'event_map_sample',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_dtype
    )

    ligand_data_dtype = [
        ('idx', 'i8'), ('canonical_smiles', '<U300'), ('atom_ids', '<U5', (100,)), ('connectivity', '?', (100, 100,))
    ]
    ligand_data_table = root.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(1,),
        dtype=ligand_data_dtype
    )

    #
    rprint(f"Querying events...")
    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query_events = [_x for _x in pony.orm.select(
            (event, event.annotations, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)]
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

        #
        rprint(f"Querying processed events...")
        idx_z_map = 0
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
                rprint(f"No experiment hit results for {experiment.path}. Skipping!")
                continue

            rprint(f"{experiment.system.name} : {experiment.path} : {experiment_num_datasets[experiment.path]}")

            # Get the known hits structures
            known_hit_structures = _get_known_hit_structures(
                experiment.model_dir,
                experiment_hit_datasets
            )
            rprint(f"Got {len(known_hit_structures)} known hit structures")

            # Get the known hits
            known_hits = _get_known_hits(known_hit_structures)
            rprint(f"Got {len(known_hits)} known hits")

            # Get the known hit centroids
            known_hit_centroids: dict[str, dict[str, np.ndarray]] = _get_known_hit_centroids(known_hits)

            # Get the closest annotated event to the known hit
            for known_hit_dataset in known_hits:
                rprint(f"Got {len(known_hits[known_hit_dataset])} hits in dataset {known_hit_dataset}!")

                # 1. Get the events that are close to any of the fragment hits in the dataset
                # 2. blobfind in the associated z maps
                # 3. match blobs to residues (positives) or be unable to (negatives)

                # Get the close events for each residue
                close_events = []
                for known_hit_residue in known_hits[known_hit_dataset]:
                    # Get the associated event
                    close_events += _get_close_events(
                        known_hit_centroids[known_hit_dataset][known_hit_residue],
                        [x for x in experiment_hit_results if x[0].dtag == known_hit_dataset],
                    )
                rprint(f'Got {len(close_events)} close events')

                # 2. Get the blobs for each zmap
                zblobs = {}
                for event in close_events:
                    dataset = XRayDataset.from_paths(
                        event[0].initial_structure,
                        event[0].initial_reflections,
                        None
                    )
                    zmap = _load_xmap_from_path(event[0].z_map)
                    zmap_array = np.array(zmap, copy=False)
                    # Resample the zmap to the reference frame

                    reference_frame = DFrame(
                        dataset,
                        None,
                    )
                    all_coords = np.argwhere(np.ones(reference_frame.spacing))
                    coordinate_array = all_coords / ( np.array(reference_frame.spacing) / zmap_array.shape )
                    resampling = map_coordinates(
                        np.array(zmap, copy=False,),
                        coordinate_array
                    )
                    rprint(resampling.shape)
                    rprint(reference_frame.spacing)
                    rprint(reference_frame.unit_cell)
                    rprint(zmap)
                    rprint(zmap.unit_cell)
                    events, cutoff = ClusterDensityDBSCAN()(
                        zmap,
                        reference_frame,
                    )
                    zblobs[event[0].id] = {
                        'events': events,
                        'cutoff': cutoff
                    }
                rprint(f'Got {len([y for x in zblobs for y in x])} z blobs')

                # 3. Match blobs to the residues or not
                distances = {}
                for _event_id, _zblobs in zblobs.items():
                    blobs, cutoff = _zblobs['events'], _zblobs['cutoff']
                    for _blob_id, _blob in blobs.items():
                        blob_centroid = _blob.centroid
                        residue_centroid = ...

                        # Get the distance from the blob centroid to residue centroid
                        distances[(_event_id, _blob_id)] = np.linalg.norm(blob_centroid-residue_centroid)


                rprint(f'Got {len(distances)} distances')



                exit()


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
                        atom_ids_array = np.zeros((100,), dtype='<U5')
                        atom_ids = _get_atom_ids(matched_cif)
                        atom_ids_array[:len(atom_ids)] = atom_ids[:len(atom_ids)]
                        connectivity_array = np.zeros((100, 100), dtype='?')
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

                        # Get the sample transform
                        centroid = np.array([_event[0].x, _event[0].y, _event[0].z])
                        transform = gemmi.Transform()
                        transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                        transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

                        # Record the 2fofc map sample
                        mtz_grid = _load_xmap_from_mtz_path(_event[0].initial_reflections)
                        mtz_sample_array = _sample_xmap_and_scale(mtz_grid, transform,
                                                                  np.zeros((90, 90, 90), dtype=np.float32))

                        mtz_sample = np.array(
                            [(
                                idx_event,
                                _event[0].id,
                                known_hit_residue,
                                mtz_sample_array
                            )],
                            dtype=mtz_sample_dtype
                        )

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
                        size = min(100, _rmsd_target.shape[0])
                        atom_array = np.zeros(100, dtype='<U5')
                        elements_array = np.zeros(100, dtype=np.int32)
                        pose_array = np.zeros((100, 3))
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
                        for pose, atom, element, rmsd in zip(poses, atoms, elements, rmsds):
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