import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import random

pd.set_option('display.max_rows', 500)
import pony
import joblib
import pickle
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import tables
import zarr
from numcodecs import Blosc, Delta
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator


from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale, \
    _get_ligand_mask_float
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
    # rprint(distances)
    close_events = []
    for j, dist in distances.items():
        if dist < delta:
            close_events.append(experiment_hit_results[j])

    return close_events


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
    root = zarr.open('output/event_data.zarr', 'w')

    z_map_sample_metadata_dtype = [
        ('idx', '<i4'),
        ('event_idx', '<i4'),
        ('res_id', '<U32'),
        ('ligand_data_idx', 'i8'),
        ('pose_data_idx', 'i8')
    ]
    table_z_map_sample_metadata = root.create_dataset(
        'z_map_sample_metadata',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_metadata_dtype
    )

    z_map_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]
    table_z_map_sample = root.create_dataset(
        'z_map_sample',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_dtype,
        compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
    )
    # z_map_sample_dtype = '<f4'
    # table_z_map_sample = root.create_dataset(
    #     'z_map_sample',
    #     shape=(0,90,90,90),
    #     chunks=(1,None, None, None),
    #     dtype=z_map_sample_dtype,
    #     compressor=Blosc(cname='zlib', clevel=9, shuffle=Blosc.SHUFFLE),
    #     filters=[Delta(dtype=z_map_sample_dtype)]
    # )

    ligand_data_dtype = [
        ('idx', 'i8'),
        ('num_heavy_atoms', 'i8'),
        ('canonical_smiles', '<U300'),
        ('atom_ids', '<U5', (100,)),
        ('connectivity', '?', (100, 100,))
    ]
    ligand_data_table = root.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(1,),
        dtype=ligand_data_dtype
    )

    known_hit_pose_sample_dtype = [
        ('idx', '<i8'),
        ('positions', '<f4', (100, 3)),
        ('atoms', '<U5', (100,)),
        ('elements', '<i4', (100,)),
        # ('rmsd', '<f4')
    ]
    table_known_hit_pose_sample = root.create_dataset(
        'known_hit_pose',
        shape=(0,),
        chunks=(1,),
        dtype=known_hit_pose_sample_dtype
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
        idx_ligand_data = 0
        idx_pose = 0
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
            # 1. Get the events that are close to any of the fragment hits in the dataset
            # 1.5 Get the associated ligand data
            # 2. blobfind in the associated z maps
            # 3. Get distances from event blobs to residues
            # 4. match blobs to residues (positives) or be unable to (negatives)
            for known_hit_dataset in known_hits:
                # if (known_hit_dataset not in ['CLIC-x0191', 'CLIC-x0132']):
                #     continue

                rprint(f"Got {len(known_hits[known_hit_dataset])} hits in dataset {known_hit_dataset}!")
                if len(known_hits[known_hit_dataset]) == 0:
                    rprint(f"SKIPPING!")
                    continue

                # 1. Get the close events for each residue
                close_events = {}
                for known_hit_residue in known_hits[known_hit_dataset]:
                    # Get the associated event
                    _close_events = _get_close_events(
                        known_hit_centroids[known_hit_dataset][known_hit_residue],
                        [x for x in experiment_hit_results if x[0].dtag == known_hit_dataset],
                    )
                    for _close_event in _close_events:
                        close_events[(known_hit_residue, _close_event[0].id)] = _close_event
                rprint(f'Got {len(close_events)} close events')
                # if len(close_events) != 2:
                #     continue
                close_event_dict = {close_event[0].id: close_event for close_event in close_events.values()}

                # Get the base event


                # Get the ligand data for each of the events found
                ligand_data = {}
                tmp_idx_ligand_data = idx_ligand_data
                for known_hit_residue in known_hits[known_hit_dataset]:
                    matched_cifs = []
                    for (_resid, _event_id), _event in close_events.items():
                        if _resid != known_hit_residue:
                            continue
                        # Get the associated ligand data
                        matched_cifs += _get_matched_cifs(
                            known_hits[known_hit_dataset][known_hit_residue],
                            _event[0],
                        )
                        # if len(matched_cifs) == 0:
                        #     rprint(f'NO MATCHED LIGAND DATA!!!!!!')
                        #     continue

                    if len(matched_cifs) == 0:
                        rprint(f'NO MATCHED LIGAND DATA!!!!!!')
                        continue

                    matched_cif = matched_cifs[0]

                    smiles = _get_smiles(matched_cif)
                    atom_ids_array = np.zeros((100,), dtype='<U5')
                    atom_ids = _get_atom_ids(matched_cif)
                    atom_ids_array[:len(atom_ids)] = atom_ids[:len(atom_ids)]
                    num_heavy_atoms = len(atom_ids)
                    connectivity_array = np.zeros((100, 100), dtype='?')
                    connectivity = _get_connectivity(matched_cif)
                    connectivity_array[:connectivity.shape[0], :connectivity.shape[1]] = connectivity[
                                                                                         :connectivity.shape[0],
                                                                                         :connectivity.shape[1]]

                    ligand_data_sample = np.array([(
                        tmp_idx_ligand_data,
                        num_heavy_atoms,
                        smiles,
                        atom_ids_array,
                        connectivity_array
                    )], dtype=ligand_data_dtype)
                    tmp_idx_ligand_data += 1
                    ligand_data[known_hit_residue] = ligand_data_sample

                    # rprint(ligand_data_sample)


                if len(ligand_data) == len(known_hits[known_hit_dataset]):
                    for known_hit_residue, ligand_data_sample in ligand_data.items():
                        ligand_data_table.append(
                            ligand_data_sample
                        )
                        idx_ligand_data += 1

                # 2. Get the blobs for each zmap
                zmaps = {}
                zblobs = {}
                for _event_id, event in close_event_dict.items():
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
                    coordinate_array = all_coords / ((np.array(reference_frame.spacing)-1) / (np.array(zmap_array.shape)-1)).reshape(1,
                                                                                                                   -1)
                    rprint(coordinate_array.shape)


                    x, y, z = np.arange(zmap_array.shape[0]), np.arange(zmap_array.shape[1]), np.arange(
                        zmap_array.shape[2]),
                    rprint(f'Zmap array shape: {zmap_array.shape}')
                    rprint(f'x,y,z shapes: {x[-1]}, {y[-1]}, {z[-1]}')
                    rprint(f'x,y,z reference shapes: {reference_frame.spacing}')
                    interp = RegularGridInterpolator((x, y, z), zmap_array)

                    clipped_coords = np.clip(coordinate_array, np.array([0,0,0]).reshape(1,3), np.array([x[-1], y[-1], z[-1]]).reshape(1,3))
                    rprint(np.min(clipped_coords, axis=0))
                    rprint(np.max(clipped_coords, axis=0))
                    resampling = interp(clipped_coords)
                    rprint(resampling.shape)

                    # resampling = map_coordinates(
                    #     np.array(zmap, copy=False,),
                    #     coordinate_array
                    # )
                    new_grid = reference_frame.get_grid()
                    new_grid_array = np.array(new_grid, copy=False)
                    new_grid_array[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = resampling
                    rprint(f'Resample shape: {resampling.shape}')
                    rprint(f'Reference spacig: {reference_frame.spacing}')
                    rprint(f'Reference unit cell: {reference_frame.unit_cell}')
                    rprint(f'Zmap: {zmap}')
                    rprint(f'Zmap unit cell: {zmap.unit_cell}')
                    rprint(f'Resampledc zmap: {new_grid}')
                    sparse_z_map = reference_frame.mask_grid(new_grid).data
                    events, cutoff = ClusterDensityDBSCAN()(
                        sparse_z_map,
                        reference_frame,
                    )
                    zblobs[event[0].id] = {
                        'events': events,
                        'cutoff': cutoff
                    }
                    zmaps[event[0].id] = new_grid

                ligand_masks = {}
                for (_known_hit_residue, _event_id), event in close_events.items():
                    ligand_mask = _get_ligand_mask_float(
                        zmaps[_event_id],
                        known_hits[known_hit_dataset][_known_hit_residue],
                        radius=1.5)
                    ligand_masks[(_known_hit_residue, event[0].id)] = ligand_mask

                rprint(f'Got {len([y for x in zblobs.values() for y in x["events"]])} z blobs')

                # 3. Get distances from event blobs to residues
                distances = {}
                for (_known_hit_residue, _event_id), event in close_events.items():

                    blobs, cutoff = zblobs[_event_id]['events'], zblobs[_event_id]['cutoff']
                    for _blob_id, _blob in blobs.items():
                        blob_centroid = _blob.centroid

                        _residue = known_hits[known_hit_dataset][_known_hit_residue]
                        residue_centroid = np.mean(_res_to_array(_residue)[0], axis=0)

                        # Get the distance from the blob centroid to residue centroid
                        cell = known_hit_structures[known_hit_dataset].cell
                        distances[(_known_hit_residue, _event_id, _blob_id,)] = cell.find_nearest_image(
                            gemmi.Position(residue_centroid[0], residue_centroid[1], residue_centroid[2]),
                            gemmi.Position(blob_centroid[0], blob_centroid[1], blob_centroid[2]),
                            gemmi.Asu.Any
                        ).dist()

                # rprint({_resid: np.mean(_res_to_array(_residue)[0], axis=0) for _resid, _residue in
                #         known_hits[known_hit_dataset].items()})
                # rprint(
                #     {_key: {'size': round(zblobs[_key[1]]['events'][_key[2]].size, 2), 'dist': round(_distance, 2)} for
                #      _key, _distance in distances.items()})
                # rprint(f'Got {len(distances)} distances')
                ref_event_distance = cell.find_nearest_image(
                    gemmi.Position(residue_centroid[0], residue_centroid[1], residue_centroid[2]),
                    gemmi.Position(event[0].x, event[0].y, event[0].z),
                    gemmi.Asu.Any
                ).dist()
                rprint(f'Reference event distance: {ref_event_distance}')

                # 4. Match blobs to the residues (or not)
                # Matching rules are:
                # Blobs further than 7A from any residue get selected as non-hits
                rprint(f"Matching blobs to residues...")
                df = pd.DataFrame(
                    [
                        {
                            '_known_hit_residue': _known_hit_residue,
                            '_event_id': _event_id,
                            '_blob_id': _blob_id,
                            '_dist': _dist
                        }
                        for (_known_hit_residue, _event_id, _blob_id), _dist
                        in distances.items()
                    ],
                ).set_index(['_event_id', '_blob_id', ])
                # rprint(df)

                grouping = df.groupby(by=['_event_id', '_blob_id'])
                # rprint(grouping)

                closest = grouping['_dist'].transform(min)
                # rprint(closest)
                # rprint(df['_dist'])
                # rprint(grouping['_dist'])
                # rprint(closest.reset_index())

                # mask = grouping == closest
                # rprint(mask)

                blob_closest_lig_df = df[df['_dist'] == closest]

                all_hits = []
                non_hits = []
                for _idx, _row in blob_closest_lig_df.iterrows():
                    all_hits.append(
                        (
                            _idx,
                            _row
                        )
                    )
                    if _row['_dist'] > 10.0:
                        non_hits.append(
                            (
                                _idx,
                                _row
                            )
                        )

                # rprint(non_hits)

                # For each non-hit add a non-hit sample to the store
                rprint(f'Have {len(non_hits)} non hits to add to store!')
                random_ints = list(range(len(non_hits)))
                random.shuffle(random_ints)
                for _x in random_ints[:10]:
                    _non_hit_idx, _row = non_hits[_x]
                    # Get the sample transform
                    blob = zblobs[_non_hit_idx[0]]['events'][_non_hit_idx[1]]
                    blob_centroid = blob.centroid
                    centroid = np.array([blob_centroid[0], blob_centroid[1], blob_centroid[2]])
                    transform = gemmi.Transform()
                    transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                    transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))



                    # Record the 2fofc map sample
                    z_map_sample_array = _sample_xmap_and_scale(
                        zmaps[_non_hit_idx[0]],
                        transform,
                        np.zeros((90, 90, 90), dtype=np.float32),
                    )
                    # rprint([(
                    #         idx_z_map,
                    #         _non_hit_idx[0],
                    #         _row['_known_hit_residue'],
                    #         ligand_data[_row['_known_hit_residue']][0]
                    #     )],)
                    z_map_sample_metadata = np.array(
                        [(
                            idx_z_map,
                            _non_hit_idx[0],
                            _row['_known_hit_residue'],
                            ligand_data[_row['_known_hit_residue']][0][0],
                            -1
                        )],
                        dtype=z_map_sample_metadata_dtype
                    )
                    table_z_map_sample_metadata.append(
                        z_map_sample_metadata
                    )

                    z_map_sample = np.array(
                        [(
                            idx_z_map,
                            z_map_sample_array
                        )],
                        dtype=z_map_sample_dtype
                    )
                    # z_map_sample = z_map_sample_array[np.newaxis, :]
                    table_z_map_sample.append(
                        z_map_sample
                    )

                    # ligand_mask = ligand_masks[(_row['_known_hit_residue'], _non_hit_idx[0])]
                    # ligand_mask_array = np.array(ligand_mask, copy=False)
                    # rprint('Point Array')
                    # rprint(blob.point_array.shape)
                    # masked_vals = ligand_mask_array[
                    #     blob.point_array[:, 0],
                    #     blob.point_array[:, 1],
                    #     blob.point_array[:, 2],
                    # ]
                    # rprint('Masked Z map vals')
                    # rprint(masked_vals)
                    # rprint('Selected ligand volume from mask array')
                    # rprint(np.sum(masked_vals))
                    # rprint(f'Total ligand volume in mask array')
                    # rprint(np.sum(ligand_mask_array))

                    idx_z_map += 1


                # For each hit residue/event pair, add a hit sample to the store
                rprint(f'Have {len(ligand_masks)} hits to add to store!')
                for (_resid, _event_id), _ligand_mask in ligand_masks.items():
                    rprint(f'{_resid} : {_event_id}')
                    # Get the corresponding event
                    event = close_event_dict[_event_id]

                    # Get the sample transform
                    centroid = np.array([event[0].x, event[0].y, event[0].z])
                    transform = gemmi.Transform()
                    transform.mat.fromlist((np.eye(3) * 0.5).tolist())
                    transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

                    # Get the pose of the ligand
                    poss, atom, elements = _res_to_array(known_hits[known_hit_dataset][_resid], )
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

                    known_hit_pos_sample = np.array([(
                        idx_pose,
                        pose_array,
                        atom_array,
                        elements_array,
                    )],
                        dtype=known_hit_pose_sample_dtype
                    )
                    # rprint(known_hit_pos_sample)
                    table_known_hit_pose_sample.append(
                        known_hit_pos_sample
                    )

                    # Sample the zmap
                    _z_map_sample = _sample_xmap_and_scale(
                        zmaps[_event_id],
                        transform,
                        np.zeros((90, 90, 90), dtype=np.float32))

                    z_map_sample_metadata = np.array(
                        [(
                            idx_z_map,
                            _event_id,
                            _resid,
                            ligand_data[_resid][0][0],
                            idx_pose
                        )],
                        dtype=z_map_sample_metadata_dtype
                    )
                    table_z_map_sample_metadata.append(
                        z_map_sample_metadata
                    )

                    z_map_sample = np.array(
                        [(
                            idx_z_map,
                            z_map_sample_array
                        )],
                        dtype=z_map_sample_dtype
                    )
                    table_z_map_sample.append(
                        z_map_sample
                    )

                    # Sample the ligand mask
                    ligand_mask_sample = _sample_xmap_and_scale(
                        _ligand_mask,
                        transform,
                        np.zeros((90, 90, 90), dtype=np.float32))
                    lig_selection = np.nonzero(ligand_mask_sample > 0.9)
                    selected_zs = _z_map_sample[lig_selection]
                    # rprint(selected_zs)
                    # rprint(_z_map_sample)
                    rprint(f'Ligand com: {com} : event com: {centroid}')
                    rprint(f'Ligand size: {lig_selection[0].size}')
                    rprint(f'Selected zs high: {_resid}, {_event_id} : {np.mean(selected_zs[selected_zs > 0.0])} ')
                    rprint(f'Zmap high: {_resid}, {_event_id} : {np.mean(_z_map_sample[_z_map_sample > 0.0])} ')

                    idx_pose += 1
                    idx_z_map += 1




if __name__ == "__main__":
    fire.Fire(main)
