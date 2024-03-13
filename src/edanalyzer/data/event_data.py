import re

import numpy as np
import gemmi
from numcodecs import Blosc, Delta

from .database import _get_st_hits, _res_to_array, _get_matched_cifs_from_dir, _get_smiles, \
    _get_atom_ids, _get_connectivity, _get_system_from_dtag
from ..datasets.base import _get_structure_from_path, _load_xmap_from_path, _sample_xmap_and_scale
from ..constants import PANDDA_ZMAP_TEMPLATE, PANDDA_INSPECT_MODEL_DIR

z_map_sample_metadata_dtype = [
    ('idx', '<i4'),
    ('event_idx', '<i4'),
    ('res_id', '<U32'),
    ('ligand_data_idx', 'i8'),
    ('pose_data_idx', 'i8')
]

z_map_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]

ligand_data_dtype = [
    ('idx', 'i8'),
    ('num_heavy_atoms', 'i8'),
    ('canonical_smiles', '<U300'),
    ('atom_ids', '<U5', (100,)),
    ('connectivity', '?', (100, 100,))
]

known_hit_pose_sample_dtype = [
    ('idx', '<i8'),
    ('positions', '<f4', (100, 3)),
    ('atoms', '<U5', (100,)),
    ('elements', '<i4', (100,)),
    # ('rmsd', '<f4')
]

annotation_dtype = [
    ('idx', '<i4'),
    ('event_map_table_idx', '<i4'),
    ('annotation', '?'),
    ('partition', 'S32')]


def _make_z_map_sample_metadata_table(group):
    table_z_map_sample_metadata = group.create_dataset(
        'z_map_sample_metadata',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_metadata_dtype
    )
    return table_z_map_sample_metadata


def _make_z_map_sample_table(group):
    table_z_map_sample = group.create_dataset(
        'z_map_sample',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_dtype,
        compressor=Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
    )

    return table_z_map_sample


def _make_ligand_data_table(group):
    ligand_data_table = group.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(1,),
        dtype=ligand_data_dtype
    )
    return ligand_data_table


def _make_known_hit_pose_table(group):
    table_known_hit_pose_sample = group.create_dataset(
        'known_hit_pose',
        shape=(0,),
        chunks=(1,),
        dtype=known_hit_pose_sample_dtype
    )

    return table_known_hit_pose_sample


def _make_annotation_table(group):
    annotation_table = group.create_dataset(
        'annotation',
        shape=(0,),
        chunks=(1,),
        dtype=annotation_dtype
    )

    return annotation_table


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


def _get_closest_hit(centroid, hits):
    distances = {}
    for resid, res in hits.items():
        res_centroid = np.mean(_res_to_array(res)[0], axis=0)
        distance = np.linalg.norm(centroid - res_centroid)
        distances[resid] = distance

    closest_resid = min(distances, key=lambda _x: distance[_x])
    return closest_resid, distances[closest_resid]


def _get_most_recent_modelled_structure_from_dataset_dir(dataset_dir):
    model_dir = dataset_dir / PANDDA_INSPECT_MODEL_DIR
    model_paths = {}
    print(dataset_dir)
    for path in model_dir.glob('*'):
        fitted_model_regex = 'fitted-v([0-9]*).pdb'
        match = re.match(fitted_model_regex, path.name)

        if match:
            model_paths[int(match[1])] = path
        else:
            model_paths[0] = path

    return model_paths[max(model_paths)]


def _get_closest_res_from_dataset_dir(
        dataset_dir,
        x, y, z
):
    st_path = _get_most_recent_modelled_structure_from_dataset_dir(dataset_dir)
    st = _get_structure_from_path(st_path)
    hits = _get_st_hits(st)
    closest_hit_resid, distance = _get_closest_hit(np.array(x, y, z), hits)

    return closest_hit_resid, hits[closest_hit_resid], distance


def _get_z_map_sample_from_dataset_dir(dataset_dir, x, y, z, idx_z_map):
    # Get the zmap
    zmap_path = dataset_dir / PANDDA_ZMAP_TEMPLATE.format(dtag=dataset_dir.name)
    zmap = _load_xmap_from_path(zmap_path)

    # Get the transform
    centroid = np.array([x, y, z])
    transform = gemmi.Transform()
    transform.mat.fromlist((np.eye(3) * 0.5).tolist())
    transform.vec.fromlist((centroid - np.array([22.5, 22.5, 22.5])))

    # Record the 2fofc map sample
    z_map_sample_array = _sample_xmap_and_scale(
        zmap,
        transform,
        np.zeros((90, 90, 90), dtype=np.float32),
    )

    z_map_sample = np.array(
        [(
            idx_z_map,
            z_map_sample_array
        )],
        dtype=z_map_sample_dtype
    )
    return z_map_sample


def _get_z_map_metadata_sample_from_dataset_dir(
        idx_z_map,
        event_idx,
        resid,
        ligand_data_idx,
        pose_data_idx
):
    z_map_sample_metadata = np.array(
        [(
            idx_z_map,
            event_idx,
            resid,
            ligand_data_idx,
            pose_data_idx
        )],
        dtype=z_map_sample_metadata_dtype
    )

    return z_map_sample_metadata


def _get_pose_sample_from_dataset_dir(
        model_dir,
        res,
        x, y, z,
        idx_pose
):
    centroid = np.array([x, y, z])
    poss, atom, elements = _res_to_array(res, )
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
    return known_hit_pos_sample


def _get_ligand_data_sample_from_dataset_dir(dataset_dir, res, idx_ligand_data):
    compound_dir = dataset_dir / 'ligand_files'

    # Get the associated ligand data
    matched_cifs = _get_matched_cifs_from_dir(
        res,
        compound_dir,

    )

    if len(matched_cifs) == 0:
        # rprint(f'NO MATCHED LIGAND DATA!!!!!!')
        return

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
        idx_ligand_data,
        num_heavy_atoms,
        smiles,
        atom_ids_array,
        connectivity_array
    )], dtype=ligand_data_dtype)

    return ligand_data_sample


def _get_annotation_sample_from_dataset_dir(dataset_dir, annotation, test_systems, annotation_idx,
                                            z_map_sample_metadata_idx):
    # Get the annotation
    if annotation == "High":
        annotation_bool = True
    else:
        annotation_bool = False

    # Get the partition
    system_name = _get_system_from_dtag(dataset_dir.name)
    if system_name in test_systems:
        partition = 'test'
    else:
        partition = 'train'

    # Update
    annotation_sample = np.array(
        [
            (
                annotation_idx,
                z_map_sample_metadata_idx,
                annotation_bool,
                partition
            )
        ],
        dtype=annotation_dtype
    )
    return annotation_sample
