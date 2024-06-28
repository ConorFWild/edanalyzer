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
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as R

from edanalyzer import constants
from edanalyzer.datasets.base import (
    _load_xmap_from_mtz_path,
    _load_xmap_from_path,
    _sample_xmap_and_scale,
    _get_ligand_mask_float,
    _sample_xmap,
    _get_identity_matrix,
    _get_random_orientation,
    _get_centroid_from_res,
    _get_transform_from_orientation_centroid,
    _get_res_from_structure_chain_res,
    _get_structure_from_path,
    _get_res_from_arrays,
    _get_grid_from_hdf5,
    _get_ed_mask_float)
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample
from edanalyzer.data.event_data import (
    _make_z_map_sample_metadata_table,
    _make_z_map_sample_table,
    _make_ligand_data_table,
    _make_known_hit_pose_table,
    _make_annotation_table,
    _get_closest_res_from_dataset_dir,
    _get_z_map_sample_from_dataset_dir,
    _get_pose_sample_from_dataset_dir,
    _get_ligand_data_sample_from_dataset_dir,
    _get_annotation_sample_from_dataset_dir,
    _get_z_map_metadata_sample_from_dataset_dir,
    _make_xmap_sample_table,
    _get_xmap_sample_from_dataset_dir
)

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
time_finish_gen = time.time()
rprint(f"Generated small rotations in: {round(time_finish_gen - time_begin_gen, 2)}")


very_small_rotations = []
identity = np.eye(3)
for j in range(20):
    rotations = R.random(100000)
    rotmat = rotations.as_matrix()
    mask = (rotmat > (0.95 * np.eye(3)))
    diag = mask[:, np.array([0, 1, 2]), np.array([0, 1, 2])]
    rot_mask = diag.sum(axis=1)
    valid_rots = rotmat[rot_mask == 3, :, :]
    rots = [x for x in valid_rots]
    very_small_rotations += rots
time_finish_gen = time.time()
rprint(f"Generated small rotations in: {round(time_finish_gen - time_begin_gen, 2)}")



meta_sample_dtype = [
    ('idx', '<i4'),
    # ('event_idx', '<i4'),
    ('res_id', '<U32'),
    ('system', '<U32'),
    ('dtag', '<U32'),
    ('event_num', 'i8'),
    ('Confidence', '<U32'),
    ('size', '<f4')
]
mtz_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]
xmap_sample_dtype = [('idx', '<i4'),  ('sample', '<f4', (90, 90, 90))]
z_map_sample_dtype = [('idx', '<i4'), ('sample', '<f4', (90, 90, 90))]
known_hit_pose_sample_dtype = [
    ('idx', '<i8'),
    ('positions', '<f4', (150, 3)),
    ('atoms', '<U5', (150,)),
    ('elements', '<i4', (150,)),
]
decoy_pose_sample_dtype = [
    ('idx', '<i4'),
    # ('database_event_idx', '<i4'),
    ('meta_idx', '<i4'),
    # ('mtz_sample_idx', '<i4'),
    ('positions', '<f4', (150, 3)),
    ('atoms', '<U5', (150,)),
    ('elements', '<i4', (150,)),
    ('rmsd', '<f4'),
    ('overlap_score', '<f4'),

]
delta_dtype = [
    ('idx', '<i4'),
    ('pose_idx', '<i4'),
    ('delta', '<f4', (150,)),
    ('delta_vec', '<f4', (150, 3)),
]
annotation_dtype = [

    ('idx', '<i4'),
    ('event_map_table_idx', '<i4'),
    ('annotation', '?'),
    ('partition', 'S32')]
ligand_data_dtype = [
    ('idx', 'i8'),
    ('num_heavy_atoms', 'i8'),
    ('canonical_smiles', '<U300'),
    ('atom_ids', '<U5', (150,)),
    ('connectivity', '?', (150, 150,))
]


def _get_event_autobuilds_paths(dataset_dir, event_idx):
    autobuild_dir = dataset_dir / 'autobuild'
    autobuild_paths = [x for x in autobuild_dir.glob('*')]
    return autobuild_paths


def _get_build_data(build_path, pose_sample, x, y, z):
    centroid = np.array([x, y, z]).reshape((1, 3))
    st = gemmi.read_structure(str(build_path))
    poss, atom, elements = _res_to_array(st[0][0][0], )
    com = np.mean(poss, axis=0).reshape((1, 3))
    event_to_lig_com = com - centroid.reshape((1, 3))
    _poss_centered = poss - com
    _rmsd_target = np.copy(_poss_centered) + np.array([22.5, 22.5, 22.5]).reshape(
        (1, 3)) + event_to_lig_com

    # rprint(f'RMSD: {pose_sample.shape} vs {_rmsd_target.shape}')
    rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(pose_sample - _rmsd_target, axis=1))) / _rmsd_target.shape[0])
    # rprint(f'Mean diff: {np.mean(np.linalg.norm(pose_sample - _rmsd_target, axis=1))}')
    return _rmsd_target, atom, elements, rmsd


def _get_pose_sample_from_res(
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
    size = min(150, _rmsd_target.shape[0])
    atom_array = np.zeros(150, dtype='<U5')
    elements_array = np.zeros(150, dtype=np.int32)
    pose_array = np.zeros((150, 3))
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

def _dep_overlap_score(decoy, pose):
    grid = gemmi.FloatGrid(90, 90, 90)
    grid.spacegroup = gemmi.SpaceGroup('P1')
    uc = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    grid.set_unit_cell(uc)
    for pos in pose:
        pos = gemmi.Position(pos[0], pos[1], pos[2])  # *
        grid.set_points_around(
            pos,
            radius=1.5,
            value=1.0,
        )

    for pos in decoy:
        pos = gemmi.Position(pos[0], pos[1], pos[2])  # *
        grid.set_points_around(
            pos,
            radius=1.5,
            value=2.0,
        )
    grid_array = np.array(grid, copy=False)

    num_missed = grid_array[grid_array == 1]
    num_total = grid_array[grid_array == 2]
    overlap = (num_total.size-num_missed.size) / num_total.size
    return overlap


def overlap_score(known_hit_predicted_density, decoy_predicted_density, known_hit_pose_residue, decoy_residue):


    # known_hit_score_mask_grid = _get_ligand_mask_float(
    #     known_hit_pose_residue,
    #     radius=2.5,
    #     n=90,
    #     r=45.0
    # )

    decoy_score_mask_grid = _get_ligand_mask_float(
        decoy_residue,
        radius=2.5,
        n=180,
        r=45.0
    )

    decoy_score_mask_arr = np.array(decoy_score_mask_grid, copy=False)
    decoy_predicted_density_arr = np.array(decoy_predicted_density, copy=False)
    # known_hit_score_mask_arr = np.array(known_hit_score_mask_grid, copy=False)
    known_hit_predicted_density_arr = np.array(known_hit_predicted_density, copy=False)


    sel = decoy_score_mask_arr > 0.0

    decoy_predicted_density_sel = decoy_predicted_density_arr[sel]
    known_hit_predicted_density_sel = known_hit_predicted_density_arr[sel]


    score = 1- ( np.sum(np.clip(decoy_predicted_density_sel - known_hit_predicted_density_sel, 0.0, None)) / np.sum(decoy_predicted_density_sel))

    return score

def setup_store(zarr_path):
    root = zarr.open(zarr_path, mode='w')

    table_meta_sample = root.create_dataset(
        'meta_sample',
        shape=(0,),
        chunks=(1,),
        dtype=meta_sample_dtype
    )
    table_mtz_sample = root.create_dataset(
        'mtz_sample',
        shape=(0,),
        chunks=(1,),
        dtype=mtz_sample_dtype
    )
    table_xmap_sample = root.create_dataset(
        'xmap_sample',
        shape=(0,),
        chunks=(1,),
        dtype=xmap_sample_dtype
    )
    table_z_map_sample = root.create_dataset(
        'z_map_sample',
        shape=(0,),
        chunks=(1,),
        dtype=z_map_sample_dtype
    )

    table_known_hit_pos_sample = root.create_dataset(
        'known_hit_pose',
        shape=(0,),
        chunks=(1,),
        dtype=known_hit_pose_sample_dtype
    )

    table_decoy_pose_sample = root.create_dataset(
        'decoy_pose_sample',
        shape=(0,),
        chunks=(1,),
        dtype=decoy_pose_sample_dtype
    )

    delta_table = root.create_dataset(
        'delta',
        shape=(0,),
        chunks=(1,),
        dtype=delta_dtype
    )

    annotation_table = root.create_dataset(
        'annotation',
        shape=(0,),
        chunks=(1,),
        dtype=annotation_dtype

    )

    ligand_data_table = root.create_dataset(
        'ligand_data',
        shape=(0,),
        chunks=(1,),
        dtype=ligand_data_dtype
    )

    return root

def _get_predicted_density_from_res(residue, event_map):
    optimized_structure = gemmi.Structure()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')

    chain.add_residue(residue)
    model.add_chain(chain)
    optimized_structure.add_model(model)

    # Get the electron density of the optimized structure
    optimized_structure.cell = event_map.unit_cell
    optimized_structure.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = 0.75  #*2
    # dencalc.rate = 1.5
    dencalc.set_grid_cell_and_spacegroup(optimized_structure)
    # dencalc.initialize_grid_to_size(event_map.nu, event_map.nv, event_map.nw)
    dencalc.put_model_density_on_grid(optimized_structure[0])
    calc_grid = dencalc.grid

    return calc_grid


def _random_mask(_decoy_poss, _decoy_elements):
    valid_mask = _decoy_elements != 0
    # rprint(f'Initial valid mask sum: {valid_mask.sum()}')
    # if _train:
    do_drop = rng.random()
    if do_drop > 0.5:
        valid_indicies = np.nonzero(valid_mask)

        u_s = rng.uniform(0.0, 0.35)
        random_drop_index = rng.integers(0, high=len(valid_indicies[0]), size=max(
            [
                3,
                int(u_s * len(valid_indicies[0]))
            ]
        ))

        drop_index = valid_indicies[0][random_drop_index]
        valid_mask[drop_index] = False
    valid_poss = _decoy_poss[valid_mask]
    valid_elements = _decoy_elements[valid_mask]

    return valid_poss, valid_elements, valid_mask

def _permute_position(_poss_pose, _poss_decoy, translation=5, small=True):
    # Get rotation and translation
    if small == 1:
        rot = R.from_matrix(small_rotations[rng.integers(0, len(small_rotations))])
    elif small == 2:
        rot = R.from_matrix(very_small_rotations[rng.integers(0, len(very_small_rotations))])
    elif small == 3:
        rot = R.random()
    _translation = rng.uniform(-translation , translation , size=3).reshape((1, 3))

    # Cetner
    com = np.mean(_poss_decoy, axis=0).reshape((1, 3))
    _poss_centered = _poss_decoy - com

    # Randomly perturb and reorient
    _rotated_poss = rot.apply(_poss_centered)
    new_com = _translation + com
    _new_poss = _rotated_poss + new_com

    # Get RMSD to original
    deltas = _poss_pose - _new_poss
    rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(deltas, axis=1))) / _new_poss.shape[0])

    return _new_poss, rmsd,  deltas, np.linalg.norm(deltas, axis=1)

def _get_augmented_decoy(
        known_hit_poss,
        atoms,
        pose_poss,
        pose_elements,
        pose_predicted_density,
        known_hit_pose_residue,
        template_grid,
        tmp_pose_idx,
                meta_idx,
        translation,
        small
):
    # Mask
    rprint(atoms)
    rprint(pose_poss)
    rprint(pose_elements)
    masked_poss, masked_elements, mask = _random_mask(pose_poss, pose_elements)
    rprint(mask)
    masked_atoms = atoms[mask]#pose_sample["atoms"][mask]

    known_hit_poss_masked = known_hit_poss[mask]
    # Permute
    _new_poss, rmsd, _delta_vecs, _delta = _permute_position(known_hit_poss_masked, masked_poss, translation, small)

    # Get Overlap
    decoy_res = _get_res_from_arrays(_new_poss, masked_elements)
    decoy_predicted_density = _get_predicted_density_from_res(
        decoy_res,
        template_grid
    )
    ol_score = overlap_score(
        pose_predicted_density,
        decoy_predicted_density,
        known_hit_pose_residue,
        decoy_res
    )

    atom_array = np.zeros(150, dtype='<U5')
    elements_array = np.zeros(150, dtype=np.int32)
    pose_array = np.zeros((150, 3))
    num_atoms = masked_elements.size
    pose_array[:num_atoms, :] = _new_poss[:num_atoms, :]
    atom_array[:num_atoms] = masked_atoms[:num_atoms]
    elements_array[:num_atoms] = masked_elements[:num_atoms]

    known_hit_pos_sample = np.array(
        [
            (
                tmp_pose_idx,
                meta_idx,
                pose_array,
                atom_array,
                elements_array,
                rmsd,
                ol_score
            )
        ],
        dtype=decoy_pose_sample_dtype
    )

    _delta_array =np.zeros((150,))
    _delta_vecs_array = np.zeros((150, 3))
    _delta_array[:num_atoms] = _delta[:num_atoms]
    _delta_vecs_array[:num_atoms, :] = _delta_vecs[:num_atoms,:]

    delta_sample = np.array([(
        tmp_pose_idx,
        # idx_pose,
        meta_idx,
        _delta_array,
        _delta_vecs_array,
    )], dtype=delta_dtype
    )

    return known_hit_pos_sample, delta_sample

def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    # zarr_path = 'output/event_data_with_mtzs_2.zarr'
    zarr_path = '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/build_data_augmented.zarr'

    root = setup_store(zarr_path)
    table_meta_sample = root['meta_sample']
    table_xmap_sample = root['xmap_sample']
    table_z_map_sample = root['z_map_sample']
    table_known_hit_pos_sample = root['known_hit_pose']
    table_decoy_pose_sample = root['decoy_pose_sample']
    delta_table = root['delta']
    annotation_table = root['annotation']
    ligand_data_table = root['ligand_data']

    # Iterate over annotated pandda 2 datasets
    meta_idx = 0
    pose_idx = 0
    # tmp_pose_idx = 0
    for pandda_dir in Path(
            '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score/').glob(
        '*'):
        rprint(f'########### Processing system: {pandda_dir.name} ###############')
        if pandda_dir.name == 'TcCS':
            continue

        # Skip if not a directory
        if not pandda_dir.is_dir():
            rprint(f'Skipping Directory!')
            continue
        rprint(f'PanDDA dir is: {pandda_dir}')

        # Skip if no inspect table
        pandda_inspect_table = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not pandda_inspect_table.exists():
            rprint(f'\tNO INSPECT TABLE! SKIPPING!')
            continue

        # Get the inspect table
        inspect_table = pd.read_csv(pandda_inspect_table)

        # Iterate the inspect table
        for _idx, _row in inspect_table.iterrows():
            # Unpack the row information
            dtag, event_idx, bdc, conf, viewed, size = _row['dtag'], _row['event_idx'], _row['1-BDC'], _row[
                constants.PANDDA_INSPECT_HIT_CONDFIDENCE], _row[constants.PANDDA_INSPECT_VIEWED], _row[
                constants.PANDDA_INSPECT_CLUSTER_SIZE]

            if conf != 'High':
                continue

            system = _get_system_from_dtag(dtag)

            x, y, z = _row['x'], _row['y'], _row['z']
            dataset_dir = pandda_dir / 'processed_datasets' / dtag

            model_dir = dataset_dir / 'modelled_structures'

            # Get the corresponding residue
            resid, res, dist = _get_closest_res_from_dataset_dir(
                dataset_dir,
                x, y, z
            )

            # Process hits only
            if dist > 6.0:
                rprint(
                    f'Could not match high confidence ligand {dtag} {event_idx} to a build!\n'
                    f'Check model in {dataset_dir} is appropriate!\n'
                    'SKIPPING!'
                )
                # raise Exception
                continue

            # Get the z map sample
            z_map_sample = _get_z_map_sample_from_dataset_dir(
                dataset_dir,
                x, y, z,
                meta_idx,
            )
            xmap_sample = _get_xmap_sample_from_dataset_dir(
                dataset_dir,
                x, y, z,
                meta_idx,
            )
            # Get the modelled pose sample
            pose_sample = _get_pose_sample_from_res(
                model_dir,
                res,
                x, y, z,
                meta_idx
            )
            known_hit_pose_elements = pose_sample["elements"][pose_sample["elements"] != 0]
            known_hit_pose_poss = pose_sample['positions'][pose_sample['elements'] != 0]
            known_hit_pose_atoms = pose_sample['atoms'][pose_sample['elements'] != 0]

            # Get the ligand data
            ligand_data_sample = _get_ligand_data_sample_from_dataset_dir(
                dataset_dir,
                res,
                meta_idx,
            )
            if not ligand_data_sample:
                rprint(f'\t\tNO LIGAND DATA! SKIPPING!')
                continue

            # Get the autobuild pose samples
            # Generate the decoy/rmsd pairs
            # poses, atoms, elements, rmsds = _get_known_hit_poses(
            #     known_hits[known_hit_dataset][known_hit_residue],
            #     event_to_lig_com
            # )
            decoy_pose_samples = []
            delta_samples = []
            tmp_pose_idx = pose_idx
            auotbuild_paths = _get_event_autobuilds_paths(dataset_dir, event_idx)
            rprint(f'Got {len(auotbuild_paths)} autobuild paths!')

            known_hit_pose_residue = _get_res_from_arrays(
                known_hit_pose_poss,
                known_hit_pose_elements,
            )
            rprint(x for x in known_hit_pose_residue)
            rprint(known_hit_pose_poss)
            template_grid = gemmi.FloatGrid(180,180,180)
            template_grid.spacegroup = gemmi.find_spacegroup_by_name("P1")
            template_grid.set_unit_cell(gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0))
            pose_predicted_density = _get_predicted_density_from_res(
                known_hit_pose_residue,
                template_grid
            )

            bins = {
                0:[],
                1:[],
                2:[],
                3:[],
                4:[],
                5:[],
                6:[],
                7:[],
                8:[],
                9:[]
            }

            superposed_builds = [known_hit_pose_residue,]
            for build_path in auotbuild_paths:
                pose, atom, element, rmsd = _get_build_data(
                    build_path,
                    known_hit_pose_poss,
                    x, y, z
                )

                if not np.array_equal(element, known_hit_pose_elements):
                    rprint(f'Ligand doesn\'t match! Skipping! {element} vs {known_hit_pose_elements}')
                    continue
                build_res = _get_res_from_arrays(
                    known_hit_pose_poss,
                    known_hit_pose_elements,
                )

                sup = gemmi.superpose_positions(
                    [atom.pos for atom in known_hit_pose_residue],
                    [atom.pos for atom in build_res]
                )

                trans = sup.transform
                for atom in build_res:
                    new_pos_vec = trans.apply(atom.pos)
                    new_pos = gemmi.Position(new_pos_vec.x, new_pos_vec.y, new_pos_vec.z)
                    atom.pos = new_pos

                superposed_builds.append(build_res)

                # Generate decoys around known hit
            for translation, small, num, from_hit in [[0.1, 2, 250, True], [0.25, 1, 100, True,], [0.5, 3, 100, False], [1.0, 1, 100, False], [3.0, 3, 100, False], [5.0, 3, 100, False]]:
                if from_hit:
                    decoy_poss, decoy_atoms, decoy_elements = _res_to_array(known_hit_pose_residue)
                    rprint([decoy_poss, decoy_atoms, decoy_elements])
                else:
                    decoy_num = rng.integers(0, high=len(superposed_builds),)
                    x = _res_to_array(superposed_builds[decoy_num])
                    decoy_poss, decoy_atoms, decoy_elements = x[0][0], x[0][1], x[0][2]
                for j in range(num):
                    decoy_sample, decoy_delta_sample = _get_augmented_decoy(
                        known_hit_pose_poss,
                        decoy_atoms,
                        decoy_poss,
                        decoy_elements,
                        pose_predicted_density,
                        known_hit_pose_residue,
                        template_grid,
                        tmp_pose_idx,
                        meta_idx,
                        translation,
                        small
                    )
                    # rprint(decoy_sample['overlap_score'])

                    bin_id = int(decoy_sample['overlap_score'][0] * 10)
                    if bin_id == 10:
                        bin_id = 9
                    bins[bin_id].append(len(decoy_pose_samples))
                    decoy_pose_samples.append(decoy_sample)
                    delta_samples.append(decoy_delta_sample)
                    tmp_pose_idx += 1

            # Generate decoys around autobuilds
            # for build_path in auotbuild_paths:
            #     pose, atom, element, rmsd = _get_build_data(
            #         build_path,
            #         known_hit_pose_poss,
            #         x, y, z
            #     )
            #
            #     if not np.array_equal(element, known_hit_pose_elements):
            #         rprint(f'Ligand doesn\'t match! Skipping! {element} vs {known_hit_pose_elements}')
            #         continue
            #     if rmsd > 15:
            #         continue
            #
            #     atom_array = np.zeros(150, dtype='<U5')
            #     elements_array = np.zeros(150, dtype=np.int32)
            #     pose_array = np.zeros((150, 3))
            #     num_atoms = element.size
            #     pose_array[:num_atoms, :] = pose[:num_atoms, :]
            #     atom_array[:num_atoms] = atom[:num_atoms]
            #     elements_array[:num_atoms] = element[:num_atoms]
            #
            #     # rprint(f'{rmsd}')
            #     for j in range(50):
            #         decoy_sample, decoy_delta_sample = _get_augmented_decoy(
            #             known_hit_pose_poss,
            #             atom,
            #             pose,
            #             element,
            #             pose_predicted_density,
            #             known_hit_pose_residue,
            #             template_grid,
            #             tmp_pose_idx,
            #             meta_idx,
            #         )
            #         bin_id = int(decoy_sample['overlap_score'][0] * 10)
            #         if bin_id == 10:
            #             bin_id = 9
            #         bins[bin_id].append(len(decoy_pose_samples))
            #         decoy_pose_samples.append(decoy_sample)
            #         delta_samples.append(decoy_delta_sample)
            #         tmp_pose_idx += 1

            rprint(f'Got {len(delta_samples)} decoy poses, of which {len([dps for dps in decoy_pose_samples if dps["rmsd"] < 2.0])} close!')

            rprint(f"Bin sizes:")
            rprint({bin_id / 10: len(bins[bin_id]) for bin_id in bins})
            smallest_num = min([len(x) for x in bins.values()])
            rprint(smallest_num)

            # Make the metadata sample
            meta_sample = np.array(
                [(
                    meta_idx,
                    # event_idx,
                    resid,
                    system,
                    dtag,
                    event_idx,
                    conf,
                    size
                )],
                dtype=meta_sample_dtype
            )
            rprint(meta_sample)

            # Add items to store
            table_meta_sample.append(meta_sample)
            table_xmap_sample.append(xmap_sample)
            table_z_map_sample.append(z_map_sample)
            table_known_hit_pos_sample.append(pose_sample)
            # annotation_table.append()
            ligand_data_table.append(ligand_data_sample)

            # Filter decoys to an even distribution over bins of overlap score


            # Add filtered decoys
            # for decoy_pose_sample, delta_sample in zip(decoy_pose_samples, delta_samples):
            for key, val in bins.items():
                for decoy_num in val[:smallest_num]:
                    table_decoy_pose_sample.append(
                        decoy_pose_samples[decoy_num]
                    )
                    delta_table.append(
                        delta_samples[decoy_num]
                    )

            meta_idx += 1
            pose_idx = tmp_pose_idx


if __name__ == "__main__":
    fire.Fire(main)
