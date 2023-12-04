import random
import itertools
import time

from edanalyzer.data import StructureReflectionsDataset, StructureReflectionsData, PanDDAEventDataset, \
    PanDDAEvent, PanDDAEventAnnotations, PanDDAEventAnnotation, PanDDAUpdatedEventAnnotations
from numpy.random import default_rng
import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
import gemmi
from torch.utils.data import Dataset
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

import networkx as nx
import networkx.algorithms.isomorphism as iso

from edanalyzer import constants



import traceback


def load_xmap_from_mtz(path):
    mtz = gemmi.read_mtz_file(str(path))
    for f, phi in constants.STRUCTURE_FACTORS:
        try:
            xmap = mtz.transform_f_phi_to_map(f, phi, sample_rate=3)
            return xmap
        except Exception as e:
            continue
    raise Exception()


def sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def annotate_data_randomly(data: StructureReflectionsData, p: float):
    rng = default_rng()
    val = rng.random_sample()

    if val < p:
        return 1
    else:
        return 0


def generate_ligand_sample(data, get_transform, sample_ligand_in_xmap):
    transform = get_transform(data)
    image = sample_ligand_in_xmap(data, transform)
    return image


def generate_xmap_ligand_sample_or_decoy(
        data: StructureReflectionsData,
        annotation: int,
        sample_ligand,
        sample_ligand_decoy):
    # Decoy
    if annotation == 0:
        image = sample_ligand(data)
    # True sample
    else:
        image = sample_ligand_decoy(data)

    return image

    # xmap = load_xmap_from_mtz(data.mtz_path)
    #
    # image = sample_xmap(xmap, transform)


def sample_ligand_density(data: StructureReflectionsData, annotater, image_sampler):
    annotation = annotater(data)
    image = image_sampler(data, annotation)

    return image, annotation


class StructureReflectionsDatasetTorch(Dataset):
    def __init__(self,
                 structure_reflections_dataset: StructureReflectionsDataset,
                 transform=None,
                 # target_transform=None,
                 ):
        self.structure_reflections_dataset = structure_reflections_dataset
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.structure_reflections_dataset.data)

    def __getitem__(self, idx):
        data = self.structure_reflections_dataset.data[idx]

        # reflections_path = data.pdb_path
        # structure_path = data.mtz_path
        # ligands = data.ligands
        image, label = self.transform(data)

        return image, label


def get_xmap_from_event(event: PanDDAEvent):
    ccp4 = gemmi.read_ccp4_map(event.event_map)
    ccp4.setup(float('nan'))
    m = ccp4.grid

    return m


def get_zmap_from_event(event: PanDDAEvent):
    zmap_path = str(Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_ZMAP_TEMPLATE.format(
        dtag=event.dtag))
    ccp4 = gemmi.read_ccp4_map(zmap_path)
    ccp4.setup(float('nan'))
    m = ccp4.grid

    return m


def get_mean_map_from_event(event: PanDDAEvent):
    zmap_path = str(Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(
        dtag=event.dtag))
    ccp4 = gemmi.read_ccp4_map(zmap_path)
    ccp4.setup(float('nan'))
    m = ccp4.grid

    return m

def get_event_map_from_event(event: PanDDAEvent):
    zmap_path = str(Path(event.event_map) )
    ccp4 = gemmi.read_ccp4_map(zmap_path)
    # ccp4.setup(float('nan'))
    ccp4.setup(0.0)
    m = ccp4.grid

    return m

def get_map_from_path(path):
    ccp4 = gemmi.read_ccp4_map(path)
    ccp4.setup(0.0)
    m = ccp4.grid

    return m

def get_sample_transform_from_event(event: PanDDAEvent,
                                    sample_distance: float,
                                    n: int,
                                    translation: float):
    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([event.x, event.y, event.z])

    # Apply random translation
    transform = event_centre_transform.combine(
        centre_grid_transform.combine(
            initial_transform
        )
    )
    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = (event.x, event.y, event.z)
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    # corner_n_pos = transform.apply(gemmi.Position(
    #     float(n),
    #     float(n),
    #     float(n),
    # )
    # )
    # corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    # corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    # average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    # event_centroid = (event.x, event.y, event.z)
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    return transform, np.zeros((n, n, n), dtype=np.float32)


def get_sample_transform_from_event_augmented(event: PanDDAEvent,
                                              sample_distance: float,
                                              n: int,
                                              translation: float):
    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
    sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Generate rotation matrix
    rotation_matrix = R.random().as_matrix()
    rotation_transform = gemmi.Transform()
    rotation_transform.mat.fromlist(rotation_matrix.tolist())

    # Apply random rotation transform to centroid
    transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
    transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])

    # Recentre transform
    rotation_recentre_transform = gemmi.Transform()
    rotation_recentre_transform.vec.fromlist(
        (sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([event.x, event.y, event.z])

    # Generate random translation transform
    rng = default_rng()
    random_translation = (rng.random(3) - 0.5) * 2 * translation
    # random_translation = np.array([0.0,0.0,0.0])
    # logger.debug(f"Random translation: {random_translation}")
    random_translation_transform = gemmi.Transform()
    random_translation_transform.vec.fromlist(random_translation.tolist())

    # Apply random translation
    transform = random_translation_transform.combine(
        event_centre_transform.combine(
            rotation_transform.combine(
                centre_grid_transform.combine(
                    initial_transform
                )
            )
        )
    )
    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = (event.x, event.y, event.z)
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    return transform, np.zeros((n, n, n), dtype=np.float32)


def get_image_from_event(event: PanDDAEvent):
    xmap = get_xmap_from_event(event)

    sample_transform, sample_array = get_sample_transform_from_event(event,
                                                                     0.5,
                                                                     30,
                                                                     3.5
                                                                     )

    image = sample_xmap(xmap, sample_transform, sample_array)

    return np.expand_dims(image, axis=0)


def get_image_from_event_augmented(event: PanDDAEvent):
    xmap = get_xmap_from_event(event)

    sample_transform, sample_array = get_sample_transform_from_event(event,
                                                                     0.5,
                                                                     30,
                                                                     3.5
                                                                     )

    image = sample_xmap(xmap, sample_transform, sample_array)

    return np.expand_dims(image, axis=0)


def get_raw_xmap_from_event(event: PanDDAEvent):
    mtz_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
        dtag=event.dtag)
    return load_xmap_from_mtz(mtz_path)
    # path = Path(event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / "xmap.ccp4"
    # ccp4 = gemmi.read_ccp4_map(str(path))
    # ccp4.setup(float('nan'))
    # m = ccp4.grid
    #
    # return m



def get_image_event_map_and_raw_from_event(event: PanDDAEvent):
    # logger.debug(f"Loading: {event.dtag}")
    sample_transform, sample_array = get_sample_transform_from_event(event,
                                                                     0.5,
                                                                     30,
                                                                     3.5
                                                                     )

    try:
        sample_array_event = np.copy(sample_array)
        xmap_event = get_xmap_from_event(event)
        image_event = sample_xmap(xmap_event, sample_transform, sample_array_event)

        sample_array_raw = np.copy(sample_array)
        xmap_raw = get_raw_xmap_from_event(event)
        image_raw = sample_xmap(xmap_raw, sample_transform, sample_array_raw)

        sample_array_zmap = np.copy(sample_array)
        xmap_zmap = get_zmap_from_event(event)
        image_zmap = sample_xmap(xmap_zmap, sample_transform, sample_array_zmap)

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_event)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)

    except Exception as e:
        print(e)
        return np.stack([sample_array, sample_array, sample_array, sample_array], axis=0), False

    return np.stack([image_event, image_raw, image_zmap, image_model], axis=0), True


def get_model_map(event: PanDDAEvent, xmap_event):
    pandda_input_pdb = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=event.dtag)
    structure = gemmi.read_structure(str(pandda_input_pdb))

    new_xmap = gemmi.FloatGrid(xmap_event.nu, xmap_event.nv, xmap_event.nw)
    new_xmap.spacegroup = xmap_event.spacegroup
    new_xmap.set_unit_cell(xmap_event.unit_cell)
    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    new_xmap.set_points_around(
                        atom.pos,
                        radius=1.0,
                        value=1.0,
                    )

    # time_begin_ns = time.time()
    # ns = gemmi.NeighborSearch(structure[0], structure.cell, 18).populate(include_h=False)
    #
    # num_sym = 0
    # num = 0
    # event_pos = gemmi.Position(event.x, event.y, event.z)
    # marks = ns.find_atoms(event_pos, '\0', radius=17)
    # # print(marks)
    # # print(len(marks))
    # mark_dists_event = []
    # mark_dists_cra = []
    # for _mark in marks:
    #     # print(_mark)
    #     # print(dir(_mark))
    #     _cra = _mark.to_cra(structure[0])
    #     mark_pos = gemmi.Position(_mark.x, _mark.y, _mark.z)
    #
    #     mark_dists_event.append(event_pos.dist(mark_pos))
    #
    #     mark_dist_cra = mark_pos.dist(_cra.atom.pos)
    #     mark_dists_cra.append(mark_dist_cra)
    #
    #
    #     if mark_dist_cra < 0.1:
    #         new_xmap.set_points_around(
    #                             mark_pos,
    #                             radius=1,
    #                             value=1.0,
    #                         )
    #         num += 1
    #     else:
    #         new_xmap.set_points_around(
    #             mark_pos,
    #             radius=1,
    #             value=-1.0,
    #         )
    #         num_sym +=1
    #
    # time_finish_ns = time.time()
    # print(
    #     f"Num: {num} : num sym: {num_sym} in {round(time_finish_ns-time_begin_ns, 2)} : {round(event.x, 2)} : {round(event.y, 2)} : {round(event.z, 2)} : {pandda_input_pdb}"
    # )
    # print(mark_dists_event[:10])
    # print(mark_dists_cra[:10])

    # For each atom, if it or a symmetry image (cryst and/or pbc +/- 1) of it are within 10A
    # event_pos = gemmi.Position(event.x, event.y, event.z)
    # time_begin_ns = time.time()
    #
    # lb = np.array([event.x, event.y, event.z]) - 10
    # ub = np.array([event.x, event.y, event.z]) + 10
    # symops = gemmi.find_spacegroup_by_name(structure.spacegroup_hm).operations()
    # num_sym = 0
    # num = 0
    # for model in structure:
    #     for chain in model:
    #         for residue in chain.get_polymer():
    #             for atom in residue:
    #                 # if event_pos.dist(atom.pos)
    #                 atom_pos = atom.pos
    #                 atom_pos_array = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
    #                 # if np.all(atom_pos_array > lb) and np.all(atom_pos_array < ub):
    #                 new_xmap.set_points_around(
    #                     atom.pos,
    #                     radius=1,
    #                     value=1.0,
    #                 )
    #                 num += 1
    #
    #                 fractional_pos = structure.cell.fractionalize(atom_pos)
    #                 for op in symops:
    #                     triplet = op.triplet()
    #                     sym_pos_frac = np.array(op.apply_to_xyz([fractional_pos.x, fractional_pos.y, fractional_pos.z]))
    #
    #                     for x,y,z in itertools.product([-1,0,1], [-1,0,1], [-1,0,1]):
    #                         if (x == 0) and (y == 0) and (z == 0) and (triplet == "x,y,z"):
    #                             # print(f"\tSkipping identity!")
    #                             continue
    #
    #                         pbc_sym_frac = sym_pos_frac + np.array([x,y,z])
    #                         pbc_sym_pos = structure.cell.orthogonalize(gemmi.Fractional(*pbc_sym_frac))
    #                         # pbc_sym_array = np.array([pbc_sym_pos.x, pbc_sym_pos.y, pbc_sym_pos.z])
    #                         new_xmap.set_points_around(
    #                             pbc_sym_pos,
    #                             radius=1,
    #                             value=-1.0,
    #                         )
    #                         num_sym += 1
    # for model in structure:
    #     for chain in model:
    #         for residue in chain.get_polymer():
    #             for atom in residue:
    #                 # if event_pos.dist(atom.pos)
    #                 atom_pos = atom.pos
    #                 atom_pos_array = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
    #                 if np.all(atom_pos_array > lb) and np.all(atom_pos_array < ub):
    #                     new_xmap.set_points_around(
    #                         atom.pos,
    #                         radius=1,
    #                         value=1.0,
    #                     )
    #                     num += 1
    #
    #                 fractional_pos = structure.cell.fractionalize(atom_pos)
    #                 for op in symops:
    #                     triplet = op.triplet()
    #                     sym_pos_frac = np.array(op.apply_to_xyz([fractional_pos.x, fractional_pos.y, fractional_pos.z]))
    #
    #                     for x,y,z in itertools.product([-1,0,1], [-1,0,1], [-1,0,1]):
    #                         if (x == y == z == 0) and (triplet == "x,y,z"):
    #                             print(f"\tSkipping identity!")
    #                             continue
    #
    #                         pbc_sym_frac = sym_pos_frac + np.array([x,y,z])
    #                         pbc_sym_pos = structure.cell.orthogonalize(gemmi.Fractional(*pbc_sym_frac))
    #                         pbc_sym_array = np.array([pbc_sym_pos.x, pbc_sym_pos.y, pbc_sym_pos.z])
    #                         if np.all(pbc_sym_array > lb) and np.all(pbc_sym_array < ub):
    #                             new_xmap.set_points_around(
    #                                 atom.pos,
    #                                 radius=1,
    #                                 value=-1.0,
    #                             )
    #                             num_sym += 1

    # time_finish_ns = time.time()
    # print(
    #     f"Num: {num} : num sym: {num_sym} in {round(time_finish_ns-time_begin_ns, 2)} : {round(event.x, 2)}, {round(event.y, 2)}, {round(event.z, 2)} : {pandda_input_pdb}"
    # )
    return new_xmap

def get_model_map_from_path(pandda_input_pdb, xmap_event):
    # pandda_input_pdb = Path(
    #     event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
    #     dtag=event.dtag)
    structure = gemmi.read_structure(str(pandda_input_pdb))

    new_xmap = gemmi.FloatGrid(xmap_event.nu, xmap_event.nv, xmap_event.nw)
    new_xmap.spacegroup = xmap_event.spacegroup
    new_xmap.set_unit_cell(xmap_event.unit_cell)
    for model in structure:
        for chain in model:
            for residue in chain.get_polymer():
                for atom in residue:
                    new_xmap.set_points_around(
                        atom.pos,
                        radius=1.0,
                        value=1.0,
                    )

    return new_xmap


def get_image_event_map_and_raw_from_event_augmented(event: PanDDAEvent):
    # logger.debug(f"Loading: {event.dtag}")
    sample_transform, sample_array = get_sample_transform_from_event_augmented(
        event,
        0.5,
        30,
        3.5
    )

    try:
        sample_array_event = np.copy(sample_array)
        xmap_event = get_xmap_from_event(event)
        image_event = sample_xmap(xmap_event, sample_transform, sample_array_event)

        sample_array_raw = np.copy(sample_array)
        xmap_raw = get_raw_xmap_from_event(event)
        image_raw = sample_xmap(xmap_raw, sample_transform, sample_array_raw)

        sample_array_zmap = np.copy(sample_array)
        xmap_zmap = get_zmap_from_event(event)
        image_zmap = sample_xmap(xmap_zmap, sample_transform, sample_array_zmap)

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_event)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)

    except Exception as e:
        print(e)
        return np.stack([sample_array, sample_array, sample_array, sample_array], axis=0), False

    return np.stack([image_event, image_raw, image_zmap, image_model], axis=0), True


def get_annotation_from_event_annotation(annotation: PanDDAEventAnnotation):
    if annotation.annotation:
        return np.array([0.0, 1.0], dtype=np.float32)
    else:
        return np.array([1.0, 0.0], dtype=np.float32)



class PanDDAEventDatasetTorch(Dataset):
    def __init__(self,
                 pandda_event_dataset: PanDDAEventDataset,
                 annotations: PanDDAEventAnnotations,
                 updated_annotations: PanDDAUpdatedEventAnnotations,
                 transform_image=lambda x: x,
                 transform_annotation=lambda x: x
                 ):
        self.pandda_event_dataset = pandda_event_dataset
        self.annotations = annotations
        self.updated_annotations = {
            (key.dtag, key.event_idx): annotation
            for key, annotation
            in zip(updated_annotations.keys, updated_annotations.annotations)
        }
        self.transform_image = transform_image
        self.transform_annotation = transform_annotation

    def __len__(self):
        return len(self.pandda_event_dataset.pandda_events)

    def __getitem__(self, idx: int):
        event = self.pandda_event_dataset.pandda_events[idx]

        key = (event.dtag, event.event_idx)

        if key in self.updated_annotations:
            logger.debug(
                f"Using updated annotation! Was {self.annotations.annotations[idx]} now {self.updated_annotations[key]}!")
            annotation = self.updated_annotations[key]
        else:
            annotation = self.annotations.annotations[idx]

        image, loaded = self.transform_image(event)

        if loaded:
            label = self.transform_annotation(annotation)

        else:
            label = np.array([1.0, 0.0], dtype=np.float32)
        return image, label, idx


def get_annotation_from_event_hit(annotation: bool):
    if annotation:
        return np.array([0.0, 1.0], dtype=np.float32)
    else:
        return np.array([1.0, 0.0], dtype=np.float32)


def get_annotation_from_event_hit_ligandmap(annotation: bool):
    if annotation:
        return np.array([0.0, 1.0], dtype=np.float32)
    else:
        return np.array([1.0, 0.0], dtype=np.float32)


def get_image_xmap_mean_map(event: PanDDAEvent, ):
    # logger.debug(f"Loading: {event.dtag}")

    sample_transform, sample_array = get_sample_transform_from_event(event,
                                                                     0.5,
                                                                     30,
                                                                     3.5
                                                                     )

    try:
        sample_array_xmap = np.copy(sample_array)
        xmap_dmap = get_raw_xmap_from_event(event)
        image_xmap_initial = sample_xmap(xmap_dmap, sample_transform, sample_array_xmap)
        image_xmap = (image_xmap_initial - np.mean(image_xmap_initial)) / np.std(image_xmap_initial)

        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_mean_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        image_mean = (image_mean_initial - np.mean(image_mean_initial)) / np.std(image_mean_initial)

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_dmap)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)

    except Exception as e:
        # print(e)
        return np.stack([sample_array, sample_array, sample_array], axis=0), False

    return np.stack([image_xmap, image_mean, image_model], axis=0), True


def get_image_xmap_mean_map_augmented(event: PanDDAEvent, ):
    # logger.debug(f"Loading: {event.dtag}")
    sample_transform, sample_array = get_sample_transform_from_event_augmented(
        event,
        0.5,
        30,
        3.5
    )

    try:
        sample_array_xmap = np.copy(sample_array)
        xmap_dmap = get_raw_xmap_from_event(event)
        image_xmap_initial = sample_xmap(xmap_dmap, sample_transform, sample_array_xmap)
        image_xmap = (image_xmap_initial - np.mean(image_xmap_initial)) / np.std(image_xmap_initial)

        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_mean_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        image_mean = (image_mean_initial - np.mean(image_mean_initial)) / np.std(image_mean_initial)

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_dmap)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)

    except Exception as e:
        # print(e)
        return np.stack([sample_array, sample_array, sample_array], axis=0), False

    return np.stack([image_xmap, image_mean, image_model], axis=0), True


class PanDDADatasetTorchXmapGroundState(Dataset):
    def __init__(self,
                 pandda_event_dataset: PanDDAEventDataset,

                 transform_image=lambda x: x,
                 transform_annotation=lambda x: x
                 ):
        self.pandda_event_dataset = pandda_event_dataset

        self.transform_image = transform_image
        self.transform_annotation = transform_annotation

    def __len__(self):
        return len(self.pandda_event_dataset.pandda_events)

    def __getitem__(self, idx: int):
        event = self.pandda_event_dataset.pandda_events[idx]

        annotation = event.hit

        image, loaded = self.transform_image(event)

        if loaded:
            label = self.transform_annotation(annotation)

        else:
            label = np.array([1.0, 0.0], dtype=np.float32)
        return image, label, idx


def parse_pdb_file_for_ligand_array(path):
    structure = gemmi.read_structure(str(path))
    poss = []
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    poss.append([pos.x, pos.y, pos.z])

    return np.array(poss).T

bond_type_cif_to_rdkit = {
    'single': Chem.rdchem.BondType.SINGLE,
    'double': Chem.rdchem.BondType.DOUBLE,
    'triple': Chem.rdchem.BondType.TRIPLE,
    'SINGLE': Chem.rdchem.BondType.SINGLE,
    'DOUBLE': Chem.rdchem.BondType.DOUBLE,
    'TRIPLE': Chem.rdchem.BondType.TRIPLE,
    'aromatic': Chem.rdchem.BondType.AROMATIC,
    # 'deloc': Chem.rdchem.BondType.OTHER
    'deloc': Chem.rdchem.BondType.SINGLE

}


def get_fragment_mol_from_dataset_cif_path(dataset_cif_path: Path):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Create a blank rdkit mol
    mol = Chem.Mol()
    editable_mol = Chem.EditableMol(mol)

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        key = "data_comp_XXX"

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
    if not atom_charge_loop:
        atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
        if not atom_charge_loop:
            atom_charge_loop = [0]*len(atom_id_loop)

    aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
    if not aromatic_atom_loop:
        aromatic_atom_loop = [None]*len(atom_id_loop)

    # Get the mapping
    id_to_idx = {}
    for j, atom_id in enumerate(atom_id_loop):
        id_to_idx[atom_id] = j

    # Iteratively add the relveant atoms
    for atom_id, atom_type, atom_charge in zip(atom_id_loop, atom_type_loop, atom_charge_loop):
        if len(atom_type) > 1:
            atom_type = atom_type[0] + atom_type[1].lower()
        atom = Chem.Atom(atom_type)
        atom.SetFormalCharge(round(float(atom_charge)))
        editable_mol.AddAtom(atom)

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
    bond_type_loop = list(cif[key].find_loop('_chem_comp_bond.type'))
    aromatic_bond_loop = list(cif[key].find_loop('_chem_comp_bond.aromatic'))
    if not aromatic_bond_loop:
        aromatic_bond_loop = [None]*len(bond_1_id_loop)

    try:
        # Iteratively add the relevant bonds
        for bond_atom_1, bond_atom_2, bond_type, aromatic in zip(bond_1_id_loop, bond_2_id_loop, bond_type_loop, aromatic_bond_loop):
            bond_type = bond_type_cif_to_rdkit[bond_type]
            if aromatic:
                if aromatic == "y":
                    bond_type = bond_type_cif_to_rdkit['aromatic']

            editable_mol.AddBond(
                id_to_idx[bond_atom_1],
                id_to_idx[bond_atom_2],
                order=bond_type
            )
    except Exception as e:
        print(e)
        print(atom_id_loop)
        print(id_to_idx)
        print(bond_1_id_loop)
        print(bond_2_id_loop)
        raise Exception

    edited_mol = editable_mol.GetMol()



    # HANDLE SULFONATES

    patt = Chem.MolFromSmarts('S(-O)(-O)(-O)')
    matches = edited_mol.GetSubstructMatches(patt)

    sulfonates = {}
    for match in matches:
        sfn = 1
        sulfonates[sfn] = {}
        on = 1
        for atom_idx in match:
            atom = edited_mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "S":
                sulfonates[sfn]["S"] = atom_idx
            else:
                atom_charge = atom.GetFormalCharge()

                if atom_charge == -1:
                    continue
                else:
                    if on == 1:
                        sulfonates[sfn]["O1"] = atom_idx
                        on += 1
                    elif on == 2:
                        sulfonates[sfn]["O2"] = atom_idx
                        on += 1
                # elif on == 3:
                #     sulfonates[sfn]["O3"] = atom_idx
    # print(f"Matches to sulfonates: {matches}")

    # atoms_to_charge = [
    #     sulfonate["O3"] for sulfonate in sulfonates.values()
    # ]
    # print(f"Atom idxs to charge: {atoms_to_charge}")
    bonds_to_double =[
        (sulfonate["S"], sulfonate["O1"]) for sulfonate in sulfonates.values()
    ] + [
        (sulfonate["S"], sulfonate["O2"]) for sulfonate in sulfonates.values()
    ]
    # print(f"Bonds to double: {bonds_to_double}")

    # Replace the bonds and update O3's charge
    new_editable_mol = Chem.EditableMol(Chem.Mol())
    for atom in edited_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        new_atom = Chem.Atom(atom.GetSymbol())
        charge = atom.GetFormalCharge()
        # if atom_idx in atoms_to_charge:
        #     charge = -1
        new_atom.SetFormalCharge(charge)
        new_editable_mol.AddAtom(new_atom)

    for bond in edited_mol.GetBonds():
        bond_atom_1 = bond.GetBeginAtomIdx()
        bond_atom_2 = bond.GetEndAtomIdx()
        double_bond = False
        for bond_idxs in bonds_to_double:
            if (bond_atom_1 in bond_idxs) & (bond_atom_2 in bond_idxs):
                double_bond = True
        if double_bond:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond_type_cif_to_rdkit['double']
            )
        else:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond.GetBondType()
            )
    new_mol = new_editable_mol.GetMol()

    return new_mol

def get_structures_from_mol(mol: Chem.Mol, dataset_cif_path, max_conformers):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Find the relevant atoms loop
    atom_id_loop = list(cif['comp_LIG'].find_loop('_chem_comp_atom.atom_id'))
    # print(f"Atom ID loop: {atom_id_loop}")


    fragment_structures = {}
    for i, conformer in enumerate(mol.GetConformers()):

        positions: np.ndarray = conformer.GetPositions()

        structure: gemmi.Structure = gemmi.Structure()
        model: gemmi.Model = gemmi.Model(f"{i}")
        chain: gemmi.Chain = gemmi.Chain(f"{i}")
        residue: gemmi.Residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(1, ' ')

        # Loop over atoms, adding them to a gemmi residue
        for j, atom in enumerate(mol.GetAtoms()):
            # Get the atomic symbol
            atom_symbol: str = atom.GetSymbol()
            # print(f"{j} : {atom_symbol}")

            # if atom_symbol == "H":
            #     continue
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            # gemmi_atom.name = atom_symbol
            gemmi_atom.name = atom_id_loop[j]
            gemmi_atom.pos = gemmi_pos
            gemmi_atom.element = gemmi_element

            # Add atom to residue
            residue.add_atom(gemmi_atom)

        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        fragment_structures[i] = structure

        if len(fragment_structures) > max_conformers:
            return fragment_structures

    return fragment_structures


def parse_cif_file_for_ligand_array(path):
    mol = get_fragment_mol_from_dataset_cif_path(path)
    mol.calcImplicitValence()

    # Generate conformers
    cids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=1000,
        pruneRmsThresh=1.5,
    )

    # Translate to structures
    fragment_structures = get_structures_from_mol(
        mol,
        path,
        10,
    )

    st = random.choice(fragment_structures)

    poss = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    poss.append([pos.x, pos.y, pos.z])

    return np.array(poss).T



LIGAND_IGNORE_REGEXES = [
    "merged",
    "LIG-[a-zA-Z]+-",
    "dimple",
    "refine",
    "init",
    "pipedream",
    "phenix",
    "None",
    "blank",
    "control",
    "DMSO",
    'tmp'
]


def get_ligand_map(
        event,
        n=30,
        step=0.5,
        translation=2.5,
):
    # Get the path to the ligand cif
    # dataset_dir = Path(event.model_building_dir) / event.dtag / "compound"
    # dataset_dir = Path(event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / "ligand_files"
    # paths = [x for x in dataset_dir.glob("*.pdb")]
    # pdb_paths = [x for x in paths if (x.exists()) and (x.stem not in LIGAND_IGNORE_REGEXES)]
    # assert len(pdb_paths) > 0, f"No pdb paths that are not to be ignored in {dataset_dir}: {[x.stem for x in paths]}"
    # path = pdb_paths[0]
    path = event.ligand

    # Get the ligand array
    ligand_array = parse_pdb_file_for_ligand_array(path)
    assert ligand_array.size > 0, f"Ligand array empty: {path}"
    rotation_matrix = R.random().as_matrix()
    rng = default_rng()
    random_translation = ((rng.random(3) - 0.5) * 2 * translation).reshape((3, 1))
    ligand_mean_pos = np.mean(ligand_array, axis=1).reshape((3, 1))
    centre_translation = np.array([step * n, step * n, step * n]).reshape((3, 1)) / 2
    zero_centred_array = ligand_array - ligand_mean_pos
    rotated_array = np.matmul(rotation_matrix, zero_centred_array)
    grid_centred_array = rotated_array + centre_translation
    augmented_array = (grid_centred_array + random_translation).T

    # Get a dummy grid to place density on
    dummy_grid = gemmi.FloatGrid(n, n, n)
    unit_cell = gemmi.UnitCell(step * n, step * n, step * n, 90.0, 90.0, 90.0)
    dummy_grid.set_unit_cell(unit_cell)

    for pos_array in augmented_array:
        assert pos_array.size == 3
        if np.all(pos_array > 0):
            if np.all(pos_array < (n * step)):
                dummy_grid.set_points_around(
                    gemmi.Position(*pos_array),
                    radius=1.0,
                    value=1.0,
                )

    return dummy_grid


def get_ligand_map_from_path(
        path,
        n=30,
        step=0.5,
        translation=2.5,
):
    # Get the path to the ligand cif
    # dataset_dir = Path(event.model_building_dir) / event.dtag / "compound"
    # dataset_dir = Path(event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / "ligand_files"
    # paths = [x for x in dataset_dir.glob("*.pdb")]
    # pdb_paths = [x for x in paths if (x.exists()) and (x.stem not in LIGAND_IGNORE_REGEXES)]
    # assert len(pdb_paths) > 0, f"No pdb paths that are not to be ignored in {dataset_dir}: {[x.stem for x in paths]}"
    # path = pdb_paths[0]
    # path = event.ligand

    # Get the ligand array
    # ligand_array = parse_pdb_file_for_ligand_array(path)
    try:
        ligand_array = parse_cif_file_for_ligand_array(path)
    except:
        ligand_array = parse_pdb_file_for_ligand_array(Path(path).parent / f"{Path(path).stem}.pdb")
    assert ligand_array.size > 0, f"Ligand array empty: {path}"
    rotation_matrix = R.random().as_matrix()
    rng = default_rng()
    random_translation = ((rng.random(3) - 0.5) * 2 * translation).reshape((3, 1))
    ligand_mean_pos = np.mean(ligand_array, axis=1).reshape((3, 1))
    centre_translation = np.array([step * n, step * n, step * n]).reshape((3, 1)) / 2
    zero_centred_array = ligand_array - ligand_mean_pos
    rotated_array = np.matmul(rotation_matrix, zero_centred_array)
    grid_centred_array = rotated_array + centre_translation
    augmented_array = (grid_centred_array + random_translation).T

    # Get a dummy grid to place density on
    dummy_grid = gemmi.FloatGrid(n, n, n)
    unit_cell = gemmi.UnitCell(step * n, step * n, step * n, 90.0, 90.0, 90.0)
    dummy_grid.set_unit_cell(unit_cell)

    for pos_array in augmented_array:
        assert pos_array.size == 3
        if np.all(pos_array > 0):
            if np.all(pos_array < (n * step)):
                dummy_grid.set_points_around(
                    gemmi.Position(*pos_array),
                    radius=1.0,
                    value=1.0,
                )

    return dummy_grid


def get_image_xmap_ligand(event: PanDDAEvent, ):
    # logger.debug(f"Loading: {event.dtag}")
    sample_transform, sample_array = get_sample_transform_from_event(
        event,
        0.5,
        30,
        3.5
    )

    try:
        print(event.dtag, event.event_idx)
        sample_array_xmap = np.copy(sample_array)
        xmap_dmap = get_raw_xmap_from_event(event)
        image_xmap_initial = sample_xmap(xmap_dmap, sample_transform, sample_array_xmap)
        xmap_mean, xmap_std = np.mean(image_xmap_initial), np.std(image_xmap_initial)
        image_xmap = (image_xmap_initial - np.mean(image_xmap_initial)) / np.std(image_xmap_initial)
        print(f"Xmap: {[xmap_mean, xmap_std]}")

        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_mean_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        mean_mean, mean_std = np.mean(image_mean_initial), np.std(image_mean_initial)
        image_mean = (image_mean_initial - np.mean(image_mean_initial)) / np.std(image_mean_initial)
        print(f"Mean: {[mean_mean, mean_std]}")

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_dmap)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)
        print(f"Model: {np.mean(image_model)}")


        # ligand_map_array = np.copy(sample_array)
        ligand_map = get_ligand_map(event)
        image_ligand = np.array(ligand_map)
        print(f"Ligand: {np.mean(image_ligand)}")

    except Exception as e:
        print(e)

        return np.stack([sample_array, sample_array, sample_array, sample_array], axis=0), False, None, None

    return np.stack([image_xmap, image_mean, image_model, image_ligand, ], axis=0), True, sample_transform, xmap_dmap


def get_image_xmap_ligand_augmented(event: PanDDAEvent, ):
    n = 30
    step = 0.5
    # logger.debug(f"Loading: {event.dtag}")
    time_begin_get_transform = time.time()
    sample_transform, sample_array = get_sample_transform_from_event_augmented(
        event,
        step,
        n,
        # 3.5
        2.0

    )
    time_finish_get_transform = time.time()
    time_get_transform = round(time_finish_get_transform - time_begin_get_transform, 2)

    try:
        sample_array_xmap = np.copy(sample_array)

        time_begin_get_xmap = time.time()
        xmap_dmap = get_raw_xmap_from_event(event)
        image_xmap_initial = sample_xmap(xmap_dmap, sample_transform, sample_array_xmap)
        image_xmap = (image_xmap_initial - np.mean(image_xmap_initial)) / np.std(image_xmap_initial)
        time_finish_get_xmap = time.time()
        time_get_xmap = round(time_finish_get_xmap - time_begin_get_xmap, 2)

        time_begin_get_mean = time.time()
        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_mean_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        std = np.std(image_mean_initial)
        if np.abs(std) < 0.0000001:
            image_mean = np.copy(sample_array)
        else:
            image_mean = (image_mean_initial - np.mean(image_mean_initial)) / std
        time_finish_get_mean = time.time()
        time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)

        time_begin_get_model = time.time()
        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, xmap_dmap, )
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)
        time_finish_get_model = time.time()
        time_get_model = round(time_finish_get_model - time_begin_get_model, 2)

        # ligand_map_array = np.copy(sample_array)
        time_begin_get_ligand = time.time()
        ligand_map = get_ligand_map(event, n=n, step=step)
        image_ligand = np.array(ligand_map)
        time_finish_get_ligand = time.time()
        time_get_ligand = round(time_finish_get_ligand - time_begin_get_ligand, 2)

    except Exception as e:
        # print(f"Exception in loading data: {traceback.format_exc()}")
        print(f"Exception in loading data: {e}")

        return np.stack([sample_array, sample_array, sample_array, sample_array], axis=0), False, None, None

    # print(f"Loaded item in: transform {time_get_transform}: xmap {time_get_xmap}: mean {time_get_mean}: model {time_get_model}: ligand {time_get_ligand}")

    return np.stack([image_xmap, image_mean, image_model, image_ligand, ], axis=0), True, sample_transform, xmap_dmap

def get_image_event_map_ligand(event: PanDDAEvent, ):
    # logger.debug(f"Loading: {event.dtag}")
    sample_transform, sample_array = get_sample_transform_from_event(
        event,
        0.5,
        30,
        3.5
    )

    try:
        # print(event.dtag, event.event_idx)

        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_event_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        mean_mean, mean_std = np.mean(image_mean_initial), np.std(image_mean_initial)
        image_event_map = (image_mean_initial - np.mean(image_mean_initial)) / np.std(image_mean_initial)
        # print(f"Mean: {[mean_mean, mean_std]}")

        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, mean_dmap)
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)
        # print(f"Model: {np.mean(image_model)}")

        # ligand_map_array = np.copy(sample_array)
        ligand_map = get_ligand_map(event)
        image_ligand = np.array(ligand_map)
        # print(f"Ligand: {np.mean(image_ligand)}")

    except Exception as e:
        print(e)

        return np.stack([sample_array, sample_array, sample_array], axis=0), False, None, None

    return np.stack([image_event_map, image_model, image_ligand, ], axis=0), True, sample_transform, mean_dmap


def get_image_event_map_ligand_augmented(event: PanDDAEvent, ):
    n = 30
    step = 0.5
    # logger.debug(f"Loading: {event.dtag}")
    time_begin_get_transform = time.time()
    sample_transform, sample_array = get_sample_transform_from_event_augmented(
        event,
        step,
        n,
        # 3.5
        2.0

    )
    time_finish_get_transform = time.time()
    time_get_transform = round(time_finish_get_transform - time_begin_get_transform, 2)

    try:
        sample_array_xmap = np.copy(sample_array)

        # time_begin_get_xmap = time.time()
        # xmap_dmap = get_raw_xmap_from_event(event)
        # image_xmap_initial = sample_xmap(xmap_dmap, sample_transform, sample_array_xmap)
        # image_xmap = (image_xmap_initial - np.mean(image_xmap_initial)) / np.std(image_xmap_initial)
        # time_finish_get_xmap = time.time()
        # time_get_xmap = round(time_finish_get_xmap - time_begin_get_xmap, 2)
        #
        # time_begin_get_mean = time.time()
        # sample_array_mean = np.copy(sample_array)
        # mean_dmap = get_mean_map_from_event(event)
        # image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        # std = np.std(image_mean_initial)
        # if np.abs(std) < 0.0000001:
        #     image_mean = np.copy(sample_array)
        # else:
        #     image_mean = (image_mean_initial - np.mean(image_mean_initial)) / std
        # time_finish_get_mean = time.time()
        # time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)

        time_begin_get_mean = time.time()
        sample_array_mean = np.copy(sample_array)
        mean_dmap = get_event_map_from_event(event)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        std = np.std(image_mean_initial)
        if np.abs(std) < 0.0000001:
            image_event_map = np.copy(sample_array)
        else:
            image_event_map = (image_mean_initial - np.mean(image_mean_initial)) / std
        time_finish_get_mean = time.time()
        time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)

        time_begin_get_model = time.time()
        sample_array_model = np.copy(sample_array)
        model_map = get_model_map(event, mean_dmap, )
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)
        time_finish_get_model = time.time()
        time_get_model = round(time_finish_get_model - time_begin_get_model, 2)

        # ligand_map_array = np.copy(sample_array)
        time_begin_get_ligand = time.time()
        ligand_map = get_ligand_map(event, n=n, step=step)
        image_ligand = np.array(ligand_map)
        time_finish_get_ligand = time.time()
        time_get_ligand = round(time_finish_get_ligand - time_begin_get_ligand, 2)

    except Exception as e:
        # print(f"Exception in loading data: {traceback.format_exc()}")
        print(f"Exception in loading data: {e}")

        return np.stack([sample_array, sample_array, sample_array], axis=0), False, None, None

    # print(f"Loaded item in: transform {time_get_transform}: xmap {time_get_xmap}: mean {time_get_mean}: model {time_get_model}: ligand {time_get_ligand}")

    return np.stack([image_event_map, image_model, image_ligand, ], axis=0), True, sample_transform, mean_dmap

# def transform_event_random_ligand(event, dataset):
#     rng = default_rng()
#     random_number = rng.random()
#     if random_number > 0.5:
#         random_ligand = ...
#         random_ligand_path = ...
#         new_event = PanDDAEvent(
#             id: event.id,
#         pandda_dir: str,
#         model_building_dir: str,
#         system_name: str,
#         dtag: str,
#         event_idx: int,
#         event_map: str,
#         x: float,
#         y: float,
#         z: float,
#         hit: bool,
#         # ligand: Ligand | None
#         ligand: str
#         )
#
#     else:
#         return event

def _get_event_mtz_path(event, sample_specification):
    sample_specification['event_mtz_path'] = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
        dtag=event.dtag)
    return sample_specification

def _get_event_map_path(event, sample_specification):
    sample_specification['event_map_path'] = event.event_map
    return sample_specification

def _get_xmap_path(event, sample_specification):

    sample_specification['xmap_path'] = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
        dtag=event.dtag)
    return sample_specification

def _get_zmap_path(event, sample_specification):

    sample_specification['zmap_path'] = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_ZMAP_TEMPLATE.format(
        dtag=event.dtag)
    return sample_specification

def _get_structure_path(event, sample_specification):
    sample_specification['structure_path'] = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=event.dtag)
    return sample_specification

def _get_bound_state_model_path(event, sample_specification):
    # model_path_4 = Path(
    #     event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INSPECT_MODEL_DIR / constants.PANDDA_MODEL_FILE.format(
    #     dtag=event.dtag)
    # model_path_1 = Path(event.pandda_dir).parent / "model_building" / event.dtag / constants.PANDDA_MODEL_FILE.format(
    #     dtag=event.dtag)
    # model_path_2 = Path(event.pandda_dir).parent / "initial_model" / event.dtag / constants.PANDDA_MODEL_FILE.format(
    #     dtag=event.dtag)
    model_path_1 = (Path(event.pandda_dir).parent / "model_building" / event.dtag / 'refine.pdb').resolve()
    model_path_2 = (Path(event.pandda_dir).parent / "initial_model" / event.dtag / 'refine.pdb').resolve()
    model_path_3 = Path("/dls/science/groups/i04-1/conor_dev/experiments/data") / event.system_name / f"{event.dtag}.pdb"
    if model_path_1.exists():
        sample_specification['bound_state_structure_path'] = str(model_path_1)
    elif model_path_2.exists():
        sample_specification['bound_state_structure_path'] = str(model_path_2)
    elif model_path_3.exists():
        sample_specification['bound_state_structure_path'] = str(model_path_3)
    # elif model_path_4.exists():
    #     sample_specification['bound_state_structure_path'] = str(model_path_4)
    else:
        # print(f"No bound state structure at: {model_path_1} or {model_path_2} or {model_path_3} or {model_path_4}")
        print(f"No bound state structure at: {model_path_1} or {model_path_2} or {model_path_3}")

        sample_specification['bound_state_structure_path'] = None
    return sample_specification

def _get_annotation_from_ntuple(event, sample_specification):
    if event.RMSD < 2.5:
        sample_specification['annotation'] = True
    else:
        sample_specification['annotation'] = False
    return sample_specification



def _get_annotation_from_event(event, sample_specification):  # Updates annotation
    sample_specification['annotation'] = event.hit
    return sample_specification

def _decide_annotation(event, sample_specification):  # Updates annotation
    rng = default_rng()
    val = rng.random()
    if val > 0.5:
        sample_specification['annotation'] = True
    else:
        sample_specification['annotation'] = False

    return sample_specification

def get_ligand_cif_graph_matches(cif_path):
    # Open the cif document with gemmi
    if not Path(cif_path).exists():
        return []
    cif = gemmi.cif.read(str(cif_path))

    keys = [
        "comp_LIG",
        "comp_XXX",
        "data_comp_XXX",
        "comp_DRG",
        "comp_UNL",
        "comp_F10",
        "comp_F18",
    ]
    key = None
    for _key in keys:
        try:
            cif[_key]
            key = _key
            break
        except Exception as e:
            continue
    if key is None:
        raise Exception([x for x in cif])

    # key = "comp_LIG"
    # try:
    #     cif[key]
    # except Exception as e:
    #     print(e)
    #     key = "data_comp_XXX"
    #     try:
    #         cif[key]
    #     except Exception as e:
    #         print(e)
    #         key = "comp_DRG"
    #         try:
    #             cif[key]
    #         except Exception as e:
    #             print(e)
    #             key = "comp_UNL"
    #             try:
    #                 cif[key]
    #             except Exception as e:
    #                 print(e)
    #                 key = "comp_F10"
    #                 try:
    #                     cif[key]
    #                 except Exception as e:
    #                     print(e)
    #                     key = "comp_XXX"
    #                     try:
    #                         cif[key]
    #                     except Exception as e:
    #                         print(e)
    #                         raise Exception([x for x in cif])
                # return []

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))

    # Construct the graph nodes
    G = nx.Graph()

    for atom_id, atom_type in zip(atom_id_loop, atom_type_loop):
        if atom_type == "H":
            continue
        G.add_node(atom_id, Z=atom_type)

    # Construct the graph edges
    for atom_id_1, atom_id_2 in zip(bond_1_id_loop, bond_2_id_loop):
        if atom_id_1 not in G:
            continue
        if atom_id_2 not in G:
            continue
        G.add_edge(atom_id_1, atom_id_2)

    # Get the isomorphisms
    GM = iso.GraphMatcher(G, G, node_match=iso.categorical_node_match('Z', 0))

    return [x for x in GM.isomorphisms_iter()]

def get_rmsd(
        pose_res,
        ref_res,
        ligand_graph
):
    # Iterate over each isorhpism, then get symmetric distance to the relevant atom
    iso_distances = []
    for isomorphism in ligand_graph:
        # print(isomorphism)
        distances = []
        for atom in ref_res:
            if atom.element.name == "H":
                continue
            try:
                pose_atom = pose_res[isomorphism[atom.name]][0]
            except:
                return None
            dist = atom.pos.dist(pose_atom.pos)
            distances.append(dist)
        # print(distances)
        rmsd = np.sqrt(np.mean(np.square(distances)))
        iso_distances.append(rmsd)
    return min(iso_distances)

def generate_ligand_pose(closest_ligand_res, min_transform=0.0, max_transform=2.0):
    # Get the ligand centroid
    poss = []
    for atom in closest_ligand_res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    initial_ligand_centroid = np.mean(poss, axis=0)

    # Get the translation
    rng = default_rng()
    val = rng.random(3)

    point = min_transform + (val*(max_transform-min_transform))

    val_2 = rng.random(3)
    if val_2[0] < 0.5:
        point[0] = -point[0]
    if val_2[1] < 0.5:
        point[1] = -point[1]
    if val_2[2] < 0.5:
        point[2] = -point[2]

    # Get the rotation
    rotation_matrix = R.random().as_matrix()

    # Create the res clone
    posed_res = closest_ligand_res.clone()

    # Generate rotation matrix
    rotation_transform = gemmi.Transform()
    rotation_transform.mat.fromlist(rotation_matrix.tolist())

    # Get the centering transform
    ligand_centre_transform = gemmi.Transform()
    ligand_centre_transform.vec.fromlist([-x for x in initial_ligand_centroid])

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([(x+point[j]) for j, x in enumerate([k for k in initial_ligand_centroid])])

    # Apply random translation
    # transform = #random_translation_transform.combine(
    transform = event_centre_transform.combine(
        rotation_transform.combine(
                ligand_centre_transform
        )
    )

    # Transform the atoms
    for atom in posed_res:
        transformed_pos = transform.apply(atom.pos)
        atom.pos = gemmi.Position(transformed_pos.x, transformed_pos.y, transformed_pos.z)

    return posed_res


def get_closest_ligand_res(st, event_centroid_pos):
    centroids = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ["LIG", "XXX"]:
                    poss = []
                    for atom in res:
                        pos = atom.pos
                        poss.append([pos.x, pos.y, pos.z])
                    centroid = np.mean(poss, axis=0)
                    centroids[f"{chain.name}_{res.seqid.num}"] = {
                        "Distance": gemmi.Position(centroid[0], centroid[1], centroid[2]).dist(event_centroid_pos),
                    "Residue": res
                    }
    if len(centroids) == 0:
        return None

    closest_res = min(centroids, key=lambda _key: centroids[_key]['Distance'])
    if centroids[closest_res]['Distance'] > 6.0:
        return None
    return centroids[closest_res]['Residue']

def _get_transformed_ligand(event, sample_specification):  # Updates ligand_res

    # Load the ligand cif
    ligand_cif_path = sample_specification['ligand_path']
    if not ligand_cif_path:
        print(f"No ligand for dataset with event map: {event.event_map}")
        sample_specification['ligand_res'] = None
        sample_specification['annotation'] = False

        return sample_specification

    # Get isomorphisms
    isomorphisms = get_ligand_cif_graph_matches(ligand_cif_path)
    if len(isomorphisms) == 0:
        print(f"No isomorphisms for: {ligand_cif_path}: Length: {len(isomorphisms)}")
        sample_specification['ligand_res'] = None
        sample_specification['annotation'] = False

        return sample_specification

    # Load the ligand bound structure
    if sample_specification['bound_state_structure_path'] is not None:
        st = gemmi.read_structure(str(sample_specification['bound_state_structure_path']))
    else:
        sample_specification['ligand_res'] = None
        sample_specification['annotation'] = False

        return sample_specification

    # Get the closest ligand in structure to the event
    closest_ligand_res = get_closest_ligand_res(st, gemmi.Position(event.x, event.y, event.z))
    if not closest_ligand_res:
        sample_specification['ligand_res'] = None
        sample_specification['annotation'] = False

        return sample_specification

    # Get the annotation
    annotation = sample_specification['annotation']

    # If a hit, try to generate a low RMSD pose
    if annotation:
        rmsd = 10.0
        j = 0
        while rmsd > 2.0:
            posed_ligand_res = generate_ligand_pose(closest_ligand_res, 0.0, 1.5)
            new_rmsds = [rmsd,]
            new_rmsds.append(
                get_rmsd(
                    posed_ligand_res,
                    closest_ligand_res,
                    isomorphisms
                )
            )
            rmsd = min([x for x in new_rmsds if x is not None])
            j += 1
            if j > 1000:
                print(f"Failed to sample a low RMSD pose! Setting annotation to False! {new_rmsds}")
                posed_ligand_res = None
                sample_specification['annotation'] = False
                break

        # if posed_ligand_res is not None:
        #     print(f"Sucessfully generated low RMSD pose")

    # If not a hit, generate a high rmsd pose
    else:
        posed_ligand_res = generate_ligand_pose(closest_ligand_res, 1.5, 12.0)

    sample_specification['ligand_res'] = posed_ligand_res

    return sample_specification

def _get_non_transformed_ligand(event, sample_specification):
    # Load the ligand bound structure
    if sample_specification['bound_state_structure_path'] is not None:
        st = gemmi.read_structure(str(sample_specification['bound_state_structure_path']))
    else:
        print(f"No bound state structure! Cannot generate ligand_res!")
        sample_specification['ligand_res'] = None
        return sample_specification

    closest_ligand_res = get_closest_ligand_res(st, gemmi.Position(event.x, event.y, event.z))

    if closest_ligand_res:
        sample_specification['ligand_res'] = closest_ligand_res
    else:
        print(f"No closest ligand res! Cannot generate ligand_res!")
        sample_specification['ligand_res'] = None
    return sample_specification

def _sample_point(lower, upper):
    rng = default_rng()
    val = rng.random(3)

    point = lower + (val*(upper-lower))
    return point

def _sample_to_ligand_distance(point, ligand_array):
    distances = np.linalg.norm(ligand_array-point.reshape((1,3)), axis=1)
    closest_distance = np.min(distances)
    return closest_distance

def _get_centroid_from_ntuple(event, sample_specification):
    sample_specification['centroid'] = [event.X_ligand, event.Y_ligand, event.Z_ligand]
    return sample_specification

def _get_centroid_relative_to_ligand(event, sample_specification):  # updates centroid and annotation
    rng = default_rng()
    val = rng.random()
    if (sample_specification['bound_state_structure_path'] is not None) & sample_specification['annotation']:
        st = gemmi.read_structure(str(sample_specification['bound_state_structure_path']))
        original_centroid = [event.x, event.y, event.z]
        original_centroid_pos = gemmi.Position(*original_centroid)
        lig_atom_poss = [[event.x, event.y, event.z]]
        for model in st:
            for chain in model:
                for residue in chain:
                    if residue.name == "LIG":
                        for atom in residue:
                            pos = atom.pos
                            if pos.dist(original_centroid_pos) < 10.0:
                                lig_atom_poss.append(
                                    [pos.x, pos.y, pos.z]
                                )

        if len(lig_atom_poss) < 3:
            sample_specification['centroid'] = [event.x, event.y, event.z]

        else:
            ligand_array = np.array(lig_atom_poss)
            lower = np.min(ligand_array, axis=0).flatten() - 4.0
            upper = np.max(ligand_array, axis=0).flatten() + 4.0


            j = 0
            # Sample a ligand point
            if val < 0.5:
                sample_specification['annotation'] = sample_specification['annotation']
                sample = _sample_point(lower, upper)
                while _sample_to_ligand_distance(sample, ligand_array) > 1.5:
                    sample = _sample_point(lower, upper)
                    j+=1
                    if j > 200:
                        print(f"Failed to get a sample point for event map: {event.event_map}!")
                        sample = [event.x, event.y, event.z]
                        break

            # Sample a non-ligand point
            else:
                sample_specification['annotation'] = False
                sample = _sample_point(lower, upper)
                while _sample_to_ligand_distance(sample, ligand_array) < 1.5:
                    sample = _sample_point(lower, upper)
                    j += 1
                    if j > 200:
                        print(f"Failed to get a sample point for event map: {event.event_map}!")
                        sample = [event.x, event.y, event.z]
                        break

            sample_specification['centroid'] = [sample[0], sample[1], sample[2]]


    else:
        sample_specification['centroid'] = [event.x, event.y, event.z]

    return sample_specification

def _get_centroid_relative_to_transformed_ligand(event, sample_specification):  # updates centroid and annotation

    ligand_res = sample_specification['ligand_res']
    if ligand_res is not None:
        original_centroid = [event.x, event.y, event.z]
        original_centroid_pos = gemmi.Position(*original_centroid)
        lig_atom_poss = [[event.x, event.y, event.z]]
        for atom in ligand_res:
            pos = atom.pos
            if pos.dist(original_centroid_pos) < 10.0:
                lig_atom_poss.append(
                    [pos.x, pos.y, pos.z]
                )

        if len(lig_atom_poss) < 3:
            sample_specification['centroid'] = [event.x, event.y, event.z]

        else:
            ligand_array = np.array(lig_atom_poss)
            lower = np.min(ligand_array, axis=0).flatten() - 4.0
            upper = np.max(ligand_array, axis=0).flatten() + 4.0


            j = 0
            # Sample a ligand point
            sample_specification['annotation'] = sample_specification['annotation']
            sample = _sample_point(lower, upper)
            while _sample_to_ligand_distance(sample, ligand_array) > 1.5:
                sample = _sample_point(lower, upper)
                j+=1
                if j > 200:
                    print(f"Failed to get a sample point for event map: {event.event_map}!")
                    sample = [event.x, event.y, event.z]
                    break

            sample_specification['centroid'] = [sample[0], sample[1], sample[2]]

    else:
        sample_specification['centroid'] = [event.x, event.y, event.z]

    return sample_specification

def _get_random_ligand_path(event, sample_specification):  # Updates ligand_path and annotation
    sample_specification['ligand_path'] = event.ligand
    return sample_specification

def _get_random_orientation(event, sample_specification):  # Updates orientation
    rotation_matrix = R.random().as_matrix()

    sample_specification['orientation'] = rotation_matrix
    return sample_specification

def _get_transform_from_ntuple(event, sample_specification):
    sample_distance: float = 0.5
    n: int = 30
    # translation: float):

    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
    sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Generate rotation matrix
    rotation_matrix = sample_specification['orientation']
    rotation_transform = gemmi.Transform()
    rotation_transform.mat.fromlist(rotation_matrix.tolist())

    # Apply random rotation transform to centroid
    transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
    transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])

    # Recentre transform
    rotation_recentre_transform = gemmi.Transform()
    rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist(sample_specification['centroid'])

    # Generate random translation transform
    # rng = default_rng()
    # random_translation = (rng.random(3) - 0.5) * 2 #* translation
    # random_translation = np.array([0.0,0.0,0.0])
    # logger.debug(f"Random translation: {random_translation}")
    # random_translation_transform = gemmi.Transform()
    # random_translation_transform.vec.fromlist(random_translation.tolist())

    # Apply random translation
    # transform = #random_translation_transform.combine(
    transform = event_centre_transform.combine(
        rotation_transform.combine(
            centre_grid_transform.combine(
                initial_transform
            )
        )
    )
    # )
    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = (event.X_ligand, event.Y_ligand, event.Z_ligand)
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    # return transform, np.zeros((n, n, n), dtype=np.float32)
    sample_specification['transform'] = transform
    sample_specification['n'] =n
    sample_specification['step'] = sample_distance

    sample_specification['sample_grid'] = np.zeros((n, n, n), dtype=np.float32)
    return sample_specification


def _get_transform(event, sample_specification):
    sample_distance: float = 0.5
    n: int = 30
    # translation: float):

    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = (np.array([n, n, n]) * sample_distance) / 2
    sample_grid_centroid_pos = gemmi.Position(*sample_grid_centroid)

    # Get centre grid transform
    centre_grid_transform = gemmi.Transform()
    centre_grid_transform.vec.fromlist([
        -sample_grid_centroid[0],
        -sample_grid_centroid[1],
        -sample_grid_centroid[2],
    ])

    # Generate rotation matrix
    rotation_matrix = sample_specification['orientation']
    rotation_transform = gemmi.Transform()
    rotation_transform.mat.fromlist(rotation_matrix.tolist())

    # Apply random rotation transform to centroid
    transformed_centroid = rotation_transform.apply(sample_grid_centroid_pos)
    transformed_centroid_array = np.array([transformed_centroid.x, transformed_centroid.y, transformed_centroid.z])

    # Recentre transform
    rotation_recentre_transform = gemmi.Transform()
    rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist(sample_specification['centroid'])

    # Generate random translation transform
    # rng = default_rng()
    # random_translation = (rng.random(3) - 0.5) * 2 #* translation
    # random_translation = np.array([0.0,0.0,0.0])
    # logger.debug(f"Random translation: {random_translation}")
    # random_translation_transform = gemmi.Transform()
    # random_translation_transform.vec.fromlist(random_translation.tolist())

    # Apply random translation
    # transform = #random_translation_transform.combine(
    transform = event_centre_transform.combine(
        rotation_transform.combine(
            centre_grid_transform.combine(
                initial_transform
            )
        )
    )
    # )
    corner_0_pos = transform.apply(gemmi.Position(0.0, 0.0, 0.0))
    corner_n_pos = transform.apply(gemmi.Position(
        float(n),
        float(n),
        float(n),
    )
    )
    corner_0 = (corner_0_pos.x, corner_0_pos.y, corner_0_pos.z)
    corner_n = (corner_n_pos.x, corner_n_pos.y, corner_n_pos.z)
    average_pos = [c0 + (cn - c0) / 2 for c0, cn in zip(corner_0, corner_n)]
    event_centroid = (event.x, event.y, event.z)
    # logger.debug(f"Centroid: {event_centroid}")
    # logger.debug(f"Corners: {corner_0} : {corner_n} : average: {average_pos}")
    # logger.debug(f"Distance from centroid to average: {gemmi.Position(*average_pos).dist(gemmi.Position(*event_centroid))}")

    # return transform, np.zeros((n, n, n), dtype=np.float32)
    sample_specification['transform'] = transform
    sample_specification['n'] =n
    sample_specification['step'] = sample_distance

    sample_specification['sample_grid'] = np.zeros((n, n, n), dtype=np.float32)
    return sample_specification


def _make_xmap_layer(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        event_mtz_path = sample_specification['event_mtz_path']
        sample_transform = sample_specification['transform']

        sample_array_mean = np.copy(sample_array)
        # mean_dmap = get_event_map_from_event(event)
        mean_dmap = load_xmap_from_mtz(event_mtz_path)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        std = np.std(image_mean_initial)
        if np.abs(std) < 0.0000001:
            image_event_map = np.copy(sample_array)
        else:
            image_event_map = (image_mean_initial - np.mean(image_mean_initial)) / std
        # time_finish_get_mean = time.time()
        # time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)
        sample_specification['xmap'] = mean_dmap
        sample_specification['xmap_layer'] = image_event_map

    except Exception as e:
        print(f"Error loading event map: {e}")
        sample_specification['xmap'] = None
        sample_specification['xmap_layer'] = None

    return sample_specification

def _make_zmap_layer(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        event_map_path = sample_specification['zmap_path']
        sample_transform = sample_specification['transform']

        sample_array_mean = np.copy(sample_array)
        # mean_dmap = get_event_map_from_event(event)
        mean_dmap = get_map_from_path(str(event_map_path))
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        std = np.std(image_mean_initial)
        if np.abs(std) < 0.0000001:
            image_event_map = np.copy(sample_array)
        else:
            image_event_map = (image_mean_initial - np.mean(image_mean_initial)) / std
        # time_finish_get_mean = time.time()
        # time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)
        sample_specification['zmap'] = mean_dmap
        sample_specification['zmap_layer'] = image_event_map

    except Exception as e:
        print(f"Error loading event map: {e}")
        sample_specification['zmap'] = None
        sample_specification['zmap_layer'] = None

    return sample_specification

def _make_event_map_layer(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        event_map_path = sample_specification['event_map_path']
        sample_transform = sample_specification['transform']

        sample_array_mean = np.copy(sample_array)
        # mean_dmap = get_event_map_from_event(event)
        mean_dmap = get_map_from_path(event_map_path)
        image_mean_initial = sample_xmap(mean_dmap, sample_transform, sample_array_mean)
        std = np.std(image_mean_initial)
        if np.abs(std) < 0.0000001:
            image_event_map = np.copy(sample_array)
        else:
            image_event_map = (image_mean_initial - np.mean(image_mean_initial)) / std
        # time_finish_get_mean = time.time()
        # time_get_mean = round(time_finish_get_mean - time_begin_get_mean, 2)
        sample_specification['event_map'] = mean_dmap
        sample_specification['event_map_layer'] = image_event_map

    except Exception as e:
        print(f"Error loading event map: {e}")
        sample_specification['event_map'] = None
        sample_specification['event_map_layer'] = None

    return sample_specification

def _make_structure_map_layer(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        structure_path = sample_specification['structure_path']
        sample_transform = sample_specification['transform']
        mean_dmap = sample_specification['event_map']

        # time_begin_get_model = time.time()
        sample_array_model = np.copy(sample_array)
        model_map = get_model_map_from_path(structure_path, mean_dmap, )
        image_model = sample_xmap(model_map, sample_transform, sample_array_model)
        # time_finish_get_model = time.time()
        # time_get_model = round(time_finish_get_model - time_begin_get_model, 2)

        sample_specification['structure_map_layer'] = image_model
    except Exception as e:
        print(f"Error loading structure map: {e}")
        sample_specification['structure_map_layer'] = None
    return sample_specification

def _make_ligand_map_layer(event, sample_specification):
    try:
        # sample_array = sample_specification['sample_grid']
        ligand_path = sample_specification['ligand_path']
        # sample_transform = sample_specification['transform']
        # mean_dmap = sample_specification['event_map_layer']
        n = sample_specification['n']
        step = sample_specification['step']
        # ligand_map_array = np.copy(sample_array)
        # time_begin_get_ligand = time.time()
        ligand_map = get_ligand_map_from_path(ligand_path, n=n, step=step)
        image_ligand = np.array(ligand_map)
        # time_finish_get_ligand = time.time()
        # time_get_ligand = round(time_finish_get_ligand - time_begin_get_ligand, 2)

        sample_specification['ligand_map_layer'] = image_ligand
    except Exception as e:
        print(f"Error loading ligand map: {e}")
        sample_array = sample_specification['sample_grid']
        sample_specification['ligand_map_layer'] = np.copy(sample_array)
        sample_specification['annotation'] = False

    return sample_specification


def get_masked_dmap(dmap, res):
    mask = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(dmap.unit_cell)

    # Get the mask
    for atom in res:
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=2.5,
            value=1,
        )

    # Get the mask array
    mask_array = np.array(mask, copy=False)

    # Get the dmap array
    dmap_array = np.array(dmap, copy=False)

    # Mask the dmap array
    dmap_array[mask_array == 0] = 0.0

    return dmap

def get_event_map(
            xmap,
            mean_map,
            bdc
        ):
    dataset_map_array = np.array(xmap, copy=False)

    mean_map_array = np.array(mean_map, copy=False)

    calc_event_map_array = (dataset_map_array - (bdc * mean_map_array)) / (1-bdc)

    event_map = gemmi.FloatGrid(
        xmap.nu, xmap.nv, xmap.nw
    )
    event_map_array = np.array(event_map, copy=False)
    event_map_array[:, :, :] = calc_event_map_array[:, :, :]
    event_map.set_unit_cell(xmap.unit_cell)
    return event_map


def _make_ligand_masked_event_map_layer_from_ntuple(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        sample_transform = sample_specification['transform']

        autobuild_structure_path = event.Build_Path
        autobuild_structure = gemmi.read_structure(autobuild_structure_path)
        res = autobuild_structure[0][0][0]

        sample_array= np.copy(sample_array)
        mean_map = get_map_from_path(event.Mean_Map_Path)
        xmap = get_map_from_path(event.Xmap_Path)
        bdc = event.BDC
        event_map = get_event_map(
            xmap,
            mean_map,
            bdc
        )
        masked_dmap = get_masked_dmap(event_map, res)
        image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
        std = np.std(image_initial)
        if np.abs(std) < 0.0000001:
            image_dmap = np.copy(sample_array)
            sample_specification['annotation'] = False
        else:
            image_dmap = (image_initial - np.mean(image_initial)) / std
        # sample_specification['event_map'] = dmap
        sample_specification['ligand_masked_event_map_layer'] = image_dmap

    except Exception as e:
        print(f"Error making masked event map: {e}")
        # sample_specification['event_map'] = None
        sample_array = sample_specification['sample_grid']
        sample_specification['annotation'] = False
        sample_specification['ligand_masked_event_map_layer'] = np.copy(sample_array)

    return sample_specification

def _make_ligand_masked_z_map_layer_from_ntuple(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        sample_transform = sample_specification['transform']

        autobuild_structure_path = event.Build_Path
        autobuild_structure = gemmi.read_structure(autobuild_structure_path)
        res = autobuild_structure[0][0][0]

        sample_array= np.copy(sample_array)
        z_map = get_map_from_path(event.Zmap_Path)

        masked_dmap = get_masked_dmap(z_map, res)
        image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
        std = np.std(image_initial)
        if np.abs(std) < 0.0000001:
            image_dmap = np.copy(sample_array)
            sample_specification['annotation'] = False
        else:
            image_dmap = (image_initial - np.mean(image_initial)) / std
        # sample_specification['event_map'] = dmap
        sample_specification['ligand_masked_z_map_layer'] = image_dmap

    except Exception as e:
        print(f"Error making masked event map: {e}")
        # sample_specification['event_map'] = None
        sample_array = sample_specification['sample_grid']
        sample_specification['annotation'] = False
        sample_specification['ligand_masked_z_map_layer'] = np.copy(sample_array)

    return sample_specification


def _make_ligand_masked_raw_xmap_map_layer_from_ntuple(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        sample_transform = sample_specification['transform']

        autobuild_structure_path = event.Build_Path
        autobuild_structure = gemmi.read_structure(autobuild_structure_path)
        res = autobuild_structure[0][0][0]

        sample_array= np.copy(sample_array)
        x_map = load_xmap_from_mtz(event.Mtz_Path)

        masked_dmap = get_masked_dmap(x_map, res)
        image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
        std = np.std(image_initial)
        if np.abs(std) < 0.0000001:
            image_dmap = np.copy(sample_array)
            sample_specification['annotation'] = False
        else:
            image_dmap = (image_initial - np.mean(image_initial)) / std
        # sample_specification['event_map'] = dmap
        sample_specification['ligand_masked_raw_xmap_map_layer'] = image_dmap

    except Exception as e:
        print(f"Error making masked event map: {e}")
        # sample_specification['event_map'] = None
        sample_array = sample_specification['sample_grid']
        sample_specification['annotation'] = False
        sample_specification['ligand_masked_raw_xmap_map_layer'] = np.copy(sample_array)

    return sample_specification

def _make_ligand_masked_event_map_layer(event, sample_specification):
    try:
        sample_array = sample_specification['sample_grid']
        event_map_path = sample_specification['event_map_path']
        sample_transform = sample_specification['transform']
        res = sample_specification['ligand_res']
        if not res:
            raise Exception(f"No ligand res!")

        sample_array= np.copy(sample_array)
        dmap = get_map_from_path(event_map_path)
        masked_dmap = get_masked_dmap(dmap, res)
        image_initial = sample_xmap(masked_dmap, sample_transform, sample_array)
        std = np.std(image_initial)
        if np.abs(std) < 0.0000001:
            image_dmap = np.copy(sample_array)
            sample_specification['annotation'] = False
        else:
            image_dmap = (image_initial - np.mean(image_initial)) / std
        # sample_specification['event_map'] = dmap
        sample_specification['ligand_masked_event_map_layer'] = image_dmap

    except Exception as e:
        print(f"Error making masked event map: {e}")
        # sample_specification['event_map'] = None
        sample_array = sample_specification['sample_grid']
        sample_specification['annotation'] = False
        sample_specification['ligand_masked_event_map_layer'] = np.copy(sample_array)

    return sample_specification


def _make_label_from_specification(sample_specification, layers):

    for layer in layers:
        if sample_specification[layer] is None:
            print(f"Missing layer: {layer}")
            sample_specification['annotation'] = False

    if sample_specification['annotation']:
        label = np.array([0.0, 1.0], dtype=np.float32)

    else:
        label = np.array([1.0, 0.0], dtype=np.float32)

    return label

def _get_event_centroid(event, sample_specification):
    sample_specification['centroid'] = [event.x, event.y, event.z]
    return sample_specification

def _get_identity_orientation(event, sample_specification):
    rotation_matrix = np.eye(3)

    sample_specification['orientation'] = rotation_matrix
    return sample_specification

class PanDDADatasetTorchLigand(Dataset):
    def __init__(self,
                 pandda_event_dataset: PanDDAEventDataset,
                # transform_event=lambda x: x,
                #  transform_image=lambda x: x,
                #  transform_annotation=lambda x: x
                 update_sample_specification,
                 layers,
                 ):
        self.pandda_event_dataset = pandda_event_dataset
        self.update_sample_specification = update_sample_specification
        self.layers = layers
        # self.transform_event = transform_event
        # self.transform_image = transform_image
        # self.transform_annotation = transform_annotation

    def __len__(self):
        return len(self.pandda_event_dataset)

    def __getitem__(self, idx: int):
        time_begin_load_item = time.time()
        event = self.pandda_event_dataset[idx]

        # annotation = event.hit

        # image, loaded, transform, dmap = self.transform_image(event)

        # if loaded:
        #     label = self.transform_annotation(annotation)
        #
        # else:
        #     label = np.array([1.0, 0.0], dtype=np.float32)


        sample_specification = {}
        for _update in self.update_sample_specification:
            sample_specification = _update(event, sample_specification)

        # augmented_event = self.augment_event
        # centroid = self.get_centroid_from_event


        # Make the image
        image = np.stack(
            [sample_specification[layer] for layer in self.layers],
            axis=0,
        )

        # Make the annotation
        label = _make_label_from_specification(sample_specification, self.layers)

        time_finish_load_item = time.time()
        # print(f"Loaded item in: {round(time_finish_load_item-time_begin_load_item, 2)}")
        return image, label, idx


def get_image_ligandmap_augmented(event, xmap_event, transform, n=30):
    sample_array = np.zeros((n, n, n), dtype=np.float32)
    model_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INSPECT_MODEL_DIR / constants.PANDDA_MODEL_FILE.format(
        dtag=event.dtag)
    if model_path.exists() and transform:
        st = gemmi.read_structure(str(model_path))

        new_xmap = gemmi.FloatGrid(xmap_event.nu, xmap_event.nv, xmap_event.nw)
        new_xmap.spacegroup = xmap_event.spacegroup
        new_xmap.set_unit_cell(xmap_event.unit_cell)
        for model in st:
            for chain in model:
                for residue in chain:
                    if residue.name == "LIG":
                        for atom in residue:
                            new_xmap.set_points_around(
                                atom.pos,
                                radius=1.5,
                                value=1.0,
                            )
        image_ligandmap = sample_xmap(new_xmap, transform, sample_array)

        return image_ligandmap, True
    else:
        return sample_array, False


class PanDDADatasetTorchLigandmap(Dataset):
    def __init__(self,
                 pandda_event_dataset: PanDDAEventDataset,

                 transform_image=lambda x: x,
                 transform_annotation=lambda x: x,
                transform_ligandmap=lambda x: x
                 ):
        self.pandda_event_dataset = pandda_event_dataset

        self.transform_image = transform_image
        self.transform_annotation = transform_annotation
        self.transform_ligandmap = transform_ligandmap

    def __len__(self):
        return len(self.pandda_event_dataset.pandda_events)

    def __getitem__(self, idx: int):
        time_begin_load_item = time.time()
        event = self.pandda_event_dataset.pandda_events[idx]

        annotation = event.hit

        image, loaded_classification, transform, xmap_event = self.transform_image(event)

        ligandmap, loaded_ligandmap = self.transform_ligandmap(event, xmap_event, transform, )

        label = self.transform_annotation(annotation, )

        time_finish_load_item = time.time()
        # print(f"Loaded item in: {round(time_finish_load_item-time_begin_load_item, 2)}")
        # Censor if a hit and no ligandmap
        if event.hit and (not loaded_ligandmap):
            return np.zeros((4,30,30,30), dtype=np.float32), label, np.zeros((30,30,30), dtype=np.float32), loaded_classification, loaded_ligandmap, idx
        else:
            return image, label, ligandmap, loaded_classification, loaded_ligandmap, idx

