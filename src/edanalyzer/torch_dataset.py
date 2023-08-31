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
]


def get_ligand_map(
        event,
        n=30,
        step=0.5,
        translation=2.5,
):
    # Get the path to the ligand cif
    # dataset_dir = Path(event.model_building_dir) / event.dtag / "compound"
    dataset_dir = Path(event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / "ligand_files"
    paths = [x for x in dataset_dir.glob("*.pdb")]
    pdb_paths = [x for x in paths if (x.exists()) and (x.stem not in LIGAND_IGNORE_REGEXES)]
    assert len(pdb_paths) > 0, f"No pdb paths that are not to be ignored in {dataset_dir}: {[x.stem for x in paths]}"
    path = pdb_paths[0]

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


class PanDDADatasetTorchLigand(Dataset):
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
        time_begin_load_item = time.time()
        event = self.pandda_event_dataset.pandda_events[idx]

        annotation = event.hit

        image, loaded, transform, dmap = self.transform_image(event)

        if loaded:
            label = self.transform_annotation(annotation)

        else:
            label = np.array([1.0, 0.0], dtype=np.float32)

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

