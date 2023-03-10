import constants
from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand, PanDDAEventDataset, \
    PanDDAEvent, PanDDAEventAnnotations, PanDDAEventAnnotation, PanDDAUpdatedEventAnnotations
from numpy.random import default_rng
import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
import gemmi
from torch.utils.data import Dataset
from pathlib import Path


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
                        radius=1,
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
