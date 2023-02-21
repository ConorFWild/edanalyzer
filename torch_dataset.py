from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand, PanDDAEventDataset, \
    PanDDAEvent, PanDDAEventAnnotations, PanDDAEventAnnotation
from numpy.random import default_rng
import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger
import gemmi
from torch.utils.data import Dataset



def load_xmap_from_mtz(path):
    ...


def sample_xmap(xmap, transform, sample_array):
    xmap.sample_values(sample_array, transform)
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


def get_sample_transform_from_event(event: PanDDAEvent,
                                    sample_distance: float,
                                    n: int,
                                    translation: float):
    # Get basic sample grid transform
    initial_transform = gemmi.Transform()
    scale_matrix = np.eye(3) * sample_distance
    initial_transform.mat.fromlist(scale_matrix.tolist())

    # Get sample grid centroid
    sample_grid_centroid = np.array([n, n, n]) * sample_distance
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
    rotation_recentre_transform.vec.fromlist((sample_grid_centroid - transformed_centroid_array).tolist())

    # Event centre transform
    event_centre_transform = gemmi.Transform()
    event_centre_transform.vec.fromlist([event.x, event.y, event.z])

    # Generate random translation transform
    rng = default_rng()
    random_translation = (rng.random_sample(3) - 0.5) * 2 * translation
    random_translation_transform = gemmi.Transform()
    random_translation_transform.vec.fromlist(random_translation.tolist())

    # Apply random translation
    transform = random_translation_transform.combine(
        event_centre_transform.combine(
            rotation_transform.combine(
                event_centre_transform.combine(
                    initial_transform
                )
            )
        )
    )
    corner_0 = gemmi.Position(0.0,0.0,0.0)
    corner_n = gemmi.Position(float(n),float(n),float(n))
    logger.debug(f"Corners: {transform.apply(corner_0)} : {transform.apply(corner_n)}")

    return transform, np.zeros((n,n,n))


def get_image_from_event(event: PanDDAEvent):
    xmap = get_xmap_from_event(event)

    sample_transform, sample_array = get_sample_transform_from_event(event,
                                                                     0.5,
                                                                     30,
                                                                     3.5
                                                                     )

    image = sample_xmap(xmap, sample_transform, sample_array)

    return image


def get_annotation_from_event_annotation(annotation: PanDDAEventAnnotation):
    if annotation.annotation:
        return np.array([0.0, 1.0])
    else:
        return np.array([1.0, 0.0])


class PanDDAEventDatasetTorch(Dataset):
    def __init__(self,
                 pandda_event_dataset: PanDDAEventDataset,
                 annotations: PanDDAEventAnnotations,
                 transform_image=lambda x: x,
                 transform_annotation=lambda x: x
                 ):
        self.pandda_event_dataset = pandda_event_dataset
        self.annotations = annotations
        self.transform_image = transform_image
        self.transform_annotation = transform_annotation

    def __len__(self):
        return len(self.pandda_event_dataset.pandda_events)

    def __getitem__(self, idx: int):
        event = self.pandda_event_dataset.pandda_events[idx]

        annotation = self.annotations.annotations[idx]

        image = self.transform_image(event)

        label = self.transform_annotation(annotation)

        return image, label

