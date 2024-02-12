import random
from pathlib import Path
import dataclasses

import numpy as np
import torch

from torch.utils.data import Dataset

from .base import (
    _load_xmap_from_path,
    _sample_xmap_and_scale,
    _get_identity_matrix,
    _get_transform_from_orientation_centroid,
    _get_ligand_from_dir,
    _get_ligand_map
)


@dataclasses.dataclass
class EventScoringDatasetItem:
    dtag: str
    event_idx: int
    x: float
    y: float
    z: float
    bdc: float
    initial_structure: str
    initial_reflections: str
    structure: str
    event_map: str
    z_map: str
    viewed: bool
    hit_confidence: str

    # ligand = Optional("LigandORM", cascade_delete=True)
    # dataset = Optional("DatasetORM", column="dataset_id")
    # pandda = Required("PanDDAORM", column="pandda_id")
    # annotations = Set("AnnotationORM")
    # partitions = Set("PartitionORM", table=constants.TABLE_EVENT_PARTITION, column="partition_id")


class EventScoringDataset(Dataset):
    def __init__(
            self,
            data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )

        try:
            z_map = _load_xmap_from_path(sample.z_map)
            ligand_array = _get_ligand_from_dir(Path(sample.event_map).parent.resolve())
            assert ligand_array.size > 0, f"Ligand array empty!"
        except Exception as e:
            print(e)
            image = np.stack(
                [
                    sample_array,
                    sample_array,
                ],
                axis=0,
            )
            image_float = image.astype(np.float32)

            label = np.array([1.0, 0.0])
            label_float = label.astype(np.float32)
            return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)

        # Get sampling transform
        orientation = _get_identity_matrix()
        centroid = np.array([sample.x, sample.y, sample.z])
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get sample image
        z_map_sample = _sample_xmap_and_scale(
            z_map, transform, np.copy(sample_array)
        )
        ligand_map = _get_ligand_map(ligand_array)
        ligand_map_sample = np.array(ligand_map, copy=True)

        # Make the image
        image = np.stack(
            [
                z_map_sample,
                ligand_map_sample
            ],
            axis=0,
        )
        image_float = image.astype(np.float32)

        # Make the annotation
        if sample.hit_confidence not in ['low', 'Low']:
            label = np.array([0.0, 1.0])
        else:
            label = np.array([1.0, 0.0])
        label = np.array(label)
        label_float = label.astype(np.float32)

        return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)
