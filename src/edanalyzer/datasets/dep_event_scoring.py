import time
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
    _get_random_orientation,
    _get_transform_from_orientation_centroid,
    _get_ligand_from_dir,
    _get_ligand_map
)


@dataclasses.dataclass
class EventScoringDatasetItem:
    annotation: bool
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


class EventScoringDataset(Dataset):
    def __init__(
            self,
            data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        time_begin_load = time.time()
        sample = self.data[idx]

        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )
        time_begin_data = time.time()
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
        time_finish_data = time.time()

        # Get sampling transform
        # orientation = _get_identity_matrix()
        orientation = _get_random_orientation()
        centroid = np.array([sample.x, sample.y, sample.z])
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get sample image
        time_begin_sample = time.time()
        z_map_sample = _sample_xmap_and_scale(
            z_map, transform, np.copy(sample_array)
        )
        ligand_map = _get_ligand_map(ligand_array)
        ligand_map_sample = np.array(ligand_map, copy=True)
        time_finish_sample=time.time()

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
        if sample.annotation:
            label = np.array([0.0, 1.0])
        else:
            label = np.array([1.0, 0.0])
        label = np.array(label)
        label_float = label.astype(np.float32)

        time_finish_load = time.time()

        # rprint(f"Loaded in: TOT: {}; DATA: {}; SAMP: {}")

        return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)
