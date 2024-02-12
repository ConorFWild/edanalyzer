import random
from pathlib import Path
import dataclasses

import numpy as np
import torch

from torch.utils.data import Dataset

from .base import (
    _load_xmap_from_mtz_path,
    _load_xmap_from_path,
    _sample_xmap_and_scale,
    _get_ligand_mask_float,
    _sample_xmap,
    _get_identity_matrix,
    _get_centroid_from_res,
    _get_transform_from_orientation_centroid,
    _get_res_from_structure_chain_res,
    _get_structure_from_path,
    _get_ligand_from_dir,
    _get_ligand_map
)

@dataclasses.dataclass
class BuildScoringDatasetItem:
    experiment_model_dir: str
    pandda_path: str
    dtag: str
    model_idx : int
    event_idx : int
    known_hit_key: str
    ligand_key: str
    rmsd : float
    score: float
    size : float
    local_strength : float
    rscc : float
    signal : float
    noise : float
    signal_noise : float
    x_ligand : float
    y_ligand : float
    z_ligand : float
    x : float
    y : float
    z: float
    build_path: str
    bdc : float
    xmap_path : str
    mean_map_path: str
    mtz_path : str
    zmap_path : str
    train_test : str



class BuildScoringDataset(Dataset):
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
            z_map = _load_xmap_from_path(sample.zmap_path)
            ligand_array = _get_ligand_from_dir((Path(sample.event_map) / '..').resolve())
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

            label = np.array([1.0,0.0])
            label_float = label.astype(np.float32)
            return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)


        # Get sampling transform
        orientation = _get_identity_matrix()
        centroid = _get_centroid_from_res(residue)
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
            label = np.array([0.0,1.0])
        else:
            label = np.array([1.0,0.0])
        label = np.array(label)
        label_float = label.astype(np.float32)

        return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)
