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
    _get_random_orientation,
    _get_centroid_from_res,
    _get_transform_from_orientation_centroid,
    _get_res_from_structure_chain_res,
    _get_structure_from_path
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
            structure = _get_structure_from_path(sample.build_path)
            xmap = _load_xmap_from_path(sample.xmap_path)
            mean_map = _load_xmap_from_path(sample.mean_map_path)
            z_map = _load_xmap_from_path(sample.zmap_path)
            raw_xmap = _load_xmap_from_mtz_path(sample.mtz_path)
        except Exception as e:
            print(e)
            image = np.stack(
                [
                    sample_array,
                    sample_array,
                    sample_array,
                    sample_array,
                ],
                axis=0,
            )
            image_float = image.astype(np.float32)

            label = np.array(3.0)
            label_float = label.astype(np.float32)
            return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)


        residue = _get_res_from_structure_chain_res(
            structure,
            0,
            0
        )

        # Get sampling transform
        # orientation = _get_identity_matrix()
        orientation = _get_random_orientation()
        centroid = _get_centroid_from_res(residue)
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get sample image


        # event_map = _load_xmap_from_path(sample.event_map_path)
        # event_map_sample = _sample_xmap_and_scale(
        #     event_map, transform, np.copy(sample_array)
        # )

        xmap_sample = _sample_xmap_and_scale(
            xmap, transform, np.copy(sample_array)
        )

        mean_map_sample = _sample_xmap_and_scale(
            mean_map, transform, np.copy(sample_array)
        )

        z_map_sample = _sample_xmap_and_scale(
            z_map, transform, np.copy(sample_array)
        )

        raw_xmap_sample = _sample_xmap_and_scale(
            raw_xmap, transform, np.copy(sample_array)
        )

        ligand_mask_grid = _get_ligand_mask_float(xmap, residue)
        image_ligand_mask = _sample_xmap(
            ligand_mask_grid,
            transform,
            np.copy(sample_array)
        )
        image_ligand_mask[image_ligand_mask < 0.9] = 0.0
        image_ligand_mask[image_ligand_mask > 0.9] = 1.0

        # Make the image
        image = np.stack(
            [
                xmap_sample * image_ligand_mask,
                mean_map_sample * image_ligand_mask,
                z_map_sample * image_ligand_mask,
                raw_xmap_sample * image_ligand_mask,
            ],
            axis=0,
        )
        image_float = image.astype(np.float32)


        # Make the annotation
        if sample.rmsd > 3.0:
            rmsd = 3.0
        else:
            rmsd = sample.rmsd
        label = np.array(rmsd)
        label_float = label.astype(np.float32)

        return idx, torch.from_numpy(image_float), torch.from_numpy(label_float)
