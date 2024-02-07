import numpy as np

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
    _get_structure_from_path
)


class PanDDADatasetTorchLigand(Dataset):
    def __init__(
            self,
            data
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        structure = _get_structure_from_path(sample.bound_state_structure)
        residue = _get_res_from_structure_chain_res(
            structure,
            sample.chain,
            sample.res
        )

        # Get sampling transform
        orientation = _get_identity_matrix()
        centroid = _get_centroid_from_res(residue)
        transform = _get_transform_from_orientation_centroid(
            orientation,
            centroid
        )

        # Get sample image
        sample_array = np.zeros(
            (30, 30, 30),
            dtype=np.float32,
        )

        event_map = _load_xmap_from_path(sample.event_map_path)
        event_map_sample = _sample_xmap_and_scale(
            event_map, transform, np.copy(sample_array)
        )

        z_map = _load_xmap_from_path(sample.z_map_path)
        z_map_sample = _sample_xmap_and_scale(
            z_map, transform, np.copy(sample_array)
        )

        xmap = _load_xmap_from_mtz_path(sample.mtz_path)
        xmap_sample = _sample_xmap_and_scale(
            xmap, transform, np.copy(sample_array)
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
                event_map_sample * image_ligand_mask,
                z_map_sample * image_ligand_mask,
                xmap_sample * image_ligand_mask,
            ],
            axis=0,
        )[np.newaxis, :]
        image_float = image.astype(np.float32)

        # Make the annotation
        label = sample.rmsd

        return image_float, label
