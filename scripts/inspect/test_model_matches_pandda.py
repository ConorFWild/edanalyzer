from pathlib import Path

import fire
import yaml
from rich import print as rprint
import lightning as lt
from torch.utils.data import DataLoader
import pony
import numpy as np
import tables
import zarr
import torch

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
    _get_structure_from_path
)
from edanalyzer.models.build_scoring import LitBuildScoringCorrelation


def main(model_path, structure_path, event_map_path, mtz_path):
    # rprint(f'Running collate_database from config file: {config_path}')
    # # Load config
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)

    # Load the data
    st = _get_structure_from_path(structure_path)
    event_map_grid = _load_xmap_from_path(event_map_path)
    mtz_grid = _load_xmap_from_mtz_path(mtz_path)

    # Get the residue
    residue = st[0][0][0]
    # residue = _get_res_from_hdf5(pose_data)

    # Get the event from the database
    # event = self.data[event_map_data['event_idx']]

    # Get sampling transform for the event map
    sample_array = np.zeros(
        (30, 30, 30),
        dtype=np.float32,
    )
    orientation = np.eye(3)
    centroid = _get_centroid_from_res(residue)
    transform = _get_transform_from_orientation_centroid(
        orientation,
        centroid
    )

    # Get sample image
    event_map_sample = _sample_xmap_and_scale(
        event_map_grid, transform, np.copy(sample_array)
    )

    mtz_map_sample = _sample_xmap_and_scale(
        mtz_grid, transform, np.copy(sample_array)
    )

    ligand_mask_grid = _get_ligand_mask_float(event_map_grid, residue)
    image_ligand_mask = _sample_xmap(
        ligand_mask_grid,
        transform,
        np.copy(sample_array)
    )

    image_ligand_mask[image_ligand_mask < 0.9] = 0.0
    image_ligand_mask[image_ligand_mask >= 0.9] = 1.0

    # masked_event_map = event_map_sample * image_ligand_mask

    image = np.stack(
        [
            event_map_sample * image_ligand_mask,
            mtz_map_sample * image_ligand_mask
        ],
        axis=0
    )

    image_float = image.astype(np.float32)

    # Load the model
    model = LitBuildScoringCorrelation()
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    # Add model to device
    model.to('cpu')

    model.eval()


    # Run the model
    cnn = model.float()

    image_t = torch.from_numpy(image_float)
    image_c = image_t.to('cpu')
    model_annotation = cnn(image_c)
    # print(f'Annotation shape: {model_annotation.shape}')

    annotation = model_annotation.to(torch.device("cpu")).detach().numpy()
    rprint(annotation)


if __name__ == "__main__":
    fire.Fire(main)
