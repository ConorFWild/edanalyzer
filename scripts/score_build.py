import dataclasses

import fire
import numpy as np
from rich import print as rprint
import torch


from edanalyzer.torch_network_resnet import resnet18

from edanalyzer.torch_dataset import (
    _get_zmap_path,
    _get_event_mtz_path,
    _get_event_map_path,
    _get_xmap_path,
    _get_structure_path,
    _get_bound_state_model_path,
    _get_annotation_from_event,  # Updates annotation
    _get_centroid_relative_to_ligand,  # updates centroid and annotation
    _get_random_ligand_path,  # Updates ligand_path and annotation
    _get_random_orientation,  # Updates orientation
    _get_transform,  # Updates transform,
    _make_zmap_layer,
    _make_xmap_layer,
    _make_event_map_layer,
    _make_structure_map_layer,
    _make_ligand_map_layer,
    _get_event_centroid,
    _get_identity_orientation,
    _decide_annotation,  # Updates annotation
    _get_transformed_ligand,  # Updates ligand_res
    _get_non_transformed_ligand,
    _get_centroid_relative_to_transformed_ligand,  # updates centroid and annotation
    _make_ligand_masked_event_map_layer,  # Updates ligand_masked_event_map_layer
    _get_annotation_from_ntuple,  # Updates annotation
    _get_centroid_from_ntuple,  # updates centroid and annotation
    _get_transform_from_ntuple,
    _get_autobuild_res_from_ntuple,
    _make_ligand_masked_event_map_layer_from_ntuple,  # Updates ligand_masked_event_map_layer
    _make_ligand_masked_z_map_layer_from_ntuple,  # Updates ligand_masked_z_map_layer
    _make_ligand_masked_raw_xmap_map_layer_from_ntuple,  # Updates ligand_masked_raw_xmap_map_layer
    load_xmap_from_mtz,
_get_res_from_autobuild_structure_path,
_get_identity_matrix,
_get_centroid_from_res,
_get_transform_from_orientation_centroid,
get_map_from_path,
_make_ligand_masked_dmap_layer,

)



def _score_build(
       structural_model_path,
        event_map_path,
        z_map_path,
        mtz_path,
       cnn_model_path
):


    num_layers = 3


    # Create the test dataset

    res = _get_res_from_autobuild_structure_path(structural_model_path)
    rotation = _get_identity_matrix()
    centroid = _get_centroid_from_res(res)
    transform = _get_transform_from_orientation_centroid(rotation, centroid)
    sample_array = np.zeros((30,30,30), dtype=np.float32)

    dmap_event = get_map_from_path(event_map_path)
    image_event_map = _make_ligand_masked_dmap_layer(
        dmap_event,
        res,
        transform,
        sample_array
    )
    dmap_z = get_map_from_path(z_map_path)
    image_z_map = _make_ligand_masked_dmap_layer(
        dmap_z,
        res,
        transform,
        sample_array
    )
    dmap_mtz = load_xmap_from_mtz(mtz_path)
    image_raw_xmap = _make_ligand_masked_dmap_layer(
        dmap_mtz,
        res,
        transform,
        sample_array
    )
    image = np.stack(
        [image_event_map, image_z_map, image_raw_xmap],
        axis=0
    )[np.newaxis,:]
    rprint(f"Image shape is: {image.shape}")

    # Get the device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    # if model_type == "resnet+ligand":
    model = resnet18(num_classes=2, num_input=num_layers)
    model.load_state_dict(torch.load(cnn_model_path, map_location=dev))
    model.to(dev)
    model.eval()

    image_c = image.to(dev)
    annotation = model(image_c)
    rprint(f'Annotation is: {annotation}')

    ...


if __name__ == "__main__":
    fire.Fire()