import numpy as np
import gemmi

# from ..interfaces import *

from edanalyzer import constants

def _load_xmap_from_mtz_path(path):
    mtz = gemmi.read_mtz_file(str(path))
    for f, phi in constants.STRUCTURE_FACTORS:
        try:
            xmap = mtz.transform_f_phi_to_map(f, phi, sample_rate=3)
            return xmap
        except Exception as e:
            continue
    raise Exception()

def _load_xmap_from_path(path):
    ccp4 = gemmi.read_ccp4_map(str(path))
    ccp4.setup(float('nan'))
    m = ccp4.grid

    return m


def _get_structure_from_path(path):
    return gemmi.read_structure(str(path))


def _get_res_from_structure_chain_res(structure, chain, res):
    return structure[0][chain][res]


def _get_identity_matrix():
    return np.eye(3)


def _get_centroid_from_res(res):
    poss = []
    for atom in res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.mean(poss, axis=0)


def _combine_transforms(new_transform, old_transform):
    new_transform_mat = new_transform.mat
    new_transform_vec = new_transform.vec

    old_transform_mat = old_transform.mat
    old_transform_vec = old_transform.vec

    combined_transform_mat = new_transform_mat.multiply(old_transform_mat)
    combined_transform_vec = new_transform_vec + new_transform_mat.multiply(old_transform_vec)

    combined_transform = gemmi.Transform()
    combined_transform.vec.fromlist(combined_transform_vec.tolist())
    combined_transform.mat.fromlist(combined_transform_mat.tolist())

    return combined_transform


def _get_transform_from_orientation_centroid(orientation, centroid):
    sample_distance: float = 0.5
    n: int = 30
    # translation: float):

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
    rotation_matrix = orientation
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
    event_centre_transform.vec.fromlist(centroid)

    transform = _combine_transforms(
        event_centre_transform,
        _combine_transforms(
            rotation_transform,
            _combine_transforms(
                centre_grid_transform,
                    initial_transform)))
    return transform


def _get_ligand_mask(dmap, res):
    mask = gemmi.Int8Grid(dmap.nu, dmap.nv, dmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(dmap.unit_cell)

    # Get the mask
    for atom in res:
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=2.5,
            value=1,
        )

    return mask

def _get_ligand_mask_float(dmap, res):
    mask = gemmi.FloatGrid(dmap.nu, dmap.nv, dmap.nw)
    mask.spacegroup = gemmi.find_spacegroup_by_name("P1")
    mask.set_unit_cell(dmap.unit_cell)

    # Get the mask
    for atom in res:
        pos = atom.pos
        mask.set_points_around(
            pos,
            radius=2.5,
            value=1.0,
        )

    return mask

def _get_masked_dmap(dmap, res):
    mask = _get_ligand_mask(dmap, res)

    # Get the mask array
    mask_array = np.array(mask, copy=False)

    # Get the dmap array
    dmap_array = np.array(dmap, copy=False)

    # Mask the dmap array
    dmap_array[mask_array == 0] = 0.0

    return dmap


def _sample_xmap(xmap, transform, sample_array):
    xmap.interpolate_values(sample_array, transform)
    return sample_array


def _sample_xmap_and_scale(masked_dmap, sample_transform, sample_array):
    image_initial = _sample_xmap(masked_dmap, sample_transform, sample_array)
    std = np.std(image_initial)
    if np.abs(std) < 0.0000001:
        image_dmap = np.copy(sample_array)

    else:
        image_dmap = (image_initial - np.mean(image_initial)) / std

    return image_dmap

def _make_ligand_masked_dmap_layer(
        dmap,
        res,
        sample_transform,
        sample_array
):
    # Get the masked dmap
    masked_dmap = _get_masked_dmap(dmap, res)

    # Get the image
    image_dmap = _sample_xmap_and_scale(masked_dmap, sample_transform, sample_array)

    return image_dmap

