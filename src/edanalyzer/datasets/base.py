from pathlib import Path
import random

import numpy as np
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem

# from ..interfaces import *

from edanalyzer import constants


def _get_ligand_path_from_dir(path):
    ligand_pdbs = [
        x
        for x
        in (path / 'ligand_files').glob("*.cif")
        if (x.exists()) and (x.stem not in constants.LIGAND_IGNORE_REGEXES)
    ]
    if len(ligand_pdbs) != 1:
        raise Exception(f'Have {len(ligand_pdbs)} valid pdbs in directory: {path /"ligand_files"}')
    return ligand_pdbs[0]

def get_fragment_mol_from_dataset_cif_path(dataset_cif_path: Path):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Create a blank rdkit mol
    mol = Chem.Mol()
    editable_mol = Chem.EditableMol(mol)

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        key = "data_comp_XXX"

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
    if not atom_charge_loop:
        atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
        if not atom_charge_loop:
            atom_charge_loop = [0]*len(atom_id_loop)

    aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
    if not aromatic_atom_loop:
        aromatic_atom_loop = [None]*len(atom_id_loop)

    # Get the mapping
    id_to_idx = {}
    for j, atom_id in enumerate(atom_id_loop):
        id_to_idx[atom_id] = j

    # Iteratively add the relveant atoms
    for atom_id, atom_type, atom_charge in zip(atom_id_loop, atom_type_loop, atom_charge_loop):
        if len(atom_type) > 1:
            atom_type = atom_type[0] + atom_type[1].lower()
        atom = Chem.Atom(atom_type)
        atom.SetFormalCharge(round(float(atom_charge)))
        editable_mol.AddAtom(atom)

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
    bond_type_loop = list(cif[key].find_loop('_chem_comp_bond.type'))
    aromatic_bond_loop = list(cif[key].find_loop('_chem_comp_bond.aromatic'))
    if not aromatic_bond_loop:
        aromatic_bond_loop = [None]*len(bond_1_id_loop)

    try:
        # Iteratively add the relevant bonds
        for bond_atom_1, bond_atom_2, bond_type, aromatic in zip(bond_1_id_loop, bond_2_id_loop, bond_type_loop, aromatic_bond_loop):
            bond_type = constants.bond_type_cif_to_rdkit[bond_type]
            if aromatic:
                if aromatic == "y":
                    bond_type = constants.bond_type_cif_to_rdkit['aromatic']

            editable_mol.AddBond(
                id_to_idx[bond_atom_1],
                id_to_idx[bond_atom_2],
                order=bond_type
            )
    except Exception as e:
        print(e)
        print(atom_id_loop)
        print(id_to_idx)
        print(bond_1_id_loop)
        print(bond_2_id_loop)
        raise Exception

    edited_mol = editable_mol.GetMol()



    # HANDLE SULFONATES

    patt = Chem.MolFromSmarts('S(-O)(-O)(-O)')
    matches = edited_mol.GetSubstructMatches(patt)

    sulfonates = {}
    for match in matches:
        sfn = 1
        sulfonates[sfn] = {}
        on = 1
        for atom_idx in match:
            atom = edited_mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "S":
                sulfonates[sfn]["S"] = atom_idx
            else:
                atom_charge = atom.GetFormalCharge()

                if atom_charge == -1:
                    continue
                else:
                    if on == 1:
                        sulfonates[sfn]["O1"] = atom_idx
                        on += 1
                    elif on == 2:
                        sulfonates[sfn]["O2"] = atom_idx
                        on += 1
                # elif on == 3:
                #     sulfonates[sfn]["O3"] = atom_idx
    # print(f"Matches to sulfonates: {matches}")

    # atoms_to_charge = [
    #     sulfonate["O3"] for sulfonate in sulfonates.values()
    # ]
    # print(f"Atom idxs to charge: {atoms_to_charge}")
    bonds_to_double =[
        (sulfonate["S"], sulfonate["O1"]) for sulfonate in sulfonates.values()
    ] + [
        (sulfonate["S"], sulfonate["O2"]) for sulfonate in sulfonates.values()
    ]
    # print(f"Bonds to double: {bonds_to_double}")

    # Replace the bonds and update O3's charge
    new_editable_mol = Chem.EditableMol(Chem.Mol())
    for atom in edited_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        new_atom = Chem.Atom(atom.GetSymbol())
        charge = atom.GetFormalCharge()
        # if atom_idx in atoms_to_charge:
        #     charge = -1
        new_atom.SetFormalCharge(charge)
        new_editable_mol.AddAtom(new_atom)

    for bond in edited_mol.GetBonds():
        bond_atom_1 = bond.GetBeginAtomIdx()
        bond_atom_2 = bond.GetEndAtomIdx()
        double_bond = False
        for bond_idxs in bonds_to_double:
            if (bond_atom_1 in bond_idxs) & (bond_atom_2 in bond_idxs):
                double_bond = True
        if double_bond:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=constants.bond_type_cif_to_rdkit['double']
            )
        else:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond.GetBondType()
            )
    new_mol = new_editable_mol.GetMol()

    return new_mol

def get_structures_from_mol(mol: Chem.Mol, dataset_cif_path, max_conformers):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(dataset_cif_path))

    # Find the relevant atoms loop
    atom_id_loop = list(cif['comp_LIG'].find_loop('_chem_comp_atom.atom_id'))
    # print(f"Atom ID loop: {atom_id_loop}")


    fragment_structures = {}
    for i, conformer in enumerate(mol.GetConformers()):

        positions: np.ndarray = conformer.GetPositions()

        structure: gemmi.Structure = gemmi.Structure()
        model: gemmi.Model = gemmi.Model(f"{i}")
        chain: gemmi.Chain = gemmi.Chain(f"{i}")
        residue: gemmi.Residue = gemmi.Residue()
        residue.name = "LIG"
        residue.seqid = gemmi.SeqId(1, ' ')

        # Loop over atoms, adding them to a gemmi residue
        for j, atom in enumerate(mol.GetAtoms()):
            # Get the atomic symbol
            atom_symbol: str = atom.GetSymbol()
            # print(f"{j} : {atom_symbol}")

            # if atom_symbol == "H":
            #     continue
            gemmi_element: gemmi.Element = gemmi.Element(atom_symbol)

            # Get the position as a gemmi type
            pos: np.ndarray = positions[j, :]
            gemmi_pos: gemmi.Position = gemmi.Position(pos[0], pos[1], pos[2])

            # Get the
            gemmi_atom: gemmi.Atom = gemmi.Atom()
            # gemmi_atom.name = atom_symbol
            gemmi_atom.name = atom_id_loop[j]
            gemmi_atom.pos = gemmi_pos
            gemmi_atom.element = gemmi_element

            # Add atom to residue
            residue.add_atom(gemmi_atom)

        chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)

        fragment_structures[i] = structure

        if len(fragment_structures) > max_conformers:
            return fragment_structures

    return fragment_structures

def _parse_cif_file_for_ligand_array(path):
    mol = get_fragment_mol_from_dataset_cif_path(path)
    mol.calcImplicitValence()

    # Generate conformers
    cids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=1000,
        pruneRmsThresh=1.5,
    )

    # Translate to structures
    fragment_structures = get_structures_from_mol(
        mol,
        path,
        10,
    )

    st = random.choice(fragment_structures)

    poss = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    pos = atom.pos
                    poss.append([pos.x, pos.y, pos.z])

    return np.array(poss).T

def _get_ligand_from_dir(path):
    ligand_path = _get_ligand_path_from_dir(path)

    ligand_array = _parse_cif_file_for_ligand_array(ligand_path)
    return ligand_array


def _get_ligand_map(ligand_array, n=30,step=0.5, translation=2.5):
    rotation_matrix = R.random().as_matrix()
    rng = default_rng()
    random_translation = ((rng.random(3) - 0.5) * 2 * translation).reshape((3, 1))
    ligand_mean_pos = np.mean(ligand_array, axis=1).reshape((3, 1))
    centre_translation = np.array([step * n, step * n, step * n]).reshape((3, 1)) / 2
    zero_centred_array = ligand_array - ligand_mean_pos
    rotated_array = np.matmul(rotation_matrix, zero_centred_array)
    grid_centred_array = rotated_array + centre_translation
    augmented_array = (grid_centred_array + random_translation).T

    # Get a dummy grid to place density on
    dummy_grid = gemmi.FloatGrid(n, n, n)
    unit_cell = gemmi.UnitCell(step * n, step * n, step * n, 90.0, 90.0, 90.0)
    dummy_grid.set_unit_cell(unit_cell)

    for pos_array in augmented_array:
        assert pos_array.size == 3
        if np.all(pos_array > 0):
            if np.all(pos_array < (n * step)):
                dummy_grid.set_points_around(
                    gemmi.Position(*pos_array),
                    radius=1.0,
                    value=1.0,
                )

    return dummy_grid


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

