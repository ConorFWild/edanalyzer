import time
from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import joblib
import pickle
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import tables
import zarr
from rdkit import Chem
from rdkit.Chem import AllChem

from edanalyzer import constants
from edanalyzer.datasets.base import _load_xmap_from_mtz_path, _load_xmap_from_path, _sample_xmap_and_scale
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag, _get_known_hit_structures, \
    _get_known_hits, _get_known_hit_centroids, _res_to_array
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer.data.build_data import PoseSample, MTZSample, EventMapSample

bond_type_cif_to_rdkit = {
    'single': Chem.rdchem.BondType.SINGLE,
    'double': Chem.rdchem.BondType.DOUBLE,
    'triple': Chem.rdchem.BondType.TRIPLE,
    'SINGLE': Chem.rdchem.BondType.SINGLE,
    'DOUBLE': Chem.rdchem.BondType.DOUBLE,
    'TRIPLE': Chem.rdchem.BondType.TRIPLE,
    'aromatic': Chem.rdchem.BondType.AROMATIC,
    # 'deloc': Chem.rdchem.BondType.OTHER
    'deloc': Chem.rdchem.BondType.SINGLE

}


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
        try:
            key = "data_comp_XXX"
            cif[key]
        except:
            try:
                key = 'comp_UNL'
                cif[key]
            except:
                raise Exception

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
    if not atom_charge_loop:
        atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
        if not atom_charge_loop:
            atom_charge_loop = [0] * len(atom_id_loop)

    aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
    if not aromatic_atom_loop:
        aromatic_atom_loop = [None] * len(atom_id_loop)

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
        aromatic_bond_loop = [None] * len(bond_1_id_loop)




    try:
        # Iteratively add the relevant bonds
        for bond_atom_1, bond_atom_2, bond_type, aromatic in zip(bond_1_id_loop, bond_2_id_loop, bond_type_loop,
                                                                 aromatic_bond_loop):
            bond_type = bond_type_cif_to_rdkit[bond_type]
            if aromatic:
                if aromatic == "y":
                    bond_type = bond_type_cif_to_rdkit['aromatic']

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
    # for atom in edited_mol.GetAtoms():
    #     print(atom.GetSymbol())
    #     for bond in atom.GetBonds():
    #         print(f"\t\t{bond.GetBondType()}")
    # for bond in edited_mol.GetBonds():
    #     ba1 = bond.GetBeginAtomIdx()
    #     ba2 = bond.GetEndAtomIdx()
    #     print(f"{bond.GetBondType()} : {edited_mol.GetAtomWithIdx(ba1).GetSymbol()} : {edited_mol.GetAtomWithIdx(ba2).GetSymbol()}")  #*}")
    # print(Chem.MolToMolBlock(edited_mol))

    # HANDLE SULFONATES
    # forward_mol = Chem.ReplaceSubstructs(
    #     edited_mol,
    #     Chem.MolFromSmiles('S(O)(O)(O)'),
    #     Chem.MolFromSmiles('S(=O)(=O)(O)'),
    #     replaceAll=True,)[0]
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
    print(f"Matches to sulfonates: {matches}")

    # atoms_to_charge = [
    #     sulfonate["O3"] for sulfonate in sulfonates.values()
    # ]
    # print(f"Atom idxs to charge: {atoms_to_charge}")
    bonds_to_double = [
                          (sulfonate["S"], sulfonate["O1"]) for sulfonate in sulfonates.values()
                      ] + [
                          (sulfonate["S"], sulfonate["O2"]) for sulfonate in sulfonates.values()
                      ]
    print(f"Bonds to double: {bonds_to_double}")

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
                order=bond_type_cif_to_rdkit['double']
            )
        else:
            new_editable_mol.AddBond(
                bond_atom_1,
                bond_atom_2,
                order=bond.GetBondType()
            )
    new_mol = new_editable_mol.GetMol()
    # print(Chem.MolToMolBlock(new_mol))

    new_mol.UpdatePropertyCache()
    # Chem.SanitizeMol(new_mol)
    return new_mol


def main(config_path):
    # Get the database
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the Database
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Open the store
    zarr_path = 'output/build_data_correlation.zarr'
    root = zarr.open(zarr_path, mode='a')

    # Open the event map store
    event_map_table = root['event_map_sample']
    pose_table = root['known_hit_pose']

    # Try to create the atom name/canonical smiles array
    # try:
    try:
        del root['ligand_data']
    except:
        rprint(f'No ligand data array! Making one!')
    ligand_data = root.create_dataset(
        'ligand_data',
        shape=event_map_table.shape,
        chunks=(20,),
        dtype=[
            ('idx', 'i8'), ('canonical_smiles', '<U300'), ('atom_ids', '<U5', (60,)), ('connectivity', '?', (60,60,))
        ]
    )
    # except:
    #     rprint(f"Already created ligand data table!")
    rprint(f'Getting pose table mapping...')
    pose_table_idxs = pose_table['idx']
    pose_table_database_events = pose_table['database_event_idx']
    rprint(f'Got pose table mapping!')

    with pony.orm.db_session:

        # Iterate over event maps
        for _record in event_map_table:

            # Get corresponding event
            database_event_idx = _record['event_idx']
            database_event = EventORM[database_event_idx]

            # Get a pose with the event
            rprint(f"Getting matched poses...")
            matching_pose_mask = pose_table_database_events == database_event_idx
            matching_pose_table_idxs = pose_table_idxs[matching_pose_mask]
            rprint(f'Got {len(matching_pose_table_idxs)} matching poses')
            matched_pose = pose_table[matching_pose_table_idxs[0]]
            matched_pose_atom_array = matched_pose['elements'][matched_pose['elements'] != 0]

            # Get event cif
            dtag_dir = Path(database_event.pandda.path) / 'processed_datasets' / database_event.dtag / 'ligand_files'
            smiles = [x for x in dtag_dir.glob('*.smiles')]
            cif_paths = [x for x in dtag_dir.glob('*.cif') if x.stem not in constants.LIGAND_IGNORE_REGEXES]
            rprint(f'{database_event.dtag}: {len(smiles)} smiles : {len(cif_paths)} cifs')

            # See which cif matches the element sequence
            matched_cifs = []
            for _cif_path in cif_paths:
                cif = gemmi.cif.read(str(_cif_path))

                key = "comp_LIG"
                try:
                    cif['comp_LIG']
                except:
                    try:
                        key = "data_comp_XXX"
                        cif[key]
                    except:
                        try:
                            key = "comp_UNL"
                            cif[key]
                        except:
                            rprint(_cif_path)

                # Find the relevant atoms loop
                atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
                atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
                atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))
                if not atom_charge_loop:
                    atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.partial_charge'))
                    if not atom_charge_loop:
                        atom_charge_loop = [0] * len(atom_id_loop)

                aromatic_atom_loop = list(cif[key].find_loop('_chem_comp_atom.aromatic'))
                if not aromatic_atom_loop:
                    aromatic_atom_loop = [None] * len(atom_id_loop)

                # Get the mapping
                id_to_idx = {}
                for j, atom_id in enumerate(atom_id_loop):
                    id_to_idx[atom_id] = j

                atom_ints = np.array(
                    [gemmi.Element(_atom_name).atomic_number for _atom_name in atom_type_loop if
                     gemmi.Element(_atom_name).atomic_number != 1]
                )
                rprint(atom_ints)
                rprint(matched_pose_atom_array)
                if np.array_equal(atom_ints, matched_pose_atom_array):
                    matched_cifs.append(
                        (
                            _cif_path,
                            cif[key]
                        )
                        )

            if len(matched_cifs) == 0:
                rprint(f"MATCH FAILED!")
                continue
            else:
                rprint(f'MATCHED!')
            ligand_data = matched_cifs[0]

            # Make atom name array
            # atom_element_array = [_x for _x in block.find_loop('_chem_comp_atom.type_symbol')]
            # atom_name_array = np.array([x for j, x in enumerate(block.find_loop('_chem_comp_atom.atom_id')) if atom_element_array[j] != 'H'])
            # rprint(atom_name_array)

            # Get Mol
            mol = get_fragment_mol_from_dataset_cif_path(ligand_data[0])
            rprint(mol)

            # Get canoncial smiles
            smiles = Chem.MolToSmiles(mol)
            rprint(smiles)

            # Store canonical smiles

            # Store the atom array
            atom_element_array = [_x for _x in ligand_data[1].find_loop('_chem_comp_atom.type_symbol')]
            atom_id_array = np.array([x for j, x in enumerate(ligand_data[1].find_loop('_chem_comp_atom.atom_id')) if atom_element_array[j] != 'H'])
            atom_array = np.zeros((60,), dtype='<U5')
            atom_array[:atom_id_array.size] = atom_id_array[:]

            # Store the connectivity matrix
            id_to_idx = {}
            for j, atom_id in enumerate(atom_id_array):
                id_to_idx[atom_id] = j
            bond_matrix = np.zeros(
                (
                    60,
                    60
                ),
                dtype='?')
            bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
            bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
            for _bond_1_id, _bond_2_id in zip(bond_1_id_loop, bond_2_id_loop):
                if _bond_1_id not in atom_id_array:
                    continue
                if _bond_2_id not in atom_id_array:
                    continue
                _bond_1_idx, _bond_2_idx = id_to_idx[_bond_1_id], id_to_idx[_bond_2_id]
                bond_matrix[_bond_1_idx, _bond_2_idx] = True
                bond_matrix[_bond_2_idx, _bond_1_idx] = True

            #

            new_ligand_data = (
                _record['idx'],
                smiles,
                atom_array,
                bond_matrix,
            )
            rprint(new_ligand_data)



    ...


if __name__ == "__main__":
    fire.Fire(main)
