import dataclasses
import time

import numpy as np
from rich import print as rprint
import gemmi
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from rdkit.Chem import AllChem

from edanalyzer import constants


@dataclasses.dataclass
class Ligand:
    path: str
    smiles: str
    chain: str
    residue: int
    num_atoms: int
    x: float
    y: float
    z: float


@dataclasses.dataclass
class Event:
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
    ligand: None
    viewed: None
    hit_confidence: str
    annotation: bool


def _get_system_from_dtag(dtag):
    hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    if len(hyphens) == 0:
        return None
    else:
        last_hypen_pos = hyphens[-1]
        system_name = dtag[:last_hypen_pos]

        return system_name


def try_open_map(path):
    try:
        m = gemmi.read_ccp4_map(str(path))
        return True
    except:
        return False


def try_open_structure(path):
    try:
        st = gemmi.read_structure(str(path))
        return True
    except:
        return False


def try_open_reflections(path):
    try:
        mtz = gemmi.read_mtz_file(str(path))
        return True
    except:
        return False


def _has_parsable_pdb(compound_dir):
    event_added = False
    # ligand_files_dir = processed_dataset_dir / "ligand_files"
    if compound_dir.exists():
        ligand_pdbs = []
        for ligand_pdb in compound_dir.glob("*.pdb"):
            if ligand_pdb.exists():
                if ligand_pdb.stem not in constants.LIGAND_IGNORE_REGEXES:
                    try:
                        st = gemmi.read_structure(str(ligand_pdb))
                    except:
                        return False
                    num_atoms = 0
                    for model in st:
                        for chain in model:
                            for residue in chain:
                                for atom in residue:
                                    num_atoms += 1

                    if num_atoms > 3:
                        return True
        # ]
        # if len(ligand_pdbs) > 0:
        #     return True

    return False


def get_ligand_num_atoms(ligand):
    num_atoms = 0
    for atom in ligand:
        num_atoms += 1

    return num_atoms


def get_ligand_centroid(ligand):
    poss = []
    for atom in ligand:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    pos_array = np.array(poss)

    return np.mean(pos_array, axis=0)


def get_structure_ligands(pdb_path):
    # logger.info(f"")
    structure = gemmi.read_structure(pdb_path)
    structure_ligands = []
    for model in structure:
        for chain in model:
            ligands = chain.get_ligands()
            for res in chain:
                if res.name == "LIG":
                    num_atoms = get_ligand_num_atoms(res)

                    ligand_centroid = get_ligand_centroid(res)

                    # smiles = parse_ligand(
                    #     structure,
                    #     chain,
                    #     ligand,
                    # )
                    smiles = "~"

                    lig = Ligand(
                        path=str(pdb_path),
                        smiles=str(smiles),
                        chain=str(chain.name),
                        residue=int(res.seqid.num),
                        num_atoms=int(num_atoms),
                        x=float(ligand_centroid[0]),
                        y=float(ligand_centroid[1]),
                        z=float(ligand_centroid[2])
                    )
                    structure_ligands.append(lig)

    return structure_ligands


def get_event_ligand(inspect_model_path, x, y, z, cutoff=10.0):
    structure_ligands = get_structure_ligands(str(inspect_model_path))

    ligand_distances = {}
    ligand_dict = {}
    for lig in structure_ligands:
        ligand_distances[(lig.chain, lig.residue)] = gemmi.Position(lig.x, lig.y, lig.z).dist(gemmi.Position(x, y, z))

        ligand_dict[(lig.chain, lig.residue)] = lig

    if len(ligand_dict) == 0:
        # logger.warning(f"Modelled structure but no ligands: {inspect_model_path}!")
        return None

    min_dist_id = min(ligand_distances, key=lambda _id: ligand_distances[_id])

    if ligand_distances[min_dist_id] < cutoff:
        # logger.warning(f"Modelled structure has ligand")
        return ligand_dict[min_dist_id]
    else:
        return None


def _parse_inspect_table_row(
        row,
        pandda_dir,
):
    dtag = str(row[constants.PANDDA_INSPECT_DTAG])
    event_idx = row[constants.PANDDA_INSPECT_EVENT_IDX]
    bdc = row[constants.PANDDA_INSPECT_BDC]
    x = row[constants.PANDDA_INSPECT_X]
    y = row[constants.PANDDA_INSPECT_Y]
    z = row[constants.PANDDA_INSPECT_Z]
    viewed = row[constants.PANDDA_INSPECT_VIEWED]

    if viewed != True:
        rprint(f"Dataset not viewed! Skipping {dtag} {event_idx} {pandda_dir}!")
        return None

    hit_confidence = row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE]
    if hit_confidence == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
        hit_confidence_class = True
    else:
        hit_confidence_class = False

    pandda_processed_datasets_dir = pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    processed_dataset_dir = pandda_processed_datasets_dir / dtag
    compound_dir = processed_dataset_dir / "ligand_files"
    if not _has_parsable_pdb(compound_dir):
        rprint(f"No parsable pdb at {compound_dir}! Skipping!")
        return None

    inspect_model_dir = processed_dataset_dir / constants.PANDDA_INSPECT_MODEL_DIR
    event_map_path = processed_dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
        dtag=dtag,
        event_idx=event_idx,
        bdc=bdc
    )
    z_map_path = processed_dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=dtag)
    if not z_map_path.exists():
        z_map_path = None
    else:
        z_map_path = str(z_map_path)

    mean_map_path = processed_dataset_dir / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(dtag=dtag)
    if not try_open_map(mean_map_path):
        return None

    initial_structure = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)
    if not try_open_structure(initial_structure):
        initial_structure = None
    else:
        initial_structure = str(initial_structure)
    if not initial_structure:
        return None

    initial_reflections = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag)
    if not try_open_reflections(initial_reflections):
        initial_reflections = None
    else:
        initial_reflections = str(initial_reflections)
    if not initial_reflections:
        return None

    if not try_open_map(event_map_path):
        return None

    # if not viewed:
    #     return None

    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=dtag)
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    # if inspect_model_path.exists():
    #     inspect_model_path = str(inspect_model_path)
    # else:
    #     inspect_model_path = None

    ligand = None
    if inspect_model_path.exists():
        ligand = get_event_ligand(
            inspect_model_path,
            x,
            y,
            z,
        )
        inspect_model_path = str(inspect_model_path)
        # if ligand:
        #     ligand_orm = LigandORM(
        #         path=str(ligand.path),
        #         smiles=str(ligand.smiles),
        #         chain=str(ligand.chain),
        #         residue=int(ligand.residue),
        #         num_atoms=int(ligand.num_atoms),
        #         x=float(ligand.x),
        #         y=float(ligand.y),
        #         z=float(ligand.z),
        #     )
        # else:
        #     ligand_orm = None
    else:
        ligand_orm = None
        inspect_model_path = None

    # hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    # if len(hyphens) == 0:
    #     return None
    # else:
    #     last_hypen_pos = hyphens[-1]
    #     system_name = dtag[:last_hypen_pos + 1]
    # ligand = None

    if hit_confidence not in ["Low", "low"]:
        annotation_value = True
    else:
        annotation_value = False

    if ligand and annotation_value:
        rprint(
            f"For {(dtag, event_idx)}, updating event centroid using associated ligand centroid from {(x, y, z)} to {(ligand.x, ligand.y, ligand.z)}")
        x, y, z = ligand.x, ligand.y, ligand.z

    rprint(f"\tAdding event: {(dtag, event_idx)}!")
    event = Event(
        dtag=str(dtag),
        event_idx=int(event_idx),
        x=float(x),
        y=float(y),
        z=float(z),
        bdc=float(bdc),
        initial_structure=initial_structure,
        initial_reflections=initial_reflections,
        structure=inspect_model_path,
        event_map=str(event_map_path),
        z_map=z_map_path,
        ligand=ligand,
        viewed=viewed,
        hit_confidence=hit_confidence,
        annotation=annotation_value
    )

    return event


def _get_known_hit_structures(
        model_dir,
        experiment_hit_datasets
):
    known_hit_structures = {}
    for hit_dtag in experiment_hit_datasets:
        hit_structure = Path(model_dir) / hit_dtag / 'refine.pdb'
        known_hit_structures[hit_dtag] = gemmi.read_structure(str(hit_structure))

    return known_hit_structures


def _get_st_hits(structure):
    hits = {}
    for model in structure:
        for chain in model:
            for res in chain.first_conformer():
                if res.name in ["LIG", "XXX"]:
                    hits[f"{chain.name}_{res.seqid.num}"] = res
    return hits


def _get_known_hits(known_hit_structures):
    centroids = {}
    for structure_key, structure in known_hit_structures.items():
        centroids[structure_key] = _get_st_hits(structure)

    return centroids


def _res_to_array(res):
    poss = []
    atoms = []
    elements = []
    for atom in res.first_conformer():
        pos = atom.pos
        element = atom.element.atomic_number
        # if atom.has_altloc():
        #     raise Exception
        if element == 1:
            continue
        poss.append([pos.x, pos.y, pos.z])
        atoms.append(atom.name)
        elements.append(element)

    return np.array(poss), np.array(atoms), np.array(elements)


def _get_known_hit_centroids(known_hits):
    centroids = {}
    for dtag in known_hits:
        centroids[dtag] = {}
        for res_id in known_hits[dtag]:
            poss, atom, elements = _res_to_array(known_hits[dtag][res_id])
            centroids[dtag][res_id] = np.mean(poss, axis=0)

    return centroids


rng = np.random.default_rng()

rprint(f'Generating small rotations')
time_begin_gen = time.time()
small_rotations = []
identity = np.eye(3)
for j in range(20):
    rotations = R.random(100000)
    rotmat = rotations.as_matrix()
    mask = (rotmat > (0.9 * np.eye(3)))
    diag = mask[:, np.array([0, 1, 2]), np.array([0, 1, 2])]
    rot_mask = diag.sum(axis=1)
    valid_rots = rotmat[rot_mask == 3, :, :]
    rots = [x for x in valid_rots]
    small_rotations += rots
# while len(small_rotations) < 10000:
#     rot = R.random()
#     if np.allclose(rot.as_matrix(), identity, atol=0.1, rtol=0.0):
#         small_rotations.append(rot)
#         rprint(len(small_rotations))
time_finish_gen = time.time()
rprint(f"Generated small rotations in: {round(time_finish_gen - time_begin_gen, 2)}")


def _get_known_hit_poses(
        res,
        event_to_lig_com,
        centroid=np.array([22.5, 22.5, 22.5]).reshape((1, 3)),
        translation=10,
        num_poses=50
):
    # Get pos array
    poss, atoms, elements = _res_to_array(res)

    size = min(100, poss.shape[0])

    elements_array = np.zeros(100, dtype=np.int32)
    elements_array[:size] = elements[:size]

    atom_array = np.zeros(100, dtype='<U5')
    atom_array[:size] = atoms[:size]

    # Iterate over poses
    poses = []
    rmsds = []
    for cutoff in [0.25, 0.5, 1.0, 2.0, 3.0, 10.0]:
        num_sampled = 0
        translation = cutoff
        rprint(f"Cutoff: {cutoff}")
        while True:
            # Copy the pos array
            _poss = np.copy(poss)

            # Get rotation and translation
            if cutoff <= 0.5:
                rot = R.from_matrix(small_rotations[rng.integers(0, len(small_rotations))])
            else:
                rot = R.random()

            _translation = rng.uniform(-translation / 3, translation / 3, size=3).reshape((1, 3))

            # Cetner
            com = np.mean(_poss, axis=0).reshape((1, 3))
            _poss_centered = _poss - com

            # Get target
            _rmsd_target = np.copy(_poss_centered) + centroid + event_to_lig_com

            # Randomly perturb and reorient

            _rotated_poss = rot.apply(_poss_centered)
            new_com = _translation + centroid + event_to_lig_com
            _new_poss = _rotated_poss + new_com

            # Get RMSD to original
            rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(_rmsd_target - _new_poss, axis=1))) / _new_poss.shape[0])

            if rmsd < cutoff:
                num_sampled += 1
            else:
                continue

            rmsds.append(rmsd)

            # Pad the poss to a uniform size
            pose_array = np.zeros((100, 3))
            pose_array[:size, :] = _new_poss[:size, :]
            poses.append(pose_array)

            if num_sampled >= num_poses:
                break

    return poses, [atom_array] * 6, [elements_array] * 6 * num_poses, rmsds


def _get_lig_block_from_path(path):
    cif = gemmi.cif.read(str(path.resolve()))

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        try:
            key = "comp_XXX"
            cif[key]
        except:
            try:
                key = "comp_UNL"
                cif[key]
            except:
                rprint(path)
    return cif[key]


def _match_atoms(atom_name_array, block):
    atom_id_loop = list(block.find_loop('_chem_comp_atom.atom_id'))
    atom_element_loop = list(block.find_loop('_chem_comp_atom.type_symbol'))
    # rprint(atom_id_loop)
    # rprint(atom_name_array)

    filtered_atom_id_loop = [_x for _x, _el in zip(atom_id_loop, atom_element_loop) if _el != 'H']
    # rprint(filtered_atom_id_loop)
    # rprint(atom_name_array)

    if len(filtered_atom_id_loop) != len(atom_name_array):
        rprint(f"Different number of atoms! No Match!!")
        return None

    match = {}
    for _j, atom_1_id in enumerate([_x for _x, _el in zip(atom_id_loop, atom_element_loop) if _el != 'H']):
        for _k, atom_2_id in enumerate(atom_name_array):
            if atom_1_id == atom_2_id:
                match[_j] = _k

    if len(match) != len(filtered_atom_id_loop):
        rprint(f"Only partial match {len(match)} / {len(filtered_atom_id_loop)}! Skipping!")
        return None

    else:
        return match


def _get_cif_paths_from_dir(dtag_dir):
    cif_paths = [x for x in dtag_dir.glob('*.cif') if x.stem not in constants.LIGAND_IGNORE_REGEXES]
    return cif_paths


def _get_event_cifs(event):
    dtag_dir = Path(event.pandda.path) / 'processed_datasets' / event.dtag / 'ligand_files'
    cif_paths = _get_cif_paths_from_dir(dtag_dir)
    # rprint(f'Got {len(cif_paths)} ligand cif paths!')

    return cif_paths


def _get_matched_cifs(cif_paths, known_hit_residue):
    atom_name_array = [atom.name for atom in known_hit_residue.first_conformer() if atom.element.name != 'H']
    matched_paths = []
    for _cif_path in cif_paths:
        block = _get_lig_block_from_path(_cif_path)
        match = _match_atoms(atom_name_array, block)

        if match:
            matched_paths.append((_cif_path, block, match))
    return matched_paths


def _get_matched_cifs_from_event(
        known_hit_residue,
        event,
):
    cif_paths = _get_event_cifs(event)

    matched_paths = _get_matched_cifs(cif_paths, known_hit_residue)

    return matched_paths


def _get_matched_cifs_from_dir(
        known_hit_residue,
        compound_dir,
):
    cif_paths = _get_cif_paths_from_dir(compound_dir)

    matched_paths = _get_matched_cifs(cif_paths, known_hit_residue)

    return matched_paths


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
            key = "comp_XXX"
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
    # print(f"Matches to sulfonates: {matches}")

    # atoms_to_charge = [
    #     sulfonate["O3"] for sulfonate in sulfonates.values()
    # ]
    # print(f"Atom idxs to charge: {atoms_to_charge}")
    bonds_to_double = [
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


def _get_smiles(matched_cif):
    mol = get_fragment_mol_from_dataset_cif_path(matched_cif[0])
    smiles = Chem.MolToSmiles(mol)
    return smiles


def _get_atom_ids(matched_cif):
    atom_ids = list(matched_cif[1].find_loop('_chem_comp_atom.atom_id'))
    atom_types = list(matched_cif[1].find_loop('_chem_comp_atom.type_symbol'))

    return [_x for _x, _y in zip(atom_ids, atom_types) if _y != 'H']


def _get_connectivity(matched_cif):
    atom_id_array = _get_atom_ids(matched_cif)
    block = matched_cif[1]

    id_to_idx = {}
    for j, atom_id in enumerate(atom_id_array):
        id_to_idx[atom_id] = j
    bond_matrix = np.zeros(
        (
            150,
            150
        ),
        dtype='?')
    bond_1_id_loop = list(block.find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(block.find_loop('_chem_comp_bond.atom_id_2'))
    for _bond_1_id, _bond_2_id in zip(bond_1_id_loop, bond_2_id_loop):
        if _bond_1_id not in atom_id_array:
            continue
        if _bond_2_id not in atom_id_array:
            continue
        _bond_1_idx, _bond_2_idx = id_to_idx[_bond_1_id], id_to_idx[_bond_2_id]
        bond_matrix[_bond_1_idx, _bond_2_idx] = True
        bond_matrix[_bond_2_idx, _bond_1_idx] = True

    return bond_matrix
