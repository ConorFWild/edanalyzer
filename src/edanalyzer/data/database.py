import dataclasses

import numpy as np
from rich import print as rprint
import gemmi
from pathlib import Path

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


def _get_known_hits(known_hit_structures):
    centroids = {}
    for structure_key, structure in known_hit_structures.items():
        centroids[structure_key] = {}
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.name in ["LIG", "XXX"]:
                        centroids[structure_key][f"{chain.name}_{res.seqid.num}"] = res

    return centroids

def _res_to_array(res):
    poss = []
    elements = []
    for atom in res:
        pos = atom.pos
        element = atom.element.atomic_number
        if element == 1:
            continue
        poss.append([pos.x, pos.y, pos.z])
        elements.append(element)

    return np.array(poss), np.array(elements)

def _get_known_hit_centroids(known_hits):
    centroids = {}
    for dtag in known_hits:
        centroids[dtag] = {}
        for res_id in known_hits[dtag]:
            poss, elements = _res_to_array(known_hits[dtag][res_id])
            centroids[dtag][res_id] = np.mean(poss, axis=0)

    return centroids