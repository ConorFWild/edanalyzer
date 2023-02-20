import re

import fire
from pathlib import Path
import subprocess
from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand, PanDDAEvent, PanDDAEventDataset
import constants
from loguru import logger
from openbabel import pybel
import gemmi
# from rdkit import Chem
from numpy.random import default_rng
# from torch_dataset import *
import numpy as np
import traceback
import pandas as pd


def download_dataset(options: Options):
    data_dir = Path(options.working_dir) / constants.DATA_DIR

    datatype = "structures/divided/pdb"
    logger.info(f"Downloading pdbs to: {data_dir}/{datatype}")
    p = subprocess.Popen(
        constants.RSYNC_COMMAND.format(
            datatype_dir=datatype,
            data_dir=data_dir

        ),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    p.communicate()

    datatype = "structures/divided/structure_factors"
    logger.info(f"Downloading structure factors to: {data_dir}/{datatype}")
    p = subprocess.Popen(
        constants.RSYNC_COMMAND.format(
            datatype_dir=datatype,
            data_dir=data_dir

        ),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    p.communicate()

    logger.info(f"RSYNC'd all pdb data and structure factors!")


def get_structure_factors(dt: StructureReflectionsData):
    doc = gemmi.cif.read(dt.mtz_path)
    rblocks = gemmi.as_refln_blocks(doc)
    rblock = rblocks[0]
    cols = rblock.column_labels()
    print(cols)

    for col in cols:
        for f, phi in constants.STRUCTURE_FACTORS:
            if col == f:
                return f, phi

    return None, None


def parse_dataset(options: Options, ):
    logger.info(f"Parsing dataset...")
    pdbs_dir = Path(options.working_dir) / constants.DATA_DIR / "structures" / "divided" / "pdb"
    sfs_dir = Path(options.working_dir) / constants.DATA_DIR / "structures" / "divided" / "structure_factors"

    pdbs = {}
    sfs = {}

    for sub in sfs_dir.glob("*"):
        for entry in sub.glob("*"):
            match = re.match(constants.MTZ_REGEX, entry.name)
            code = match.group(1)
            sfs[code] = entry
    logger.info(f"Found {len(sfs)} structure factors...")

    for sub in pdbs_dir.glob("*"):
        for entry in sub.glob("*"):
            match = re.match(constants.PDB_REGEX, entry.name)
            code = match.group(1)
            if code in sfs:
                pdbs[code] = entry
    logger.info(f"Found {len(pdbs)} that could be associated with structure factors...")

    datas = []
    id = 0
    j = 0
    for entry_name, path in sfs.items():
        j += 1
        logger.debug(f"Processing dataset: {entry_name}: {j} / {len(sfs)}")
        if entry_name in pdbs:
            pdb = pdbs[entry_name]
        else:
            continue
        dt = StructureReflectionsData(
            id=id,
            name=entry_name,
            pdb_path=str(pdb),
            mtz_path=str(path),
            ligands=[],
            partition=0,
            f="",
            phi=""
        )
        # Check structure factors
        try:
            f, phi = get_structure_factors(dt)
        except Exception as e:
            logger.debug(f"Could not get structure factors, skipping!")
            logger.debug(traceback.format_exc())
            continue
        if f is None:
            logger.debug(f"No recognisable structure factors!")
            continue
        logger.info(f"Structure factors are: {f} {phi}")
        dt.f = f
        dt.phi = phi

        # Get ligands
        try:
            ligands = get_structure_ligands(dt.pdb_path)
        except Exception as e:
            logger.debug("Could not get ligands, skipping!")
            logger.debug(traceback.format_exc())
            continue
        if len(ligands) == 0:
            logger.debug("Did not find any ligands!")
            continue
        logger.debug(f"Found {len(ligands)} ligands")
        dt.ligands = ligands

        id += 1
        datas.append(dt)

    logger.info(f"Found {len(datas)} complete datasets!")

    dataset = StructureReflectionsDataset(data=datas)
    dataset.save(options.working_dir)
    # return Dataset(datas)


def parse_ligand(structure_template, chain, ligand_residue):
    structure = structure_template.clone()
    chains_to_remove = []
    for model in structure:
        for _chain in model:
            chains_to_remove.append(_chain.name)
            # if _chain.name != chain.name:

            # model.remove_chain(_chain.name)
    for model in structure:
        for _chain_name in chains_to_remove:
            model.remove_chain(_chain_name)

    new_chain = gemmi.Chain(chain.name)
    new_chain.add_residue(ligand_residue)
    for model in structure:
        model.add_chain(new_chain)

    pdb_string = structure.make_minimal_pdb()

    pybel_mol = pybel.readstring("pdb", pdb_string)

    smiles = pybel_mol.write("can")
    # print(smiles)

    # smiles = Chem.MolToSmiles(mol)

    return smiles


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
    id = 0
    for model in structure:
        for chain in model:
            ligands = chain.get_ligands()
            for ligand in ligands:
                # structure_ligands.append(

                num_atoms = get_ligand_num_atoms(ligand)

                ligand_centroid = get_ligand_centroid(ligand)

                smiles = parse_ligand(
                    structure,
                    chain,
                    ligand,
                )
                logger.debug(f"Ligand smiles: {smiles}")
                logger.debug(f"Num atoms: {num_atoms}")
                logger.debug(f"Centroid: {ligand_centroid}")
                lig = Ligand(
                    id=id,
                    smiles=smiles,
                    chain=chain.name,
                    residue=ligand.seqid.num,
                    num_atoms=num_atoms,
                    x=ligand_centroid[0],
                    y=ligand_centroid[1],
                    z=ligand_centroid[2]
                )
                id += 1
                structure_ligands.append(lig)

    return structure_ligands


def generate_smiles(options: Options, dataset: StructureReflectionsDataset):
    logger.info(f"Generating smiles for dataset")
    for data in dataset.data:
        ligands = get_structure_ligands(data)
        data.ligands = ligands

    logger.info(f"Generated smiles, saving to {options.working_dir}")
    dataset.save(options.working_dir)


def partition_dataset(options: Options, dataset: StructureReflectionsDataset, prob=0.1):
    logger.info(f"Assigning datasets to test set with probability {prob}")
    num_data = len(dataset.data)
    rng = default_rng()
    vals = rng.random_sample(num_data)
    for data, val in zip(dataset.data, vals):
        if val < prob:
            data.partition = 1
        else:
            data.partition = 0

    num_test = len([data for data in dataset.data if data.partition == 1])
    logger.info(f"Assigned {num_test} of {num_data} to test set")
    dataset.save(options.working_dir)


def train(options: Options, dataset: StructureReflectionsDataset):
    # Get the dataset
    dataset_torch = StructureReflectionsDatasetTorch(
        dataset,
        transform=lambda data: sample_ligand_density(
            data,
            lambda _data: annotate_data_randomly(_data, 0.5),
            lambda _data, _annotation: generate_xmap_ligand_sample_or_decoy(
                _data,
                _annotation,
                sample_ligand=lambda __data: generate_ligand_sample(
                    __data,
                    get_ligand_decoy_transform,
                    sample_xmap_from_data
                ),
                sample_ligand_decoy=lambda __data: generate_ligand_sample(
                    __data,
                    get_ligand_transform,
                    sample_xmap_from_data,

                )
            )
        )
    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=1, shuffle=True)

    # Trainloop

    ...


def test(options: Options, dataset: StructureReflectionsDataset):
    ...


def get_event_ligand(inspect_model_path, x, y, z, cutoff=5.0):
    structure_ligands = get_structure_ligands(inspect_model_path)

    ligand_distances = {}
    ligand_dict = {}
    for lig in structure_ligands:
        ligand_distances[lig.id] = gemmi.Position(lig.x, lig.y, lig.z).dist(gemmi.Position(x, y, z))

        ligand_dict[lig.id] = lig

    if len(ligand_dict) == 0:
        return None

    min_dist_id = min(ligand_distances, key=lambda _id: ligand_distances[_id])

    if ligand_distances[min_dist_id] < cutoff:
        return ligand_dict[min_dist_id]
    else:
        return None


def parse_inspect_table_row(row, pandda_dir, pandda_processed_datasets_dir, model_building_dir):
    dtag = row[constants.PANDDA_INSPECT_DTAG]
    event_idx = row[constants.PANDDA_INSPECT_EVENT_IDX]
    bdc = row[constants.PANDDA_INSPECT_BDC]
    x = row[constants.PANDDA_INSPECT_X]
    y = row[constants.PANDDA_INSPECT_Y]
    z = row[constants.PANDDA_INSPECT_Z]

    hit_confidence = row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE]
    if hit_confidence == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
        hit_confidence_class = True
    else:
        hit_confidence_class = False

    processed_dataset_dir = pandda_processed_datasets_dir / dtag
    inspect_model_dir = processed_dataset_dir / constants.PANDDA_INSPECT_MODEL_DIR
    event_map_path = processed_dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
        dtag=dtag,
        event_idx=event_idx,
        bdc=bdc
    )
    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    ligand = get_event_ligand(
        inspect_model_path,
        x,
        y,
        z,
    )

    event = PanDDAEvent(
        id=0,
        pandda_dir=str(pandda_dir),
        model_building_dir=str(model_building_dir),
        dtag=dtag,
        event_idx=int(event_idx),
        event_map=str(event_map_path),
        x=float(x),
        y=float(y),
        z=float(z),
        hit=hit_confidence_class,
        ligand=ligand
    )

    return event


def parse_pandda_inspect_table(pandda_inspect_table_file,
                               potential_pandda_dir,
                               pandda_processed_datasets_dir,
                               model_building_dir,
                               ):
    pandda_inspect_table = pd.read_csv(pandda_inspect_table_file)

    events = []
    for index, row in pandda_inspect_table.iterrow():
        possible_event = parse_inspect_table_row(
            row, potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir)
        if possible_event:
            events.append(possible_event)

    if len(events) > 0:
        return events
    else:
        return None


def parse_potential_pandda_dir(potential_pandda_dir, model_building_dir):
    pandda_analysis_dir = potential_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    pandda_inspect_table_file = pandda_analysis_dir / constants.PANDDA_INSPECT_TABLE_FILE
    pandda_processed_datasets_dir = pandda_analysis_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    if pandda_analysis_dir.exists():
        if pandda_inspect_table_file.exists():
            events = parse_pandda_inspect_table(
                pandda_inspect_table_file,
                potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir

            )
            return events

    return None


def parse_pandda_dataset(options: Options):

    pandda_data_root_dir = Path(constants.PANDDA_DATA_ROOT_DIR)
    logger.info(f"Looking for PanDDAs under dir: {pandda_data_root_dir}")

    pandda_events = []
    for year_dir_or_project_superdir in pandda_data_root_dir.glob("*"):
        logger.info(f"Checking superdir: {year_dir_or_project_superdir}")
        for project_dir in year_dir_or_project_superdir.glob("*"):
            logger.info(f"Checking project dir: {project_dir}")

            analysis_dir = project_dir / constants.DIAMOND_PROCESSING_DIR / constants.DIAMOND_ANALYSIS_DIR

            model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_NEW
            if not model_building_dir.exists():
                model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_OLD
                if not model_building_dir.exists():
                    logger.debug(f"No model building dir: skipping!")
                    continue

            logger.debug(f"Model building dir is: {model_building_dir}")

            for potential_pandda_dir in analysis_dir.glob("*"):
                logger.debug(f"Checking folder {potential_pandda_dir} ")
                potential_pandda_data = parse_potential_pandda_dir(
                    potential_pandda_dir,
                    model_building_dir,
                )
                if potential_pandda_data:
                    pandda_events += potential_pandda_data
                    logger.info(f"Found {len(potential_pandda_data)} events with models!")

                else:
                    logger.debug(f"Discovered no events with models: skipping!")

    pandda_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    pandda_dataset.save(options.working_dir)


def partition_pandda_dataset(dataset):
    system_split = get_system_split(dataset, 0.2)
    smiles_split = get_smiles_split(dataset, 0.2)


class CLI:

    def download_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        download_dataset(options)

    def parse_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        # dataset = Dataset.load(options.working_dir)
        parse_dataset(options)

    def generate_smiles(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = StructureReflectionsDataset.load(options.working_dir)

        generate_smiles(options, dataset)

    # def generate_conformations(self):
    #     ...
    #
    # def generate_plausible_decoys(self):
    #     ...
    #
    # def generate_implausible_decoys(self):
    #     ...

    def partition_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = StructureReflectionsDataset.load(options.working_dir)

    def train(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = StructureReflectionsDataset.load(options.working_dir)

    def test(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = StructureReflectionsDataset.load(options.working_dir)

    def parse_pandda_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        parse_pandda_dataset(options)

    def partition_pandda_dataset(self, options_json_path: str = "./options.json"):
        ...

    def train_pandda(self, options_json_path: str = "./options.json"):
        ...

    def test_pandda(self, options_json_path: str = "./options.json"):
        ...

    def generate_reannotate_table(self):
        ...

    def reannotate(self):
        ...

    def parse_reannotations(self):
        ...


if __name__ == "__main__":
    fire.Fire(CLI)
