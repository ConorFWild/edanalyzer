import fire
from pathlib import Path
import subprocess
from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand
import constants
from loguru import logger
from openbabel import pybel
import gemmi
# from rdkit import Chem
from numpy.random import default_rng
# from torch_dataset import *


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
            sfs[entry.stem] = entry
    logger.info(f"Found {len(sfs)} structure factors...")

    for sub in pdbs_dir.glob("*"):
        for entry in sub.glob("*"):
            if entry.stem in sfs:
                pdbs[entry.stem] = entry
    logger.info(f"Found {len(pdbs)} that could be associated with structure factors...")

    datas = []
    id = 0
    for entry_name, path in sfs.items():
        logger.debug(f"Processing dataset: {entry_name}")
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
        f, phi = get_structure_factors(dt)
        if f is None:
            logger.debug(f"No recognisable structure factors!")
            continue
        logger.info(f"Structure factors are: {f} {phi}")
        dt.f = f
        dt.phi = phi

        # Get ligands
        ligands = get_structure_ligands(dt)
        if len(ligands) ==0:
            logger.debug("Did not find any ligands!")
            continue
        logger.debug(f"Found {len(ligands)} ligands")
        dt.ligands = ligands

        id += 1
        datas.append(dt)

    logger.info(f"Found {len(datas)} complete datasets!")

    dataset = StructureReflectionsDataset(datas)
    dataset.save(options.working_dir)
    # return Dataset(datas)


def parse_ligand(structure_template, chain, ligand_residue):
    structure = structure_template.clone()
    for model in structure:
        for _chain in chain:
            if _chain.name != chain.name:
                model.remove_chain(_chain.name)

    new_chain = gemmi.Chain()
    new_chain.name = chain.name
    new_chain.add_residue(ligand_residue)

    pdb_string = structure.make_minimal_pdb()

    pybel_mol = pybel.readstring("pdb", pdb_string)

    smiles = pybel_mol.write("can")

    # smiles = Chem.MolToSmiles(mol)

    return smiles


def get_structure_ligands(data: StructureReflectionsData):
    # logger.info(f"")
    structure = gemmi.read_structure(data.pdb_path)
    structure_ligands = []
    id = 0
    for model in structure:
        for chain in model:
            ligands = chain.get_ligands()
            for ligand in ligands:
                # structure_ligands.append(

                smiles = parse_ligand(
                    structure,
                    chain,
                    ligand,
                )
                lig = Ligand(
                    id=id,
                    smiles=smiles,
                    chain=chain.name,
                    residue=ligand.seqid.num
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


if __name__ == "__main__":
    fire.Fire(CLI)
