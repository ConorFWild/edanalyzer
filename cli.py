import os
import re

import fire
from pathlib import Path
import subprocess
from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand, PanDDAEvent, \
    PanDDAEventDataset, PanDDAEventAnnotations, PanDDAEventAnnotation
import constants
from torch_dataset import PanDDAEventDatasetTorch, get_image_from_event, get_annotation_from_event_annotation, \
    get_image_event_map_and_raw_from_event, get_image_event_map_and_raw_from_event_augmented

from loguru import logger
# from openbabel import pybel
import gemmi
# from rdkit import Chem
from numpy.random import default_rng
# from torch_dataset import *
import numpy as np
import traceback
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_network import squeezenet1_1, resnet18
import download_dataset
import dataclasses

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
    structure_ligands = get_structure_ligands(str(inspect_model_path))

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
    dtag = str(row[constants.PANDDA_INSPECT_DTAG])
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
    if not event_map_path.exists():
        return None

    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=dtag)
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    if inspect_model_path.exists() & hit_confidence_class:
        ligand = get_event_ligand(
            inspect_model_path,
            x,
            y,
            z,
        )
        inspect_model_path_string = str(inspect_model_path)
    else:
        ligand = None
        inspect_model_path_string = None

    hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    if len(hyphens) == 0:
        return None
    else:
        last_hypen_pos = hyphens[-1]
        system_name = dtag[:last_hypen_pos + 1]

    event = PanDDAEvent(
        id=0,
        pandda_dir=str(pandda_dir),
        model_building_dir=str(model_building_dir),
        system_name=system_name,
        dtag=dtag,
        event_idx=int(event_idx),
        event_map=str(event_map_path),
        x=float(x),
        y=float(y),
        z=float(z),
        hit=hit_confidence_class,
        inspect_model_path=inspect_model_path_string,
        ligand=ligand
    )

    return event


def parse_pandda_inspect_table(
        pandda_inspect_table_file,
        potential_pandda_dir,
        pandda_processed_datasets_dir,
        model_building_dir,
):
    try:
        pandda_inspect_table = pd.read_csv(pandda_inspect_table_file)
    except Exception as e:
        logger.warning(f"Failed to read table: {pandda_inspect_table_file} : {e}")
        return None

    events = []
    for index, row in pandda_inspect_table.iterrows():
        possible_event = parse_inspect_table_row(
            row, potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir)
        if possible_event:
            events.append(possible_event)

    events_with_models = len([event for event in events if event.ligand is not None])

    if events_with_models > 0:
        return events
    else:
        return None


def parse_potential_pandda_dir(potential_pandda_dir, model_building_dir):
    pandda_analysis_dir = potential_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    pandda_inspect_table_file = pandda_analysis_dir / constants.PANDDA_INSPECT_TABLE_FILE
    pandda_processed_datasets_dir = potential_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    if not os.access(pandda_analysis_dir, os.R_OK):
        return None
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
                    logger.info(f"Found {len(potential_pandda_data)} events!")
                    num_events_with_ligands = len(
                        [event for event in potential_pandda_data if event.ligand is not None])
                    logger.info(f"Events which are modelled: {num_events_with_ligands}")

                else:
                    logger.debug(f"Discovered no events with models: skipping!")

    logger.info(f"Found {len(pandda_events)} events!")
    num_events_with_ligands = len([event for event in pandda_events if event.ligand is not None])
    logger.info(f"Found {num_events_with_ligands} events with ligands modelled!")

    pandda_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    pandda_dataset.save(Path(options.working_dir))


def split_dataset_on(dataset, f, fraction):
    positive_events = [event for event in dataset.pandda_events if event.ligand]
    num_dataset = len(positive_events)
    clss = []
    for data in dataset.pandda_events:
        cls = f(data)
        clss.append(cls)

    cls_set = list(set(clss))
    logger.debug(f"Num systems: {cls_set}")

    while True:
        rng = default_rng()
        choice = rng.choice(cls_set, size=int(fraction * len(cls_set)), replace=False)
        logger.debug(f"Choice: {choice}")
        choice_events = [data for data in positive_events if (data.system_name in choice)]

        logger.debug(f"Num choice events: {len(choice_events)}")
        not_choice = [x for x in cls_set if x not in choice]
        logger.debug(f"Not choice: {not_choice}")
        non_choice_events = [data for data in positive_events if (data.system_name not in choice)]
        logger.debug(f"Num non-choice events: {len(non_choice_events)}")

        choice_fraction = float(len(choice_events)) / num_dataset
        logger.debug(f"Choice fraction: {choice_fraction}")

        if np.abs(choice_fraction - fraction) < 0.025:
            return choice


def partition_pandda_dataset(options, dataset):
    system_split = split_dataset_on(dataset, lambda data: data.system_name, 0.2)
    logger.info(system_split)
    events_in_split = [event for event in dataset.pandda_events if event.system_name in system_split]
    events_not_in_split = [event for event in dataset.pandda_events if event.system_name not in system_split]
    logger.info(len(events_in_split))
    logger.info(len(events_not_in_split))

    # annotations = []
    # for event in dataset.pandda_events:
    #     if event.system_name in system_split:
    #         annotations.append(PanDDAEventAnnotation(annotation=True))
    #     else:
    #         annotations.append(PanDDAEventAnnotation(annotation=False))

    train_set = PanDDAEventDataset(pandda_events=events_not_in_split)
    train_set_annotations = []
    for event in train_set.pandda_events:
        if event.hit:
            train_set_annotations.append(PanDDAEventAnnotation(annotation=True))
        else:
            train_set_annotations.append(PanDDAEventAnnotation(annotation=False))

    train_set.save(Path(options.working_dir) / constants.TRAIN_SET_FILE)

    PanDDAEventAnnotations(annotations=train_set_annotations).save(
        Path(options.working_dir) / constants.TRAIN_SET_ANNOTATION_FILE)

    test_set = PanDDAEventDataset(pandda_events=events_in_split)
    test_set_annotations = []
    for event in train_set.pandda_events:
        if event.hit:
            test_set_annotations.append(PanDDAEventAnnotation(annotation=True))
        else:
            test_set_annotations.append(PanDDAEventAnnotation(annotation=False))

    test_set.save(Path(options.working_dir) / constants.TEST_SET_FILE)

    PanDDAEventAnnotations(annotations=test_set_annotations).save(
        Path(options.working_dir) / constants.TEST_SET_ANNOTATION_FILE)

    # smiles_split = get_smiles_split(dataset, 0.2)


def train_pandda(
        options: Options,
        dataset: PanDDAEventDataset,
        annotations: PanDDAEventAnnotations):
    num_epochs = 100
    logger.info(f"Training on {len(dataset.pandda_events)} events!")

    # Get the dataset
    dataset_torch = PanDDAEventDatasetTorch(
        dataset,
        annotations,
        transform_image=get_image_event_map_and_raw_from_event_augmented,
        transform_annotation=get_annotation_from_event_annotation

    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=True, num_workers=12)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=2)


    model = model.train()

    # Define loss function
    criterion = nn.BCELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=0.00001)

    optimizer.zero_grad()

    running_loss = 0

    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    model.to(dev)
    # Trainloop

    running_loss = []

    for epoch in range(num_epochs):
        i = 0
        print(f"Epoch: {epoch}")
        for image, annotation, idx in train_dataloader:
            print(f"\tBatch: {i}")
            # print(image)
            # print(annotation)
            # print(image.shape)
            image_c = image.to(dev)
            annotation_c = annotation.to(dev)

            optimizer.zero_grad()

            # forward + backward + optimize
            model_annotation = model(image_c)
            # print(outputs.to("cpu").detach().numpy())
            loss = criterion(model_annotation, annotation_c)
            loss.backward()
            optimizer.step()

            # RECORD LOSS
            running_loss.append(loss.item())

            # print statistics per epoch
            i += 1
            if i % 100 == 99:  # print every 100 mini-batches

                model_annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in model_annotation]
                annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in annotation]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss[-90:]) / 90}")

                for model_annotation_np, annotation_np in zip(model_annotations_np, annotations_np):
                    print(f"{round(float(model_annotation_np[1]), 2)} : {round(float(annotation_np[1]), 2)}")
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        # torch.save(model.state_dict(), Path(options.working_dir) / constants.MODEL_FILE)


def try_make_dir(path: Path):
    if path.exists():
        os.mkdir(path)


def symlink(source_path: Path, target_path: Path):
    os.symlink(source_path, target_path)


def make_fake_processed_dataset_dir(event: PanDDAEvent, processed_datasets_dir: Path):
    processed_dataset_dir = processed_datasets_dir / event.dtag
    try_make_dir(processed_dataset_dir)

    pandda_model_dir = processed_dataset_dir / constants.PANDDA_INSPECT_MODEL_DIR
    try_make_dir(pandda_model_dir)

    initial_pdb_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=event.dtag)
    fake_initial_pdb_path = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=event.dtag)
    symlink(initial_pdb_path, fake_initial_pdb_path)

    inital_mtz_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
        dtag=event.dtag)
    fake_inital_mtz_path = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=event.dtag)
    symlink(inital_mtz_path, fake_inital_mtz_path)

    event_map_path = Path(event.event_map)
    fake_event_map_path = processed_dataset_dir / event_map_path.name
    symlink(event_map_path, fake_event_map_path)

    zmap_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_ZMAP_TEMPLATE.format(
        dtag=event.dtag)
    fake_zmap_path = processed_dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=event.dtag)
    symlink(zmap_path, fake_zmap_path)


@dataclasses.dataclass()
class EventTableRecord:
    dtag: str
    event_idx: int
    bdc: float
    cluster_size: int
    global_correlation_to_average_map: float
    global_correlation_to_mean_map: float
    local_correlation_to_average_map: float
    local_correlation_to_mean_map: float
    site_idx: int
    x: float
    y: float
    z: float
    z_mean: float
    z_peak: float
    applied_b_factor_scaling: float
    high_resolution: float
    low_resolution: float
    r_free: float
    r_work: float
    analysed_resolution: float
    map_uncertainty: float
    analysed: bool
    interesting: bool
    exclude_from_z_map_analysis: bool
    exclude_from_characterisation: bool

    @staticmethod
    def from_event(event: PanDDAEvent):
        matches = re.findall(
            "_1-BDC_([0-9.]+)_map\.native\.ccp4",
            event.event_map
        )
        bdc = float(matches[0])

        return EventTableRecord(
            dtag=event.dtag,
            event_idx=event.event_idx,
            bdc=bdc,
            cluster_size=0,
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=0,
            x=event.x,
            y=event.y,
            z=event.z,
            z_mean=0.0,
            z_peak=0.0,
            applied_b_factor_scaling=0.0,
            high_resolution=0.0,
            low_resolution=0.0,
            r_free=0.0,
            r_work=0.0,
            analysed_resolution=0.0,
            map_uncertainty=0.0,
            analysed=False,
            interesting=False,
            exclude_from_z_map_analysis=False,
            exclude_from_characterisation=False,
        )


@dataclasses.dataclass()
class EventTable:
    records: list[EventTableRecord]

    @staticmethod
    def from_pandda_event_dataset(pandda_event_dataset: PanDDAEventDataset):
        records = []
        for event in pandda_event_dataset.pandda_events:
            event_record = EventTableRecord.from_event(event)
            records.append(event_record)

        return EventTable(records)

    def save(self, path: Path):
        records = []
        for record in self.records:
            event_dict = dataclasses.asdict(record)
            event_dict["1-BDC"] = round(1 - event_dict["bdc"], 2)
            records.append(event_dict)
        table = pd.DataFrame(records)
        table.to_csv(str(path))


def make_fake_event_table(dataset: PanDDAEventDataset, path: Path):
    event_table = EventTable.from_pandda_event_dataset(dataset)
    event_table.save(path)


@dataclasses.dataclass()
class SiteTableRecord:
    site_idx: int
    centroid: tuple[float, float, float]

    @staticmethod
    def from_site_id(site_id: int, centroid: tuple[float, float, float]):
        return SiteTableRecord(
            site_idx=site_id,
            centroid=(centroid[0], centroid[1], centroid[2],),
        )


@dataclasses.dataclass()
class SiteTable:
    site_record_list: list[SiteTableRecord]

    def __iter__(self):
        for record in self.site_record_list:
            yield record

    @staticmethod
    def from_pandda_event_dataset(pandda_event_dataset: PanDDAEventDataset):

        records = [SiteTableRecord(0, (0.0, 0.0, 0.0))]

        return SiteTable(records)

    def save(self, path: Path):
        records = []
        for site_record in self.site_record_list:
            site_record_dict = dataclasses.asdict(site_record)
            records.append(site_record_dict)

        table = pd.DataFrame(records)

        table.to_csv(str(path))


def make_fake_site_table(dataset: PanDDAEventDataset, path: Path):
    site_table = SiteTable.from_pandda_event_dataset(dataset)
    site_table.save(path)


def make_fake_pandda(dataset: PanDDAEventDataset, path: Path):
    fake_pandda_dir = path
    fake_analyses_dir = fake_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    fake_event_table_path = fake_analyses_dir / constants.PANDDA_EVENT_TABLE_PATH
    fake_site_table_path = fake_analyses_dir / constants.PANDDA_SITE_TABLE_PATH
    fake_processed_datasets_dir = path / constants.PANDDA_PROCESSED_DATASETS_DIR

    try_make_dir(fake_pandda_dir)
    try_make_dir(fake_analyses_dir)
    try_make_dir(fake_processed_datasets_dir)

    make_fake_event_table(dataset, fake_event_table_path)
    make_fake_site_table(dataset, fake_site_table_path)

    fake_processed_dataset_dirs = {}
    for event in dataset.pandda_events:
        make_fake_processed_dataset_dir(event, fake_processed_datasets_dir)


def annotate_test_set(options: Options, dataset: PanDDAEventDataset, annotations: PanDDAEventAnnotations):
    # Get the dataset
    dataset_torch = PanDDAEventDatasetTorch(
        dataset,
        annotations,
        transform_image=get_image_event_map_and_raw_from_event,
        transform_annotation=get_annotation_from_event_annotation
    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

    model = squeezenet1_1(num_classes=2, num_input=2)
    model.load_state_dict(torch.load(Path(options.working_dir) / constants.MODEL_FILE))
    model.eval()

    records = {}
    for image, annotation, idx in train_dataloader:
        image_c = image.to(dev)
        annotation_c = annotation.to(dev)

        # forward
        model_annotation = model(image_c)

        annotation_np = annotation.to(torch.device("cpu")).detach().numpy()
        model_annotation_np = model_annotation.to(torch.device("cpu")).detach().numpy()
        idx_np = idx.to(torch.device("cpu")).detach().numpy()

        #
        for _annotation, _model_annotation, _idx in zip(annotation_np, model_annotation_np, idx_np):
            records[_idx] = {"annotation": _annotation[1], "model_annotation": _model_annotation[1]}

    # Save a model annotations json
    # pandda_event_model_annotations = PanDDAEventModelAnnotations(
    #     annotations={
    #         _idx: records[_idx]["model_annotation"] for _idx in records
    #     }
    # )

    # Sort by model annotation
    sorted_idxs = sorted(records, key=lambda x: records[x]["model_annotation"], reverse=True)

    # Get highest scoring non-hits
    high_scoring_non_hits = []
    for sorted_idx in sorted_idxs:
        if len(high_scoring_non_hits) > 1000:
            continue
        if records[sorted_idx]["annotation"] == 0.0:
            high_scoring_non_hits.append(sorted_idx)

    # Get the lowest scoring hits
    low_scoring_hits = []
    for sorted_idx in reversed(sorted_idxs):
        if len(low_scoring_hits) > 1000:
            continue
        if records[sorted_idx]["annotation"] == 1.0:
            low_scoring_hits.append(sorted_idx)

    # Make fake PanDDA and inspect table for high scoring non hits
    high_scoring_non_hit_dataset = PanDDAEventDataset(pandda_events=[
        dataset[_idx] for _idx in high_scoring_non_hits
    ])
    make_fake_pandda(high_scoring_non_hit_dataset, Path(options.working_dir) / HIGH_SCORING_NON_HIT_DATASET_FILE)

    # Make fake PanDDA and inspect table for low scoring hits
    low_scoring_hit_dataset = PanDDAEventDataset(pandda_events=[
        dataset[_idx] for _idx in high_scoring_non_hits
    ])
    make_fake_pandda(low_scoring_hit_dataset, Path(options.working_dir) / LOW_SCORING_HIT_DATASET_FILE)


# def

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
        options = Options.load(options_json_path)
        dataset = PanDDAEventDataset.load(Path(options.working_dir) / constants.PANDDA_DATASET_FILE)
        partition_pandda_dataset(options, dataset)

    def train_pandda(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = PanDDAEventDataset.load(Path(options.working_dir) / constants.TRAIN_SET_FILE)
        annotations = PanDDAEventAnnotations.load(Path(options.working_dir) / constants.TRAIN_SET_ANNOTATION_FILE)

        train_pandda(options, dataset, annotations)

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
