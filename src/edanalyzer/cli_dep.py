import os
import pickle
import re
import shutil

import fire
from pathlib import Path
import subprocess
from edanalyzer.data import (
    StructureReflectionsDataset, Options, StructureReflectionsData, Ligand, PanDDAEvent,
    PanDDAEventDataset, PanDDAEventAnnotations, PanDDAEventAnnotation, PanDDAUpdatedEventAnnotations,
    PanDDAEventKey,load_model
)
from edanalyzer import constants
from edanalyzer.torch_dataset import (
    PanDDAEventDatasetTorch, PanDDADatasetTorchXmapGroundState, get_annotation_from_event_annotation,
    get_image_event_map_and_raw_from_event, get_image_event_map_and_raw_from_event_augmented,
get_annotation_from_event_hit, get_image_xmap_mean_map_augmented, get_image_xmap_mean_map,
get_image_xmap_ligand_augmented, PanDDADatasetTorchLigand, get_image_xmap_ligand, get_image_ligandmap_augmented,
PanDDADatasetTorchLigandmap
)
from edanalyzer.database import (
    populate_from_diamond, initialize_database, populate_partition_from_json,
    parse_old_annotation_update_dir, populate_from_custom_panddas, EventORM, PanDDAORM, AnnotationORM,

)
from edanalyzer.losses import categorical_loss

from edanalyzer.database_pony import *

from pony.orm import *


from loguru import logger
# from openbabel import pybel
import gemmi
# from rdkit import Chem
from numpy.random import default_rng
# from torch_dataset import *
import numpy as np
import traceback
import pandas as pd
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from edanalyzer.torch_network import resnet18
from edanalyzer.torch_network_resnet import resnet18
from edanalyzer.torch_network_resnet_ligandmap import resnet18_ligandmap
from edanalyzer.torch_network_squeezenet import squeezenet1_1, squeezenet1_0
from edanalyzer.torch_network_mobilenet import mobilenet_v3_large_3d
# import download_dataset
import dataclasses
import time

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload


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

                # smiles = parse_ligand(
                #     structure,
                #     chain,
                #     ligand,
                # )
                smiles = ""
                # logger.debug(f"Ligand smiles: {smiles}")
                # logger.debug(f"Num atoms: {num_atoms}")
                # logger.debug(f"Centroid: {ligand_centroid}")
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


# def train(options: Options, dataset: StructureReflectionsDataset):
#     # Get the dataset
#     dataset_torch = StructureReflectionsDatasetTorch(
#         dataset,
#         transform=lambda data: sample_ligand_density(
#             data,
#             lambda _data: annotate_data_randomly(_data, 0.5),
#             lambda _data, _annotation: generate_xmap_ligand_sample_or_decoy(
#                 _data,
#                 _annotation,
#                 sample_ligand=lambda __data: generate_ligand_sample(
#                     __data,
#                     get_ligand_decoy_transform,
#                     sample_xmap_from_data
#                 ),
#                 sample_ligand_decoy=lambda __data: generate_ligand_sample(
#                     __data,
#                     get_ligand_transform,
#                     sample_xmap_from_data,
#
#                 )
#             )
#         )
#     )
#
#     # Get the dataloader
#     train_dataloader = DataLoader(dataset_torch, batch_size=1, shuffle=True)
#
#     # Trainloop
#
#     ...


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
    viewed = row[constants.PANDDA_INSPECT_VIEWED]

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

    if not viewed:
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
        annotations: PanDDAEventAnnotations,
        updated_annotations: PanDDAUpdatedEventAnnotations,
        begin_epoch,
        model_file,
        num_workers=36,
        update=False
):
    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    num_epochs = 100
    logger.info(f"Training on {len(dataset.pandda_events)} events!")

    # Get the dataset
    dataset_torch = PanDDAEventDatasetTorch(
        dataset,
        annotations,
        updated_annotations,
        transform_image=get_image_event_map_and_raw_from_event_augmented,
        transform_annotation=get_annotation_from_event_annotation

    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=True, num_workers=num_workers)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=4)
    model.to(dev)

    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=dev),
                              )
    model = model.train()

    # Define loss function
    criterion = categorical_loss

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=0.00001)

    optimizer.zero_grad()

    running_loss = 0

    # Trainloop

    running_loss = []



    for epoch in range(begin_epoch+1, begin_epoch+num_epochs):
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
            begin_annotate = time.time()
            model_annotation = model(image_c)
            finish_annotate = time.time()
            logger.debug(f"Annotated 12 datasets in {finish_annotate - begin_annotate}")
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
                print([(x, type(x)) for x in annotation])
                idxs = [int(x) for x in idx]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss[-90:]) / 90}")

                for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
                    mod_an = round(float(model_annotation_np[1]), 2)
                    an = round(float(annotation_np[1]), 2)
                    event = dataset[_idx]
                    event_path = event.event_map

                    print(
                        f"{mod_an} : {an} : {event_path}"
                    )
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        torch.save(model.state_dict(), Path(options.working_dir) / constants.MODEL_FILE_EPOCH.format(epoch=epoch))

def train_pandda_from_dataset(
        options: Options,
        dataset: PanDDAEventDataset,
        begin_epoch,
        model_file,
        num_workers=36,
        update=False
):
    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    num_epochs = 500
    logger.info(f"Training on {len(dataset.pandda_events)} events!")

    # Get the dataset
    dataset_torch = PanDDADatasetTorchXmapGroundState(
        dataset,
        transform_image=get_image_xmap_mean_map_augmented,
        transform_annotation=get_annotation_from_event_hit

    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=True, num_workers=num_workers)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=3)
    model.to(dev)

    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=dev),
                              )
    model = model.train()

    # Define loss function
    criterion = categorical_loss

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           # lr=0.001,
                           )

    optimizer.zero_grad()

    running_loss = 0

    # Trainloop

    running_loss = []

    for epoch in range(begin_epoch+1, begin_epoch+num_epochs):
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
            begin_annotate = time.time()
            model_annotation = model(image_c)
            finish_annotate = time.time()
            logger.debug(f"Annotated 12 datasets in {finish_annotate - begin_annotate}")
            # print(outputs.to("cpu").detach().numpy())
            loss = criterion(model_annotation, annotation_c)
            loss.backward()
            optimizer.step()

            # RECORD LOSS
            running_loss.append(loss.item())

            # print statistics per epoch
            i += 1
            if i % 1000 == 999:  # print every 100 mini-batches

                model_annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in model_annotation]
                annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in annotation]
                print([(x, type(x)) for x in annotation])
                idxs = [int(x) for x in idx]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss[-998:]) / 998}")

                for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
                    mod_an = round(float(model_annotation_np[1]), 2)
                    an = round(float(annotation_np[1]), 2)
                    event = dataset[_idx]
                    event_path = event.event_map

                    print(
                        f"{mod_an} : {an} : {event_path}"
                    )
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        torch.save(model.state_dict(), Path(options.working_dir) / constants.MODEL_FILE_EPOCH_XMAP_MEAN.format(epoch=epoch))

def train_pandda_from_dataset_ligand(
        options: Options,
        dataset: PanDDAEventDataset,
        begin_epoch,
        model_file,
        num_workers=36,
        update=False
):
    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    num_epochs = 1000
    logger.info(f"Training on {len(dataset.pandda_events)} events!")

    # Get the dataset
    dataset_torch = PanDDADatasetTorchLigand(
        dataset,
        transform_image=get_image_xmap_ligand_augmented,
        transform_annotation=get_annotation_from_event_hit
    )


    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=True, num_workers=num_workers)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=4)
    model.to(dev)

    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=dev),
                              )
    model = model.train()

    # Define loss function
    criterion = categorical_loss

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           # lr=0.001,
                           )

    optimizer.zero_grad()

    running_loss = 0

    # Trainloop

    running_loss = []

    for epoch in range(begin_epoch + 1, begin_epoch + num_epochs):
        i = 0
        print(f"Epoch: {epoch}")
        for image, annotation, idx in train_dataloader:
            # print(f"\tBatch: {i}")
            # print(image)
            # print(annotation)
            # print(image.shape)
            image_c = image.to(dev)
            annotation_c = annotation.to(dev)

            optimizer.zero_grad()

            # forward + backward + optimize
            begin_annotate = time.time()
            model_annotation = model(image_c)
            finish_annotate = time.time()
            # logger.debug(f"Annotated 12 datasets in {finish_annotate - begin_annotate}")
            # print(outputs.to("cpu").detach().numpy())
            loss = criterion(model_annotation, annotation_c)
            loss.backward()
            optimizer.step()

            # RECORD LOSS
            running_loss.append(loss.item())

            # print statistics per epoch
            i += 1
            if i % 1000 == 999:  # print every 100 mini-batches

                model_annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in model_annotation]
                annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in annotation]
                print([(x, type(x)) for x in annotation])
                idxs = [int(x) for x in idx]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss[-998:]) / 998}")
                logger.debug(f"Recent loss is: {sum(running_loss[-998:]) / 998}")

                for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
                    mod_an = round(float(model_annotation_np[1]), 2)
                    an = round(float(annotation_np[1]), 2)
                    event = dataset[_idx]
                    event_path = event.event_map

                    print(
                        f"{mod_an} : {an} : {event_path}"
                    )
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        torch.save(model.state_dict(),
                   Path(options.working_dir) / constants.MODEL_FILE_EPOCH_XMAP_LIGAND.format(epoch=epoch))

def train(
        # options,
        working_dir,
        train_dataset_torch,
        test_dataset_torch,
        model,
        initial_epoch,
        model_key,
        dev,
        test_interval,
        batch_size=12,
        num_workers=20,
        num_epochs=1000,
):

    # Get the dataloader
    test_dataloader = DataLoader(
        test_dataset_torch,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset_torch,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    # Define loss function
    criterion = categorical_loss

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
                           # lr=0.001,
                           )

    optimizer.zero_grad()

    running_loss = 0

    # Trainloop

    running_loss = []

    # for epoch in range(initial_epoch + 1, initial_epoch + num_epochs):
    annotations = {}
    for epoch in range(initial_epoch + 1, num_epochs):
        i = 0
        print(f"Epoch: {epoch}")
        model.train()

        for image, annotation, idx in train_dataloader:
            # if i > 1000:
            #     if i < 4180:
            #         continue
            # print(f"\tBatch: {i}")
            # print(image)
            # print(annotation)
            # print(image.shape)
            image_c = image.to(dev)
            annotation_c = annotation.to(dev)

            optimizer.zero_grad()

            # forward + backward + optimize
            begin_annotate = time.time()
            model_annotation = model(image_c)

            # print(f"Image: {image}")
            # print(f"Annotation: {annotation}")
            # print(f"Model annotation: {model_annotation}")
            # print(model_annotation)

            finish_annotate = time.time()
            # logger.debug(f"Annotated 12 datasets in {finish_annotate - begin_annotate}")
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
                # print([(x, type(x)) for x in annotation])
                idxs = [int(x) for x in idx]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss[-98:]) / 98}")
                logger.debug(f"Recent loss is: {sum(running_loss[-98:]) / 98}")

                for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
                    mod_an = round(float(model_annotation_np[1]), 2)
                    an = round(float(annotation_np[1]), 2)
                    # event = dataset[_idx]
                    # event_path = event.event_map
                    print(
                        f"{mod_an} : {an} "
                    )
                    # print(
                    #     f"{mod_an} : {an} : {event_path}"
                    # )
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")



        if epoch % test_interval == 0:
            logger.info(f"Evaluating on test dataset!")
            model.eval()
            # annotations[i] = {}
            annotations[epoch] = {}
            for image, annotation, idx in test_dataloader:
                image_c = image.to(dev)
                annotation_c = annotation.to(dev)

                optimizer.zero_grad()

                # forward + backward + optimize
                # begin_annotate = time.time()
                model_annotation = model(image_c)
                event = test_dataset_torch.pandda_event_dataset[idx]
                # print(event)
                # annotations[i][(event.pandda_dir, event.dtag, event.event_idx)] = (
                # annotations[epoch][(event.pandda_dir, event.dtag, event.event_idx)] = (
                #     float(annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
                #     float(model_annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
                # )
                annotations[epoch][(event.PanDDA_Path, event.Dtag, event.Event_IDX)] = (
                    float(annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
                    float(model_annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
                )
            # print(annotations[i])

            with open( Path(working_dir)/ f"annotations_{model_key}.pickle", 'wb') as f:
                pickle.dump(annotations, f)

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        torch.save(
            model.state_dict(),
            Path(working_dir) / f"{model_key}{epoch}.pt",
        )

def save_example_ligandmap(ligandmap, output_path):
    grid = gemmi.FloatGrid(30,30,30)

    grid_array = np.array(grid, copy=False)
    grid_array[:,:,:] = ligandmap[:,:,:]
    grid.set_unit_cell(gemmi.UnitCell(15, 15, 15, 90.0, 90.0, 90.0))
    grid.spacegroup = gemmi.find_spacegroup_by_name("P1")
    m = gemmi.Ccp4Map()
    m.grid = grid
    m.update_ccp4_header()
    # m.set_extent(...)
    m.write_ccp4_map(output_path)



def train_ligandmap(
        options,
        dataset_torch,
        model,
        initial_epoch,
        model_key,
        dev,
        batch_size=12,
        num_workers=20,
        num_epochs=1000,
):

    # Get the dataloader
    train_dataloader = DataLoader(
        dataset_torch,
        batch_size=batch_size,
        # batch_size=12,
        shuffle=True,
        num_workers=num_workers,
    )

    # Define loss function
    criterion1 = categorical_loss
    criterion2 = torch.nn.MSELoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(),
                           # lr=0.001,
                           )

    optimizer.zero_grad()

    running_loss = 0

    # Trainloop

    running_loss_classification = []
    running_loss_ligandmap = []

    for epoch in range(initial_epoch + 1, initial_epoch + num_epochs):
        i = 0
        print(f"Epoch: {epoch}")
        for image, annotation, ligandmap, loaded_classification, loaded_ligandmap, idx in train_dataloader:
            # print(f"\tBatch: {i}")
            # print(image)
            # print(annotation)
            # print(image.shape)
            # if not loaded_classification or loaded_ligandmap:
            #     print(f"Failed to load!")
            #     continue
            image_c = image.to(dev)
            annotation_c = annotation.to(dev)
            ligandmap_c = ligandmap.to(dev)

            optimizer.zero_grad()

            # forward + backward + optimize
            begin_annotate = time.time()
            model_annotation_classification, model_annotation_ligandmap = model(image_c)
            # print(model_annotation)

            finish_annotate = time.time()
            # logger.debug(f"Annotated 12 datasets in {finish_annotate - begin_annotate}")
            # print(outputs.to("cpu").detach().numpy())
            # if loaded_classification:
            loss_classification = criterion1(model_annotation_classification, annotation_c)
            # loss1.backward(retain_graph=True)
            running_loss_classification.append(loss_classification.item())

            # if loaded_ligandmap:
            loss_ligandmap = 5*criterion2(model_annotation_ligandmap, ligandmap_c)
            # loss.backward()
            running_loss_ligandmap.append(loss_ligandmap.item())

            loss = loss_classification + loss_ligandmap
            loss.backward()

            optimizer.step()

            # RECORD LOSS

            # print statistics per epoch
            i += 1
            if i % 100 == 99:  # print every 100 mini-batches

                model_annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in model_annotation_classification]
                annotations_np = [x.to(torch.device("cpu")).detach().numpy() for x in annotation]
                ligandmaps_np = [x.to(torch.device("cpu")).detach().numpy() for x in model_annotation_ligandmap]

                print([(x, type(x)) for x in annotation])
                idxs = [int(x) for x in idx]
                # print("Loss at epoch {}, iteration {} is {}".format(epoch,
                #                                                     i,
                #                                                     running_loss / i) + "\n")
                print(f"Recent loss is: {sum(running_loss_classification[-98:]) / 98}")
                logger.debug(f"Recent loss is: {sum(running_loss_classification[-98:]) / 98}")
                print(f"Recent ligandmap loss is: {sum(running_loss_ligandmap[-98:]) / 98}")
                logger.debug(f"Recent ligandmap loss is: {sum(running_loss_ligandmap[-98:]) / 98}")

                save_example_ligandmap(ligandmaps_np[0], "./example.ccp4")

                for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
                    mod_an = round(float(model_annotation_np[1]), 2)
                    an = round(float(annotation_np[1]), 2)
                    # event = dataset[_idx]
                    # event_path = event.event_map
                    print(
                        f"{mod_an} : {an} "
                    )
                    # print(
                    #     f"{mod_an} : {an} : {event_path}"
                    # )
                    # print("{}".format() + "\n")
                print("#################################################" + "\n")
            #     model_annotation_np = model_annotation_classification.to(torch.device("cpu")).detach().numpy()
            #     annotation_np = annotation.to(torch.device("cpu")).detach().numpy()
            #     # print([(x, type(x)) for x in annotation])
            #     _idx = int(idx)
            #     # print("Loss at epoch {}, iteration {} is {}".format(epoch,
            #     #                                                     i,
            #     #                                                     running_loss / i) + "\n")
            #     print(f"Recent loss is: {sum(running_loss_classification[-98:]) / 98}")
            #     logger.debug(f"Recent loss is: {sum(running_loss_classification[-98:]) / 98}")
            #     print(f"Recent reconstruction loss is: {sum(running_loss_ligandmap[-98:]) / 98}")
            #     logger.debug(f"Recent reconstruction loss is: {sum(running_loss_ligandmap[-98:]) / 98}")
            #
            #     # for model_annotation_np, annotation_np, _idx in zip(model_annotations_np, annotations_np, idxs):
            #     mod_an = round(float(model_annotation_np[1]), 2)
            #     an = round(float(annotation_np[1]), 2)
            #     # event = dataset[_idx]
            #     # event_path = event.event_map
            #     print(
            #         f"{mod_an} : {an} "
            #     )
            #     # print(
            #     #     f"{mod_an} : {an} : {event_path}"
            #     # )
            #     # print("{}".format() + "\n")
            # print("#################################################" + "\n")

        logger.info(f"Saving state dict for model at epoch: {epoch}")
        torch.save(
            model.state_dict(),
            Path(options.working_dir) / f"{model_key}{epoch}.pt",
        )

def try_make_dir(path: Path):
    if not path.exists():
        os.mkdir(path)


def symlink(source_path: Path, target_path: Path):
    if not target_path.exists():
        os.symlink(source_path, target_path)


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
    def from_event(event: PanDDAEvent, site_idx=None):
        matches = re.findall(
            "_1-BDC_([0-9.]+)_map\.native\.ccp4",
            event.event_map
        )
        bdc = float(matches[0])

        if site_idx:
            _site_idx = site_idx
        else:
            _site_idx = 0

        return EventTableRecord(
            dtag=event.dtag,
            event_idx=event.event_idx,
            bdc=1 - bdc,
            cluster_size=0,
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=_site_idx,
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

    @staticmethod
    def from_event_database(event: PanDDAEvent, site_idx=None):
        matches = re.findall(
            "_1-BDC_([0-9.]+)_map\.native\.ccp4",
            event.event_map
        )
        bdc = float(matches[0])

        if site_idx:
            _site_idx = site_idx
        else:
            _site_idx = 0

        return EventTableRecord(
            dtag=str(event.id),
            event_idx=1,
            bdc=bdc,
            cluster_size=0,
            global_correlation_to_average_map=0,
            global_correlation_to_mean_map=0,
            local_correlation_to_average_map=0,
            local_correlation_to_mean_map=0,
            site_idx=_site_idx,
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
        for j, event in enumerate(pandda_event_dataset.pandda_events):
            event_record = EventTableRecord.from_event(event, int(j / 100))
            records.append(event_record)

        return EventTable(records)

    @staticmethod
    def from_pandda_event_dataset_database(pandda_event_dataset: PanDDAEventDataset):
        records = []
        for j, event in enumerate(pandda_event_dataset.pandda_events):
            event_record = EventTableRecord.from_event_database(event, int(j / 100))
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
    event_table = EventTable.from_pandda_event_dataset_database(dataset)
    event_table.save(path)
    return event_table


def make_fake_processed_dataset_dir(event: PanDDAEvent, event_table_record: EventTableRecord, processed_datasets_dir: Path):
    # processed_dataset_dir = processed_datasets_dir / event.dtag
    processed_dataset_dir = processed_datasets_dir / str(event.id)
    try_make_dir(processed_dataset_dir)

    pandda_model_dir = processed_dataset_dir / constants.PANDDA_INSPECT_MODEL_DIR
    try_make_dir(pandda_model_dir)

    initial_pdb_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=event.dtag)
    fake_initial_pdb_path = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=event.id)
    symlink(initial_pdb_path, fake_initial_pdb_path)

    inital_mtz_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
        dtag=event.dtag)
    fake_inital_mtz_path = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=event.id)
    symlink(inital_mtz_path, fake_inital_mtz_path)

    event_map_path = Path(event.event_map)
    # fake_event_map_path = processed_dataset_dir / event_map_path.name
    fake_event_map_path = processed_dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
        dtag=event.id,
        event_idx=1,
        bdc=round(1-event_table_record.bdc, 2)
    )

    symlink(event_map_path, fake_event_map_path)

    zmap_path = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_ZMAP_TEMPLATE.format(
        dtag=event.dtag)
    fake_zmap_path = processed_dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=event.id)
    symlink(zmap_path, fake_zmap_path)

    pandda_model_file = Path(
        event.pandda_dir) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag / constants.PANDDA_INSPECT_MODEL_DIR / constants.PANDDA_MODEL_FILE.format(
        dtag=event.dtag)
    # fake_model_file = pandda_model_dir / pandda_model_file.name
    fake_model_file = pandda_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=event.id)

    if pandda_model_file.exists():
        symlink(pandda_model_file, fake_model_file)




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
    def from_pandda_event_dataset(pandda_event_dataset: PanDDAEventDataset, event_table: EventTable):

        site_ids = []
        for _record in event_table.records:
            if _record.site_idx in site_ids:
                continue
            else:
                site_ids.append(_record.site_idx)

        records = []
        for site_id in site_ids:
            records.append(SiteTableRecord(site_id, (0.0, 0.0, 0.0)))

        return SiteTable(records)

    def save(self, path: Path):
        records = []
        for site_record in self.site_record_list:
            site_record_dict = dataclasses.asdict(site_record)
            records.append(site_record_dict)

        table = pd.DataFrame(records)

        table.to_csv(str(path))


def make_fake_site_table(dataset: PanDDAEventDataset, path: Path, event_table):
    site_table = SiteTable.from_pandda_event_dataset(dataset, event_table)
    site_table.save(path)


def make_fake_pandda(dataset: PanDDAEventDataset, path: Path):
    fake_pandda_dir = path
    fake_analyses_dir = fake_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    fake_event_table_path = fake_analyses_dir / constants.PANDDA_EVENT_TABLE_PATH
    fake_site_table_path = fake_analyses_dir / constants.PANDDA_SITE_TABLE_PATH
    fake_processed_datasets_dir = fake_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR

    try_make_dir(fake_pandda_dir)
    try_make_dir(fake_analyses_dir)
    try_make_dir(fake_processed_datasets_dir)

    event_table = make_fake_event_table(dataset, fake_event_table_path)
    make_fake_site_table(dataset, fake_site_table_path, event_table)

    fake_processed_dataset_dirs = {}
    for event, event_table_record in zip(dataset.pandda_events, event_table.records):
        logger.debug(f"Copying event dir: {event.dtag} {event.event_idx}")
        make_fake_processed_dataset_dir(
            event,
            event_table_record,
            fake_processed_datasets_dir,
        )


def annotate_test_set(
        options: Options,
        dataset: PanDDAEventDataset,
        annotations: PanDDAEventAnnotations,
        updated_annotations: PanDDAUpdatedEventAnnotations,
        test_annotation_dir: Path,
events: dict[int, EventORM]
):
    logger.info(f"Output directory is: {test_annotation_dir}")
    if not test_annotation_dir.exists():
        os.mkdir(test_annotation_dir)

    records_file = test_annotation_dir / "train_records.pickle"
    logger.info(f"Record file is: {records_file}")

    if not records_file.exists():
        logger.info(f"No record file to parse: Annotating dataset!")
        # Get the dataset
        dataset_torch = PanDDAEventDatasetTorch(
            dataset,
            annotations,
            updated_annotations=updated_annotations,
            transform_image=get_image_event_map_and_raw_from_event,
            transform_annotation=get_annotation_from_event_annotation
        )

        # Get the dataloader
        train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

        # model = squeezenet1_1(num_classes=2, num_input=2)
        model = resnet18(num_classes=2, num_input=4)
        model.load_state_dict(torch.load(Path(options.working_dir) / constants.MODEL_FILE))
        model.eval()

        if torch.cuda.is_available():
            logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            logger.info(f"Using cpu!")
            dev = "cpu"

        model.to(dev)
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
                event = dataset.pandda_events[_idx]
                logger.debug(f"{event.dtag} {event.event_idx} {_annotation[1]} {_model_annotation[1]}")

        # Save a model annotations json
        # pandda_event_model_annotations = PanDDAEventModelAnnotations(
        #     annotations={
        #         _idx: records[_idx]["model_annotation"] for _idx in records
        #     }
        # )

        with open(records_file, "wb") as f:
            pickle.dump(records, f)

    else:
        logger.info(f"Records file for CNN annotations exists: loading!")
        with open(records_file, "rb") as f:

            records = pickle.load(f)

    for cutoff in np.linspace(0.0,1.0, 100):
        fp = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == False) & (_record["model_annotation"] > cutoff))
        ]
        tp = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == True) & (_record["model_annotation"] > cutoff))
        ]
        fn = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == True) & (_record["model_annotation"] < cutoff))
        ]
        tn = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == False) & (_record["model_annotation"] < cutoff))
        ]
        if len(tp+fp) != 0:
            precission = len(tp) / len(tp+fp)
        else:
            precission = 0.0
        if len(tp+fn):
            recall = len(tp) / len(tp+fn)
        else:
            recall =0.0
        logger.info(f"Cutoff: {cutoff}: Precission: {precission} : Recall: {recall}")

    # Sort by model annotation
    sorted_idxs = sorted(records, key=lambda x: records[x]["model_annotation"], reverse=True)

    # Get highest scoring non-hits
    high_scoring_non_hits = []
    for sorted_idx in sorted_idxs:
        event = dataset.pandda_events[sorted_idx]
        event_orm = events[event.id]
        if "manual" in [annotation.source for annotation in event_orm.annotations]:
            logger.debug(f"Already have manual annotation!")
            continue

        if len(high_scoring_non_hits) > 5000:
            continue
        if records[sorted_idx]["annotation"] == 0.0:
            high_scoring_non_hits.append(sorted_idx)
    logger.info(f"Got {len(high_scoring_non_hits)} high scoring non-hits!")

    # Get the lowest scoring hits
    low_scoring_hits = []
    for sorted_idx in reversed(sorted_idxs):

        # Skip if already manually annotated
        event = dataset.pandda_events[sorted_idx]
        event_orm = events[event.id]
        if "manual" in [annotation.source for annotation in event_orm.annotations]:
            logger.debug(f"Already have manual annotation!")
            continue

        if len(low_scoring_hits) > 5000:
            continue
        if records[sorted_idx]["annotation"] == 1.0:
            low_scoring_hits.append(sorted_idx)
    logger.info(f"Got {len(low_scoring_hits)} low scoring hits!")

    # Make fake PanDDA and inspect table for high scoring non hits
    pandda_events = []
    dtag_event_ids = []
    for _idx in high_scoring_non_hits:
        event = dataset.pandda_events[_idx]
        # key = (event.dtag, event.event_idx)
        # if key in dtag_event_ids:
        #     continue
        # else:
        pandda_events.append(event)
            # dtag_event_ids.append(key)
    high_scoring_non_hit_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    make_fake_pandda(
        high_scoring_non_hit_dataset,
        test_annotation_dir / constants.HIGH_SCORING_NON_HIT_DATASET_DIR,
    )

    # Make fake PanDDA and inspect table for low scoring hits
    pandda_events = []
    dtag_event_ids = []
    for _idx in low_scoring_hits:
        event = dataset.pandda_events[_idx]
        # key = (event.dtag, event.event_idx)
        # if key in dtag_event_ids:
        #     continue
        # else:
        pandda_events.append(event)
            # dtag_event_ids.append(key)

    low_scoring_hit_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    make_fake_pandda(
        low_scoring_hit_dataset,
        test_annotation_dir / constants.LOW_SCORING_HIT_DATASET_DIR,
    )

def annotate_dataset_ligand(
        dataset: PanDDAEventDataset,
        test_annotation_dir: Path,
        model_file,
        events: dict[int, EventORM]
):
    logger.info(f"Output directory is: {test_annotation_dir}")
    if not test_annotation_dir.exists():
        os.mkdir(test_annotation_dir)

    records_file = test_annotation_dir / "train_records.pickle"
    logger.info(f"Record file is: {records_file}")

    if not records_file.exists():
        logger.info(f"No record file to parse: Annotating dataset!")
        # Get the dataset
        dataset_torch = PanDDADatasetTorchLigand(
            dataset,
            transform_image=get_image_xmap_ligand,
            transform_annotation=get_annotation_from_event_hit
        )

        # Get the dataloader
        train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

        # model = squeezenet1_1(num_classes=2, num_input=2)
        model = resnet18(num_classes=2, num_input=4)
        model.load_state_dict(torch.load(model_file))
        model.eval()

        if torch.cuda.is_available():
            logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            logger.info(f"Using cpu!")
            dev = "cpu"

        model.to(dev)
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
                event = dataset.pandda_events[_idx]
                logger.debug(f"{event.dtag} {event.event_idx} {_annotation[1]} {_model_annotation[1]}")

        # Save a model annotations json
        # pandda_event_model_annotations = PanDDAEventModelAnnotations(
        #     annotations={
        #         _idx: records[_idx]["model_annotation"] for _idx in records
        #     }
        # )

        with open(records_file, "wb") as f:
            pickle.dump(records, f)

    else:
        logger.info(f"Records file for CNN annotations exists: loading!")
        with open(records_file, "rb") as f:

            records = pickle.load(f)

    # Sort by model annotation
    sorted_idxs = sorted(records, key=lambda x: records[x]["model_annotation"], reverse=True)

    # Get highest scoring non-hits
    high_scoring_non_hits = []
    for sorted_idx in sorted_idxs:
        event = dataset.pandda_events[sorted_idx]
        event_orm = events[event.id]
        if "manual" in [annotation.source for annotation in event_orm.annotations]:
            logger.debug(f"Already have manual annotation!")
            continue

        if len(high_scoring_non_hits) > 5000:
            continue
        if records[sorted_idx]["annotation"] == 0.0:
            high_scoring_non_hits.append(sorted_idx)
    logger.info(f"Got {len(high_scoring_non_hits)} high scoring non-hits!")

    # Get the lowest scoring hits
    low_scoring_hits = []
    for sorted_idx in reversed(sorted_idxs):

        # Skip if already manually annotated
        event = dataset.pandda_events[sorted_idx]
        event_orm = events[event.id]
        if "manual" in [annotation.source for annotation in event_orm.annotations]:
            logger.debug(f"Already have manual annotation!")
            continue

        if len(low_scoring_hits) > 5000:
            continue
        if records[sorted_idx]["annotation"] == 1.0:
            low_scoring_hits.append(sorted_idx)
    logger.info(f"Got {len(low_scoring_hits)} low scoring hits!")

    # Make fake PanDDA and inspect table for high scoring non hits
    pandda_events = []
    dtag_event_ids = []
    for _idx in high_scoring_non_hits:
        event = dataset.pandda_events[_idx]
        # key = (event.dtag, event.event_idx)
        # if key in dtag_event_ids:
        #     continue
        # else:
        pandda_events.append(event)
            # dtag_event_ids.append(key)
    high_scoring_non_hit_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    make_fake_pandda(
        high_scoring_non_hit_dataset,
        test_annotation_dir / constants.HIGH_SCORING_NON_HIT_DATASET_DIR,
    )

    # Make fake PanDDA and inspect table for low scoring hits
    pandda_events = []
    for _idx in low_scoring_hits:
        event = dataset.pandda_events[_idx]
        pandda_events.append(event)

    low_scoring_hit_dataset = PanDDAEventDataset(pandda_events=pandda_events)
    make_fake_pandda(
        low_scoring_hit_dataset,
        test_annotation_dir / constants.LOW_SCORING_HIT_DATASET_DIR,
    )

def annotate_dataset(
        options: Options,
        dataset: PanDDAEventDataset,
        annotations: PanDDAEventAnnotations,
        updated_annotations: PanDDAUpdatedEventAnnotations,
        model_file: Path
):

    logger.info(f"No record file to parse: Annotating dataset!")
    # Get the dataset
    dataset_torch = PanDDAEventDatasetTorch(
        dataset,
        annotations,
        updated_annotations=updated_annotations,
        transform_image=get_image_event_map_and_raw_from_event,
        transform_annotation=get_annotation_from_event_annotation
    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=4)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    model.to(dev)
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
            event = dataset.pandda_events[_idx]
            # logger.debug(f"{event.dtag} {event.event_idx} {_annotation[1]} {_model_annotation[1]}")

    # Save a model annotations json
    # pandda_event_model_annotations = PanDDAEventModelAnnotations(
    #     annotations={
    #         _idx: records[_idx]["model_annotation"] for _idx in records
    #     }
    # )

    return records

def get_annotations_from_dataset(
        dataset: PanDDAEventDataset,
        model_file: Path
):

    logger.info(f"No record file to parse: Annotating dataset!")
    # Get the dataset
    dataset_torch = PanDDADatasetTorchXmapGroundState(
        dataset,
        transform_image=get_image_xmap_mean_map,
        transform_annotation=get_annotation_from_event_hit
    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=3)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    model.to(dev)
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
            event = dataset.pandda_events[_idx]
            # logger.debug(f"{event.dtag} {event.event_idx} {_annotation[1]} {_model_annotation[1]}")

    # Save a model annotations json
    # pandda_event_model_annotations = PanDDAEventModelAnnotations(
    #     annotations={
    #         _idx: records[_idx]["model_annotation"] for _idx in records
    #     }
    # )

    return records

def get_annotations_from_dataset_ligand(
        dataset: PanDDAEventDataset,
        model_file: Path
):

    logger.info(f"No record file to parse: Annotating dataset!")
    # Get the dataset
    dataset_torch = PanDDADatasetTorchLigand(
        dataset,
        transform_image=get_image_xmap_ligand,
        transform_annotation=get_annotation_from_event_hit
    )

    # Get the dataloader
    train_dataloader = DataLoader(dataset_torch, batch_size=12, shuffle=False, num_workers=12)

    # model = squeezenet1_1(num_classes=2, num_input=2)
    model = resnet18(num_classes=2, num_input=4)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    if torch.cuda.is_available():
        logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        logger.info(f"Using cpu!")
        dev = "cpu"

    model.to(dev)
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
            event = dataset.pandda_events[_idx]
            records[_idx] = {"annotation": _annotation[1], "model_annotation": _model_annotation[1], "event": event}

            # logger.debug(f"{event.dtag} {event.event_idx} {_annotation[1]} {_model_annotation[1]}")

    # Save a model annotations json
    # pandda_event_model_annotations = PanDDAEventModelAnnotations(
    #     annotations={
    #         _idx: records[_idx]["model_annotation"] for _idx in records
    #     }
    # )

    return records


def precission_recall(records):
    precission_recalls = {}
    for cutoff in np.linspace(0.0,1.0, 100):
        fp = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == False) & (_record["model_annotation"] > cutoff))
        ]
        tp = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == True) & (_record["model_annotation"] > cutoff))
        ]
        fn = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == True) & (_record["model_annotation"] < cutoff))
        ]
        tn = [
            _idx for _idx, _record in records.items()
            if ((_record["annotation"] == False) & (_record["model_annotation"] < cutoff))
        ]
        if len(tp+fp) != 0:
            precission = len(tp) / len(tp+fp)
        else:
            precission = 0.0
        if len(tp+fn):
            recall = len(tp) / len(tp+fn)
        else:
            recall =0.0
        precission_recalls[round(cutoff, 3)] = (round(precission, 3), round(recall, 3))

    return precission_recalls
        # logger.info(f"Cutoff: {round(cutoff, 3)}: Precission: {round(precission, 3)} : Recall: {round(recall, 3)}")


def dataset_and_annotations_from_database(options):
    engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

    with Session(engine) as session:
        logger.info(f"Loading events")
        events_stmt = select(EventORM).options(
            selectinload(EventORM.annotations),
            selectinload(EventORM.partitions),
            selectinload(EventORM.pandda).options(
                selectinload(PanDDAORM.system),
                selectinload(PanDDAORM.experiment),
            ),
        )
        events = session.scalars(events_stmt).unique().all()
        logger.info(f"Loaded {len(events)} events!")

        logger.info(f"Num events with partitions: {len([event for event in events if event.partitions])}")
        logger.info(f"Num events without partitions: {len([event for event in events if not event.partitions])}")

        events_with_partitions = [event for event in events if event.partitions]

        # Load train partition events
        train_partition_events = [
            event
            for event
            in events_with_partitions
            if constants.INITIAL_TRAIN_PARTITION == event.partitions.name
        ]
        logger.info(f"Got {len(train_partition_events)} finetune events!")

        # Load finetune train partition events
        finetune_train_partition_events = [
            event
            for event
            in events_with_partitions
            if constants.FINETUNE_TRAIN_PARTITION == event.partitions.name
        ]
        logger.info(f"Got {len(finetune_train_partition_events)} finetune events!")

        events_pyd = []
        annotations_pyd = []
        for event_orm in train_partition_events + finetune_train_partition_events:
            if event_orm.hit_confidence not in ["Low", "low"]:
                hit = True
            else:
                hit = False

            event_pyd = PanDDAEvent(
                id=event_orm.id,
                pandda_dir=event_orm.pandda.path,
                model_building_dir=event_orm.pandda.experiment.model_dir,
                system_name=event_orm.pandda.system.name,
                dtag=event_orm.dtag,
                event_idx=event_orm.event_idx,
                event_map=event_orm.event_map,
                x=event_orm.x,
                y=event_orm.y,
                z=event_orm.z,
                hit=hit,
                ligand=None
            )
            events_pyd.append(event_pyd)

            event_annotations = {
                annotation.source: annotation
                for annotation
                in event_orm.annotations
            }
            if "manual" in event_annotations:
                annotation_orm = event_annotations["manual"]
            else:
                annotation_orm = event_annotations["auto"]

            annotation_pyd = PanDDAEventAnnotation(
                annotation=annotation_orm.annotation
            )
            annotations_pyd.append(annotation_pyd)

        # Make the dataset
        dataset = PanDDAEventDataset(
            pandda_events=events_pyd
        )
        logger.info(f"Got {len(events_pyd)} events")

        # Make the annotations
        annotation_dataset = PanDDAEventAnnotations(annotations=annotations_pyd)
        logger.info(f"Got {len(annotations_pyd)} annotations")
        hits = [annotation_pyd for annotation_pyd in annotations_pyd if annotation_pyd.annotation]
        logger.info(f"Got {len(hits)} events annotated as hits")
        non_hits = [annotation_pyd for annotation_pyd in annotations_pyd if not annotation_pyd.annotation]
        logger.info(f"Got {len(non_hits)} events annotated as hits")

        # Make a blank updated annotations
        updated_annotations = PanDDAUpdatedEventAnnotations(
            keys=[],
            annotations=[]

        )

    return dataset, annotation_dataset, updated_annotations, {event.id: event for event in train_partition_events + finetune_train_partition_events}

def get_events(options):
    engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

    with Session(engine) as session:
        logger.info(f"Loading events")
        events_stmt = select(EventORM).options(
            selectinload(EventORM.annotations),
            selectinload(EventORM.partitions),
            selectinload(EventORM.pandda).options(
                selectinload(PanDDAORM.system),
                selectinload(PanDDAORM.experiment),
            ),
        )
        events = session.scalars(events_stmt).unique().all()
        logger.info(f"Loaded {len(events)} events!")

        logger.info(f"Num events with partitions: {len([event for event in events if event.partitions])}")
        logger.info(f"Num events without partitions: {len([event for event in events if not event.partitions])}")

    return events

def get_dataset_annotations_from_events(events):
    events_pyd = []
    annotations_pyd = []
    for event_id, event_orm in events.items():
        if event_orm.hit_confidence not in ["Low", "low"]:
            hit = True
        else:
            hit = False

        event_pyd = PanDDAEvent(
            id=event_orm.id,
            pandda_dir=event_orm.pandda.path,
            model_building_dir=event_orm.pandda.experiment.model_dir,
            system_name=event_orm.pandda.system.name,
            dtag=event_orm.dtag,
            event_idx=event_orm.event_idx,
            event_map=event_orm.event_map,
            x=event_orm.x,
            y=event_orm.y,
            z=event_orm.z,
            hit=hit,
            ligand=None
        )
        events_pyd.append(event_pyd)

        event_annotations = {
            annotation.source: annotation
            for annotation
            in event_orm.annotations
        }
        if "manual" in event_annotations:
            annotation_orm = event_annotations["manual"]
        else:
            annotation_orm = event_annotations["auto"]

        annotation_pyd = PanDDAEventAnnotation(
            annotation=annotation_orm.annotation
        )
        annotations_pyd.append(annotation_pyd)

    # Make the dataset
    dataset = PanDDAEventDataset(
        pandda_events=events_pyd
    )
    logger.info(f"Got {len(events_pyd)} events")

    # Make the annotations
    annotation_dataset = PanDDAEventAnnotations(annotations=annotations_pyd)
    logger.info(f"Got {len(annotations_pyd)} annotations")
    hits = [annotation_pyd for annotation_pyd in annotations_pyd if annotation_pyd.annotation]
    logger.info(f"Got {len(hits)} events annotated as hits")
    non_hits = [annotation_pyd for annotation_pyd in annotations_pyd if not annotation_pyd.annotation]
    logger.info(f"Got {len(non_hits)} events annotated as hits")

    # Make a blank updated annotations
    updated_annotations = PanDDAUpdatedEventAnnotations(
        keys=[],
        annotations=[],
    )


    return dataset, annotation_dataset, updated_annotations

def test_dataset_and_annotations_from_database(options):
    engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

    with Session(engine) as session:
        logger.info(f"Loading events")
        events_stmt = select(EventORM).options(
            selectinload(EventORM.annotations),
            selectinload(EventORM.partitions),
            selectinload(EventORM.pandda).options(
                selectinload(PanDDAORM.system),
                selectinload(PanDDAORM.experiment),
            ),
        )
        events = session.scalars(events_stmt).unique().all()
        logger.info(f"Loaded {len(events)} events!")

        logger.info(f"Num events with partitions: {len([event for event in events if event.partitions])}")
        logger.info(f"Num events without partitions: {len([event for event in events if not event.partitions])}")

        events_with_partitions = [event for event in events if event.partitions]

        # Load train partition events
        test_partition_events = [
            event
            for event
            in events_with_partitions
            if constants.INITIAL_TEST_PARTITION == event.partitions.name
        ]
        logger.info(f"Got {len(test_partition_events)} finetune events!")

        # Load finetune train partition events
        finetune_test_partition_events = [
            event
            for event
            in events_with_partitions
            if constants.FINETUNE_TEST_PARTITION == event.partitions.name
        ]
        logger.info(f"Got {len(finetune_test_partition_events)} finetune events!")

        events_pyd = []
        annotations_pyd = []
        for event_orm in test_partition_events + finetune_test_partition_events:
            if event_orm.hit_confidence not in ["Low", "low"]:
                hit = True
            else:
                hit = False

            event_pyd = PanDDAEvent(
                id=event_orm.id,
                pandda_dir=event_orm.pandda.path,
                model_building_dir=event_orm.pandda.experiment.model_dir,
                system_name=event_orm.pandda.system.name,
                dtag=event_orm.dtag,
                event_idx=event_orm.event_idx,
                event_map=event_orm.event_map,
                x=event_orm.x,
                y=event_orm.y,
                z=event_orm.z,
                hit=hit,
                ligand=None
            )
            events_pyd.append(event_pyd)

            event_annotations = {
                annotation.source: annotation
                for annotation
                in event_orm.annotations
            }
            if "manual" in event_annotations:
                annotation_orm = event_annotations["manual"]
            else:
                annotation_orm = event_annotations["auto"]

            annotation_pyd = PanDDAEventAnnotation(
                annotation=annotation_orm.annotation
            )
            annotations_pyd.append(annotation_pyd)

        # Make the dataset
        dataset = PanDDAEventDataset(
            pandda_events=events_pyd
        )
        logger.info(f"Got {len(events_pyd)} events")

        # Make the annotations
        annotation_dataset = PanDDAEventAnnotations(annotations=annotations_pyd)
        logger.info(f"Got {len(annotations_pyd)} annotations")
        hits = [annotation_pyd for annotation_pyd in annotations_pyd if annotation_pyd.annotation]
        logger.info(f"Got {len(hits)} events annotated as hits")
        non_hits = [annotation_pyd for annotation_pyd in annotations_pyd if not annotation_pyd.annotation]
        logger.info(f"Got {len(non_hits)} events annotated as hits")

        # Make a blank updated annotations
        updated_annotations = PanDDAUpdatedEventAnnotations(
            keys=[],
            annotations=[],
        )

    return dataset, annotation_dataset, updated_annotations, {event.id: event for event in test_partition_events + finetune_test_partition_events}


def check_accessible(event):
    processed_dataset_path = Path(event.pandda.path) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag

    ground_state_structure_path = processed_dataset_path / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=event.dtag)

    mtz_path = processed_dataset_path / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=event.dtag)

    ground_state_map_path = processed_dataset_path / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(dtag=event.dtag)

    bound_state_structure_path = processed_dataset_path / constants.PANDDA_INSPECT_MODEL_DIR / constants.PANDDA_MODEL_FILE.format(
        dtag=event.dtag)

    annotations = {annotation.source: annotation for annotation in event.annotations}
    if "manual" in annotations:
        annotation = annotations["manual"]
    else:
        annotation = annotations["auto"]

    # if ground_state_structure_path.exists() & ground_state_map_path.exists() & mtz_path.exists():
    try:
        gemmi.read_structure(str(ground_state_structure_path))
        gemmi.read_mtz_file(str(mtz_path))
        gemmi.read_ccp4_map(str(ground_state_map_path))
        return event

    except Exception as e:
        print(e)
        return None


class CLI:

    # def download_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     download_dataset(options)
    #
    # def parse_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     # dataset = Dataset.load(options.working_dir)
    #     parse_dataset(options)
    #
    # def generate_smiles(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = StructureReflectionsDataset.load(options.working_dir)
    #
    #     generate_smiles(options, dataset)

    # def generate_conformations(self):
    #     ...
    #
    # def generate_plausible_decoys(self):
    #     ...
    #
    # def generate_implausible_decoys(self):
    #     ...

    # def partition_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = StructureReflectionsDataset.load(options.working_dir)

    # def train(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = StructureReflectionsDataset.load(options.working_dir)

    # def test(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = StructureReflectionsDataset.load(options.working_dir)

    # def parse_pandda_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     parse_pandda_dataset(options)
    #
    # def partition_pandda_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = PanDDAEventDataset.load(Path(options.working_dir) / constants.PANDDA_DATASET_FILE)
    #     partition_pandda_dataset(options, dataset)
    #
    # def train_pandda(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = PanDDAEventDataset.load(Path(options.working_dir) / constants.TRAIN_SET_FILE)
    #     annotations = PanDDAEventAnnotations.load(Path(options.working_dir) / constants.TRAIN_SET_ANNOTATION_FILE)
    #     updated_annotations = PanDDAUpdatedEventAnnotations.load(
    #         Path(options.working_dir) / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE)
    #     train_pandda(options, dataset, annotations, updated_annotations)
    #
    # def annotate_train_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = PanDDAEventDataset.load(Path(options.working_dir) / constants.TRAIN_SET_FILE)
    #     annotations = PanDDAEventAnnotations.load(Path(options.working_dir) / constants.TRAIN_SET_ANNOTATION_FILE)
    #     updated_annotations = PanDDAUpdatedEventAnnotations.load(
    #         Path(options.working_dir) / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE)
    #
    #     annotate_test_set(options, dataset, annotations, updated_annotations, )
    #
    # def annotate_test_dataset(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = PanDDAEventDataset.load(Path(options.working_dir), name=constants.TEST_SET_FILE)
    #     annotations = PanDDAEventAnnotations.load(Path(options.working_dir) / constants.TEST_SET_ANNOTATION_FILE)
    #     updated_event_annotation_path = Path(options.working_dir) / constants.PANDDA_UPDATED_TEST_EVENT_ANNOTATIONS_FILE
    #     if updated_event_annotation_path.exists():
    #         updated_annotations = PanDDAUpdatedEventAnnotations.load(updated_event_annotation_path)
    #     else:
    #         updated_annotations = PanDDAUpdatedEventAnnotations(keys=[], annotations=[])
    #     test_annotations_dir = Path(options.working_dir) / constants.PANDDA_TEST_ANNOTATION_DIR
    #
    #     annotate_test_set(options, dataset, annotations, updated_annotations, test_annotations_dir)
    #
    # def parse_updated_annotations(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     output_dir = Path(options.working_dir)
    #     if (output_dir / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE).exists():
    #         pandda_updated_annotations = PanDDAUpdatedEventAnnotations.load(
    #             output_dir / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE)
    #     else:
    #         pandda_updated_annotations = PanDDAUpdatedEventAnnotations(keys=[], annotations=[])
    #
    #     logger.info(f"Currently {len(pandda_updated_annotations.keys)} updated annotations")
    #
    #     keys = []
    #     annotations = []
    #
    #     high_scoring_non_hit_dir = output_dir / constants.HIGH_SCORING_NON_HIT_DATASET_DIR
    #     high_scoring_non_hit_inspect_table_path = high_scoring_non_hit_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE
    #     high_scoring_non_hit_inspect_table = pd.read_csv(high_scoring_non_hit_inspect_table_path)
    #
    #     logger.info(f"Length of high scoring non hit inspect table: {len(high_scoring_non_hit_inspect_table)}")
    #     keys = []
    #     annotations = []
    #     for idx, row in high_scoring_non_hit_inspect_table.iterrows():
    #         dtag = row[constants.PANDDA_INSPECT_DTAG]
    #         event_idx = int(row[constants.PANDDA_INSPECT_EVENT_IDX])
    #         key = PanDDAEventKey(dtag=dtag, event_idx=event_idx)
    #         # print(row[constants.PANDDA_INSPECT_VIEWED])
    #         if row[constants.PANDDA_INSPECT_VIEWED] == True:
    #             if row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
    #                 annotation = PanDDAEventAnnotation(annotation=True)
    #             else:
    #                 annotation = PanDDAEventAnnotation(annotation=False)
    #             keys.append(key)
    #             annotations.append(annotation)
    #
    #     logger.info(f"Got {len(keys)} new high scoring non hit annotations!")
    #     for key, annotation in zip(keys, annotations):
    #         if key not in pandda_updated_annotations.keys:
    #             pandda_updated_annotations.keys.append(key)
    #             pandda_updated_annotations.annotations.append(annotation)
    #
    #     low_scoring_hit_dir = output_dir / constants.LOW_SCORING_HIT_DATASET_DIR
    #     low_scoring_hit_inspect_table_path = low_scoring_hit_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE
    #     low_scoring_hit_inspect_table = pd.read_csv(low_scoring_hit_inspect_table_path)
    #
    #     logger.info(f"Length of low scoring hit inspect table: {len(low_scoring_hit_inspect_table)}")
    #     keys = []
    #     annotations = []
    #     for idx, row in low_scoring_hit_inspect_table.iterrows():
    #         dtag = row[constants.PANDDA_INSPECT_DTAG]
    #         event_idx = int(row[constants.PANDDA_INSPECT_EVENT_IDX])
    #         key = PanDDAEventKey(dtag=dtag, event_idx=event_idx)
    #         if row[constants.PANDDA_INSPECT_VIEWED] == True:
    #             if row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
    #                 annotation = PanDDAEventAnnotation(annotation=True)
    #             else:
    #                 annotation = PanDDAEventAnnotation(annotation=False)
    #             keys.append(key)
    #             annotations.append(annotation)
    #
    #     logger.info(f"Got {len(keys)} total new annotations!")
    #
    #     for key, annotation in zip(keys, annotations):
    #         if key not in pandda_updated_annotations.keys:
    #             pandda_updated_annotations.keys.append(key)
    #             pandda_updated_annotations.annotations.append(annotation)
    #
    #     logger.info(f"Now {len(pandda_updated_annotations.keys)} updated annotations!")
    #
    #     pandda_updated_annotations.save(output_dir / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE)
    #     ...
    #
    # def test_pandda(self, options_json_path: str = "./options.json"):
    #     ...
    #
    # def generate_reannotate_table(self):
    #     ...
    #
    # def reannotate_pandda(self, pandda_dir, options_json_path: str = "./options.json", ):
    #
    #     options = Options.load(options_json_path)
    #
    #     records_file = Path(options.working_dir) / "annotate_pandda_records.pickle"
    #
    #     pandda_dir = Path(pandda_dir)
    #
    #     # Get the event table
    #
    #     # Make a copy of the event table
    #     analyse_table_path = pandda_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_EVENT_TABLE_PATH
    #     inspect_table_path = pandda_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE
    #
    #     deprecated_analyse_table_path = pandda_dir / constants.PANDDA_ANALYSIS_DIR / "pandda_analyse_events_dep.csv"
    #     deprecated_inspect_table_path = pandda_dir / constants.PANDDA_ANALYSIS_DIR / "pandda_inspect_events_dep.csv"
    #
    #     if not deprecated_analyse_table_path.exists():
    #         shutil.copyfile(analyse_table_path, deprecated_inspect_table_path)
    #
    #     if not deprecated_inspect_table_path.exists():
    #         shutil.copyfile(inspect_table_path, deprecated_inspect_table_path)
    #
    #     # Parse the event table
    #     table = pd.read_csv(inspect_table_path)
    #
    #     if not records_file.exists():
    #         logger.info(f"Performing annotation!")
    #
    #         # Get the device
    #         if torch.cuda.is_available():
    #             logger.info(f"Using cuda!")
    #             dev = "cuda:0"
    #         else:
    #             logger.info(f"Using cpu!")
    #             dev = "cpu"
    #
    #         # Load the model
    #         model = resnet18(num_classes=2, num_input=3)
    #         model.load_state_dict(torch.load(Path(options.working_dir) / constants.MODEL_FILE))
    #
    #         # Add model to device
    #         model.to(dev)
    #         model.eval()
    #
    #         # Iterate the event table, rescoring
    #         records = {}
    #         logger.info(f"Annotating {len(table)} events!")
    #         for idx, row in table.iterrows():
    #             # Get an event model
    #             event = parse_inspect_table_row(
    #                 row,
    #                 pandda_dir,
    #                 pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR,
    #                 "",
    #             )
    #
    #             # Load the image
    #             image, loaded = get_image_event_map_and_raw_from_event(event)
    #
    #             # Get tensor
    #             image_t = torch.unsqueeze(torch.from_numpy(image), 0)
    #
    #             # Move tensors to device
    #             image_c = image_t.to(dev)
    #             # Run model
    #             model_annotation = model(image_c)
    #
    #             # Track score
    #             model_annotation_np = model_annotation.to(torch.device("cpu")).detach().numpy()
    #
    #             #
    #             records[idx] = {"event": event, "model_annotation": model_annotation_np[0][1]}
    #             logger.debug(f"{idx} / {len(table)} : {event.dtag} {event.event_idx} {model_annotation[0][1]}")
    #
    #         # Cache scores
    #         with open(records_file, "wb") as f:
    #             pickle.dump(records, f)
    #
    #     # Load scores if they are there
    #     else:
    #         logger.info(f"Loading pre-analysed records!")
    #         with open(records_file, "rb") as f:
    #             records = pickle.load(f)
    #
    #     # Sort by score
    #     table["score"] = [record["model_annotation"] for record in records.values()]
    #     sorted_table = table.sort_values(by="score", ascending=False)
    #     logger.info(f"Sorted events by CNN score!")
    #     print(sorted_table)
    #
    #     # Save new table
    #     sorted_table.to_csv(analyse_table_path)
    #
    #     ...
    #
    # def parse_finetune_dataset(self, options_json_path: str = "./options.json"):
    #     # Get options
    #     options = Options.load(options_json_path)
    #
    #     # Get events for each finettune dataset path
    #     events = []
    #     for finetune_pandda_and_source_path in options.finetune_datasets_train:
    #         finetune_dataset_events = parse_potential_pandda_dir(
    #             Path(finetune_pandda_and_source_path.pandda),
    #             Path(finetune_pandda_and_source_path.source)
    #         )
    #         events += finetune_dataset_events
    #         logger.info(f"Got {len(finetune_dataset_events)} events from {finetune_pandda_and_source_path.pandda}")
    #
    #     logger.info(f"Got {len(events)} annotated events!")
    #
    #     pandda_dataset = PanDDAEventDataset(pandda_events=events)
    #     pandda_dataset.save(Path(options.working_dir), constants.FINETUNE_TRAIN_EVENTS_FILE)
    #
    #     pandda_dataset_annotations = []
    #     for event in pandda_dataset.pandda_events:
    #         if event.hit:
    #             pandda_dataset_annotations.append(PanDDAEventAnnotation(annotation=True))
    #         else:
    #             pandda_dataset_annotations.append(PanDDAEventAnnotation(annotation=False))
    #
    #     hits = [a for a in pandda_dataset_annotations if a.annotation]
    #     non_hits = [a for a in pandda_dataset_annotations if not a.annotation]
    #
    #     logger.info(f"Got {len(hits)} hits and {len(non_hits)} non-hits!")
    #
    #     PanDDAEventAnnotations(annotations=pandda_dataset_annotations).save(
    #         Path(options.working_dir) / constants.FINETUNE_TRAIN_SET_ANNOTATION_FILE)
    #
    # def finetune(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     dataset = PanDDAEventDataset.load(Path(options.working_dir), name=constants.FINETUNE_TRAIN_EVENTS_FILE)
    #     annotations = PanDDAEventAnnotations.load(
    #         Path(options.working_dir) / constants.FINETUNE_TRAIN_SET_ANNOTATION_FILE)
    #     updated_annotations = PanDDAUpdatedEventAnnotations.load(
    #         Path(options.working_dir) / constants.PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE)
    #     train_pandda(options, dataset, annotations, updated_annotations, update=True)
    #
    # def parse_reannotations(self):
    #     ...
    #
    # def initialize_database(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     initialize_database(f"{options.working_dir}/{constants.SQLITE_FILE}")
    #
    # def populate_database_diamond(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     with Session(engine) as session:
    #         populate_from_diamond(session)
    #
    # def populate_partitions(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     train_dataset = PanDDAEventDataset.load(Path(options.working_dir), name=constants.TRAIN_SET_FILE)
    #
    #     test_dataset = PanDDAEventDataset.load(Path(options.working_dir), name=constants.TEST_SET_FILE)
    #
    #     with Session(engine) as session:
    #         populate_partition_from_json(
    #             session,
    #             train_dataset,
    #             test_dataset,
    #         )
    #
    # def parse_old_annotation_update_dir(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     old_annotation_update_dirs = options.old_updated_annotation_dirs
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     for old_annotation_update_dir in old_annotation_update_dirs:
    #         with Session(engine) as session:
    #             logger.info(f"Parsing old update dir: {old_annotation_update_dir}")
    #             parse_old_annotation_update_dir(
    #                 session,
    #                 Path(old_annotation_update_dir),
    #             )
    #
    # def populate_from_custom_panddas_finetune_train(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     partition_name = constants.FINETUNE_TRAIN_PARTITION
    #
    #     funetune_train_datasets = options.finetune_datasets_train
    #     populate_from_custom_panddas(
    #         engine,
    #         funetune_train_datasets,
    #         partition_name
    #     )
    #
    # def populate_from_custom_panddas_finetune_test(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     partition_name = constants.FINETUNE_TEST_PARTITION
    #
    #     funetune_test_datasets = options.finetune_datasets_test
    #     populate_from_custom_panddas(
    #         engine,
    #         funetune_test_datasets,
    #         partition_name
    #     )
    #
    # def train_default_and_finetune(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     with Session(engine) as session:
    #         logger.info(f"Loading events")
    #         events_stmt = select(EventORM).options(
    #             selectinload(EventORM.annotations),
    #             selectinload(EventORM.partitions),
    #             selectinload(EventORM.pandda).options(
    #                 selectinload(PanDDAORM.system),
    #                 selectinload(PanDDAORM.experiment),
    #             ),
    #         )
    #         events = session.scalars(events_stmt).unique().all()
    #         logger.info(f"Loaded {len(events)} events!")
    #
    #         logger.info(f"Num events with partitions: {len([event for event in events if event.partitions])}")
    #         logger.info(f"Num events without partitions: {len([event for event in events if not event.partitions])}")
    #
    #         events_with_partitions = [event for event in events if event.partitions]
    #
    #         # Load train partition events
    #         train_partition_events = [
    #             event
    #             for event
    #             in events_with_partitions
    #             if constants.INITIAL_TRAIN_PARTITION == event.partitions.name
    #         ]
    #         logger.info(f"Got {len(train_partition_events)} finetune events!")
    #
    #         # Load finetune train partition events
    #         finetune_train_partition_events = [
    #             event
    #             for event
    #             in events_with_partitions
    #             if constants.FINETUNE_TRAIN_PARTITION == event.partitions.name
    #         ]
    #         logger.info(f"Got {len(finetune_train_partition_events)} finetune events!")
    #
    #         events_pyd = []
    #         annotations_pyd = []
    #         for event_orm in train_partition_events + finetune_train_partition_events:
    #             if event_orm.hit_confidence not in ["Low", "low"]:
    #                 hit = True
    #             else:
    #                 hit = False
    #
    #             event_pyd = PanDDAEvent(
    #                 id=event_orm.id,
    #                 pandda_dir=event_orm.pandda.path,
    #                 model_building_dir=event_orm.pandda.experiment.model_dir,
    #                 system_name=event_orm.pandda.system.name,
    #                 dtag=event_orm.dtag,
    #                 event_idx=event_orm.event_idx,
    #                 event_map=event_orm.event_map,
    #                 x=event_orm.x,
    #                 y=event_orm.y,
    #                 z=event_orm.z,
    #                 hit=hit,
    #                 ligand=None
    #             )
    #             events_pyd.append(event_pyd)
    #
    #             event_annotations = {
    #                 annotation.source: annotation
    #                 for annotation
    #                 in event_orm.annotations
    #             }
    #             if "manual" in event_annotations:
    #                 annotation_orm = event_annotations["manual"]
    #             else:
    #                 annotation_orm = event_annotations["auto"]
    #
    #             annotation_pyd = PanDDAEventAnnotation(
    #                 annotation=annotation_orm.annotation
    #             )
    #             annotations_pyd.append(annotation_pyd)
    #
    #         # Make the dataset
    #         dataset = PanDDAEventDataset(
    #             pandda_events=events_pyd
    #         )
    #         logger.info(f"Got {len(events_pyd)} events")
    #
    #         # Make the annotations
    #         annotation_dataset = PanDDAEventAnnotations(annotations=annotations_pyd)
    #         logger.info(f"Got {len(annotations_pyd)} annotations")
    #         hits = [annotation_pyd for annotation_pyd in annotations_pyd if annotation_pyd.annotation]
    #         logger.info(f"Got {len(hits)} events annotated as hits")
    #         non_hits = [annotation_pyd for annotation_pyd in annotations_pyd if not annotation_pyd.annotation]
    #         logger.info(f"Got {len(non_hits)} events annotated as hits")
    #
    #         # Make a blank updated annotations
    #         updated_annotations = PanDDAUpdatedEventAnnotations(
    #             keys=[],
    #             annotations=[]
    #
    #         )
    #
    #         model_files = {}
    #         for model_file in Path(options.working_dir).glob("*"):
    #             file_name = model_file.name
    #             match = re.match(constants.MODEL_FILE_REGEX, file_name)
    #             if match:
    #                 epoch = int(match[1])
    #                 model_files[epoch] = model_file
    #
    #         if len(model_files) > 0:
    #             model_file = model_files[max(model_files)]
    #             epoch = max(model_files)
    #         else:
    #             model_file = None
    #             epoch = 0
    #
    #         logger.info(f"Beggining from epoch: {epoch}")
    #
    #         train_pandda(
    #             options,
    #             dataset,
    #             annotation_dataset,
    #             updated_annotations,
    #             epoch,
    #         model_file,
    #             num_workers=12
    #         )
    #
    # def train_from_dataset_path(self, dataset_path, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     # Make the dataset
    #     # dataset = PanDDAEventDataset.load(dataset_path)
    #     dataset = load_model(dataset_path, PanDDAEventDataset)
    #
    #
    #     # Get the output model file
    #     model_files = {}
    #     for model_file in Path(options.working_dir).glob("*"):
    #         file_name = model_file.name
    #         match = re.match(constants.MODEL_FILE_REGEX_XMAP_MEAN, file_name)
    #         if match:
    #             epoch = int(match[1])
    #             model_files[epoch] = model_file
    #
    #     if len(model_files) > 0:
    #         model_file = model_files[max(model_files)]
    #         epoch = max(model_files)
    #     else:
    #         model_file = None
    #         epoch = 0
    #
    #     logger.info(f"Beggining from epoch: {epoch}")
    #
    #     train_pandda_from_dataset(
    #         options,
    #         dataset,
    #         epoch,
    #         model_file,
    #         num_workers=12
    #     )
    #
    # def train_from_dataset_path_ligand(self, dataset_path, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #
    #     # Make the dataset
    #     # dataset = PanDDAEventDataset.load(dataset_path)
    #     dataset = load_model(dataset_path, PanDDAEventDataset)
    #
    #
    #     # Get the output model file
    #     model_files = {}
    #     for model_file in Path(options.working_dir).glob("*"):
    #         file_name = model_file.name
    #         match = re.match(constants.MODEL_FILE_REGEX_XMAP_LIGAND, file_name)
    #         if match:
    #             epoch = int(match[1])
    #             model_files[epoch] = model_file
    #
    #     if len(model_files) > 0:
    #         model_file = model_files[max(model_files)]
    #         epoch = max(model_files)
    #     else:
    #         model_file = None
    #         epoch = 0
    #
    #     logger.info(f"Beggining from epoch: {epoch}")
    #
    #     train_pandda_from_dataset_ligand(
    #         options,
    #         dataset,
    #         epoch,
    #         model_file,
    #         num_workers=20
    #     )

    def make_train_dataset(
            self,
    ):
        DATASET_ID = "pandda_2_2023_06_27"

        # get the events
        engine = create_engine("sqlite:///test/database.db")
        session = Session(engine)
        events_stmt = select(EventORM).options(
            selectinload(EventORM.partitions),
            selectinload(EventORM.annotations),
            selectinload(EventORM.ligand),
            selectinload(EventORM.pandda).options(
                selectinload(PanDDAORM.system),
                selectinload(PanDDAORM.experiment)

            )
        )
        events = session.scalars(events_stmt).unique().all()
        print(f"Number of events: {len(events)}")


        # Get the events with partitions
        partitioned_events = [event for event in events if event.partitions]

        # Get which events have accessible data
        with Parallel(n_jobs=-2, prefer="threads", verbose=10) as parallel:
            possible_events = parallel(
                delayed(check_accessible)(_event)
                for _event
                in partitioned_events
            )

        complete_events = [_event for _event in possible_events if _event is not None]
        print(f"Number of compeleye events: {len(complete_events)}")


        # Get the train systems
        train_systems = {
            event.pandda.system.name: event.pandda.system
            for event
            in complete_events
            if event.partitions.name == constants.INITIAL_TRAIN_PARTITION
        }
        print(f"Number of train systems: {len(train_systems)}")

        # Define which partitions to use for train data
        partitions_used = ["pandda_2_2023_04_28", "pandda_2_2023_06_27", "train"]

        # Get the potential train events
        train_events = [
            event
            for event
            in complete_events
            if (event.pandda.system.name in train_systems) and (event.partitions.name in partitions_used)]
        # print(len(train_events))
        print(f"Number of potential train events: {len(train_systems)}")


        # Get the train events with parsable pdbs
        LIGAND_IGNORE_REGEXES = [
            "merged",
            "LIG-[a-zA-Z]+-",
            "dimple",
            "refine",
            "init",
            "pipedream",
            "phenix",
            "None",
            "blank",
            "control",
            "DMSO",
        ]

        train_events_with_parsable_ligand_pdbs = []
        for event in train_events:
            # processed_dataset_dir = Path(event.pandda.path) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag
            dataset_dir = Path(event.pandda.experiment.model_dir) / event.dtag / "compound"

            event_added = False
            # ligand_files_dir = processed_dataset_dir / "ligand_files"
            if dataset_dir.exists():
                ligand_pdbs = [
                    x
                    for x
                    in dataset_dir.glob("*.pdb")
                    if (x.exists()) and (x.stem not in LIGAND_IGNORE_REGEXES)
                ]
                if len(ligand_pdbs) > 0:
                    train_events_with_parsable_ligand_pdbs.append(event)

        # print(len(train_events_with_parsable_ligand_pdbs))
        print(f"Number of train events with ligand pdbs: {len(train_events_with_parsable_ligand_pdbs)}")

        # Get the hit and non-hit events
        hits = []
        non_hits = []
        for event in train_events_with_parsable_ligand_pdbs:

            annotations = {annotation.source: annotation for annotation in event.annotations}
            if "manual" in annotations:
                # print("manual!")
                annotation = annotations["manual"]
            else:
                annotation = annotations["auto"]

            # Only get non hits from old PanDDA
            if event.partitions.name == "pandda_2_2023_04_28":
                if annotation.annotation:
                    continue
                else:
                    non_hits.append(event)


            elif event.partitions.name == "pandda_2_2023_06_27":

                # print(annotation.annotation)
                if event.hit_confidence == "High":
                    hits.append(event)
                else:
                    non_hits.append(event)

            elif event.partitions.name == "train":

                # print(annotation.annotation)
                if annotation.annotation:
                    hits.append(event)
                else:
                    non_hits.append(event)
        # print(len(hits))
        # print(len(non_hits))
        print(f"Number of hits: {len(hits)}")
        print(f"Number of non-hits: {len(non_hits)}")


        # Balance the dataset by repeating hits
        num_hits = len(hits)
        num_non_hits = len(non_hits)
        repeated_hits = (hits * int(num_non_hits / num_hits)) + hits[: num_non_hits % num_hits]

        # Make the dataset
        train_events_pyd = []
        num_ligand_centroids = 0
        for event in repeated_hits + non_hits:
            annotations = {annotation.source: annotation for annotation in event.annotations}

            if "manual" in annotations:
                annotation = annotations["manual"]
            else:
                annotation = annotations["auto"]

            if event.partitions.name == "pandda_2_2023_04_28":
                if annotation.annotation:
                    continue
                else:
                    x, y, z = event.x, event.y, event.z


            elif event.partitions.name == "pandda_2_2023_06_27":
                if (event.ligand is not None) & (event.hit_confidence == "High"):
                    x, y, z = event.ligand.x, event.ligand.y, event.ligand.z
                    num_ligand_centroids += 1
                else:
                    x, y, z = event.x, event.y, event.z

            elif event.partitions.name == "train":
                x, y, z = event.x, event.y, event.z

            else:
                continue

            event_pyd = PanDDAEvent(
                id=event.id,
                pandda_dir=event.pandda.path,
                model_building_dir=event.pandda.experiment.model_dir,
                system_name=event.pandda.system.name,
                dtag=event.dtag,
                event_idx=event.event_idx,
                event_map=event.event_map,
                x=x,
                y=y,
                z=z,
                hit=annotation.annotation,
                ligand=None,
            )
            train_events_pyd.append(event_pyd)

        # print(len(train_events_pyd))
        # print(num_ligand_centroids)
        print(f"Number of events to train on: {len(train_events_pyd)}")
        print(f"Number of events with updated centroid: {num_ligand_centroids}")



        # Output the dataset
        train_dataset = PanDDAEventDataset(pandda_events=train_events_pyd)
        train_dataset.save(path=Path("."), name=f"train_dataset_{DATASET_ID}.json")

        # Get the test systems
        test_systems = {
            event.pandda.system.name: event.pandda.system
            for event
            in complete_events
            if event.partitions.name == constants.INITIAL_TEST_PARTITION
        }
        # print(len(test_systems))
        print(f"Number of test stsyems: {len(test_systems)}")

        # Define the partitions to use for test data
        partitions_used = ["pandda_2_2023_06_27", ]
        test_events = [
            event
            for event
            in complete_events
            if (event.pandda.system.name in test_systems) and (event.partitions.name in partitions_used)
        ]
        # print(len(test_events))
        print(f"Number of potential test events: {len(test_events)}")


        # Get the test events with ligand data
        test_events_with_parsable_ligand_pdbs = []
        for event in test_events:
            # processed_dataset_dir = Path(event.pandda.path) / constants.PANDDA_PROCESSED_DATASETS_DIR / event.dtag
            dataset_dir = Path(event.pandda.experiment.model_dir) / event.dtag / "compound"

            event_added = False
            # ligand_files_dir = processed_dataset_dir / "ligand_files"
            if dataset_dir.exists():
                ligand_pdbs = [
                    x
                    for x
                    in dataset_dir.glob("*.pdb")
                    if (x.exists()) and (x.stem not in LIGAND_IGNORE_REGEXES)
                ]
                if len(ligand_pdbs) > 0:
                    test_events_with_parsable_ligand_pdbs.append(event)

        # print(len(test_events_with_parsable_ligand_pdbs))
        print(f"Number of test events with ligand pdbs: {len(test_events)}")

        # Get the hit and non-hit test events
        hits = []
        non_hits = []
        for event in test_events_with_parsable_ligand_pdbs:
            annotations = {annotation.source: annotation for annotation in event.annotations}
            if "manual" in annotations:
                # print("manual!")
                annotation = annotations["manual"]
            else:
                annotation = annotations["auto"]

            # print(annotation.annotation)
            if annotation.annotation:
                hits.append(event)
            else:
                non_hits.append(event)
        # print(len(hits))
        # print(len(non_hits))
        print(f"Number of hits: {len(hits)}")
        print(f"Number of non-hits: {len(non_hits)}")
        # Get the baseline precisiion of the test set
        baseline_precission = len(hits) / (len(hits) + len(non_hits))
        # print(baseline_precission)
        print(f"Precission: {baseline_precission}")


        # Make the test dataset
        train_events_pyd = []
        # for event in (hits+[x for x in rng.choice(non_hits, len(hits), replace=False)]):
        # for event in test_events_with_parsable_ligand_pdbs:

        num_ligand_centroids = 0
        for event in test_events_with_parsable_ligand_pdbs:

            annotations = {annotation.source: annotation for annotation in event.annotations}

            if "manual" in annotations:
                annotation = annotations["manual"]
            else:
                annotation = annotations["auto"]

            if event.partitions.name == "pandda_2_2023_04_28":
                if annotation.annotation:
                    continue
                else:
                    x, y, z = event.x, event.y, event.z


            elif event.partitions.name == "pandda_2_2023_06_27":
                if (event.ligand is not None) & (event.hit_confidence == "High"):
                    x, y, z = event.ligand.x, event.ligand.y, event.ligand.z
                    num_ligand_centroids += 1
                else:
                    x, y, z = event.x, event.y, event.z

            elif event.partitions.name == "test":
                x, y, z = event.x, event.y, event.z

            else:
                continue

            event_pyd = PanDDAEvent(
                id=event.id,
                pandda_dir=event.pandda.path,
                model_building_dir=event.pandda.experiment.model_dir,
                system_name=event.pandda.system.name,
                dtag=event.dtag,
                event_idx=event.event_idx,
                event_map=event.event_map,
                x=x,
                y=y,
                z=z,
                hit=annotation.annotation,
                ligand=None,
            )
            train_events_pyd.append(event_pyd)

        # print(len(train_events_pyd))
        print(f"Number of events to test on: {len(train_events_pyd)}")
        print(f"Number of events with updated centroid: {num_ligand_centroids}")

        # Output the test dataset
        train_dataset = PanDDAEventDataset(pandda_events=train_events_pyd)
        train_dataset.save(path=Path("."), name=f"test_dataset_{DATASET_ID}.json")
    def train(self,
              dataset_path,
              model_path=None,
              data_type="ligand",
              # data_type="ligandmap",
              # model_type="squeeze+ligand",
              model_type="resnet+ligand",
              # model_type="resnet+ligandmap",
              # model_type="mobilenet+ligand",
              # model_key="squeeze_ligand_",
              model_key="resnet_ligand_masked_",
              # model_key="resnet_ligandmap_",
              # model_key="mobilenet_ligand_",

              options_json_path: str = "./options.json"):
        logger.info(f"Model type: {model_type}")
        logger.info(f"Model key: {model_key}")
        logger.info(f"Options path is: {options_json_path}")
        logger.info(f"Dataset path is: {dataset_path}")

        options = Options.load(options_json_path)

        # Make the dataset
        # dataset = PanDDAEventDataset.load(dataset_path)
        dataset = load_model(
            dataset_path,
            PanDDAEventDataset,
        )
        logger.info(f"Training on {len(dataset.pandda_events)} events!")

        if data_type == "ligand":
            dataset_torch = PanDDADatasetTorchLigand(
                dataset,
                transform_image=get_image_xmap_ligand_augmented,
                transform_annotation=get_annotation_from_event_hit
            )
        elif data_type == "ligandmap":
            dataset_torch = PanDDADatasetTorchLigandmap(
                dataset,
                transform_image=get_image_xmap_ligand_augmented,
                transform_annotation=get_annotation_from_event_hit,
                transform_ligandmap=get_image_ligandmap_augmented,
            )

        else:
            raise Exception

        # Get the output model file
        if model_path:
            model_path = Path(model_path)
            file_name = model_path.name
            match = re.match(
                model_key,
                file_name,
            )
            if match:
                re.match(
                    ".*([0-9]).pt",
                    file_name
                )
                epoch = int(match[1])
                model_file = model_path
            else:
                epoch = 0
                model_file = None
        else:
            epoch = 0
            model_file = None

        logger.info(f"Beggining from epoch: {epoch}")

        # Get the device
        if torch.cuda.is_available():
            logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            logger.info(f"Using cpu!")
            dev = "cpu"

        # Load the model
        if model_type == "squeeze+ligand":
            # model = squeezenet1_0(num_classes=2, num_features=4)
            model = squeezenet1_1(num_classes=2, num_features=4)

            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "resnet+ligand":
            model = resnet18(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "resnet+ligandmap":
            model = resnet18_ligandmap(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "mobilenet+ligand":
            model = mobilenet_v3_large_3d(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )

        else:
            raise Exception
        model = model.train()

        if data_type == "ligand":
            train(
            options,
            dataset_torch,
            model,
            epoch,
            model_key,
            dev,
            num_workers=20,
        )
        elif data_type == "ligandmap":
            train_ligandmap(
                options,
                dataset_torch,
                model,
                epoch,
                model_key,
                dev,
                num_workers=20,
            )

        # train_pandda_from_dataset_ligand(
        #     options,
        #     dataset,
        #     epoch,
        #     model_file,
        #     num_workers=20
        # )

    def test(self,
              dataset_path,
              model_path=None,
              data_type="ligand",
              # data_type="ligandmap",
              # model_type="squeeze+ligand",
              model_type="resnet+ligand",
              # model_type="resnet+ligandmap",
              # model_type="mobilenet+ligand",
              # model_key="squeeze_ligand_",
              model_key="resnet_ligand_masked_",
              # model_key="resnet_ligandmap_",
              # model_key="mobilenet_ligand_",

              options_json_path: str = "./options.json"):
        logger.info(f"Model type: {model_type}")
        logger.info(f"Model key: {model_key}")
        logger.info(f"Options path is: {options_json_path}")
        logger.info(f"Dataset path is: {dataset_path}")

        options = Options.load(options_json_path)

        # Make the dataset
        # dataset = PanDDAEventDataset.load(dataset_path)
        dataset = load_model(
            dataset_path,
            PanDDAEventDataset,
        )
        logger.info(f"Training on {len(dataset.pandda_events)} events!")

        if data_type == "ligand":
            dataset_torch = PanDDADatasetTorchLigand(
                dataset,
                transform_image=get_image_xmap_ligand_augmented,
                transform_annotation=get_annotation_from_event_hit
            )
        elif data_type == "ligandmap":
            dataset_torch = PanDDADatasetTorchLigandmap(
                dataset,
                transform_image=get_image_xmap_ligand_augmented,
                transform_annotation=get_annotation_from_event_hit,
                transform_ligandmap=get_image_ligandmap_augmented,
            )

        else:
            raise Exception

        # Get the output model file
        if model_path:
            model_path = Path(model_path)
            file_name = model_path.name
            match = re.match(
                model_key,
                file_name,
            )
            if match:
                re.match(
                    ".*([0-9]).pt",
                    file_name
                )
                epoch = int(match[1])
                model_file = model_path
            else:
                epoch = 0
                model_file = None
        else:
            epoch = 0
            model_file = None

        logger.info(f"Beggining from epoch: {epoch}")

        # Get the device
        if torch.cuda.is_available():
            logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            logger.info(f"Using cpu!")
            dev = "cpu"

        # Load the model
        if model_type == "squeeze+ligand":
            # model = squeezenet1_0(num_classes=2, num_features=4)
            model = squeezenet1_1(num_classes=2, num_features=4)

            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "resnet+ligand":
            model = resnet18(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "resnet+ligandmap":
            model = resnet18_ligandmap(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type== "mobilenet+ligand":
            model = mobilenet_v3_large_3d(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )

        else:
            raise Exception
        model = model.train()

        if data_type == "ligand":
            train(
            options,
            dataset_torch,
            model,
            epoch,
            model_key,
            dev,
            num_workers=20,
        )
        elif data_type == "ligandmap":
            train_ligandmap(
                options,
                dataset_torch,
                model,
                epoch,
                model_key,
                dev,
                num_workers=20,
            )

    def annotate_train_dataset_all(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset, annotations, updated_annotations, events = dataset_and_annotations_from_database(options)

        train_annotations_dir = Path(options.working_dir) / constants.PANDDA_TRAIN_ANNOTATION_DIR

        annotate_test_set(
            options,
            dataset,
            annotations,
            updated_annotations,
            train_annotations_dir,
            events,
        )

    def annotate_test_dataset_all(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset, annotations, updated_annotations, events = test_dataset_and_annotations_from_database(options)

        test_annotations_dir = Path(options.working_dir) / constants.PANDDA_TEST_ANNOTATION_DIR

        annotate_test_set(
            options,
            dataset,
            annotations,
            updated_annotations,
            test_annotations_dir,
            events
        )

    def annotate_dataset_ligand(self, dataset_path, model_file, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)

        engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

        with Session(engine) as session:
            logger.info(f"Loading events")
            events_stmt = select(EventORM).options(
                selectinload(EventORM.annotations),
                selectinload(EventORM.partitions),
                selectinload(EventORM.pandda).options(
                    selectinload(PanDDAORM.system),
                    selectinload(PanDDAORM.experiment),
                ),
            )
            events_list = session.scalars(events_stmt).unique().all()
            events = {event.id: event for event in events_list}

        dataset = load_model(dataset_path, PanDDAEventDataset)
        test_annotations_dir = Path(options.working_dir) / f"annotations_{Path(dataset_path).stem}"

        annotate_dataset_ligand(
            dataset,
            test_annotations_dir,
            model_file,
            events,
        )

    # def update_from_annotations_v2(self, options_json_path: str = "./options.json"):
    #     options = Options.load(options_json_path)
    #     engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")
    #
    #     with Session(engine) as session:
    #
    #         # Get the events
    #         events_stmt = select(EventORM).options(
    #             selectinload(EventORM.annotations)
    #         )
    #         logger.info(f"Loading events...")
    #         events = {event.id: event for event in session.scalars(events_stmt).unique().all()}
    #         logger.info(f"Loaded {len(events)} events")
    #
    #         # Update the high scoring non-hit annotations
    #         high_scoring_non_hit_pandda_path = Path(
    #             options.working_dir) / constants.PANDDA_TRAIN_ANNOTATION_DIR / constants.HIGH_SCORING_NON_HIT_DATASET_DIR
    #         high_scoring_non_hit_annotations = update_from_annotations_v2_get_annotations(
    #             events,
    #             high_scoring_non_hit_pandda_path
    #         )
    #         logger.info(f"Got {len(high_scoring_non_hit_annotations)} new high scoring non-hit annotations")
    #
    #         # Update the low scoring hit annotations
    #         low_scoring_hit_pandda_path = Path(
    #             options.working_dir) / constants.PANDDA_TRAIN_ANNOTATION_DIR / constants.LOW_SCORING_HIT_DATASET_DIR
    #         low_scoring_hit_annotations = update_from_annotations_v2_get_annotations(
    #             events,
    #             low_scoring_hit_pandda_path
    #         )
    #         logger.info(f"Got {len(low_scoring_hit_annotations)} new low scoring hit annotations")
    #
    #         # session.add_all(high_scoring_non_hit_annotations)
    #         # session.add_all(low_scoring_hit_annotations)
    #         session.commit()

    def update_test_from_annotations_v2(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

        with Session(engine) as session:

            # Get the events
            events_stmt = select(EventORM).options(
                selectinload(EventORM.annotations)
            )
            logger.info(f"Loading events...")
            events = {event.id: event for event in session.scalars(events_stmt).unique().all()}
            logger.info(f"Loaded {len(events)} events")

            # Update the high scoring non-hit annotations
            high_scoring_non_hit_pandda_path = Path(
                options.working_dir) / constants.PANDDA_TEST_ANNOTATION_DIR / constants.HIGH_SCORING_NON_HIT_DATASET_DIR
            high_scoring_non_hit_annotations = update_from_annotations_v2_get_annotations(
                events,
                high_scoring_non_hit_pandda_path
            )
            logger.info(f"Got {len(high_scoring_non_hit_annotations)} new high scoring non-hit annotations")

            # Update the low scoring hit annotations
            low_scoring_hit_pandda_path = Path(
                options.working_dir) / constants.PANDDA_TEST_ANNOTATION_DIR / constants.LOW_SCORING_HIT_DATASET_DIR
            low_scoring_hit_annotations = update_from_annotations_v2_get_annotations(
                events,
                low_scoring_hit_pandda_path
            )
            logger.info(f"Got {len(low_scoring_hit_annotations)} new low scoring hit annotations")

            # session.add_all(high_scoring_non_hit_annotations)
            # session.add_all(low_scoring_hit_annotations)
            session.commit()

    def update_from_annotations_dir(self, annotation_dir, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

        with Session(engine) as session:

            # Get the events
            events_stmt = select(EventORM).options(
                selectinload(EventORM.annotations)
            )
            logger.info(f"Loading events...")
            events = {event.id: event for event in session.scalars(events_stmt).unique().all()}
            logger.info(f"Loaded {len(events)} events")

            # Update the high scoring non-hit annotations
            high_scoring_non_hit_pandda_path = Path(annotation_dir) / constants.HIGH_SCORING_NON_HIT_DATASET_DIR
            high_scoring_non_hit_annotations = update_from_annotations_v2_get_annotations(
                events,
                high_scoring_non_hit_pandda_path
            )
            logger.info(f"Got {len(high_scoring_non_hit_annotations)} new high scoring non-hit annotations")

            # Update the low scoring hit annotations
            low_scoring_hit_pandda_path = Path(annotation_dir) / constants.LOW_SCORING_HIT_DATASET_DIR
            low_scoring_hit_annotations = update_from_annotations_v2_get_annotations(
                events,
                low_scoring_hit_pandda_path
            )
            logger.info(f"Got {len(low_scoring_hit_annotations)} new low scoring hit annotations")

            # session.add_all(high_scoring_non_hit_annotations)
            # session.add_all(low_scoring_hit_annotations)
            session.commit()

    def score_models_on_test_set(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset, annotations, updated_annotations, events = test_dataset_and_annotations_from_database(options)

        model_files = {}
        for model_file in Path(options.working_dir).glob("*"):
            file_name = model_file.name
            match = re.match(constants.MODEL_FILE_REGEX, file_name)
            if match:
                epoch = int(match[1])
                model_files[epoch] = model_file

        for epoch in sorted(model_files):
            model_file = model_files[epoch]
            logger.info(f"######## Testing model for epoch: {epoch} ########")
            records = annotate_dataset(options, dataset, annotations,updated_annotations, model_file)
            precission_recall(records)

    def score_models_on_finetune_sets(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        events = get_events(options)

        def filter_finetune_test(_event: EventORM):
            if _event.partitions:
                if _event.partitions.name == constants.FINETUNE_TEST_PARTITION:
                    return True

            return False

        def filter_finetune_train(_event: EventORM):
            if _event.partitions:
                if _event.partitions.name == constants.FINETUNE_TRAIN_PARTITION:
                    return True

            return False

        events_finetune_test = {
            event.id: event
            for event
            in filter(
                filter_finetune_test,
                events,
            )
        }
        logger.info(f"Got {len(events_finetune_test)} finetune test events")
        dataset_ftte, annotations_ftte, updated_annotations_ftte = get_dataset_annotations_from_events(events_finetune_test)

        events_finetune_train = {
            event.id: event
            for event
            in filter(
                filter_finetune_train,
                events,
            )
        }
        logger.info(f"Got {len(events_finetune_train)} finetune train events")
        dataset_fttr, annotations_fttr, updated_annotations_fttr = get_dataset_annotations_from_events(events_finetune_train)


        # dataset, annotations, updated_annotations, events = test_dataset_and_annotations_from_database(options)

        model_pr = {}
        for model_file in Path(options.working_dir).glob("*"):
            file_name = model_file.name
            match = re.match(constants.MODEL_FILE_REGEX, file_name)
            if match:
                epoch = match[1]
                logger.info(f"######## Testing model for epoch: {epoch} ########")

                records = annotate_dataset(options, dataset_ftte, annotations_ftte, updated_annotations_ftte, model_file)

                for cutoff, (precission, recall) in precission_recall(records).items():

                    model_pr[(epoch, cutoff)] = (precission, recall)

        # Filter by precission > 0.4
        def filter_precission(_key):
            if model_pr[_key][0] > 0.4:
                return True
            else:
                return False
        filtered_model_pr = {_key: model_pr[_key] for _key in filter(filter_precission, model_pr)}

        # Rank by highest recall at precission > 0.4
        for epoch, cutoff in sorted(filtered_model_pr, key=lambda _key: filtered_model_pr[_key][1]):
            precission, recall = filtered_model_pr[(epoch, cutoff)]
            logger.info(
                f"Epoch: {epoch} : Cutoff: {cutoff} : Precission : {precission} : Recall : {recall}"
            )

    def score_models_on_dataset(self, test_dataset_path, epoch=0, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)

        epoch = int(epoch)

        test_dataset = load_model(test_dataset_path, PanDDAEventDataset)
        print(f"Got dataset with: {len(test_dataset.pandda_events)} events!")

        # model_files = {}
        # for model_file in Path(options.working_dir).glob("*"):
        #     file_name = model_file.name
        #     match = re.match(constants.MODEL_FILE_REGEX_XMAP_MEAN, file_name)
        #     if match:
        #         epoch = int(match[1])
        #         model_files[epoch] = model_file

        model_file = (Path(options.working_dir) / constants.MODEL_FILE_EPOCH_XMAP_MEAN.format(epoch=epoch)).resolve()
        print(f"Initial model path is: {model_file}")
        model_pr = {}
        while model_file.exists():

            # model_file = model_files[epoch]

            # file_name = model_file.name
            # match = re.match(constants.MODEL_FILE_REGEX_XMAP_MEAN, file_name)
            # if match:
            #     epoch = match[1]
            logger.info(f"######## Testing model for epoch: {epoch} ########")

            records = get_annotations_from_dataset(
                test_dataset,
                model_file,
            )

            for cutoff, (precission, recall) in precission_recall(records).items():

                model_pr[(epoch, cutoff)] = (precission, recall)

            results_this_epoch = {_key[1]: model_pr[_key] for _key in model_pr if _key[0] == epoch}
            selected_key = min(
                results_this_epoch,
                key=lambda _key: float(np.abs(results_this_epoch[_key][1]-0.95)),
            )
            print(f"Epoch {epoch}: Precission at recall: {results_this_epoch[selected_key][1]} is: {results_this_epoch[selected_key][0]} at cutoff: {selected_key}")

            epoch += 1
            # epoch -= 10

            model_file = Path(options.working_dir) /constants.MODEL_FILE_EPOCH_XMAP_MEAN.format(epoch=epoch)

        # Filter by precission > 0.4
        def filter_precission(_key):
            if model_pr[_key][0] > 0.4:
                return True
            else:
                return False
        filtered_model_pr = {_key: model_pr[_key] for _key in filter(filter_precission, model_pr)}

        # Rank by highest recall at precission > 0.4
        for epoch, cutoff in sorted(filtered_model_pr, key=lambda _key: filtered_model_pr[_key][1]):
            precission, recall = filtered_model_pr[(epoch, cutoff)]
            logger.info(
                f"Epoch: {epoch} : Cutoff: {cutoff} : Precission : {precission} : Recall : {recall}"
            )


    def score_models_on_dataset_ligand(self, test_dataset_path, model_key="resnet_ligand_masked_", epoch=1, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)

        epoch = int(epoch)

        test_dataset = load_model(test_dataset_path, PanDDAEventDataset)
        print(f"Got dataset with: {len(test_dataset.pandda_events)} events!")

        # model_files = {}
        # for model_file in Path(options.working_dir).glob("*"):
        #     file_name = model_file.name
        #     match = re.match(constants.MODEL_FILE_REGEX_XMAP_MEAN, file_name)
        #     if match:
        #         epoch = int(match[1])
        #         model_files[epoch] = model_file

        # model_file = (Path(options.working_dir) / constants.MODEL_FILE_EPOCH_XMAP_LIGAND.format(epoch=epoch)).resolve()
        model_file = Path(options.working_dir) / f"{model_key}{epoch}.pt"

        print(f"Initial model path is: {model_file}")
        model_pr = {}
        while model_file.exists():

            # model_file = model_files[epoch]

            # file_name = model_file.name
            # match = re.match(constants.MODEL_FILE_REGEX_XMAP_MEAN, file_name)
            # if match:
            #     epoch = match[1]
            logger.info(f"######## Testing model for epoch: {epoch} ########")

            records = get_annotations_from_dataset_ligand(
                test_dataset,
                model_file,
            )

            for cutoff, (precission, recall) in precission_recall(records).items():

                model_pr[(epoch, cutoff)] = (precission, recall)

            results_this_epoch = {_key[1]: model_pr[_key] for _key in model_pr if _key[0] == epoch}
            selected_key = min(
                results_this_epoch,
                key=lambda _key: float(np.abs(results_this_epoch[_key][1]-0.95)),
            )
            selected_key_high_prec = min(
                results_this_epoch,
                key=lambda _key: float(np.abs(results_this_epoch[_key][0]-0.90)),
            )
            print(f"Epoch {epoch}: Precission at recall: {results_this_epoch[selected_key][1]} is: {results_this_epoch[selected_key][0]} at cutoff: {selected_key}")
            print(
                f"\tEpoch {epoch}: Recall at precission: {results_this_epoch[selected_key_high_prec][0]} is: {results_this_epoch[selected_key_high_prec][1]} at cutoff: {selected_key_high_prec}")
            # epoch -= 10
            epoch += 1
            # epoch += 2

            # model_file = Path(options.working_dir) /constants.MODEL_FILE_EPOCH_XMAP_LIGAND.format(epoch=epoch)
            model_file =Path(options.working_dir) / f"{model_key}{epoch}.pt"


        # Filter by precission > 0.4
        def filter_precission(_key):
            if model_pr[_key][0] > 0.4:
                return True
            else:
                return False
        filtered_model_pr = {_key: model_pr[_key] for _key in filter(filter_precission, model_pr)}

        # Rank by highest recall at precission > 0.4
        for epoch, cutoff in sorted(filtered_model_pr, key=lambda _key: filtered_model_pr[_key][1]):
            precission, recall = filtered_model_pr[(epoch, cutoff)]
            logger.info(
                f"Epoch: {epoch} : Cutoff: {cutoff} : Precission : {precission} : Recall : {recall}"
            )

    def score_against_historical_hits(self,
                                      test_partition_key="pandda_2_2023_06_27",
                                      model_key="resnet_ligand_masked_",
                                      options_json_path="./options.json"
                                      ):


        options = Options.load(options_json_path)
        # from edanalyzer.database_pony import *
        import pony

        db.bind(provider='sqlite', filename=f"{options.working_dir}/{constants.SQLITE_FILE}")
        db.generate_mapping()

        # engine = create_engine(f"sqlite:///{options.working_dir}/{constants.SQLITE_FILE}")

        # Get the models
        models = _get_models(model_key, Path(options.working_dir))
        print(f"Got {len(models)} models!")

        # Get the test events

        # Get the database session
        # with Session(engine) as session:
        # session = Session(engine)

        print(f"Getting events... (slow)")
        events_pickle = Path(options.working_dir) / "events.pickle"
        # if events_pickle.exists():
        #     with open(events_pickle, "rb") as f:
        #         events = pickle.load(f)
        # else:
        # events_stmt = select(EventORM).options(
        #     selectinload(EventORM.partitions),
        #     selectinload(EventORM.annotations),
        #     selectinload(EventORM.ligand),
        #     selectinload(EventORM.pandda).options(
        #         selectinload(PanDDAORM.system),
        #         selectinload(PanDDAORM.experiment)
        #
        #     )
        # )
        # events = session.scalars(events_stmt).unique().all()

            # with open(events_pickle, 'wb') as f:
            #     pickle.dump(events, f)

        with pony.orm.db_session:
            event_datas = pony.orm.select((event, event.partitions, event.annotations, event.ligand, event.pandda,
                                           event.pandda.system, event.pandda.experiment) for event in EventORM)[:]

            events = [event_data[0] for event_data in event_datas]
            print(f"Got {len(events)} total events from database")

            partitioned_events = [event for event in events if event.partitions]
            print(f"Got {len(partitioned_events)} total partitioned events from database")

            # for event in partitioned_events:
            #     print([x for x in event.partitions][0])
            #     print(event.partitions.name)

            test_systems = {
                event.pandda.system.name: event.pandda.system
                for event
                in partitioned_events
                if [x for x in event.partitions][0].name == constants.INITIAL_TEST_PARTITION
            }

            test_experiments = {
                system_name: [experiment for experiment in system.experiments]
                for system_name, system
                in test_systems.items()
            }
            print(f"Got {len(test_systems)} test systems")

            # Get the new PanDDA events in the partition
            # initial_test_events = [
            #     event
            #     for event
            #     in partitioned_events
            #     if (event.pandda.system.name in test_systems) and (event.partitions.name == test_partition_key)
            # ]
            #
            # # Assign them to their system
            # test_events = {}

            # Get the historical events
            initial_reference_events = [
                event
                for event
                in partitioned_events
                if (event.pandda.system.name in test_systems) and ([x for x in event.partitions][0].name == "test")
            ]

            # Match them to their system
            reference_events = {
                system_name: [event for event in initial_reference_events if event.pandda.system.name == system_name]
                for system_name
                in test_systems
            }

            panddas_list = pony.orm.select(pandda for pandda in PanDDAORM)[:]

            panddas = {pandda.path: pandda for pandda in panddas_list}

            print(f"For a total of {len(initial_reference_events)} reference events")

            records = []

            # Iterate the systems
            for system_name, system in test_systems.items():
                print(f"########## System: {system_name} ##########˚")

                # Get the corresponding reference events
                reference_system_events = reference_events[system_name]

                reference_hits = []
                for event in reference_system_events:
                    event_annotations = {annotation.source: annotation for annotation in event.annotations}

                    if "manual" in event_annotations:
                        annotation_orm = event_annotations["manual"]
                    else:
                        annotation_orm = event_annotations["auto"]

                    # Reference event is a hit: check for match
                    if annotation_orm.annotation:
                        reference_hits.append(event)

                hit_dtags = [event.dtag for event in reference_hits]

                print(f"Got {len(reference_system_events)} reference events, of which {len(reference_hits)} are hits!")

                for experiment in test_experiments[system_name]:
                    print(f"# Experiment: {experiment.path}")

                    pandda_path = str(Path(experiment.path) / "processing" / "analysis" / test_partition_key / "1")
                    print(f"Deleting previous pandda if in database...")
                    if pandda_path in panddas:
                        pandda =panddas[pandda_path]
                        pandda.delete()
                    # else:
                    pandda = PanDDAORM(
                        path=pandda_path,
                        events=[],
                        datasets=[],
                        system=system,
                        experiment=experiment,
                    )

                    print(f"Getting test events...")
                    experiment_events = _get_test_events(
                        experiment,
                        pandda,
                        test_partition_key,
                    )
                    if not experiment_events:
                        print(f"Got not find experiment events! Skipping")
                        continue
                    print(f"Got {len(experiment_events)} events from experiment PanDDA")

                    # Get the matched events
                    print(f"Matching events...")
                    matched_events = _get_matched_events(experiment_events, reference_system_events)
                    num_matched = len([event for event in matched_events if event.hit_confidence == "High"])

                    print(f"Got {len(matched_events)} events of which {num_matched} were matched")

                    if num_matched == 0:
                        print(f"Matched none! Skipping!")
                        continue
                    # Score the events against each model
                    model_scores = {}
                    for model_number, model in models.items():
                        # if model_number != 10:
                        #     continue
                        model_scores[model_number] = _get_model_scores(model, matched_events)

                    # Get the scoring statistics
                    scoring_statistics, pr_curve_tables = _get_scoring_statistics(model_scores)


                    # Render scoring statistics
                    _print_scoring_statistics(scoring_statistics)

                    # for model_number, model_statistics in scoring_statistics.items():
                    #     for recall, recall_statistics in model_statistics.items():
                    #         records.append(
                    #             {
                    #                 "System": str(system_name),
                    #                 "Experiment": str(experiment.path),
                    #                 "Model": model_number,
                    #                 "Recall": recall,
                    #                 "Precision": recall_statistics["precision"],
                    #                 "Cutoff": recall_statistics["cutoff"],
                    #                 "Actual Recall": recall_statistics["actual_recall"]
                    #             }
                    #         )
                    for model_number, pr_curve_table in pr_curve_tables.items():
                        for idx, row in pr_curve_table.iterrows():
                            records.append(
                                {
                                    "System": str(system_name),
                                    "Experiment": str(experiment.path),
                                    "Model": model_number,
                                    "Recall": row['recall'],
                                    "Precision": row["precision"],
                                    "Cutoff": row["cutoff"],
                                }
                            )

                    table = pd.DataFrame(records)
                    table.to_csv(Path(options.working_dir) / "model_statistics.csv")

            db.rollback()




def _get_models(_model_key, _working_dir):
    models = {}
    for path in _working_dir.glob("*"):
        match = re.match(f"{_model_key}([0-9]+).pt", path.name)
        if match is not None:
            # print(match)
            models[int(match.group(1))] = path

    return models


from edanalyzer.database_pony_utils import parse_analyse_table_row


def _get_test_events(
        experiment,
        pandda,
        test_partition_key,
):

    test_events = []
    # for system_name, experiments in test_experiments.items():
    #     test_events[system_name] = {}

    # for experiment in experiments:
    test_system_panddas_dir = Path(experiment.path) / "processing" / "analysis" / test_partition_key

    if not test_system_panddas_dir.exists():
        print(f"Could not find test system panddas dir: {test_system_panddas_dir}")

        return None

    test_system_pandda_dir = test_system_panddas_dir / "1"

    if not test_system_pandda_dir.exists():
        print(f"Could not find test system pandda dir: {test_system_pandda_dir}")
        return None

    analyses_dir = test_system_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    if not analyses_dir.exists():
        print(f"Could not find analysis dir: {analyses_dir}")

        return None

    analysis_table_path = analyses_dir / constants.PANDDA_EVENT_TABLE_PATH
    if not analysis_table_path.exists():
        print(f"Could not find analysis table: {analysis_table_path}")
        return None

    analysis_table = pd.read_csv(analysis_table_path)
    # test_events[system_name][experiment.path] = []

    for idx, row in analysis_table.iterrows():
        # if row[constants.PANDDA_INSPECT_DTAG] not in hit_dtags:
        #     continue
        event = parse_analyse_table_row(
            row,
            None,
            test_system_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR,
            None,
            pandda,
            annotation_type="auto",
        )

        #
        if event:
            test_events.append(event)

    return test_events


def _get_matched_events(_test_system_events, _reference_system_events):
    matched_events = []

    for test_event in _test_system_events:
        matched = False
        if test_event.ligand:
            x, y, z = test_event.ligand.x, test_event.ligand.y, test_event.ligand.z
        else:
            x, y, z = test_event.x, test_event.y, test_event.z

        for reference_event in _reference_system_events:
            if reference_event.dtag != test_event.dtag:
                continue

            event_annotations = {annotation.source: annotation for annotation in reference_event.annotations}

            if "manual" in event_annotations:
                annotation_orm = event_annotations["manual"]
            else:
                annotation_orm = event_annotations["auto"]

            # Reference event is a hit: check for match
            if annotation_orm.annotation:
                rx, ry, rz = reference_event.x, reference_event.y, reference_event.z

                distance = np.linalg.norm(np.array([x - rx, y - ry, z - rz]))
                # Successful match
                if distance < 3.0:
                    test_event.hit_confidence = "High"
                    matched = True
                    break

        if not matched:
            test_event.hit_confidence = "Low"

        matched_events.append(test_event)

    return matched_events

    ...


def _get_model_scores(_model, _matched_events):
    events_pyd = []
    for j, event in enumerate(_matched_events):
        if event.ligand:
            ligand = event.ligand
            x, y, z = ligand.x, ligand.y, ligand.z
        else:
            x, y, z = event.x, event.y, event.z

        if event.hit_confidence == "High":
            hit = True
        elif event.hit_confidence == "Low":
            hit = False
        else:
            raise Exception

        event_pyd = PanDDAEvent(
                # id=event.id,
                id=j,
                pandda_dir=event.pandda.path,
                model_building_dir=event.pandda.experiment.model_dir,
                system_name=event.pandda.system.name,
                dtag=event.dtag,
                event_idx=event.event_idx,
                event_map=event.event_map,
                x=x,
                y=y,
                z=z,
                hit=hit,
                ligand=None,
            )
        events_pyd.append(event_pyd)
    dataset = PanDDAEventDataset(pandda_events=events_pyd)
    records = get_annotations_from_dataset_ligand(dataset, _model)
    # for record in records:
    #     print(record)

    return records


def _get_scoring_statistics(_model_scores, recalls=[0.95, 0.975, 0.99, 1.0]):

    scoring_statistics = {}
    pr_curve_tables = {}
    for model_number, model_scores in _model_scores.items():
        scoring_statistics[model_number] = {}
        cutoff_precission_recall = []
        for cutoff in np.linspace(0.0,1.0,num=101):
            cutoff = round(cutoff, 2)
            tp, tn, fp, fn = 0,0,0,0
            for record_idx, record in model_scores.items():
                if record["model_annotation"] > cutoff:
                    if record["annotation"] == 0.0:
                        fp += 1
                    elif record["annotation"] == 1.0:
                        tp += 1
                    else:
                        raise Exception
                else:
                    if record["annotation"] == 0.0:
                        tn += 1
                    elif record["annotation"] == 1.0:
                        fn += 1
                    else:
                        raise Exception

            p = tp + fn
            n = tn + fp
            if p != 0:
                recall = tp / p
            else:
                recall = 0

            ap = tp+fp
            an = tn+fn
            if ap != 0:
                prec = tp/ap
            else:
                prec = 0

            cutoff_precission_recall.append(
                {
                    "cutoff": cutoff,
                    "precision": prec,
                    "recall": recall
                }
            )


        cutoff_precission_recall_table = pd.DataFrame(cutoff_precission_recall)
        pr_curve_tables[model_number] = cutoff_precission_recall_table
        # print(cutoff_precission_recall_table)
        for recall in recalls:
            delta_recall_series = (cutoff_precission_recall_table["recall"] - recall).abs()
            highest_recall_table = cutoff_precission_recall_table[delta_recall_series == delta_recall_series.min()]
            highest_recall_row = highest_recall_table[highest_recall_table["cutoff"] == highest_recall_table["cutoff"].max()].iloc[0]
            # print(highest_recall_row)
            precision = round(highest_recall_row["precision"], 3)
            cutoff = round(highest_recall_row["cutoff"], 3)
            observed_recall = round(highest_recall_row["recall"], 3)
            scoring_statistics[model_number][recall] = {"cutoff": cutoff, "precision": precision, "actual_recall": observed_recall}

    return scoring_statistics, pr_curve_tables

    ...


def _print_scoring_statistics(model_scoring_statistics):
    for model_number, scoring_statistics in model_scoring_statistics.items():
        for recall, statistics in scoring_statistics.items():
            precision, cutoff, actual_recall = statistics['precision'], statistics['cutoff'], statistics['actual_recall']
            print(f"\tModel Number: {model_number}: Recall: {recall} : Actual Recall: {actual_recall} :  Precision : {precision} : Cutoff : {cutoff}")

def update_from_annotations_v2_get_annotations(
    events: dict[int, EventORM],
    pandda_path: Path,
):

    # Get the table path
    table_path = pandda_path / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE

    # Load the table
    table = pd.read_csv(table_path)
    logger.info(f"Event table has {len(table)} rows")

    # Iterate the table making new annotations
    annotations = []
    for idx, row in table.iterrows():
        # Unpack the row
        dtag = row[constants.PANDDA_INSPECT_DTAG]
        confidence = row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE]
        viewed = row[constants.PANDDA_INSPECT_VIEWED]
        bdc = row[constants.PANDDA_INSPECT_BDC]

        # Get the event
        event = events[int(dtag)]

        # Skip if not viewed
        if not viewed == True:
            # logger.debug(f"Event {dtag} not viewed, skipping!")
            continue

        # Check event map path matches
        observed_event_map_path = pandda_path / constants.PANDDA_PROCESSED_DATASETS_DIR / str(dtag) / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
            dtag=str(dtag),
            event_idx=str(1),
            bdc=str(round(bdc, 2)),
        )
        assert observed_event_map_path.exists(), f"{observed_event_map_path} vs {event.event_map}"
        assert str(observed_event_map_path.readlink()) == event.event_map, f"{observed_event_map_path} vs {event.event_map}"

        # Check does not already have a manual annotation
        annotations = {annotation.source: annotation for annotation in event.annotations}
        if "manual" in [annotation.source for annotation in event.annotations]:
            logger.warning(f"Event {dtag} already has manual in its {len(event.annotations)} annotations! Skipping!")
            # continue

        annotation = annotations["manual"]
        # annotation.source = "deprecated"

        # Get the newly assigned annotation
        if confidence == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
            # annotation = AnnotationORM(
            #     annotation=True,
            #     source="manual",
            #     event=event,
            # )
            print(f"Setting annotation to True!")
            annotation.annotation = True

        elif confidence == constants.PANDDA_INSPECT_TABLE_LOW_CONFIDENCE:
            # annotation = AnnotationORM(
            #     annotation=False,
            #     source="manual",
            #     event=event,
            # )
            annotation.annotation = False


        else:
            raise Exception(f"Failed to parse annotation label in table! Confidence was: {confidence}")

        # Append
        # annotations.append(annotation)

    return annotations


if __name__ == "__main__":
    fire.Fire(CLI)
