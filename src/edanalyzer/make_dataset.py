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
    PanDDAEventKey, load_model, save_model
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
import download_dataset
import dataclasses
import time

import yaml

@dataclasses.dataclass
class MakeDatasetOptions:
    working_dir: Path
    out_path: Path
    partitions: list[str]


def _parse_make_dataset_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        dic = yaml.safe_load(f)

    return MakeDatasetOptions(
        Path(dic["working_dir"]),
        Path(dic["out_path"]),
        dic["partitions"]
    )


def _check_accessible(
        dtag,
        pandda_path,
        annotations,
):
    processed_dataset_path = Path(pandda_path) / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag

    ground_state_structure_path = processed_dataset_path / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
        dtag=dtag)

    mtz_path = processed_dataset_path / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag)

    ground_state_map_path = processed_dataset_path / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(dtag=dtag)

    bound_state_structure_path = processed_dataset_path / constants.PANDDA_INSPECT_MODEL_DIR / constants.PANDDA_MODEL_FILE.format(
        dtag=dtag)

    # annotations = {annotation.source: annotation for annotation in event.annotations}
    if "manual" in annotations:
        annotation = annotations["manual"]
    else:
        annotation = annotations["auto"]

    # if ground_state_structure_path.exists() & ground_state_map_path.exists() & mtz_path.exists():
    try:
        gemmi.read_structure(str(ground_state_structure_path))
        gemmi.read_mtz_file(str(mtz_path))
        gemmi.read_ccp4_map(str(ground_state_map_path))
        return True

    except Exception as e:
        print(e)
        return False


def _is_system_event(event: EventORM, systems):
    if event.pandda.system.name in systems:
        return True
    else:
        return False


def _has_parsable_pdb(event: EventORM):
    dataset_dir = Path(event.pandda.experiment.model_dir) / event.dtag / "compound"

    event_added = False
    # ligand_files_dir = processed_dataset_dir / "ligand_files"
    if dataset_dir.exists():
        ligand_pdbs = [
            x
            for x
            in dataset_dir.glob("*.pdb")
            if (x.exists()) and (x.stem not in constants.LIGAND_IGNORE_REGEXES)
        ]
        if len(ligand_pdbs) > 0:
            return True

    return False


def _get_hits_and_non_hits(parsable_pdb_events: list[EventORM]):
    # Get the hit and non-hit events
    hits = []
    non_hits = []
    for event in parsable_pdb_events:

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

    return hits, non_hits


def _event_to_event_pyd(event: EventORM):
    annotations = {annotation.source: annotation for annotation in event.annotations}

    if "manual" in annotations:
        annotation = annotations["manual"]
    else:
        annotation = annotations["auto"]

    if event.partitions.name == "pandda_2_2023_04_28":
        if annotation.annotation:
            return None
        else:
            x, y, z = event.x, event.y, event.z

    elif event.partitions.name == "pandda_2_2023_06_27":
        if (event.ligand is not None) & (event.hit_confidence == "High"):
            x, y, z = event.ligand.x, event.ligand.y, event.ligand.z
            # num_ligand_centroids += 1
        else:
            x, y, z = event.x, event.y, event.z

    elif event.partitions.name == "train":
        x, y, z = event.x, event.y, event.z

    else:
        return None

    return PanDDAEvent(
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


def _make_dataset(events: list[EventORM], options: MakeDatasetOptions):
    # Get the events which have partitions
    partitioned_events = [event for event in events if event.partitions]
    partition_events = [event for event in partitioned_events if event.partitions.name in options.partitions]

    # Of these get those events have accessible data: the "complete" events
    with Parallel(n_jobs=-2, verbose=10) as parallel:
        possible_events_mask = parallel(
            delayed(_check_accessible)(
                _event.dtag,
                _event.pandda.path,
                {annotation.source: annotation for annotation in _event.annotations}
            )
            for _event
            in partition_events
        )

    complete_events = [_event for _event, _event_mask in zip(partition_events, possible_events_mask) if _event_mask]
    print(f"Number of compeleye events: {len(complete_events)}")

    # Filter the events by system
    train_systems = {
        event.pandda.system.name: event.pandda.system
        for event
        in complete_events
        if event.partitions.name == constants.INITIAL_TRAIN_PARTITION
    }
    print(f"Number of train systems: {len(train_systems)}")

    system_events = [_event for _event in complete_events if _is_system_event(_event, train_systems)]

    # Filter the events by having a parsable ligand pdb
    parsable_pdb_events = [_event for _event in system_events if _has_parsable_pdb(_event)]

    # Split into hits and non-hits, using the ligand if present
    hits, non_hits = _get_hits_and_non_hits(parsable_pdb_events)

    # Balance the dataset by repeating hits
    num_hits = len(hits)
    num_non_hits = len(non_hits)
    repeated_hits = (hits * int(num_non_hits / num_hits)) + hits[: num_non_hits % num_hits]

    # Make the dataset
    train_events_pyd = []
    num_ligand_centroids = 0
    for event in repeated_hits + non_hits:

        event_pyd = _event_to_event_pyd(event)
        if event_pyd:
            train_events_pyd.append(event_pyd)

    # print(len(train_events_pyd))
    # print(num_ligand_centroids)
    print(f"Number of events to train on: {len(train_events_pyd)}")
    print(f"Number of events with updated centroid: {num_ligand_centroids}")

    # Output the dataset
    dataset = PanDDAEventDataset(pandda_events=train_events_pyd)
    save_model(options.out_path, dataset, )
    # train_dataset.save(path=Path("."), name=f"train_dataset_{DATASET_ID}.json")
