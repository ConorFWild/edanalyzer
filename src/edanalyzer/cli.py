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
    PanDDAEventKey, load_model
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

from edanalyzer.make_dataset import _make_dataset, _parse_make_dataset_yaml
from edanalyzer.rescore import _rescore, _parse_rescore_options, _pandda_dir_to_dataset

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

import pony


class CLI:
    def make_dataset(
            self,
            # partitions=("pandda_2_2023_06_27",),
            options_yaml: str = "./options.yaml",
    ):
        # options = Options.load(options_json_path)
        options = _parse_make_dataset_yaml(options_yaml)

        # get the events
        db.bind(provider='sqlite', filename=f"{options.working_dir}/{constants.SQLITE_FILE}")
        db.generate_mapping()

        with pony.orm.db_session:
            events = pony.orm.select((event, event.partitions, event.annotations, event.ligand, event.pandda,
                                      event.pandda.system, event.pandda.experiment) for event in EventORM)[:]
            print(f"Number of events: {len(events)}")

            # Get the events with partitions
            _make_dataset(events, options)

    def rescore(
            self,
            # pandda_dir,
            # data_type="ligand",
            # model_type="resnet+ligand",
            # model_file="resnet_ligand_masked_",
            rescore_options_yaml
    ):
        options = _parse_rescore_options(rescore_options_yaml)

        # Transform PanDDA dir to dataset
        dataset, initial_scores = _pandda_dir_to_dataset(options.pandda_dir, options.data_dir)

        if options.data_type == "ligand":
            dataset_torch = PanDDADatasetTorchLigand(
                dataset,
                transform_image=get_image_xmap_ligand,
                transform_annotation=get_annotation_from_event_hit
            )
        elif options.data_type == "ligandmap":
            dataset_torch = PanDDADatasetTorchLigandmap(
                dataset,
                transform_image=get_image_xmap_ligand_augmented,
                transform_annotation=get_annotation_from_event_hit,
                transform_ligandmap=get_image_ligandmap_augmented,
            )

        else:
            raise Exception

        # Get the device
        if torch.cuda.is_available():
            logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            logger.info(f"Using cpu!")
            dev = "cpu"

        # Load the model
        if options.model_type == "squeeze+ligand":
            # model = squeezenet1_0(num_classes=2, num_features=4)
            model = squeezenet1_1(num_classes=2, num_features=4)

            model.to(dev)

            if options.model_file:
                model.load_state_dict(torch.load(options.model_file, map_location=dev),
                                      )
        elif options.model_type == "resnet+ligand":
            model = resnet18(num_classes=2, num_input=4)
            model.to(dev)

            if options.model_file:
                model.load_state_dict(torch.load(options.model_file, map_location=dev),
                                      )
        elif options.model_type == "resnet+ligandmap":
            model = resnet18_ligandmap(num_classes=2, num_input=4)
            model.to(dev)

            if options.model_file:
                model.load_state_dict(torch.load(options.model_file, map_location=dev),
                                      )
        elif options.model_type == "mobilenet+ligand":
            model = mobilenet_v3_large_3d(num_classes=2, num_input=4)
            model.to(dev)

            if options.model_file:
                model.load_state_dict(torch.load(options.model_file, map_location=dev),
                                      )

        else:
            raise Exception
        model = model.eval()

        scores: dict[tuple[str, int], float] = _rescore(
            dataset, dataset_torch, model, dev
        )

        for event_id, new_score in scores.items():
            print(f"{event_id[0]} {event_id[1]} : Old Score: {initial_scores[event_id]} : New Score: {new_score}")

        ...

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

              options_json_path: str = "./options.json",
              ):
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
        elif model_type == "resnet+ligand":
            model = resnet18(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type == "resnet+ligandmap":
            model = resnet18_ligandmap(num_classes=2, num_input=4)
            model.to(dev)

            if model_file:
                model.load_state_dict(torch.load(model_file, map_location=dev),
                                      )
        elif model_type == "mobilenet+ligand":
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

    def calibrate(self):
        ...


if __name__ == "__main__":
    fire.Fire(CLI)
