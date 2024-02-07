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
import lightning as ln

class LitResnet(ln.LightningModule):
