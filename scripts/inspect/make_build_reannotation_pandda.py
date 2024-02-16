import dataclasses
import os
import pickle
import pathlib
import shutil
import subprocess
import time
from pathlib import Path
import re

import yaml
import fire
import pony
import rich
from rich import print as rprint
from rich.panel import Panel
from rich.align import Align
from rich.padding import Padding
import pandas as pd
import joblib
import gemmi
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# from edanalyzer.torch_network import resnet18
from edanalyzer.torch_network_resnet import resnet18
from edanalyzer.torch_network_resnet_ligandmap import resnet18_ligandmap
from edanalyzer.torch_network_squeezenet import squeezenet1_1, squeezenet1_0
from edanalyzer.torch_network_mobilenet import mobilenet_v3_large_3d

from edanalyzer.database_pony import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM  # import *
from edanalyzer import constants
from edanalyzer.cli_dep import train
from edanalyzer.data import PanDDAEvent, PanDDAEventDataset
from edanalyzer.torch_dataset import (
    PanDDAEventDatasetTorch, PanDDADatasetTorchXmapGroundState, get_annotation_from_event_annotation,
    get_image_event_map_and_raw_from_event, get_image_event_map_and_raw_from_event_augmented,
    get_annotation_from_event_hit, get_image_xmap_mean_map_augmented, get_image_xmap_mean_map,
    get_image_xmap_ligand_augmented, PanDDADatasetTorchLigand, get_image_xmap_ligand, get_image_ligandmap_augmented,
    PanDDADatasetTorchLigandmap, get_image_event_map_ligand, get_image_event_map_ligand_augmented
)


def _get_model(closest_pose):
    st = gemmi.Structure()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()

    for pose_row, element in zip(closest_pose['positions'], closet_pose['elements']):


def try_make(path):
    try:
        os.mkdir(path)
    except Exception as e:
        return


def try_link(source_path, target_path):
    try:
        os.symlink(Path(source_path).resolve(), target_path)
    except Exception as e:
        # print(e)
        return

def _make_test_dataset_psuedo_pandda(
        config_path
):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'rb') as f:
        custom_annotations = pickle.load(f)

    # Open a file in "w"rite mode
    fileh = tables.open_file("output/build_data.h5", mode="r")

    # Get the HDF5 root group
    root = fileh.root
    table_mtz_sample = root.mtz_sample
    table_event_map_sample = root.event_map_sample
    table_known_hit_pos_sample = root.known_hit_pose

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Make the directories
    psuedo_pandda_dir = working_dir / "test_datasets_pandda"
    analyses_dir = psuedo_pandda_dir / "analyses"
    processed_datasets_dir = psuedo_pandda_dir / "processed_datasets"
    analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
    inspect_table_path = analyses_dir / "pandda_inspect_events.csv"
    analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"
    inspect_site_table_path = analyses_dir / "pandda_inspect_sites.csv"

    # Spoof the main directories
    try_make(psuedo_pandda_dir)
    try_make(analyses_dir)
    try_make(processed_datasets_dir)

    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = [_x for _x in pony.orm.select(_event for _event in EventORM)]

        # Iterate over the event maps
        for event_map_sample in table_event_map_sample.iterrows():
            # Get the corresponding poses
            event_map_sample_idx = event_map_sample['idx']
            poses = [x for x in table_known_hit_pos_sample.where(f'event_map_sample_idx == {event_map_sample_idx}')]

            # Get the closest pose
            closest_pose = min(poses, key=lambda _x: _x['rmsd'])

            # Get the corresponding event
            event = query[closest_pose['database_event_idx']]
            event_id = event.id

            # Make the dataset dir
            try_make(processed_datasets_dir / f'{event_id}')

            # Get the model
            model = _get_model(closest_pose)

            # Write the model

            # Get the event map

            # Write the event map



        # # Select events and Organise by dtag
        # events = {}
        # for res in query:
        #     if res[2].name == test_partition:
        #         event = res[0]
        #         annotation = res[1]
        #         dtag = event.dtag
        #         event_idx = event.event_idx
        #         if dtag not in events:
        #             events[dtag] = {}
        #
        #         table = inspect_tables[event.pandda.path]
        #         row = table[
        #             (table[constants.PANDDA_INSPECT_DTAG] == dtag)
        #             & (table[constants.PANDDA_INSPECT_EVENT_IDX] == event_idx)
        #             ]
        #
        #         events[dtag][event_idx] = {
        #             'event': event,
        #             'annotation': annotation,
        #             'row': row
        #         }
        # rprint(f"Got {len(events)} events!")



        # Spoof the dataset directories
        for dtag in events:
            dtag_dir = processed_datasets_dir / dtag
            try_make(dtag_dir)
            modelled_structures_dir = dtag_dir / "modelled_structures"
            try_make(modelled_structures_dir)

            for event_idx, event_info in events[dtag].items():
                event, annotation = event_info['event'], event_info['annotation']
                try_link(event.initial_structure, dtag_dir / Path(event.initial_structure).name)
                try_link(event.initial_reflections, dtag_dir / Path(event.initial_reflections).name)
                try_link(event.event_map, dtag_dir / Path(event.event_map).name)
                if event.structure:
                    try_link(event.structure, modelled_structures_dir / Path(event.structure).name)

        # Spoof the event table
        rows = []
        j = 0
        for dtag in events:
            for event_idx, event_info in events[dtag].items():
                row = event_info['row']
                # row.loc[0, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1
                rows.append(row)
                # j = j + 1

        event_table = pd.concat(rows).reset_index()
        for j in range(len(event_table)):
            event_table.loc[j, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1

        event_table.drop(["index", "Unnamed: 0"], axis=1, inplace=True)
        event_table.to_csv(analyse_table_path, index=False)
        event_table.to_csv(inspect_table_path, index=False)

        # Spoof the site table
        site_records = []
        num_sites = ((j) // 100) + 1
        print(f"Num sites is: {num_sites}")
        for site_id in np.arange(0, num_sites + 1):
            site_records.append(
                {
                    "site_idx": int(site_id) + 1,
                    "centroid": (0.0, 0.0, 0.0),
                    "Name": None,
                    "Comment": None
                }
            )
        print(len(site_records))
        site_table = pd.DataFrame(site_records)
        site_table.to_csv(analyse_site_table_path, index=False)
        site_table.to_csv(inspect_site_table_path, index=False)


# def _make_psuedo_pandda(psuedo_pandda_dir, events, rows, annotations):
#     # psuedo_pandda_dir = working_dir / "test_datasets_pandda"
#     analyses_dir = psuedo_pandda_dir / "analyses"
#     processed_datasets_dir = psuedo_pandda_dir / "processed_datasets"
#     analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
#     inspect_table_path = analyses_dir / "pandda_inspect_events.csv"
#     analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"
#     inspect_site_table_path = analyses_dir / "pandda_inspect_sites.csv"
#
#     # Spoof the main directories
#     try_make(psuedo_pandda_dir)
#     try_make(analyses_dir)
#     try_make(processed_datasets_dir)
#
#     # Spoof the dataset directories
#     _j = 0
#     for event, row in zip(events, rows):
#         dtag_dir = processed_datasets_dir / str(_j)
#         try_make(dtag_dir)
#         modelled_structures_dir = dtag_dir / "modelled_structures"
#         try_make(modelled_structures_dir)
#
#         # event, annotation = event_info['event'], event_info['annotation']
#         try_link(
#             event.initial_structure,
#             dtag_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=_j),
#         )
#         try_link(
#             event.initial_reflections,
#             dtag_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=_j),
#         )
#         try_link(
#             event.event_map,
#             dtag_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
#                 dtag=_j,
#                 event_idx=1,
#                 bdc=row.iloc[0]["1-BDC"],
#             ),
#         )
#         ligand_files_dir = dtag_dir / "ligand_files"
#
#         try_make(ligand_files_dir)
#         original_compound_dir = Path(event.event_map).parent / "ligand_files"
#         if original_compound_dir.exists():
#             for path in original_compound_dir.glob("*"):
#                 if path.stem not in constants.LIGAND_IGNORE_REGEXES:
#                     try_link(
#                         path,
#                         ligand_files_dir / path.name
#                     )
#         if event.structure:
#             try_link(
#                 event.structure,
#                 modelled_structures_dir / constants.PANDDA_MODEL_FILE.format(dtag=_j),
#             )
#         _j += 1
#
#     # Spoof the event table, changing the site, dtag and eventidx
#     # rows = []
#     # j = 0
#     # for dtag in events:
#     #     for event_idx, event_info in events[dtag].items():
#     #         row = event_info['row']
#     #         # row.loc[0, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1
#     #         rows.append(row)
#     # j = j + 1
#
#     event_table = pd.concat(rows).reset_index()
#
#     rprint(event_table)
#     rprint(len(annotations))
#     rprint(len(event_table))
#     rprint(len(rows))
#     for _j in range(len(event_table)):
#         assert events[_j].dtag == event_table.loc[_j, constants.PANDDA_INSPECT_DTAG]
#         event_table.loc[_j, constants.PANDDA_INSPECT_DTAG] = str(_j)
#         event_table.loc[_j, constants.PANDDA_INSPECT_EVENT_IDX] = 1
#         event_table.loc[_j, constants.PANDDA_INSPECT_SITE_IDX] = (_j // 100) + 1
#         event_table.loc[_j, constants.PANDDA_INSPECT_Z_PEAK] = annotations[_j]
#         event_table.loc[_j, constants.PANDDA_INSPECT_VIEWED] = False
#
#     rprint(event_table['z_peak'])
#
#     if "index" in event_table.columns:
#         event_table.drop(["index", ], axis=1, inplace=True)
#     if "Unnamed: 0" in event_table.columns:
#         event_table.drop(["Unnamed: 0"], axis=1, inplace=True)
#
#     event_table.to_csv(analyse_table_path, index=False)
#     event_table.to_csv(inspect_table_path, index=False)
#
#     # Spoof the site table
#     site_records = []
#     num_sites = ((_j) // 100) + 1
#     print(f"Num sites is: {num_sites}")
#     for site_id in np.arange(0, num_sites + 1):
#         site_records.append(
#             {
#                 "site_idx": int(site_id) + 1,
#                 "centroid": (0.0, 0.0, 0.0),
#                 "Name": None,
#                 "Comment": None
#             }
#         )
#     print(len(site_records))
#     site_table = pd.DataFrame(site_records)
#     site_table.to_csv(analyse_site_table_path, index=False)
#     site_table.to_csv(inspect_site_table_path, index=False)


if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)