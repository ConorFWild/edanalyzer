import dataclasses
from pathlib import Path

import yaml
import pandas as pd

import torch
from torch.utils.data import DataLoader

from edanalyzer import constants
from edanalyzer.data import PanDDAEvent, PanDDAEventDataset



@dataclasses.dataclass
class RescoreOptions:
    pandda_dir: Path
    data_dir: Path
    model_type: str
    model_file: Path
    data_type: str


def _parse_rescore_options(yaml_path):
    with open(yaml_path, "r") as f:
        dic = yaml.safe_load(f)

    return RescoreOptions(
        Path(dic["pandda_dir"]),
        Path(dic["data_dir"]),
        dic["model_type"],
        Path(dic["model_file"]),
        dic["data_type"]
    )


def _pandda_dir_to_dataset(pandda_dir: Path, data_dir: Path):
    analyses_dir = pandda_dir / constants.PANDDA_ANALYSIS_DIR

    processed_datasets_dir = pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR

    pandda_analyse_events_file = analyses_dir / constants.PANDDA_EVENT_TABLE_PATH

    pandda_event_table = pd.read_csv(pandda_analyse_events_file)

    events_pyd = []
    scores = {}
    for _idx, _row in pandda_event_table.iterrows():
        dtag = _row["dtag"]
        event_idx = int(_row["event_idx"])
        bdc = _row["1-BDC"]
        score = _row["z_peak"]
        scores[(dtag, event_idx)] = score

        event_map_path = constants.PANDDA_EVENT_MAP_TEMPLATE.format(
            dtag=dtag,
            event_idx=event_idx,
            bdc=bdc
        )

        dataset_dir = processed_datasets_dir / dtag

        x, y, z = _row["x"], _row["y"], _row["z"]

        event_pyd = PanDDAEvent(
            id=0,
            pandda_dir=str(pandda_dir),
            model_building_dir=str(data_dir),
            system_name="SYSTEM",
            dtag=dtag,
            event_idx=event_idx,
            event_map=str(event_map_path),
            x=x,
            y=y,
            z=z,
            hit=False,
            ligand=None,
        )
        events_pyd.append(event_pyd)

    return PanDDAEventDataset(pandda_events=events_pyd), scores


def _rescore(dataset, dataset_torch, model, dev, initial_scores):
    # Get the dataloader
    train_dataloader = DataLoader(
        dataset_torch,
    )

    new_scores = {}
    for image, annotation, _idx in train_dataloader:

        image_c = image.to(dev)
        # annotation_c = annotation.to(dev)

        # forward
        model_annotation = model(image_c)

        # Get corresponding event
        event = dataset.pandda_events[_idx]

        event_id = (event.dtag, event.event_idx)
        new_scores[event_id] = model_annotation.to(torch.device("cpu")).detach().numpy()[1]

        print(f"{event_id[0]} {event_id[1]} : Old Score: {initial_scores[event_id]} : New Score: {new_scores[event_id]}")
