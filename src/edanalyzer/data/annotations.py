import re

import pandas as pd
import pony

from edanalyzer import constants
from edanalyzer.database_pony import db, EventORM


def _get_custom_annotations_from_database(path):
    # get the events
    try:
        db.bind(provider='sqlite', filename=f"{path}")
        db.generate_mapping()
    except Exception as e:
        print(e)

    with pony.orm.db_session:
        events = pony.orm.select((event, event.partitions, event.annotations, event.ligand, event.pandda,
                                  event.pandda.system, event.pandda.experiment) for event in EventORM)[:]

        custom_annotations = {}
        for event_info in events:
            event = event_info[0]
            if event.pandda:
                event_id = (
                    str(event.pandda.path),
                    str(event.dtag),
                    int(event.event_idx)
                )
                if event.annotations:
                    annotations = {_a.source: _a.annotation for _a in event.annotations}

                    if "manual" in annotations:
                        annotation = annotations["manual"]
                        # else:
                        #     annotation = annotations["auto"]
                        custom_annotations[event_id] = annotation

    db.disconnect()
    return custom_annotations


def _get_custom_annotations_from_pandda(path):
    analyses_dir = path / constants.PANDDA_ANALYSIS_DIR
    processed_datasets = path / constants.PANDDA_PROCESSED_DATASETS_DIR
    pandda_inspect_path = analyses_dir / constants.PANDDA_INSPECT_TABLE_FILE
    custom_annotations = {}
    if pandda_inspect_path.exists():
        inspect_table = pd.read_csv(pandda_inspect_path)

        for idx, row in inspect_table.iterrows():
            viewed = row[constants.PANDDA_INSPECT_VIEWED]
            if viewed:
                confidence = row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE]
                if confidence not in ["low", "Low"]:
                    annotation = True
                else:
                    annotation = False
                event_identifier = row[constants.PANDDA_INSPECT_DTAG]
                dataset_dir = processed_datasets / str(event_identifier)
                event_map_path = [x for x in dataset_dir.glob("*.ccp4")][0]
                real_event_map_path = event_map_path.resolve()
                match = re.match("(.+)-event_([0-9]+)_1-BDC_[.0-9]+_", real_event_map_path.name)
                dtag, event_idx = match.groups()
                pandda_path = real_event_map_path.parent.parent.parent
                custom_annotations[(str(pandda_path), dtag, event_idx)] = annotation

    db.disconnect()
    return custom_annotations
