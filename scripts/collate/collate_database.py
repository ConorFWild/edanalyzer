from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import joblib
import pickle

from edanalyzer import constants
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag
from edanalyzer.data.daabase_pony import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'r') as f:
        custom_annotations = pickle.load(f)

    #
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Get the possible pandda paths
    possible_pandda_paths = [
        path
        for dataset_pattern
        in config['datasets']
        for path
        in Path('/').glob(dataset_pattern[1:])
        if not any([path.match(exclude_pattern) for exclude_pattern in config['exclude']])

    ]
    rprint(f"Got {len(possible_pandda_paths)} pandda paths!")
    rprint(possible_pandda_paths)

    # Get the pandda event tables
    inspect_tables = {}
    for possible_pandda_path in possible_pandda_paths:
        analyse_table_path = possible_pandda_path / "analyses" / "pandda_inspect_events.csv"
        if analyse_table_path.exists():
            try:
                analyse_table = pd.read_csv(analyse_table_path)
                if len(analyse_table[analyse_table[constants.PANDDA_INSPECT_VIEWED] == True]) < 15:
                    continue
                if len(analyse_table[analyse_table[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == "High"]) < 2:
                    continue
                inspect_tables[possible_pandda_path] = analyse_table
            except Exception as e:
                print(f"\tERROR READING INSPECT TABLE: {analyse_table_path} : {e}")
        else:
            print(f"\tERROR READING INSPECT TABLE : {analyse_table_path} : NO SUCH TABLE!")
    rprint(f"Got {len(inspect_tables)} pandda inspect tables!")

    systems = {}
    experiments = {}
    panddas = {}
    annotations = {}
    partitions = {}
    datasets = {}
    events = {}
    with pony.orm.db_session:
        # Multiprocess PanDDAs, returning valid events for addition to the
        with joblib.Parallel(n_jobs=-1, verbose=50) as parallel:
            # j = 0
            for pandda_path, inspect_table in inspect_tables.items():
                print(f"### {pandda_path} ")

                # if j > 10:
                #     continue
                # j += 1
                pandda_events: list[Event] = parallel(
                    joblib.delayed(_parse_inspect_table_row)(
                        row,
                        pandda_path
                    )
                    for idx, row
                    in inspect_table.iterrows()
                )
                rprint(
                    f"Got {len(pandda_events)} of which {len([x for x in pandda_events if x is not None])} are not None!")
                for pandda_event in pandda_events:
                    if pandda_event:
                        dtag, event_idx = pandda_event.dtag, pandda_event.event_idx
                        system_name = _get_system_from_dtag(dtag)
                        if not system_name:
                            continue
                        if system_name in systems:
                            system = systems[system_name]
                        else:
                            system = SystemORM(
                                name=system_name,
                                experiments=[],
                                panddas=[],
                                datasets=[],
                            )
                            systems[system_name] = system

                        structure_path = Path(pandda_event.initial_structure).absolute().resolve()
                        dataset_dir_index = [j for j, part in enumerate(structure_path.parts) if part == dtag]
                        dataset_path = Path(*structure_path.parts[:dataset_dir_index[0] + 1])
                        experiment_path = dataset_path.parent
                        if experiment_path in experiments:
                            experiment = experiments[experiment_path]
                        else:
                            experiment = ExperimentORM(
                                path=str(experiment_path),
                                model_dir=str(experiment_path),
                                panddas=[],
                                system=system,
                                datasets=[]
                            )
                            experiments[experiment_path] = experiment

                        if dtag in datasets:
                            dataset = datasets[dtag]
                        else:
                            dataset = DatasetORM(
                                dtag=pandda_event.dtag,
                                path=str(dataset_path),
                                structure=str(Path(pandda_event.initial_structure).absolute().resolve()),
                                reflections=str(Path(pandda_event.initial_reflections).absolute().resolve()),
                                system=system,
                                experiment=experiment,
                                panddas=[]
                            )
                            datasets[dtag] = dataset

                        if pandda_path in panddas:
                            pandda = panddas[pandda_path]
                        else:
                            pandda = PanDDAORM(
                                path=str(pandda_path),  # *
                                events=[],
                                datasets=[dataset, ],
                                system=system,
                                experiment=experiment,
                            )
                            panddas[pandda_path] = pandda

                        if (str(pandda_path), dtag, event_idx) in custom_annotations:
                            print(
                                f"\tUpdating annotation of {(str(pandda_path), dtag, event_idx)} using custom annotation!")
                            _annotation = custom_annotations[(str(pandda_path), dtag, event_idx)]
                        else:
                            _annotation = pandda_event.annotation

                        event = EventORM(
                            dtag=pandda_event.dtag,
                            event_idx=pandda_event.event_idx,
                            x=pandda_event.x,
                            y=pandda_event.y,
                            z=pandda_event.z,
                            bdc=pandda_event.bdc,
                            initial_structure=pandda_event.initial_structure,
                            initial_reflections=pandda_event.initial_reflections,
                            structure=pandda_event.structure,
                            event_map=pandda_event.event_map,
                            z_map=pandda_event.z_map,
                            viewed=pandda_event.viewed,
                            hit_confidence=pandda_event.hit_confidence,
                            ligand=None,
                            dataset=dataset,
                            pandda=pandda,
                            annotations=[],
                            partitions=[]
                        )

                        if pandda_event.ligand:
                            ligand_orm = LigandORM(
                                path=str(pandda_event.ligand.path),
                                smiles=str(pandda_event.ligand.smiles),
                                chain=str(pandda_event.ligand.chain),
                                residue=int(pandda_event.ligand.residue),
                                num_atoms=int(pandda_event.ligand.num_atoms),
                                x=float(pandda_event.ligand.x),
                                y=float(pandda_event.ligand.y),
                                z=float(pandda_event.ligand.z),
                                event=event
                            )
                        else:
                            ligand_orm = None

                        pickled_data_dir = pandda_path / "pickled_data"
                        pandda_done = pandda_path / "pandda.done"
                        statistical_maps = pandda_path / "statistical_maps"
                        pickled_panddas_dir = pandda_path / "pickled_panddas"
                        if pickled_data_dir.exists():
                            source = "pandda_1"
                        elif pandda_done.exists():
                            source = "pandda_1"
                        elif statistical_maps.exists():
                            source = "pandda_1"
                        elif pickled_panddas_dir.exists():
                            source = "pandda_1"
                        else:
                            source = "pandda_2"
                        AnnotationORM(
                            annotation=_annotation,
                            source=source,
                            event=event
                        )

                        events[(pandda_path, pandda_event.dtag, pandda_event.event_idx)] = event

        rprint(
            f"Got {len(events)} of which {len([x for x in events.values() if x.hit_confidence == 'High'])} are high confidence!")
        rprint(systems)
        # for event_id, event in events.items():
        #     print(event)

    db.disconnect()


if __name__ == "__main__":
    fire.Fire(main)