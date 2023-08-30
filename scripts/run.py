import dataclasses
import os
import pickle
import pathlib
from pathlib import Path

import yaml
import fire
import pony
import rich
from rich import print as rprint
import pandas as pd
import joblib
import gemmi
import numpy as np

from edanalyzer.database_pony import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM  # import *
from edanalyzer import constants


# from pony.orm import *

@dataclasses.dataclass
class Ligand:
    path: str
    smiles: str
    chain: str
    residue: int
    num_atoms: int
    x: float
    y: float
    z: float


@dataclasses.dataclass
class Event:
    dtag: str
    event_idx: int
    x: float
    y: float
    z: float
    bdc: float
    initial_structure: str
    initial_reflections: str
    structure: str
    event_map: str
    z_map: str
    ligand: None
    viewed: None
    hit_confidence: str
    annotation: bool


@dataclasses.dataclass
class ConfigTrain:
    max_epochs: int
    # def __init__(self, dic):
    #     self.max_epochs = dic['max_epochs']


@dataclasses.dataclass
class ConfigTest:
    test_interval: int
    test_convergence_interval: int
    # def __init__(self, dic):
    #     self.test_interval = dic['test_interval']
    #     self.test_convergence_interval = dic['test_convergence_interval']


@dataclasses.dataclass
class Config:
    name: str
    steps: list[str]
    working_directory: Path
    datasets: list[str]
    exclude: list[str]
    train: ConfigTrain
    test: ConfigTest
    custom_annotations: Path
    cpus: int

    # def __init__(self, dic):
    #     self.name = dic["name"]
    #     self.steps = dic['steps']
    #     self.working_directory = Path(dic['working_directory'])
    #     self.datasets = [x for x in dic['datasets']]
    #     self.exclude = [x for x in dic['exclude']]
    #     self.train = ConfigTrain(dic['train'])
    #     self.test = ConfigTest(dic['test'])
    #     self.custom_annotations = dic['custom_annotations']
    #     self.cpus = dic['cpus']


def _get_custom_annotations(path):
    # get the events
    db.bind(provider='sqlite', filename=f"{path}")
    db.generate_mapping()

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


def try_open_structure(path):
    try:
        st = gemmi.read_structure(str(path))
        return True
    except:
        return False


def try_open_reflections(path):
    try:
        mtz = gemmi.read_mtz_file(str(path))
        return True
    except:
        return False


def try_open_map(path):
    try:
        m = gemmi.read_ccp4_map(str(path))
        return True
    except:
        return False


def _has_parsable_pdb(compound_dir):
    event_added = False
    # ligand_files_dir = processed_dataset_dir / "ligand_files"
    if compound_dir.exists():
        ligand_pdbs = []
        for ligand_pdb in compound_dir.glob("*.pdb"):
            if ligand_pdb.exists():
                if ligand_pdb.stem not in constants.LIGAND_IGNORE_REGEXES:
                    try:
                        st = gemmi.read_structure(str(ligand_pdb))
                    except:
                        return False
                    num_atoms = 0
                    for model in st:
                        for chain in model:
                            for residue in chain:
                                for atom in residue:
                                    num_atoms += 1

                    if num_atoms > 3:
                        return True
        # ]
        # if len(ligand_pdbs) > 0:
        #     return True

    return False


def _get_system_from_dtag(dtag):
    hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    if len(hyphens) == 0:
        return None
    else:
        last_hypen_pos = hyphens[-1]
        system_name = dtag[:last_hypen_pos]

        return system_name


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
    for model in structure:
        for chain in model:
            ligands = chain.get_ligands()
            for res in chain:
                # structure_ligands.append(
                if res.name == "LIG":
                    num_atoms = get_ligand_num_atoms(res)

                    ligand_centroid = get_ligand_centroid(res)

                    # smiles = parse_ligand(
                    #     structure,
                    #     chain,
                    #     ligand,
                    # )
                    smiles = "~"
                    # logger.debug(f"Ligand smiles: {smiles}")
                    # logger.debug(f"Num atoms: {num_atoms}")
                    # logger.debug(f"Centroid: {ligand_centroid}")
                    lig = Ligand(
                        path=str(pdb_path),
                        smiles=str(smiles),
                        chain=str(chain.name),
                        residue=int(res.seqid.num),
                        num_atoms=int(num_atoms),
                        x=float(ligand_centroid[0]),
                        y=float(ligand_centroid[1]),
                        z=float(ligand_centroid[2])
                    )
                    structure_ligands.append(lig)

    return structure_ligands


# def generate_smiles(options: Options, dataset: StructureReflectionsDataset):
#     logger.info(f"Generating smiles for dataset")
#     for data in dataset.data:
#         ligands = get_structure_ligands(data)
#         data.ligands = ligands
#
#     logger.info(f"Generated smiles, saving to {options.working_dir}")
#     dataset.save(options.working_dir)


def get_event_ligand(inspect_model_path, x, y, z, cutoff=10.0):
    structure_ligands = get_structure_ligands(str(inspect_model_path))

    ligand_distances = {}
    ligand_dict = {}
    for lig in structure_ligands:
        ligand_distances[(lig.chain, lig.residue)] = gemmi.Position(lig.x, lig.y, lig.z).dist(gemmi.Position(x, y, z))

        ligand_dict[(lig.chain, lig.residue)] = lig

    if len(ligand_dict) == 0:
        # logger.warning(f"Modelled structure but no ligands: {inspect_model_path}!")
        return None

    min_dist_id = min(ligand_distances, key=lambda _id: ligand_distances[_id])

    if ligand_distances[min_dist_id] < cutoff:
        # logger.warning(f"Modelled structure has ligand")
        return ligand_dict[min_dist_id]
    else:
        return None


def _parse_inspect_table_row(
        row,
        pandda_dir,
):
    dtag = str(row[constants.PANDDA_INSPECT_DTAG])
    event_idx = row[constants.PANDDA_INSPECT_EVENT_IDX]
    bdc = row[constants.PANDDA_INSPECT_BDC]
    x = row[constants.PANDDA_INSPECT_X]
    y = row[constants.PANDDA_INSPECT_Y]
    z = row[constants.PANDDA_INSPECT_Z]
    viewed = row[constants.PANDDA_INSPECT_VIEWED]

    if viewed != True:
        rprint(f"Dataset not viewed! Skipping {dtag} {event_idx} {pandda_dir}!")
        return None

    hit_confidence = row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE]
    if hit_confidence == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
        hit_confidence_class = True
    else:
        hit_confidence_class = False

    pandda_processed_datasets_dir = pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    processed_dataset_dir = pandda_processed_datasets_dir / dtag
    compound_dir = processed_dataset_dir / "ligand_files"
    if not _has_parsable_pdb(compound_dir):
        rprint(f"No parsable pdb at {compound_dir}! Skipping!")
        return None

    inspect_model_dir = processed_dataset_dir / constants.PANDDA_INSPECT_MODEL_DIR
    event_map_path = processed_dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
        dtag=dtag,
        event_idx=event_idx,
        bdc=bdc
    )
    z_map_path = processed_dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=dtag)
    if not z_map_path.exists():
        z_map_path = None
    else:
        z_map_path = str(z_map_path)

    mean_map_path = processed_dataset_dir / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(dtag=dtag)
    if not try_open_map(mean_map_path):
        return None

    initial_structure = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)
    if not try_open_structure(initial_structure):
        initial_structure = None
    else:
        initial_structure = str(initial_structure)
    if not initial_structure:
        return None

    initial_reflections = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag)
    if not try_open_reflections(initial_reflections):
        initial_reflections = None
    else:
        initial_reflections = str(initial_reflections)
    if not initial_reflections:
        return None

    if not try_open_map(event_map_path):
        return None

    # if not viewed:
    #     return None

    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=dtag)
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    # if inspect_model_path.exists():
    #     inspect_model_path = str(inspect_model_path)
    # else:
    #     inspect_model_path = None

    ligand = None
    if inspect_model_path.exists():
        ligand = get_event_ligand(
            inspect_model_path,
            x,
            y,
            z,
        )
        inspect_model_path = str(inspect_model_path)
        # if ligand:
        #     ligand_orm = LigandORM(
        #         path=str(ligand.path),
        #         smiles=str(ligand.smiles),
        #         chain=str(ligand.chain),
        #         residue=int(ligand.residue),
        #         num_atoms=int(ligand.num_atoms),
        #         x=float(ligand.x),
        #         y=float(ligand.y),
        #         z=float(ligand.z),
        #     )
        # else:
        #     ligand_orm = None
    else:
        ligand_orm = None
        inspect_model_path = None

    # hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    # if len(hyphens) == 0:
    #     return None
    # else:
    #     last_hypen_pos = hyphens[-1]
    #     system_name = dtag[:last_hypen_pos + 1]
    # ligand = None

    if hit_confidence not in ["Low", "low"]:
        annotation_value = True
    else:
        annotation_value = False

    if ligand and annotation_value:
        rprint(
            f"For {(dtag, event_idx)}, updating event centroid using associated ligand centroid from {(x, y, z)} to {(ligand.x, ligand.y, ligand.z)}")
        x, y, z = ligand.x, ligand.y, ligand.z

    rprint(f"\tAdding event: {(dtag, event_idx)}!")
    event = Event(
        dtag=str(dtag),
        event_idx=int(event_idx),
        x=float(x),
        y=float(y),
        z=float(z),
        bdc=float(bdc),
        initial_structure=initial_structure,
        initial_reflections=initial_reflections,
        structure=inspect_model_path,
        event_map=str(event_map_path),
        z_map=z_map_path,
        ligand=ligand,
        viewed=viewed,
        hit_confidence=hit_confidence,
        annotation=annotation_value
    )

    return event


def _make_database(
        name,
        working_directory,
        datasets,
        exclude,
        cpus,
        custom_annotations
):
    database_path = working_directory / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Get the possible pandda paths
    possible_pandda_paths = [
        path
        for dataset_pattern
        in datasets
        for path
        in Path('/').glob(dataset_pattern[1:])
        if not any([path.match(exclude_pattern) for exclude_pattern in exclude])

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
            j = 0
            for pandda_path, inspect_table in inspect_tables.items():
                if j > 10:
                    continue
                j += 1
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
                            print(f"\tUpdating annotation of {(dtag, event_idx)} using custom annotation!")
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

                        AnnotationORM(
                            annotation=_annotation,
                            source='auto',
                            event=event
                        )

                        events[(pandda_path, pandda_event.dtag, pandda_event.event_idx)] = event

        # Partition the datasets
        # print(f"System")
        # pony.orm.select(p for p in SystemORM).show()
        # print(f"Experiment")
        # [print(k) for k in pony.orm.select(p.path for p in ExperimentORM)]
        # print(f"Datasets")
        # [print(k) for k in pony.orm.select(p.path for p in DatasetORM)]
        # pony.orm.select(p.path for p in PanDDAORM).show()
        # pony.orm.select((p.dtag, p.event_idx, p.pandda) for p in EventORM).show()
        rprint(f"Got {len(events)} of which {len([x for x in events.values() if x.hit_confidence == 'High'])} are high confidence!")

        # for event_id, event in events.items():
        #     print(event)

        ...


def __main__(config_yaml="config.yaml"):
    # Initialize the config
    with open(config_yaml, "r") as f:
        dic = yaml.safe_load(f)

        config = Config(
            name=dic["name"],
            steps=dic['steps'],
            working_directory=Path(dic['working_directory']),
            datasets=[x for x in dic['datasets']],
            exclude=[x for x in dic['exclude']],
            train=ConfigTrain(
                dic['train']['max_epochs'],
            ),
            test=ConfigTest(
                dic['test']['test_interval'],
                dic['test']['test_convergence_interval'],
            ),
            custom_annotations=dic['custom_annotations'],
            cpus=dic['cpus']
        )
        rprint(config)

    if not config.working_directory.exists():
        os.mkdir(config.working_directory)

    # Parse custom annotations
    if "Annotations" in config.steps:
        custom_annotations_path = config.working_directory / "custom_annotations.pickle"
        if custom_annotations_path.exists():
            with open(custom_annotations_path, 'rb') as f:
                custom_annotations = pickle.load(f)
        else:
            custom_annotations: dict[tuple[str, str, int], bool] = _get_custom_annotations(config.custom_annotations)
            with open(custom_annotations_path, "wb") as f:
                pickle.dump(custom_annotations, f)
        rprint(f"\tGot {len(custom_annotations)} custom annotations!")

    # Construct the dataset
    if "Collate" in config.steps:
        _make_database(
            config.name,
            config.working_directory,
            config.datasets,
            config.exclude,
            config.cpus,
            custom_annotations
        )

    # Run training/testing
    if 'Train+Test' in config.steps:
        ...
    # Summarize train/test results
    if 'Summarize' in config.steps:
        ...


if __name__ == "__main__":
    fire.Fire(__main__)
