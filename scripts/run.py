import dataclasses
import os
import pickle
import pathlib
import time
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
    PanDDADatasetTorchLigandmap
)
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
    model_file: Path | None
    # def __init__(self, dic):
    #     self.max_epochs = dic['max_epochs']


@dataclasses.dataclass
class ConfigTest:
    initial_epoch: int
    test_interval: int
    test_convergence_interval: int
    partition: int
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

        # Partition the datasets
        # print(f"System")
        # pony.orm.select(p for p in SystemORM).show()
        # print(f"Experiment")
        # [print(k) for k in pony.orm.select(p.path for p in ExperimentORM)]
        # print(f"Datasets")
        # [print(k) for k in pony.orm.select(p.path for p in DatasetORM)]
        # pony.orm.select(p.path for p in PanDDAORM).show()
        # pony.orm.select((p.dtag, p.event_idx, p.pandda) for p in EventORM).show()
        rprint(
            f"Got {len(events)} of which {len([x for x in events.values() if x.hit_confidence == 'High'])} are high confidence!")

        # for event_id, event in events.items():
        #     print(event)

        ...


def _test_partition_solution(_partition_vector, _num_items_vector):
    sums = {}
    for j in range(0, 10):
        sums[j] = np.sum(_num_items_vector[_partition_vector == j])

    # print(_partition_vector)
    # print(sums)
    res = np.std([x for x in sums.values()])
    # print(res)/
    return res


def partition_events(query):
    # systems = pony.orm.select(system for system in SystemORM)
    systems = {}
    for res in query:
        _system = res[1]
        _hit = res[2].annotation
        if _hit:
            if _system.name not in systems:
                systems[_system.name] = 0
            systems[_system.name] += 1
    rprint(systems)
    rprint(f"Got {len(systems)}")
    # rprint(
    #     {
    #         system.name: len(system.datasets)
    #         for system
    #         in systems
    #     }
    # )

    # Get an approximate partitioning
    # num_items_vector = np.array([len(x.datasets) for x in systems])
    num_items_vector = np.array([x for x in systems.values()])
    rprint(num_items_vector)
    result = scipy.optimize.differential_evolution(
        func=lambda _test_partition_vector: _test_partition_solution(
            _test_partition_vector,
            num_items_vector,
        ),
        bounds=[(0, 9) for x in range(len(systems))],
        integrality=np.array([True for x in range(len(systems))])
    )
    rprint([result.x, result.fun])

    partitions = {}
    for x, system_name in zip([int(j) for j in result.x], systems):
        if not x in partitions:
            partitions[x] = []
        partitions[x].append(system_name)
    rprint(partitions)

    sums = {}
    for j, _system_names in partitions.items():
        sums[j] = sum([systems[_system_name] for _system_name in _system_names])

    rprint(sums)

    return partitions


def _print_pandda_2_systems(working_directory):
    database_path = working_directory / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:

        partitions = pony.orm.select(p for p in PartitionORM)

        query = pony.orm.select((event, event.pandda.system, event.annotations) for event in EventORM)

        pandda_2_results = [
            result
            for result
            in query
            if (result[2].source == "pandda_2")
        ]
        rprint(f"Got {len(pandda_2_results)} pandda 2 results")

        systems = {}
        for res in pandda_2_results:
            _system = res[1]
            _hit = res[2].annotation
            if _hit:
                if _system.name not in systems:
                    systems[_system.name] = 0
                systems[_system.name] += 1
        rprint(systems)


def _partition_dataset(working_directory):
    database_path = working_directory / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:

        partitions = pony.orm.select(p for p in PartitionORM)
        if len(partitions) > 0:
            rprint(f"Already have {len(partitions)} partitions!")
            return

        query = pony.orm.select((event, event.pandda.system, event.annotations) for event in EventORM)

        # PanDDA 1 partitions
        pandda_1_partitions = partition_events([x for x in query if x[2].source == "pandda_1"])
        for partition_number, system_names in pandda_1_partitions.items():
            partition = PartitionORM(
                name=f"pandda_1_{partition_number}",
                events=[result[0] for result in query if (result[1].name in system_names) and (result[2].source == "pandda_1")],
            )

        # PanDDA 2 partitions
        pandda_2_partitions = partition_events([x for x in query if x[2].source == "pandda_2"])
        for partition_number, system_names in pandda_2_partitions.items():
            partition = PartitionORM(
                name=f"pandda_2_{partition_number}",
                events=[result[0] for result in query if (result[1].name in system_names) and (result[2].source == "pandda_2")],
            )


def try_make(path):
    try:
        os.mkdir(path)
    except Exception as e:
        return


def try_link(source_path, target_path):
    try:
        os.symlink(source_path, target_path)
    except Exception as e:
        # print(e)
        return

def _make_psuedo_pandda(psuedo_pandda_dir, events, rows, annotations):
    # psuedo_pandda_dir = working_dir / "test_datasets_pandda"
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

    # Spoof the dataset directories
    _j = 0
    for event, row in zip(events, rows):
        dtag_dir = processed_datasets_dir / str(_j)
        _j += 1
        try_make(dtag_dir)
        modelled_structures_dir = dtag_dir / "modelled_structures"
        try_make(modelled_structures_dir)

        # event, annotation = event_info['event'], event_info['annotation']
        try_link(
            event.initial_structure,
            dtag_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=_j),
        )
        try_link(
            event.initial_reflections,
            dtag_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=_j),
        )
        try_link(
            event.event_map,
            dtag_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
                dtag=_j,
                event_idx=1,
                bdc=row.loc[0, "1-BDC"],
            ),
        )
        if event.structure:
            try_link(
                event.structure,
                modelled_structures_dir / constants.PANDDA_MODEL_FILE.format(dtag=_j),
            )


    # Spoof the event table, changing the site, dtag and eventidx
    # rows = []
    # j = 0
    # for dtag in events:
    #     for event_idx, event_info in events[dtag].items():
    #         row = event_info['row']
    #         # row.loc[0, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1
    #         rows.append(row)
            # j = j + 1

    event_table = pd.concat(rows).reset_index()

    rprint(event_table)
    rprint(len(annotations))
    rprint(len(event_table))
    rprint(len(rows))
    for _j in range(len(event_table)):
        event_table.loc[_j, constants.PANDDA_INSPECT_DTAG] = str(_j)
        event_table.loc[_j, constants.PANDDA_INSPECT_EVENT_IDX] = 1
        event_table.loc[_j, constants.PANDDA_INSPECT_SITE_IDX] = (_j // 100) + 1
        event_table.loc[_j, constants.PANDDA_INSPECT_SITE_IDX] = annotations[_j]


    event_table.drop(["index", "Unnamed: 0"], axis=1, inplace=True)
    event_table.to_csv(analyse_table_path, index=False)
    event_table.to_csv(inspect_table_path, index=False)

    # Spoof the site table
    site_records = []
    num_sites = ((_j) // 100) + 1
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

def _make_test_dataset_psuedo_pandda(
            working_dir,
        test_partition
        ):
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:
        partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = pony.orm.select(
            (event, event.annotations, event.partitions, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)

        # Get the test partition pandda's and their inspect tables
        inspect_tables = {}
        for res in query:
            if res[2].name == test_partition:
                pandda = res[3]
                if pandda.path in inspect_tables:
                    continue
                else:
                    inspect_tables[pandda.path] = pd.read_csv(Path(pandda.path) / "analyses" / "pandda_inspect_events.csv")
        rprint(f"Got {len(inspect_tables)} inspect tables.")
        # rprint(inspect_tables)

        # Select events and Organise by dtag
        events = {}
        for res in query:
            if res[2].name == test_partition:
                event = res[0]
                annotation = res[1]
                dtag = event.dtag
                event_idx = event.event_idx
                if dtag not in events:
                    events[dtag] = {}

                table = inspect_tables[event.pandda.path]
                row = table[
                    (table[constants.PANDDA_INSPECT_DTAG] == dtag)
                    & (table[constants.PANDDA_INSPECT_EVENT_IDX] == event_idx)
                ]

                events[dtag][event_idx] = {
                    'event': event,
                    'annotation': annotation,
                    'row': row
                }
        rprint(f"Got {len(events)} events!")

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
        for site_id in np.arange(0, num_sites+1):
            site_records.append(
                {
                    "site_idx": int(site_id)+1,
                    "centroid": (0.0,0.0,0.0),
                    "Name": None,
                    "Comment": None
                }
            )
        print(len(site_records))
        site_table = pd.DataFrame(site_records)
        site_table.to_csv(analyse_site_table_path, index=False)
        site_table.to_csv(inspect_site_table_path, index=False)

def _make_dataset_from_events(query):
    train_dataset_torch = PanDDADatasetTorchLigand(
        PanDDAEventDataset(
            pandda_events=[
                PanDDAEvent(
                    id=res[0].id,
                    pandda_dir=res[0].pandda.path,
                    model_building_dir=res[0].pandda.experiment.model_dir,
                    system_name=res[0].pandda.system.name,
                    dtag=res[0].dtag,
                    event_idx=res[0].event_idx,
                    event_map=res[0].event_map,
                    x=res[0].x,
                    y=res[0].y,
                    z=res[0].z,
                    hit=res[1].annotation,
                    ligand=None
                )
                for res
                in query
                # if res[0].pandda.system.name not in test_partition_event_systems
            ]
        ),
        transform_image=get_image_xmap_ligand,
        transform_annotation=get_annotation_from_event_hit
    )
    # train_dataloader = DataLoader(
    #     train_dataset_torch,
    #     batch_size=12,
    #     shuffle=False,
    #     num_workers=36,
    #     drop_last=True
    # )

    return train_dataset_torch

def _get_model_annotations(model, test_dataset, dev):

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=20,
        drop_last=False
    )
    annotations = {}
    _j = 0
    for image, annotation, idx in test_dataloader:
        rprint(f"{_j} / {len(test_dataloader)}")
        _j += 1
        image_c = image.to(dev)
        annotation_c = annotation.to(dev)

        # optimizer.zero_grad()

        # forward + backward + optimize
        # begin_annotate = time.time()
        model_annotation = model(image_c)
        event = test_dataset.pandda_event_dataset[idx]
        # print(event)
        # annotations[i][(event.pandda_dir, event.dtag, event.event_idx)] = (
        annotations[(event.pandda_dir, event.dtag, event.event_idx)] = (
            float(annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
            float(model_annotation.to(torch.device("cpu")).detach().numpy()[0][1]),
        )
    return annotations

def _make_reannotation_psuedo_pandda(
            working_dir,
        model_file
        ):
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:
        partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = pony.orm.select(
            (event, event.annotations, event.partitions, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)

        # Get the test partition pandda's and their inspect tables
        inspect_tables = {}
        for res in query:
            pandda = res[3]
            if pandda.path in inspect_tables:
                continue
            else:
                inspect_path = Path(pandda.path) / "analyses" / "pandda_inspect_events.csv"
                if inspect_path.exists():
                    inspect_tables[pandda.path] = pd.read_csv(inspect_path)
        rprint(f"Got {len(inspect_tables)} inspect tables.")

        # Get the device
        if torch.cuda.is_available():
            # logger.info(f"Using cuda!")
            dev = "cuda:0"
        else:
            # logger.info(f"Using cpu!")
            dev = "cpu"

        # if model_type == "resnet+ligand":
        model = resnet18(num_classes=2, num_input=4)
        model.to(dev)
        model.eval()

        if model_file:
            model.load_state_dict(torch.load(model_file, map_location=dev))

        # Create a dataset from all events
        dataset = _make_dataset_from_events([res for res in query if res[3].path in inspect_tables])

        # Annotate it
        annotation_file = working_dir / "model_annotations.pickle"
        if annotation_file.exists():
            with open(annotation_file, 'rb') as f:
                model_annotations = pickle.load(f)
        else:
            model_annotations = _get_model_annotations(model, dataset, dev)
            with open(annotation_file, 'wb') as f:
                pickle.dump(model_annotations, f)

        rprint(model_annotations)

        # Pick out the highest ranking non-hits and lowest ranking hits
        positives, negatives = [], []
        for res in [_res for _res in query if _res[3].path in inspect_tables]:
            event = res[0]
            pandda_path, dtag, event_idx = res[3].path, event.dtag, event.event_idx
            human_annotation, model_annotation = model_annotations[(pandda_path, dtag, event_idx)]
            if human_annotation > 0.5:
                positives.append(res)
            else:
                negatives.append(res)

        hrnh_events, hrnh_rows, hrnh_annotations = [], [], []
        for res in sorted(
                positives,
                key=lambda _res: model_annotations[(res[3].path, event.dtag, event.event_idx)][1],
                reverse=True
        ):
            event = res[0]
            pandda_path, dtag, event_idx = res[3].path, event.dtag, event.event_idx
            human_annotation, model_annotation = model_annotations[(pandda_path, dtag, event_idx)]
            table = inspect_tables[res[3].path]
            row = table[
                (table[constants.PANDDA_INSPECT_DTAG] == dtag)
                & (table[constants.PANDDA_INSPECT_EVENT_IDX] == event_idx)
                ]
            print(row)
            # row.loc[0, constants.PANDDA_INSPECT_Z_PEAK] = float(model_annotation)
            hrnh_annotations.append(float(model_annotation))
            hrnh_events.append(event)
            hrnh_rows.append(row)

        lrh_events, lrh_rows, lrh_annotations = [], [], []
        for res in sorted(
                negatives,
                key=lambda _res: model_annotations[(res[3].path, event.dtag, event.event_idx)][1],
        ):
            event = res[0]
            pandda_path, dtag, event_idx = res[3].path, event.dtag, event.event_idx
            human_annotation, model_annotation = model_annotations[(pandda_path, dtag, event_idx)]
            table = inspect_tables[res[3].path]
            row = table[
                (table[constants.PANDDA_INSPECT_DTAG] == dtag)
                & (table[constants.PANDDA_INSPECT_EVENT_IDX] == event_idx)
                ]
            lrh_annotations.append(float(model_annotation))
            lrh_events.append(event)
            lrh_rows.append(row)

        # Create the fake panddas
        _make_psuedo_pandda(working_dir / "high_ranking_non_hits", hrnh_events, hrnh_rows, hrnh_annotations)
        _make_psuedo_pandda(working_dir / "low_ranking_hits", lrh_events, lrh_rows, lrh_annotations)


def _make_dataset(
            working_dir,
        test_partition
        ):
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:
        partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = pony.orm.select(
            (event, event.annotations, event.partitions, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)

        test_partition_events = partitions[test_partition]
        test_partition_event_systems = set([event.pandda.system.name for event in test_partition_events.events])
        rprint(f"Test partition systems are: {test_partition_event_systems}")

        print(partitions)
        for partition_name, partition in partitions.items():
            print(f"{partition_name} : {len(partition.events)}")

        train_dataset_torch = PanDDADatasetTorchLigand(
            PanDDAEventDataset(
                pandda_events=[
                    PanDDAEvent(
                        id=res[0].id,
                        pandda_dir=res[0].pandda.path,
                        model_building_dir=res[0].pandda.experiment.model_dir,
                        system_name=res[0].pandda.system.name,
                        dtag=res[0].dtag,
                        event_idx=res[0].event_idx,
                        event_map=res[0].event_map,
                        x=res[0].x,
                        y=res[0].y,
                        z=res[0].z,
                        hit=res[1].annotation,
                        ligand=None
                    )
                    for res
                    in query
                    if res[0].pandda.system.name not in test_partition_event_systems
                ]
            ),
            transform_image=get_image_xmap_ligand_augmented,
            transform_annotation=get_annotation_from_event_hit
        )
    train_dataloader = DataLoader(
        train_dataset_torch,
        batch_size=12,
        shuffle=False,
        num_workers=36,
        drop_last=True
    )

    import h5py
    import hdf5plugin

    if Path("test_3.h5").exists():

        begin_read_dataset = time.time()
        with h5py.File('test_3.h5', 'r') as f:
            panddas = f["pandda_paths"]
            dtags = f["dtags"]
            event_idxs = f["event_idxs"]
            images = f["images"]
            annotations = f['annotations']
            # for j in range(len(dtags) // 12):
            #     _panddas = panddas[j*12:min((j+1)*12, len(dtags))]
            #     _dtags = dtags[j * 12:min((j + 1) * 12, len(dtags))]
            #     _event_idxs = event_idxs[j * 12:min((j + 1) * 12, len(dtags))]
            #     _images = images[j * 12:min((j + 1) * 12, len(dtags))]
            #     _annotations = annotations[j * 12:min((j + 1) * 12, len(dtags))]
            for j in range(len(dtags)):

                _panddas = panddas[j]
                _dtag = dtags[j]
                _event_idx = event_idxs[j]
                _image = images[j]
                _annotation = annotations[j]

        finish_read_dataset = time.time()
        rprint(f"Read dataset in:{finish_read_dataset-begin_read_dataset}")

    else:

        begin_make_dataset = time.time()

        with h5py.File('test_3.h5', 'w') as f:
            panddas = f.create_dataset("pandda_paths", (len(train_dataset_torch),), chunks=(1,), dtype=h5py.string_dtype(encoding='utf-8'))
            dtags = f.create_dataset("dtags", (len(train_dataset_torch),), chunks=(1,), dtype=h5py.string_dtype(encoding='utf-8'))
            event_idxs = f.create_dataset("event_idxs", (len(train_dataset_torch),), chunks=(1,), dtype='i')
            # images = f.create_dataset("images", (len(train_dataset_torch), 4, 30,30,30), chunks=(12,4,30,30,30), dtype='float32', compression="gzip", compression_opts=9)
            images = f.create_dataset(
                "images",
                (len(train_dataset_torch), 4, 30,30,30),
                chunks=(1,4,30,30,30),
                dtype='float32',
                **hdf5plugin.Blosc2(cname='blosclz', clevel=9, filters=hdf5plugin.Blosc2.BITSHUFFLE),
            )
            annotations = f.create_dataset("annotations", (len(train_dataset_torch), 2), chunks=(1,2), dtype='float32')

            # for j in range(len(train_dataset_torch)):
            _k = 0
            for image, annotation, idx in train_dataloader:
                print(f"Iteration: {_k} / {len(train_dataloader)}")
                _k = _k + 1
                # image, annotation, idx = train_dataset_torch[j]
                image_np = image.detach().numpy()
                annotation_np = annotation.detach().numpy()
                idx_np = idx.detach().numpy()

                for _j, _idx in enumerate(idx_np):
                    event = train_dataset_torch.pandda_event_dataset[int(_idx)]
                    panddas[int(_idx)] = event.pandda_dir
                    dtags[int(_idx)] = event.dtag
                    event_idxs[int(_idx)] = event.event_idx
                    images[int(_idx)] = image_np[_j]
                    annotations[int(_idx)] = annotation_np[_j][1]

        finish_make_dataset = time.time()
        print(f"Made dataset in: {finish_make_dataset-begin_make_dataset}")


    # j = 0
    # for images, annotations, idxs in train_dataloader:
    #     images_np = images.detach().numpy()
    #     annotations_np = annotations.detach().numpy()
    #     idxs_np = idxs.detach().numpy()
    #
    #     events = [train_dataset_torch.pandda_event_dataset[idx] for idx in idxs_np]
    #     panddas[j*12:(j+1)*12] = [event.pandda_dir for event in events]
    #     dtags[j*12:(j+1)*12] = [event.dtag for event in events]
    #     event_idxs[j * 12:(j + 1) * 12] = [event.event_idx for event in events]


def _train_and_test(working_dir, test_partition, initial_epoch, test_interval, model_file, model_key):
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    with pony.orm.db_session:
        partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = pony.orm.select(
            (event, event.annotations, event.partitions, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)

        test_partition_events = partitions[test_partition]
        test_partition_event_systems = set([event.pandda.system.name for event in test_partition_events.events])
        rprint(f"Test partition systems are: {test_partition_event_systems}")

        print(partitions)
        for partition_name, partition in partitions.items():
            print(f"{partition_name} : {len(partition.events)}")


        # for event in partitions[test_partition].events:
        #     print(event.pandda.path)

        # event_partitions = {}
        # for res in query:
        #     event = res[0]
        #     partition = event.partitions[0]
        #     if partition.name in event_partitions
        #     event_partitions[]
        # for res in query:
        #     print(f"Annotation: {res[0].annotations.annotation}")
        train_dataset_torch = PanDDADatasetTorchLigand(
            PanDDAEventDataset(
                pandda_events=[
                    PanDDAEvent(
                        id=res[0].id,
                        pandda_dir=res[0].pandda.path,
                        model_building_dir=res[0].pandda.experiment.model_dir,
                        system_name=res[0].pandda.system.name,
                        dtag=res[0].dtag,
                        event_idx=res[0].event_idx,
                        event_map=res[0].event_map,
                        x=res[0].x,
                        y=res[0].y,
                        z=res[0].z,
                        hit=res[1].annotation,
                        ligand=None
                    )
                    for res
                    in query
                    if res[0].pandda.system.name not in test_partition_event_systems
                ]
            ),
            transform_image=get_image_xmap_ligand_augmented,
            transform_annotation=get_annotation_from_event_hit
        )
        rprint(f"Got {len(train_dataset_torch)} train events!")

        test_dataset_torch = PanDDADatasetTorchLigand(
            PanDDAEventDataset(
                pandda_events=[
                    PanDDAEvent(
                        # id=event.id,
                        # pandda_dir=event.pandda.path,
                        # model_building_dir=event.pandda.experiment.model_dir,
                        # system_name=event.pandda.system.name,
                        # dtag=event.dtag,
                        # event_idx=event.event_idx,
                        # event_map=event.event_map,
                        # x=event.x,
                        # y=event.y,
                        # z=event.z,
                        # hit=event.annotations.annotation,
                        # ligand=None
                        id=res[0].id,
                        pandda_dir=res[0].pandda.path,
                        model_building_dir=res[0].pandda.experiment.model_dir,
                        system_name=res[0].pandda.system.name,
                        dtag=res[0].dtag,
                        event_idx=res[0].event_idx,
                        event_map=res[0].event_map,
                        x=res[0].x,
                        y=res[0].y,
                        z=res[0].z,
                        hit=res[1].annotation,
                        ligand=None
                    )
                    for res
                    in query
                    if res[2].name == test_partition
                ]),
            transform_image=get_image_xmap_ligand,
            transform_annotation=get_annotation_from_event_hit
        )
        rprint(f"Got {len(test_dataset_torch)} test events!")

    # Get the device
    if torch.cuda.is_available():
        # logger.info(f"Using cuda!")
        dev = "cuda:0"
    else:
        # logger.info(f"Using cpu!")
        dev = "cpu"

    # if model_type == "resnet+ligand":
    model = resnet18(num_classes=2, num_input=4)
    model.to(dev)

    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=dev),
                                  )

    train(
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
    )

def _get_precision_recall(epoch_results):
    pr = {}
    for cutoff in np.linspace(0.0,1.0,num=101):
        tp = [key for key, annotations in epoch_results.items() if (annotations[0] == 1.0) and (annotations[1] >= cutoff)]# *]
        fp = [key for key, annotations in epoch_results.items() if (annotations[0] == 0.0) and (annotations[1] >= cutoff)]
        tn = [key for key, annotations in epoch_results.items() if (annotations[0] == 0.0) and (annotations[1] < cutoff)]
        fn = [key for key, annotations in epoch_results.items() if (annotations[0] == 1.0) and (annotations[1] < cutoff)]

        # rprint([len(tp), len(fp), len(tn), len(fn)])
        try:
            recall = len(tp) / len(tp+fn)
        except:
            recall = 0.0
        try:
            precision = len(tp) / len(tp + fp)
        except:
            precision = 0.0

        pr[round(float(cutoff), 2)] = {'precision': round(precision, 3), 'recall': round(recall, 3)}

    return pr

def _summarize(working_dir):
    with open(Path(working_dir) / "annotations.pickle", 'rb') as f:
        test_results = pickle.load(f)

    # rprint(test_results)

    for epoch, epoch_results in test_results.items():
        precision_recall = _get_precision_recall(epoch_results)
        # rprint(precision_recall)
        recall_greater_than_95 = {cutoff: pr for cutoff, pr in precision_recall.items() if pr['recall'] > 0.95}

        if len(recall_greater_than_95) > 0:
            max_prec_cutoff = max(recall_greater_than_95, key=lambda x: recall_greater_than_95[x]['precision'])
            rprint(f"Epoch: {epoch} : Recall: {precision_recall[max_prec_cutoff]['recall']} : Precision: {precision_recall[max_prec_cutoff]['precision']}")

def __main__(config_yaml="config.yaml"):
    # Initialize the config
    with open(config_yaml, "r") as f:
        dic = yaml.safe_load(f)
        if dic['train']['model_file']:
            model_file = Path(dic['train']['model_file'])
        else:
            model_file = None
        config = Config(
            name=dic["name"],
            steps=dic['steps'],
            working_directory=Path(dic['working_directory']),
            datasets=[x for x in dic['datasets']],
            exclude=[x for x in dic['exclude']],
            train=ConfigTrain(
                dic['train']['max_epochs'],
                model_file
            ),
            test=ConfigTest(
                dic['test']['initial_epoch'],
                dic['test']['test_interval'],
                dic['test']['test_convergence_interval'],
                dic['test']['partition']
            ),
            custom_annotations=dic['custom_annotations'],
            cpus=dic['cpus']
        )
        rprint(config)
    rprint(f"Printing pandda 2 systems...")
    # _print_pandda_2_systems(config.working_directory)

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
    # Partition the data
    if "Partition" in config.steps:
        _partition_dataset(
            config.working_directory
        )

    if "MakeTestDatasetPsuedoPanDDA" in config.steps:
        _make_test_dataset_psuedo_pandda(
            config.working_directory,
            config.test.partition
        )

    if "MakeReannotationPsuedoPanDDA" in config.steps:
        _make_reannotation_psuedo_pandda(
            config.working_directory,
            config.train.model_file
        )

    if "UpdateFromReannotationDir" in config.steps:
        ...

    if "MakeDataset" in config.steps:
        _make_dataset(
            config.working_directory,
            config.test.partition
        )

    # Run training/testing
    if 'Train+Test' in config.steps:
        _train_and_test(
            config.working_directory,
            config.test.partition,
            config.test.initial_epoch,
            config.test.test_interval,
            config.train.model_file,
            config.name
        )
    # Summarize train/test results
    if 'Summarize' in config.steps:
        _summarize(config.working_directory)


if __name__ == "__main__":
    fire.Fire(__main__)
