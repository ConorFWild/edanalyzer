import time
from pathlib import Path
from typing import List, Optional
import os

from loguru import logger
import gemmi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy import String, Integer
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import lazyload
from sqlalchemy.orm import joinedload


import constants
from data import PanDDAEventReannotations, PanDDAEventAnnotations, PanDDAEventDataset


class Base(DeclarativeBase):
    pass


event_partition_association_table = Table(
    constants.TABLE_EVENT_PARTITION,
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("event_id", Integer, ForeignKey(f"{constants.TABLE_EVENT}.id"), ),
    Column("partition_id", Integer, ForeignKey(f"{constants.TABLE_PARTITION}.id"), ),
)

dataset_pandda_association_table = Table(
    constants.TABLE_DATASET_PANDDA,
    Base.metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset_id", Integer, ForeignKey(f"{constants.TABLE_DATASET}.id"), ),
    Column("pandda_id", Integer, ForeignKey(f"{constants.TABLE_PANDDA}.id"), ),
)


# class EventPartitionAssociationTable(Base):
#     __tablename__ = constants.TABLE_EVENT_PARTITION
#
#     id: Mapped[int] = mapped_column(primary_key=True)
#
#     event_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EVENT}.id"))
#     partition_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_PARTITION}.id"))
#
#
# class DatasetPanDDAAssociationTable(Base):
#     __tablename__ = constants.TABLE_DATASET_PANDDA
#
#     id: Mapped[int] = mapped_column(primary_key=True)
#
#     dataset_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_DATASET}.id"))
#     pandda_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_PANDDA}.id"))


class PanDDAORM(Base):
    __tablename__ = constants.TABLE_PANDDA

    __table_args__ = (
        UniqueConstraint("path"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    path: Mapped[str]

    events: Mapped[List["EventORM"]] = relationship(back_populates="pandda")
    datasets: Mapped[List["DatasetORM"]] = relationship(
        secondary=constants.TABLE_DATASET_PANDDA,
        back_populates="panddas",
    )

    system_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_SYSTEM}.id"))
    experiment_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EXPERIMENT}.id"))

    system: Mapped["SystemORM"] = relationship(back_populates="panddas")
    experiment: Mapped["ExperimentORM"] = relationship(back_populates="panddas")


class DatasetORM(Base):
    __tablename__ = constants.TABLE_DATASET

    __table_args__ = (
        UniqueConstraint("path"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dtag: Mapped[str]
    path: Mapped[str]
    structure: Mapped[str]
    reflections: Mapped[str]

    system_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_SYSTEM}.id"))
    experiment_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EXPERIMENT}.id"))

    system: Mapped["SystemORM"] = relationship(back_populates="datasets")
    experiment: Mapped["ExperimentORM"] = relationship(back_populates="datasets")
    panddas: Mapped[List["PanDDAORM"]] = relationship(
        secondary=constants.TABLE_DATASET_PANDDA,
        back_populates="datasets",
    )
    events: Mapped[List["EventORM"]] = relationship(back_populates="dataset")


class EventORM(Base):
    __tablename__ = constants.TABLE_EVENT

    __table_args__ = (
        UniqueConstraint("pandda_id", "dtag", "event_idx"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dtag: Mapped[str]
    event_idx: Mapped[int]
    x: Mapped[float]
    y: Mapped[float]
    z: Mapped[float]
    bdc: Mapped[float]
    initial_structure: Mapped[Optional[str]]
    initial_reflections: Mapped[Optional[str]]
    structure: Mapped[Optional[str]]
    event_map: Mapped[str]
    z_map: Mapped[Optional[str]]
    viewed: Mapped[bool]
    hit_confidence: Mapped[str]

    # ligand_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_LIGAND}.id"))
    dataset_id: Mapped[Optional[int]] = mapped_column(ForeignKey(f"{constants.TABLE_DATASET}.id"))
    pandda_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_PANDDA}.id"))

    ligand: Mapped[Optional["LigandORM"]] = relationship(back_populates="event")
    dataset: Mapped[Optional["DatasetORM"]] = relationship(back_populates="events")
    pandda: Mapped["PanDDAORM"] = relationship(back_populates="events")
    annotations: Mapped[List["AnnotationORM"]] = relationship(back_populates="event")
    partitions: Mapped["PartitionORM"] = relationship(
        secondary=constants.TABLE_EVENT_PARTITION,
        back_populates="events",
    )


class LigandORM(Base):
    __tablename__ = constants.TABLE_LIGAND

    id: Mapped[int] = mapped_column(primary_key=True)

    path: Mapped[str]
    smiles: Mapped[str]
    chain: Mapped[str]
    residue: Mapped[int]
    num_atoms: Mapped[int]
    x: Mapped[float]
    y: Mapped[float]
    z: Mapped[float]

    event_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EVENT}.id"))
    event: Mapped["EventORM"] = relationship(back_populates="ligand")


class AnnotationORM(Base):
    __tablename__ = constants.TABLE_ANNOTATION

    __table_args__ = (
        UniqueConstraint("source", "event_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    annotation: Mapped[bool]
    source: Mapped[str]

    event_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EVENT}.id"))

    event: Mapped["EventORM"] = relationship(back_populates="annotations")


class SystemORM(Base):
    __tablename__ = constants.TABLE_SYSTEM

    __table_args__ = (
        UniqueConstraint("name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    experiments: Mapped[List["ExperimentORM"]] = relationship(back_populates="system")
    panddas: Mapped[List["PanDDAORM"]] = relationship(back_populates="system")
    datasets: Mapped[List["DatasetORM"]] = relationship(back_populates="system")


class ExperimentORM(Base):
    __tablename__ = constants.TABLE_EXPERIMENT

    __table_args__ = (
        UniqueConstraint("path"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    path: Mapped[Optional[str]]
    model_dir: Mapped[str]

    system_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_SYSTEM}.id"))

    panddas: Mapped[List["PanDDAORM"]] = relationship(back_populates="experiment")
    system: Mapped["SystemORM"] = relationship(back_populates="experiments")
    datasets: Mapped[List["DatasetORM"]] = relationship(back_populates="experiment")


class PartitionORM(Base):
    __tablename__ = constants.TABLE_PARTITION

    __table_args__ = (
        UniqueConstraint("name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str]
    events: Mapped[List["EventORM"]] = relationship(
        secondary=constants.TABLE_EVENT_PARTITION,
        back_populates="partitions",
    )
    # events: Mapped[List["EventORM"]]


def populate_events_from_pandda():
    ...


def populate_from_pandda(pandda_path: Path):
    system = get_system_from_pandda_path()
    experiment = get_experiment_from_pandda_path()

    ...


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
                    smiles = ""
                    # logger.debug(f"Ligand smiles: {smiles}")
                    # logger.debug(f"Num atoms: {num_atoms}")
                    # logger.debug(f"Centroid: {ligand_centroid}")
                    lig = LigandORM(
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


def get_event_ligand(inspect_model_path, x, y, z, cutoff=5.0):
    structure_ligands = get_structure_ligands(str(inspect_model_path))

    ligand_distances = {}
    ligand_dict = {}
    for lig in structure_ligands:
        ligand_distances[lig.id] = gemmi.Position(lig.x, lig.y, lig.z).dist(gemmi.Position(x, y, z))

        ligand_dict[lig.id] = lig

    if len(ligand_dict) == 0:
        logger.warning(f"Modelled structure but no ligands: {inspect_model_path}!")
        return None

    min_dist_id = min(ligand_distances, key=lambda _id: ligand_distances[_id])

    if ligand_distances[min_dist_id] < cutoff:
        # logger.warning(f"Modelled structure has ligand")
        return ligand_dict[min_dist_id]
    else:
        return None


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


def parse_inspect_table_row(row, pandda_dir, pandda_processed_datasets_dir, model_building_dir, annotation_type):
    dtag = str(row[constants.PANDDA_INSPECT_DTAG])
    event_idx = row[constants.PANDDA_INSPECT_EVENT_IDX]
    bdc = row[constants.PANDDA_INSPECT_BDC]
    x = row[constants.PANDDA_INSPECT_X]
    y = row[constants.PANDDA_INSPECT_Y]
    z = row[constants.PANDDA_INSPECT_Z]
    viewed = row[constants.PANDDA_INSPECT_VIEWED]

    if viewed != True:
        return None

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
    z_map_path = processed_dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=dtag)
    if not z_map_path.exists():
        z_map_path = None
    else:
        z_map_path = str(z_map_path)

    initial_structure = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)
    if not try_open_structure(initial_structure):
        initial_structure = None
    else:
        initial_structure = str(initial_structure)

    initial_reflections = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag)
    if not try_open_reflections(initial_reflections):
        initial_reflections = None
    else:
        initial_reflections = str(initial_reflections)

    if not try_open_map(event_map_path):
        return None

    # if not viewed:
    #     return None

    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=dtag)
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    if inspect_model_path.exists():
        ligand = get_event_ligand(
            inspect_model_path,
            x,
            y,
            z,
        )
        inspect_model_path = str(inspect_model_path)
    else:
        ligand = None
        inspect_model_path = None

    # hyphens = [pos for pos, char in enumerate(dtag) if char == "-"]
    # if len(hyphens) == 0:
    #     return None
    # else:
    #     last_hypen_pos = hyphens[-1]
    #     system_name = dtag[:last_hypen_pos + 1]

    if hit_confidence not in ["Low", "low"]:
        annotation_value = True
    else:
        annotation_value = False
    annotation = AnnotationORM(
        annotation=annotation_value,
        source="auto"
    )

    event = EventORM(
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
        annotations=[annotation, ]
    )
    annotation.event = event

    return event


def parse_pandda_inspect_table(
        pandda_inspect_table_file,
        potential_pandda_dir,
        pandda_processed_datasets_dir,
        model_building_dir,
        annotation_type
):
    try:
        pandda_inspect_table = pd.read_csv(pandda_inspect_table_file)
    except Exception as e:
        logger.warning(f"Failed to read table: {pandda_inspect_table_file} : {e}")
        return None

    events = []
    for index, row in pandda_inspect_table.iterrows():
        possible_event = parse_inspect_table_row(
            row,
            potential_pandda_dir,
            pandda_processed_datasets_dir,
            model_building_dir,
            annotation_type,
        )
        if possible_event:
            events.append(possible_event)

    # events_with_models = len([event for event in events if event.ligand is not None])
    high_confidence_events = len([event for event in events if event.hit_confidence not in ["Low", "low"]])

    if high_confidence_events > 0:
        return events
    else:
        logger.warning(f"No events with models! Skipping!")
        return None


def parse_pandda_inspect_table_parallel(
        pandda_inspect_table_file,
        potential_pandda_dir,
        pandda_processed_datasets_dir,
        model_building_dir,
        parallel,
annotation_type,
):
    try:
        pandda_inspect_table = pd.read_csv(pandda_inspect_table_file)
    except Exception as e:
        logger.warning(f"Failed to read table: {pandda_inspect_table_file} : {e}")
        return None

    # events = []
    # for index, row in pandda_inspect_table.iterrows():
    #     possible_event = parse_inspect_table_row(
    #         row, potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir)
    #     if possible_event:
    #         events.append(possible_event)

    events_initial = parallel(
        delayed(parse_inspect_table_row)(
            row,
            potential_pandda_dir,
            pandda_processed_datasets_dir,
            model_building_dir,
            annotation_type
        )
        for index, row
        in pandda_inspect_table.iterrows()
    )

    events = [event for event in events_initial if event]
    # events_with_models = len([event for event in events if event.ligand is not None])
    high_confidence_events = len([event for event in events if event.hit_confidence not in ["Low", "low"]])

    if high_confidence_events > 0:
        return events
    else:
        logger.warning(f"No events with models! Skipping!")
        return None


def parse_potential_pandda_dir(potential_pandda_dir, model_building_dir, annotation_type):
    pandda_analysis_dir = potential_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    pandda_inspect_table_file = pandda_analysis_dir / constants.PANDDA_INSPECT_TABLE_FILE
    pandda_processed_datasets_dir = potential_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR

    if not os.access(pandda_analysis_dir, os.R_OK):
        logger.warning(f"Could not read: {pandda_analysis_dir}! Skipping!")
        return None
    if pandda_analysis_dir.exists():
        if pandda_inspect_table_file.exists():
            events = parse_pandda_inspect_table(
                pandda_inspect_table_file,
                potential_pandda_dir,
                pandda_processed_datasets_dir,
                model_building_dir,
                annotation_type
            )
            return events

    return None


def parse_potential_pandda_dir_parallel(potential_pandda_dir, model_building_dir, parallel, annotation_type):
    pandda_analysis_dir = potential_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    pandda_inspect_table_file = pandda_analysis_dir / constants.PANDDA_INSPECT_TABLE_FILE
    pandda_processed_datasets_dir = potential_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR

    if not os.access(pandda_analysis_dir, os.R_OK):
        logger.warning(f"Could not read: {pandda_analysis_dir}! Skipping!")
        return None
    if pandda_analysis_dir.exists():
        if pandda_inspect_table_file.exists():
            events = parse_pandda_inspect_table_parallel(
                pandda_inspect_table_file,
                potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir, parallel, annotation_type

            )
            return events

    return None


def get_experiment_datasets(experiment: ExperimentORM, ):
    datasets = {}
    for directory in Path(experiment.model_dir).glob("*"):
        if directory.is_dir():
            structure_path = directory / constants.MODEL_BUILDING_STRUCTURE_FILE
            reflections_path = directory / constants.MODEL_BUILDING_REFLECTIONS_FILE

            if not structure_path.exists():
                # structure_path = None
                structure_path = directory / constants.OLD_MODEL_BUILDING_STRUCTURE_FILE_TEMPLATE.format(dtag=directory.name)
                if not structure_path.exists():
                    continue

            if not reflections_path.exists():
                # reflections_path = None
                reflections_path = directory / constants.OLD_MODEL_BUILDING_REFLECTIONS_FILE_TEMPLATE.format(dtag=directory.name)
                if not reflections_path.exists():
                    continue

            dataset = DatasetORM(
                dtag=directory.name,
                path=str(directory),
                structure=str(structure_path),
                reflections=str(reflections_path)
            )
            datasets[dataset.dtag] = dataset
    return datasets


def get_system_from_datasets(datasets: List[DatasetORM]):
    for dataset in datasets:
        hyphens = [pos for pos, char in enumerate(dataset.dtag) if char == "-"]
        if len(hyphens) == 0:
            continue
        else:
            last_hypen_pos = hyphens[-1]
            system_name = dataset.dtag[:last_hypen_pos]

            return SystemORM(
                name=system_name
            )

    return None


#

def get_pandda_dir_dataset_dtags(potential_pandda_dir: Path):
    processed_dataset_dir = potential_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR

    dataset_dtags = []
    for dataset_dir in processed_dataset_dir.glob("*"):
        if dataset_dir.is_dir():
            dataset_dtags.append(dataset_dir.name)

    return dataset_dtags


def populate_from_diamond(session):
    pandda_data_root_dir = Path(constants.PANDDA_DATA_ROOT_DIR)
    logger.info(f"Looking for PanDDAs under dir: {pandda_data_root_dir}")

    with Parallel(n_jobs=-2, prefer="threads", verbose=10) as parallel:

        experiments = {}
        systems = {}
        panddas = {}
        begin_time = time.time()
        for experiment_superdir in pandda_data_root_dir.glob("*"):
            logger.info(f"Checking superdir: {experiment_superdir}")
            for experiment_dir in experiment_superdir.glob("*"):
                logger.info(f"Checking project dir: {experiment_dir}")

                analysis_dir = experiment_dir / constants.DIAMOND_PROCESSING_DIR / constants.DIAMOND_ANALYSIS_DIR

                model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_NEW
                if not model_building_dir.exists():
                    model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_OLD
                    if not model_building_dir.exists():
                        logger.warning(f"No model building dir: skipping!")
                        continue

                experiment = ExperimentORM(
                    path=str(experiment_dir),
                    model_dir=str(model_building_dir)
                )

                logger.debug(f"Model building dir is: {model_building_dir}")

                experiment_datasets = get_experiment_datasets(experiment)
                if len(experiment_datasets) == 0:
                    logger.warning(f"No datasets for experiment! Skipping!")
                    continue
                else:
                    logger.info(f"Got {len(experiment_datasets)} datasets!")
                experiment.datasets = list(experiment_datasets.values())

                logger.debug(f"Example experiment is: {experiment.datasets[0].dtag}")

                system = get_system_from_datasets(experiment.datasets)

                if not system:
                    continue
                # logger.debug(f"Example system is: {experiment.datasets[0]}")

                experiments[str(experiment_dir)] = experiment

                if system.name in systems:
                    system = systems[system.name]
                else:
                    systems[system.name] = system
                system.experiments.append(experiment)

                for dataset in experiment_datasets.values():
                    dataset.system = system
                    dataset.experiment = experiment

                for potential_pandda_dir in analysis_dir.glob("*"):
                    logger.debug(f"Checking folder {potential_pandda_dir} ")
                    pandda_events = parse_potential_pandda_dir_parallel(
                        potential_pandda_dir,
                        model_building_dir,
                        parallel
                    )

                    if not pandda_events:
                        logger.info(f"No PanDDA events! Skipping!")
                        continue
                    else:
                        logger.info(f"Got {len(pandda_events)} pandda events!")

                    for event in pandda_events:
                        if event.dtag in experiment_datasets:
                            event.dataset = experiment_datasets[event.dtag]
                        else:
                            logger.warning(
                                f"Event with dataset {event.dtag} has no corresponding dataset in experiment!")

                    pandda_dataset_dtags = get_pandda_dir_dataset_dtags(potential_pandda_dir)

                    if pandda_events:
                        logger.info(f"Found {len(pandda_events)} events!")
                        num_events_with_ligands = len(
                            [event for event in pandda_events if event.ligand is not None])
                        logger.info(f"Events which are modelled: {num_events_with_ligands}")

                        pandda = PanDDAORM(
                            path=str(potential_pandda_dir),
                            events=pandda_events,
                            datasets=[dataset for dataset in experiment_datasets.values() if
                                      dataset.dtag in pandda_dataset_dtags],
                            system=system,
                            experiment=experiment
                        )
                        panddas[str(pandda.path)] = pandda


                    else:
                        logger.debug(f"Discovered no events with models: skipping!")

                    logger.info(f"Discovered {len(pandda_events)} events for pandda!")

            # if (time.time() - begin_time) > 60*15:
            #     break

        # logger.info(f"Found {len(pandda_events)} events!")
        # num_events_with_ligands = len([event for event in pandda_events if event.ligand is not None])
        # logger.info(f"Found {num_events_with_ligands} events with ligands modelled!")

    for pandda in panddas.values():
        for event in pandda.events:
            if event.dataset is None:
                system = pandda.system
                if event.dtag in system.datasets:
                    event.dataset = system

    session.add_all([experiment for experiment in experiments.values()])
    session.add_all([system for system in systems.values()])
    session.add_all([pandda for pandda in panddas.values()])

    session.commit()


def populate_partition_from_json(
        session,
        train_dataset: PanDDAEventDataset,
        test_dataset: PanDDAEventDataset,
):
    # Get the datasets
    # datasets_stmt = select(DatasetORM)

    # Get the events
    events_stmt = select(EventORM).join(EventORM.pandda)

    # Get the train dataset keys
    train_event_keys = [str(event.event_map) for event in
                        train_dataset.pandda_events]

    # Get the events
    logger.info(f"Getting events")
    events = {event.event_map: event for event in session.scalars(events_stmt)}
    logger.info(f'Got {len(events)} events!')

    # Add partitions for train
    train_partition = PartitionORM(name=constants.INITIAL_TRAIN_PARTITION, events=[])
    logger.info(f"Iterating {len(train_event_keys)} train events")
    for event_key in train_event_keys:
        if event_key in events:
            train_partition.events.append(events[event_key])

    logger.info(f"Added {len(train_partition.events)} events to intial train partition")

    # Get the test dataset keys
    test_event_keys = [str(event.event_map) for event in test_dataset.pandda_events]

    # Add partitions for test
    test_partition = PartitionORM(name=constants.INITIAL_TEST_PARTITION, events=[])
    logger.info(f"Iterating {len(test_event_keys)} test events")
    for event_key in test_event_keys:
        if event_key in events:
            test_partition.events.append(events[event_key])

    logger.info(f"Added {len(test_partition.events)} events to intial test partition")

    # Add partitions and commit
    session.add_all([train_partition, test_partition])
    session.commit()


def populate_from_custom_pandda(session, custom_pandda, partition_name):
    # Get the experiments
    experiments_stmt = select(ExperimentORM)
    experiments = {experiment.model_dir: experiment for experiment in session.scalars(experiments_stmt).unique().all()}
    logger.info(f"Found {len(experiments)} experiments in database!")

    # Get the systems
    systems_stmt = select(SystemORM).options(joinedload((SystemORM.datasets)))
    systems = {system.name: system for system in session.scalars(systems_stmt).unique().all()}
    logger.info(f"Found {len(systems)} systems in database!")

    # Get the datasets
    datasets_stmt = select(DatasetORM).options(joinedload(DatasetORM.experiment),
                                               joinedload(DatasetORM.events),
                                               )
    datasets = {dataset.path: dataset for dataset in session.scalars(datasets_stmt).unique().all()}
    logger.info(f"Found {len(datasets)} datasets in database!")

    # Get the partitions
    partitions_stmt = select(PartitionORM)
    partitions = {partition.name: partition for partition in session.scalars(partitions_stmt).unique().all()}
    logger.info(f"Found {len(partitions)} partitions in database!")

    # Get the panddas
    panddas_stmt = select(PanDDAORM)
    panddas = {pandda.path: pandda for pandda in session.scalars(panddas_stmt).unique().all()}
    logger.info(f"Found {len(panddas)} panddas in database!")

    # Create a new partition if necessary
    if partition_name not in partitions:
        partition = PartitionORM(name=partition_name)
        logger.info(f"Using partitions: {partition_name}")
    else:
        partition = partitions[partition_name]
        logger.info(f'Created new partition: {partition_name}')

    # Loop over custom PanDDAs, adding appropriate systems, experiments, annotitions,
    # partitions and events
    new_panddas = []
    new_systems = []
    new_experiments = []
    with Parallel(n_jobs=-2, prefer="threads", verbose=10) as parallel:

        new_experiment = False
        new_system = False

        logger.info(f"PanDDA Path is: {custom_pandda.pandda}")
        # Unpack the PanDDA object
        pandda_data_source = custom_pandda.source
        pandda_path = custom_pandda.pandda

        # Check if PanDDA already added
        if pandda_path in panddas:
            logger.warning(f"Already parsed PanDDA at: {pandda_path}")
            return

        # Get the experiment or create a new one
        if pandda_data_source in experiments:
            logger.info(f"PanDDA data source {pandda_data_source} already part of an experiment")
            experiment = experiments[pandda_data_source]

            # Get the datasets
            experiment_datasets = {
                dataset.dtag: dataset
                for dataset
                in datasets.values()
                if dataset.experiment.model_dir == experiment.model_dir
            }
            logger.info(f"Found {len(experiment_datasets)} datasets for experiment")

            # Get the system
            system = experiment.system
            logger.info(f"System name is: {system.name}")


        else:
            logger.info(f"Creating new experiment: {pandda_data_source}")
            experiment = ExperimentORM(
                path=None,
                model_dir=pandda_data_source,
            )
            # new_experiments.append(experiment)
            new_experiment = True

            # Get the datasets
            experiment_datasets = get_experiment_datasets(experiment)
            logger.info(f"Found {len(experiment_datasets)} datasets for experiment")

            # Get the system
            system = get_system_from_datasets(list(experiment_datasets.values()))
            logger.info(f"System name is: {system.name}")

            # Get the system or create a new one
            if system.name in systems:
                logger.info(f"Know system, replacing with one from table!")
                system = systems[system.name]

                system.experiments.append(experiment)
            else:
                logger.info(f"New system!")
                # new_systems.append(system)
                new_system = True

            # Match the experiment to its system
            experiment.system = system

            # Update the dataset system and experiment
            for dtag, dataset in experiment_datasets.items():
                dataset.experiment = experiment
                dataset.system = system

        # Get the other system datasets
        system_datasets = {dataset.dtag: dataset for dataset in system.datasets}
        logger.info(f"System has {len(system_datasets)} datasets")

        # Get the events
        events = parse_potential_pandda_dir_parallel(
            Path(pandda_path),
            Path(pandda_data_source),
            annotation_type="manual",
            parallel=parallel
        )
        if not events:
            logger.warning(f"Did not find any events! Skipping!")
            return
        logger.info(f"Found {len(events)} events!")

        # Match events to datasets
        for event in events:
            if event.dtag in experiment_datasets:
                event.dataset = experiment_datasets[event.dtag]
            else:
                logger.warning(f"Event with dataset {event.dtag} has no corresponding dataset in experiment!")
                if event.dtag in system_datasets:
                    event.dataset = system_datasets[event.dtag]
                else:
                    logger.warning(f"Not in system datasets either!")

        # Get PanDDA dataset dtags
        pandda_dataset_dtags = get_pandda_dir_dataset_dtags(Path(pandda_path))

        # Create a new PanDDA
        pandda = PanDDAORM(
            path=str(pandda_path),
            events=events,
            datasets=[dataset for dataset in experiment_datasets.values() if
                      dataset.dtag in pandda_dataset_dtags],
            system=system,
            experiment=experiment
        )
        # new_panddas.append(pandda)

        # Add the events to the revelant partition
        for event in events:
            event.partition = partition
            partition.events.append(event)

        # Add entries
        if new_experiment:
            session.add_all([experiment, ])
        if new_system:
            session.add_all([system, ])

        session.add_all([pandda, ])

        # Commit
        session.commit()

def populate_from_custom_panddas(engine, custom_panddas, partition_name):
    for custom_pandda in custom_panddas:
        with Session(engine) as session:
            populate_from_custom_pandda(session, custom_pandda, partition_name)

def parse_old_annotation_update_dir(session, annotation_update_dir: Path):
    # Get the events
    events_stmt = select(EventORM).options(joinedload(EventORM.annotations))
    events = {event.event_map: event for event in session.scalars(events_stmt).unique().all()}
    logger.info(f"Got {len(events)} events from table")

    # Get the inspect table
    inspect_table_path = annotation_update_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE
    inspect_table = pd.read_csv(inspect_table_path)
    logger.info(f"Loaded inspect table at : {inspect_table_path}")
    num_viewed = inspect_table[inspect_table[constants.PANDDA_INSPECT_VIEWED] == True]
    logger.info(f"Inspect table has {len(inspect_table)} events of which {len(num_viewed)} are viewed")

    # Parse the table
    annotations = []
    for idx, row in inspect_table.iterrows():
        dtag = row[constants.PANDDA_INSPECT_DTAG]
        event_idx = int(row[constants.PANDDA_INSPECT_EVENT_IDX])
        bdc = row[constants.PANDDA_INSPECT_BDC]
        viewed = row[constants.PANDDA_INSPECT_VIEWED]

        if viewed != True:
            logger.debug(f"Event {dtag} {event_idx} has not been viewed! Skipping!")
            continue

        # Get the resolved event map path
        event_map_path = str(
            (
                annotation_update_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag /
                constants.PANDDA_EVENT_MAP_TEMPLATE.format(
                    dtag=dtag,
                    event_idx=event_idx,
                    bdc=bdc
        )).readlink())

        # Get the event using its event map path
        if event_map_path in events:
            event = events[event_map_path]
        else:
            continue
        # event = events[event_map_path]

        # print(len(event.annotations))
        annotated = False
        for annotation in event.annotations:
            if annotation.source == "manual":
                logger.debug(f"Already manually annotated event: {event.event_map}")
                annotated = True

        if annotated:
            continue

        # Determine the annotation
        if row[constants.PANDDA_INSPECT_VIEWED] == True:
            if row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
                annotation = AnnotationORM(annotation=True, source="manual", event=event)
            else:
                annotation = AnnotationORM(annotation=False, source="manual", event=event)

            annotations.append(annotation)

    logger.info(f"Got {len(annotations)} new annotations!")

    # Update
    session.add_all(
        [
            annotation
            for annotation
            in annotations
        ]
    )

    # Commit
    session.commit()


def parse_annotation_update_dir(session, annotation_update_dir: Path):
    # Get the events
    events_stmt = select(EventORM)
    events = {event.id: event for event in session.scalars(events_stmt)}

    # Get the inspect table
    inspect_table_path = annotation_update_dir / constants.PANDDA_ANALYSIS_DIR / constants.PANDDA_INSPECT_TABLE_FILE
    inspect_table = pd.read_csv(inspect_table_path)

    # Parse the table
    annotations = []
    for idx, row in inspect_table.iterrows():
        dtag = row[constants.PANDDA_INSPECT_DTAG]
        event = events[dtag]
        if row[constants.PANDDA_INSPECT_VIEWED] == True:
            if row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == constants.PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE:
                annotation = AnnotationORM(annotation=True, source="manual", event=event)
            else:
                annotation = AnnotationORM(annotation=False, source="manual", event=event)

            annotations.append(annotation)

    # Update
    session.add_all([annotation for annotation in annotations])

    # Commit
    session.commit()


def initialize_database(engine_path: str):
    engine = create_engine(f"sqlite:///{engine_path}")
    Base.metadata.create_all(engine)
