from pathlib import Path
from typing import List, Optional
import os

from loguru import logger
import gemmi
import numpy as np
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

import constants


class Base(DeclarativeBase):
    pass


event_partition_association_table = Table(
    "event_partition_association_table",
    Base.metadata,
    Column("event_id", ForeignKey(f"{constants.TABLE_EVENT}.id"), primary_key=True),
    Column("partition_id", ForeignKey(f"{constants.TABLE_PARTITION}.id"), primary_key=True),
)

dataset_pandda_association_table = Table(
    "dataset_pandda_association_table",
    Base.metadata,
    Column("dataset_id", ForeignKey(f"{constants.TABLE_DATASET}.id"), primary_key=True),
    Column("pandda_id", ForeignKey(f"{constants.TABLE_PANDDA}.id"), primary_key=True),
)


class PanDDAORM(Base):
    __tablename__ = constants.TABLE_PANDDA

    id: Mapped[int] = mapped_column(primary_key=True)

    path: Mapped[str]

    events: Mapped[List["EventORM"]] = relationship(back_populates="pandda")
    datasets: Mapped[List["DatasetORM"]] = relationship(
        secondary=dataset_pandda_association_table,
        back_populates="panddas",
    )
    system: Mapped["SystemORM"] = relationship(back_populates="panddas")
    experiment: Mapped["ExperimentORM"] = relationship(back_populates="panddas")


class DatasetORM(Base):
    __tablename__ = constants.TABLE_DATASET

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
        secondary=dataset_pandda_association_table,
        back_populates="datasets",
    )


class EventORM(Base):
    __tablename__ = constants.TABLE_EVENT

    id: Mapped[int] = mapped_column(primary_key=True)
    event_idx: Mapped[int]
    x: Mapped[float]
    y: Mapped[float]
    z: Mapped[float]
    bdc: Mapped[float]
    initial_structure: Mapped[str]
    initial_reflections: Mapped[str]
    structure: Mapped[str]
    event_map: Mapped[str]
    z_map: Mapped[str]

    ligand_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_LIGAND}.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_DATASET}.id"))
    pandda_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_PANDDA}.id"))

    ligand: Mapped[Optional["LigandORM"]]
    dataset: Mapped["DatasetORM"]
    pandda: Mapped["PanDDAORM"]
    annotations: Mapped["AnnotationORM"] = relationship(back_populates="event")
    partitions: Mapped["PartitionORM"] = relationship(
        secondary=event_partition_association_table,
        back_populates="events",
    )

    ...


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

    id: Mapped[int] = mapped_column(primary_key=True)
    annotation: Mapped[bool]
    source: Mapped[str]

    event_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_EVENT}.id"))

    event: Mapped["EventORM"] = relationship(back_populates="annotations")


class SystemORM(Base):
    __tablename__ = constants.TABLE_SYSTEM

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    experiments: Mapped[List["ExperimentORM"]] = relationship(back_populates="system")
    panddas: Mapped[List["PanDDAORM"]] = relationship(back_populates="system")


class ExperimentORM(Base):
    __tablename__ = constants.TABLE_EXPERIMENT

    id: Mapped[int] = mapped_column(primary_key=True)

    path: Mapped[str]
    model_dir: Mapped[str]

    system_id: Mapped[int] = mapped_column(ForeignKey(f"{constants.TABLE_SYSTEM}.id"))

    panddas: Mapped[List["PanDDAORM"]] = relationship(back_populates="experiment")
    system: Mapped["SystemORM"] = relationship(back_populates="experiments")
    datasets: Mapped[List["DatasetORM"]] = relationship(back_populates="experiment")


class PartitionORM(Base):
    __tablename__ = constants.TABLE_PARTITION

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str]
    events: Mapped[List["EventORM"]] = relationship(
        secondary=event_partition_association_table,
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
            for ligand in ligands:
                # structure_ligands.append(

                num_atoms = get_ligand_num_atoms(ligand)

                ligand_centroid = get_ligand_centroid(ligand)

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
                    smiles=smiles,
                    chain=chain.name,
                    residue=ligand.seqid.num,
                    num_atoms=num_atoms,
                    x=ligand_centroid[0],
                    y=ligand_centroid[1],
                    z=ligand_centroid[2]
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
        return None

    min_dist_id = min(ligand_distances, key=lambda _id: ligand_distances[_id])

    if ligand_distances[min_dist_id] < cutoff:
        return ligand_dict[min_dist_id]
    else:
        return None


def parse_inspect_table_row(row, pandda_dir, pandda_processed_datasets_dir, model_building_dir):
    dtag = str(row[constants.PANDDA_INSPECT_DTAG])
    event_idx = row[constants.PANDDA_INSPECT_EVENT_IDX]
    bdc = row[constants.PANDDA_INSPECT_BDC]
    x = row[constants.PANDDA_INSPECT_X]
    y = row[constants.PANDDA_INSPECT_Y]
    z = row[constants.PANDDA_INSPECT_Z]
    viewed = row[constants.PANDDA_INSPECT_VIEWED]

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
    if not initial_structure.exists():
        initial_structure = None
    else:
        initial_structure = str(initial_structure)

    initial_reflections = processed_dataset_dir / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag)
    if not initial_reflections.exists():
        initial_reflections = None
    else:
        initial_reflections = str(initial_reflections)

    if not event_map_path.exists():
        return None

    if not viewed:
        return None

    inspect_model_path = inspect_model_dir / constants.PANDDA_MODEL_FILE.format(dtag=dtag)
    # initial_model = processed_dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag)

    if inspect_model_path.exists() & hit_confidence_class:
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

    event = EventORM(
        event_idx=event_idx,
        x=x,
        y=y,
        z=z,
        bdc=bdc,
        initial_structure=initial_structure,
        initial_reflections=initial_reflections,
        structure=inspect_model_path,
        event_map=event_map_path,
        z_map=z_map_path,
        ligand=ligand,
    )

    return event


def parse_pandda_inspect_table(
        pandda_inspect_table_file,
        potential_pandda_dir,
        pandda_processed_datasets_dir,
        model_building_dir,
):
    try:
        pandda_inspect_table = pd.read_csv(pandda_inspect_table_file)
    except Exception as e:
        logger.warning(f"Failed to read table: {pandda_inspect_table_file} : {e}")
        return None

    events = []
    for index, row in pandda_inspect_table.iterrows():
        possible_event = parse_inspect_table_row(
            row, potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir)
        if possible_event:
            events.append(possible_event)

    events_with_models = len([event for event in events if event.ligand is not None])

    if events_with_models > 0:
        return events
    else:
        return None


def parse_potential_pandda_dir(potential_pandda_dir, model_building_dir):
    pandda_analysis_dir = potential_pandda_dir / constants.PANDDA_ANALYSIS_DIR
    pandda_inspect_table_file = pandda_analysis_dir / constants.PANDDA_INSPECT_TABLE_FILE
    pandda_processed_datasets_dir = potential_pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    if not os.access(pandda_analysis_dir, os.R_OK):
        return None
    if pandda_analysis_dir.exists():
        if pandda_inspect_table_file.exists():
            events = parse_pandda_inspect_table(
                pandda_inspect_table_file,
                potential_pandda_dir, pandda_processed_datasets_dir, model_building_dir

            )
            return events

    return None


def get_experiment_datasets(experiment: ExperimentORM):
    datasets = {}
    for directory in Path(experiment.path).glob("*"):
        if directory.is_dir():
            structure_path = directory / constants.MODEL_BUILDING_STRUCTURE_FILE
            reflections_path = directory / constants.MODEL_BUILDING_REFLECTIONS_FILE
            if not structure_path.exists():
                structure_path = None

            if not reflections_path.exists():
                reflections_path = None

            dataset = DatasetORM(
                dtag=directory.name,
                path=str(directory),
                structure=structure_path,
                reflections=reflections_path
            )
            datasets[dataset.dtag] = dataset
    return datasets


def get_system_from_dataset(dataset: DatasetORM):
    hyphens = [pos for pos, char in enumerate(dataset.dtag) if char == "-"]
    if len(hyphens) == 0:
        return None
    else:
        last_hypen_pos = hyphens[-1]
        system_name = dataset.dtag[:last_hypen_pos]

    return SystemORM(

    )


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

    experiments = {}
    systems = {}
    panddas = {}
    for experiment_superdir in pandda_data_root_dir.glob("*"):
        logger.info(f"Checking superdir: {experiment_superdir}")
        for experiment_dir in experiment_superdir.glob("*"):
            logger.info(f"Checking project dir: {experiment_dir}")

            analysis_dir = experiment_dir / constants.DIAMOND_PROCESSING_DIR / constants.DIAMOND_ANALYSIS_DIR

            model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_NEW
            if not model_building_dir.exists():
                model_building_dir = analysis_dir / constants.DIAMOND_MODEL_BUILDING_DIR_OLD
                if not model_building_dir.exists():
                    logger.debug(f"No model building dir: skipping!")
                    continue

            experiment = ExperimentORM(
                path=str(experiment_dir),
                model_dir=str(model_building_dir)
            )
            experiments[str(experiment_dir)] = experiment

            logger.debug(f"Model building dir is: {model_building_dir}")

            experiment_datasets = get_experiment_datasets(experiment)
            experiment.datasets = list(experiment_datasets.values())

            system = get_system_from_dataset(experiment.datasets[0])
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
                pandda_events = parse_potential_pandda_dir(
                    potential_pandda_dir,
                    model_building_dir,
                )

                for event in pandda_events:
                    event.dataset = experiment_datasets[event.dtag]

                pandda_dataset_dtags = get_pandda_dir_dataset_dtags(potential_pandda_dir)

                if pandda_events:
                    logger.info(f"Found {len(pandda_events)} events!")
                    num_events_with_ligands = len(
                        [event for event in pandda_events if event.ligand is not None])
                    logger.info(f"Events which are modelled: {num_events_with_ligands}")

                    pandda = PanDDAORM(
                        path=str(potential_pandda_dir),
                        events=pandda_events,
                        datasets=[dataset for dataset in experiment_datasets.values() if dataset.dtag in pandda_dataset_dtags],
                        system=system,
                        experiment=experiment
                    )
                    panddas[str(pandda.path)] = pandda


                else:
                    logger.debug(f"Discovered no events with models: skipping!")

    # logger.info(f"Found {len(pandda_events)} events!")
    # num_events_with_ligands = len([event for event in pandda_events if event.ligand is not None])
    # logger.info(f"Found {num_events_with_ligands} events with ligands modelled!")

    session.add_all([experiment for experiment in experiments.values()])
    session.add_all([system for system in systems.values()])
    session.add_all([pandda for pandda in panddas.values()])

    session.commit()


def populate_partition_from_json(json_path: Path):
    ...


def initialize_database(engine_path: str):
    engine = create_engine(f"sqlite:///{engine_path}")
    Base.metadata.create_all(engine)
