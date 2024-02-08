# from pony import *
from pony.orm import Database, Required, Optional, Set, PrimaryKey, composite_key, select

from edanalyzer import constants

db = Database()


class PanDDAORM(db.Entity):
    _table_ = constants.TABLE_PANDDA

    id = PrimaryKey(int, auto=True)

    path = Required(str, unique=True)  # *

    events = Set("EventORM")
    datasets = Set("DatasetORM", table=constants.TABLE_DATASET_PANDDA, column="dataset_id")

    system = Required("SystemORM", column="system_id")
    experiment = Required("ExperimentORM", column="experiment_id")


class DatasetORM(db.Entity):
    _table_ = constants.TABLE_DATASET

    id = PrimaryKey(int, auto=True)
    dtag = Required(str)
    path = Required(str, unique=True)
    structure = Required(str)
    reflections = Required(str)

    system = Required("SystemORM", column="system_id")
    experiment = Required("ExperimentORM", column="experiment_id")
    panddas = Set("PanDDAORM", table=constants.TABLE_DATASET_PANDDA, column="pandda_id")
    events = Set("EventORM")


class EventORM(db.Entity):
    _table_ = constants.TABLE_EVENT

    id = PrimaryKey(int, auto=True)

    dtag = Required(str)
    event_idx = Required(int)
    x = Required(float)
    y = Required(float)
    z = Required(float)
    bdc = Required(float)
    initial_structure = Optional(str)
    initial_reflections = Optional(str)
    structure = Optional(str, nullable=True)
    event_map = Required(str)
    z_map = Optional(str)
    viewed = Required(bool)
    hit_confidence = Required(str)

    ligand = Optional("LigandORM", cascade_delete=True)
    dataset = Optional("DatasetORM", column="dataset_id")
    pandda = Required("PanDDAORM", column="pandda_id")
    annotations = Set("AnnotationORM")
    partitions = Set("PartitionORM", table=constants.TABLE_EVENT_PARTITION, column="partition_id")

    composite_key(pandda, dtag, event_idx)


class LigandORM(db.Entity):
    _table_ = constants.TABLE_LIGAND

    id = PrimaryKey(int, auto=True)

    path = Required(str)
    smiles = Required(str)
    chain = Required(str)
    residue = Required(int)
    num_atoms = Required(int)
    x = Required(float)
    y = Required(float)
    z = Required(float)

    event = Required("EventORM", column="event_id")


class AnnotationORM(db.Entity):
    _table_ = constants.TABLE_ANNOTATION

    id = PrimaryKey(int, auto=True)

    annotation = Required(bool)
    source = Required(str)

    event = Required("EventORM", column="event_id")

    composite_key(source, event)


class SystemORM(db.Entity):
    _table_ = constants.TABLE_SYSTEM

    id = PrimaryKey(int, auto=True)
    name = Required(str, unique=True)

    experiments = Set("ExperimentORM")
    panddas = Set("PanDDAORM")
    datasets = Set("DatasetORM")


class ExperimentORM(db.Entity):
    _table_ = constants.TABLE_EXPERIMENT

    id = PrimaryKey(int, auto=True)

    path = Optional(str, unique=True)
    model_dir = Required(str)

    panddas = Set("PanDDAORM")
    system = Required("SystemORM", column="system_id")
    datasets = Set("DatasetORM")


class PartitionORM(db.Entity):
    _table_ = constants.TABLE_PARTITION

    id = PrimaryKey(int, auto=True)

    name = Required(str, unique=True)
    events = Set("EventORM", table=constants.TABLE_EVENT_PARTITION, column="event_id")

class AutobuildORM(db.Entity):
    _table_ = constants.TABLE_AUTOBUILD

    id = PrimaryKey(int, auto=True)

    experiment_model_dir = Optional(str)
    pandda_path = Optional(str)
    dtag = Optional(str)
    model_idx = Optional(int)
    event_idx = Optional(int)
    known_hit_key = Optional(str)
    ligand_key = Optional(str)
    rmsd = Optional(float)
    score = Optional(float)
    size = Optional(float)
    local_strength = Optional(float)
    rscc = Optional(float)
    signal = Optional(float)
    noise = Optional(float)
    signal_noise = Optional(float)
    x_ligand = Optional(float)
    y_ligand = Optional(float)
    z_ligand = Optional(float)
    x = Optional(float)
    y = Optional(float)
    z = Optional(float)
    build_path = Optional(str)
    bdc = Optional(float)
    xmap_path = Optional(str)
    mean_map_path = Optional(str)
    mtz_path = Optional(str)
    zmap_path = Optional(str)
    train_test = Optional(str)
