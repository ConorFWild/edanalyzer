# from pony import *
from pony import Database, Entity, Required, Optional, Set, PrimaryKey, composite_key, select

from edanalyzer import constants

db = Database()

#
# event_partition_association_table = Table(
#     constants.TABLE_EVENT_PARTITION,
#     db.Entity.metadata,
#     Column("id", Integer, primary_key=True),
#     Column("event_id", Integer, ForeignKey(f"{constants.TABLE_EVENT}.id"), ),
#     Column("partition_id", Integer, ForeignKey(f"{constants.TABLE_PARTITION}.id"), ),
# )
#
# dataset_pandda_association_table = Table(
#     constants.TABLE_DATASET_PANDDA,
#     db.Entity.metadata,
#     Column("id", Integer, primary_key=True),
#     Column("dataset_id", Integer, ForeignKey(f"{constants.TABLE_DATASET}.id"), ),
#     Column("pandda_id", Integer, ForeignKey(f"{constants.TABLE_PANDDA}.id"), ),
# )

class PanDDAORM(db.Entity):
    _table_ = constants.TABLE_PANDDA

    id = PrimaryKey(int, auto=True)

    path = Required(str, unique=True)  #*

    events = Set("EventORM")
    datasets = Set("DatasetORM", table=constants.TABLE_DATASET_PANDDA, column="dataset_id")

    system = Required("SystemORM")
    experiment = Required("Experiment")

class DatasetORM(db.Entity):
    _table_ = constants.TABLE_DATASET

    id = PrimaryKey(int, auto=True)
    dtag = Required(str)
    path = Required(str, unique=True)
    structure = Required(str)
    reflections = Required(str)

    system = Required("SystemORM")
    experiment = Required("ExperimentORM")
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
    structure = Optional(str)
    event_map = Required(str)
    z_map = Optional(str)
    viewed = Required(bool)
    hit_confidence = Required(str)

    ligand = Optional("LigandORM")
    dataset = Optional("DatasetORM")
    pandda = Required("PanDDAORM")
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

    event = Required("EventORM")

class AnnotationORM(db.Entity):
    _table_ = constants.TABLE_ANNOTATION


    id = PrimaryKey(int, auto=True)

    annotation = Required(bool)
    source = Required(str)

    event= Required("EventORM")

    composite_key(source, event)



class SystemORM(db.Entity):
    _table_ = constants.TABLE_SYSTEM


    id = PrimaryKey(int, auto=True)
    name= Required(str, unique=True)

    experiments = Set("ExperimentORM")
    panddas= Set("PanDDAORM")
    datasets= Set("DatasetORM")


class ExperimentORM(db.Entity):
    _table_ = constants.TABLE_EXPERIMENT



    id = PrimaryKey(int, auto=True)

    path = Optional(str, unique=True)
    model_dir = Required(str)

    panddas = Set("PanDDAORM")
    system = Required("SystemORM")
    datasets = Set("DatasetORM")


class PartitionORM(db.Entity):
    _table_ = constants.TABLE_PARTITION


    id = PrimaryKey(int, auto=True)


    name = Required(str, unique=True)
    events = Set("EventORM", table=constants.TABLE_EVENT_PARTITION, column="event_id")