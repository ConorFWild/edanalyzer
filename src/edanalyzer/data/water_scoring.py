# from pony import *
from pony.orm import Database, Required, Optional, Set, PrimaryKey, composite_key, select

from edanalyzer import constants

db = Database()


class WaterAnnotation(db.Entity):
    _table_ = "Annotations"

    id = PrimaryKey(int, auto=True)

    dataIdx = Required(int, unique=True)  
    landmarkIdx = Required(int)
    annotation = Required(str)
