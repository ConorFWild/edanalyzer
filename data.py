from pydantic import BaseModel

import constants


def load_model(path, model):
    return model.parse_file(path)


def save_model(path, model):
    with open(path, "w") as f:
        f.write(model.json())


# class Conformer(BaseModel):
#     id: int
#     path: str
#     rmsd: float


class Ligand(BaseModel):
    id: int
    smiles: str
    chain: str
    residue: int
    # conformers: list[Conformer]
    num_atoms: int
    x: float
    y: float
    z: float


class PanDDAEvent(BaseModel):
    id: int
    pandda_dir: str
    model_building_dir: str
    system_name: str
    dtag: str
    event_idx: int
    event_map: str
    x: float
    y: float
    z: float
    hit: bool
    ligand: Ligand | None


# class PanDDAData(BaseModel):
#     events: list[PanDDAEvent]
#
class PanDDAEventDataset(BaseModel):
    pandda_events: list[PanDDAEvent]

    def save(self, path):
        save_model(path / constants.PANDDA_DATASET_FILE, self)

    @classmethod
    def load(cls, path):
        return load_model(path / constants.PANDDA_DATASET_FILE, cls)


class PanDDAEventReannotation(BaseModel):
    reannotation: bool

class PanDDAEventReannotations(BaseModel):
    reannotations: list[PanDDAEventReannotation]

class StructureReflectionsData(BaseModel):
    id: int
    name: str
    pdb_path: str
    mtz_path: str
    ligands: list[Ligand]
    partition: int
    f: str
    phi: str


class StructureReflectionsDataset(BaseModel):
    data: list[StructureReflectionsData]

    @classmethod
    def load(cls, path):
        load_model(path / constants.DATASET_FILE, cls)

    def save(self, path):
        save_model(path / constants.DATASET_FILE, self)


class Options(BaseModel):
    working_dir: str

    @classmethod
    def load(cls, path):
        return load_model(path, cls)

    def save(self, path):
        save_model(path, self)
