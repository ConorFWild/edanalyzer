from data import StructureReflectionsDataset, Options, StructureReflectionsData, Ligand
from numpy.random import default_rng


def load_xmap_from_mtz(path):
    ...


def sample_xmap(xmap, transform):
    ...


def annotate_data_randomly(data: StructureReflectionsData, p: float):
    rng = default_rng()
    val = rng.random_sample()

    if val < p:
        return 1
    else:
        return 0


def generate_ligand_sample(data, get_transform, sample_ligand_in_xmap):
    transform = get_transform(data)
    image = sample_ligand_in_xmap(data, transform)
    return image


def generate_xmap_ligand_sample_or_decoy(
        data: StructureReflectionsData,
        annotation: int,
        sample_ligand,
        sample_ligand_decoy):
    # Decoy
    if annotation == 0:
        image = sample_ligand(data)
    # True sample
    else:
        image = sample_ligand_decoy(data)

    return image

    # xmap = load_xmap_from_mtz(data.mtz_path)
    #
    # image = sample_xmap(xmap, transform)


def sample_ligand_density(data: StructureReflectionsData, annotater, image_sampler):
    annotation = annotater(data)
    image = image_sampler(data, annotation)

    return image, annotation


class StructureReflectionsDatasetTorch(Dataset):
    def __init__(self,
                 structure_reflections_dataset: StructureReflectionsDataset,
                 transform=None,
                 # target_transform=None,
                 ):
        self.structure_reflections_dataset = structure_reflections_dataset
        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.structure_reflections_dataset.data)

    def __getitem__(self, idx):
        data = self.structure_reflections_dataset.data[idx]

        # reflections_path = data.pdb_path
        # structure_path = data.mtz_path
        # ligands = data.ligands

        image, label = self.transform(data)

        return image, label
