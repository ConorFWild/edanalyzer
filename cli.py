import fire
from pathlib import Path
import subprocess
from data import Dataset, Options, Data
import constants


def download_dataset(options: Options):
    for datatype in [
        "structures/divided/pdb",
        "structures/divided/structure_factors"
    ]:
        p = subprocess.Popen(
            constants.RSYNC_COMMAND.format(
                datatype=datatype,
                data_dir=Path(options.working_dir) / constants.DATA_DIR

            )
        )
        p.communicate()


def parse_dataset(options: Options, ):
    pdbs_dir = Path(options.working_dir) / constants.DATA_DIR / "structures" / "divided" / "pdb"
    sfs_dir = Path(options.working_dir) / constants.DATA_DIR / "structures" / "divided" / "structure_factors"

    pdbs = {}
    sfs = {}

    for entry in sfs_dir.glob("*"):
        sfs[entry.name] = entry

    for entry in pdbs_dir.glob("*"):
        if entry.name in sfs:
            pdbs[entry.name] = entry

    datas = []
    id = 0
    for entry_name, path in sfs.items():
        if entry_name in pdbs:
            pdb = pdbs[entry_name]
        else:
            continue
        dt = Data(
            id=id,
            name=entry_name,
            pdb_path=str(pdb),
            mtz_path=str(path),
            ligands=[],
            partition=0,
        )
        id += 1
        datas.append(dt)

    dataset = Dataset(datas)
    dataset.save(options.working_dir)
    # return Dataset(datas)


def generate_smiles(options: Options, dataset: Dataset):
    ...


def partition_dataset(options: Options, dataset: Dataset):
    ...


def train(options: Options, dataset: Dataset):
    ...


def test(options: Options, dataset: Dataset):
    ...


class CLI:

    def download_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        download_dataset(options)

    def parse_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        # dataset = Dataset.load(options.working_dir)
        parse_dataset(options)

    def generate_smiles(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = Dataset.load(options.working_dir)

        generate_smiles(options, dataset)

    # def generate_conformations(self):
    #     ...
    #
    # def generate_plausible_decoys(self):
    #     ...
    #
    # def generate_implausible_decoys(self):
    #     ...

    def partition_dataset(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = Dataset.load(options.working_dir)

    def train(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = Dataset.load(options.working_dir)

    def test(self, options_json_path: str = "./options.json"):
        options = Options.load(options_json_path)
        dataset = Dataset.load(options.working_dir)


if __name__ == "__main__":
    fire.Fire(CLI)
