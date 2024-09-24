import os
import pickle
import time
from pathlib import Path
import dataclasses
import typing

import yaml
import fire
import pony
from rich import print as rprint
import zarr
from rich.traceback import install
install(show_locals=True)

import pandas as pd
import gemmi
import numpy as np

import tables

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer import constants
from edanalyzer.utils import try_make, try_link

COMPARATORS_DIR = Path(
    '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score/')
COMPARATOR_DATASETS = {
    "BAZ2BA": COMPARATORS_DIR / 'BAZ2BA',
    "MRE11AA": COMPARATORS_DIR / 'MRE11AA',
    "Zika_NS3A": COMPARATORS_DIR / 'Zika_NS3A',
    "DCLRE1CA": COMPARATORS_DIR / 'DCLRE1CA',
    "GluN1N2A": COMPARATORS_DIR / 'GluN1N2A',
    "Tif6": COMPARATORS_DIR / 'Tif6',
    "AURA": COMPARATORS_DIR / 'AURA',
    "A71EV2A": COMPARATORS_DIR / 'A71EV2A',
    "SETDB1": COMPARATORS_DIR / 'SETDB1',
    "JMJD2DA": COMPARATORS_DIR / 'JMJD2DA',
    "PTP1B": COMPARATORS_DIR / 'PTP1B',
    "BRD1A": COMPARATORS_DIR / 'BRD1A',
    "PP1": COMPARATORS_DIR / 'PP1',
    "PWWP_C64S": COMPARATORS_DIR / 'PWWP_C64S',
    "NS3Hel": COMPARATORS_DIR / 'NS3Hel',
    "NSP16": COMPARATORS_DIR / 'NSP16',
}
PANDDAS_NEW_SCORE_DIR = Path(
    '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/test_systems_panddas_new_event_score_5')
PANDDAS_REANNOTATION_DIR = Path(
    '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/reannotation_pandda')
PANDDA_MODEL_FILE = "{dtag}-pandda-model.pdb"


@dataclasses.dataclass
class Event:
    event_map: typing.Callable[..., typing.Any]
    bdc: float
    row: dict


@dataclasses.dataclass
class Dataset:
    structure: typing.Callable[..., typing.Any]
    reflections: typing.Callable[..., typing.Any]
    z_map: typing.Callable[..., typing.Any]
    ligand_files: typing.Callable[..., typing.Any]
    events: dict[int, Event]
    meta: dict


def _get_ligand_files(ligand_cif_file, ligand_pdb_file):
    return ligand_cif_file, ligand_pdb_file

def _get_structure(mov_panddas_path, dtag, ):
    return gemmi.read_structure(
        str(mov_panddas_path / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(
            dtag=dtag)))


def _get_reflections(mov_panddas_path, dtag, ):
    return gemmi.read_mtz_file(
        str(mov_panddas_path / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
            dtag=dtag)))


def _get_z_map(mov_panddas_path, dtag, model_id):
    m = gemmi.read_ccp4_map(
        str(mov_panddas_path / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / "model_maps" / f'{model_id}_z.ccp4'))
    m.setup(0.0)
    return m.grid


def _get_event_map(mov_panddas_path, dtag, model_id, bdc):
    m_ground = gemmi.read_ccp4_map(
        str(mov_panddas_path / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / "model_maps" / f'{model_id}_mean.ccp4'))
    m_ground.setup(0.0)
    grid_ground = m_ground.grid
    ground_np = np.array(grid_ground, copy=False)

    m_xmap = gemmi.read_ccp4_map(str(mov_panddas_path / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / 'xmap.ccp4'))
    m_xmap.setup(0.0)
    grid_xmap = m_xmap.grid
    xmap_np = np.array(grid_xmap, copy=False)

    xmap_np_copy = xmap_np.copy()

    xmap_np[:, :, :] = (xmap_np_copy[:, :, :] - (bdc * ground_np[:, :, :])) / (1 - bdc)

    return grid_xmap


def _save_xmap(xmap, path):
    # Write the event map
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = xmap
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))


def _save_structure(st, path):
    st.write_pdb(str(path))


def _save_reflections(ref, path):
    ref.write_to_file(str(path))


# def _get_event_map_from_event(ground_state_path, xmap_path, bdc):
#     ...


def _make_dataset_dir(processed_datasets_dir, dtag, dataset: Dataset):
    rprint(f'Making dir: {processed_datasets_dir / dtag}')
    try_make(processed_datasets_dir/ dtag)

    # Save the structure
    _save_structure(dataset.structure(),
                    processed_datasets_dir/ dtag / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=dtag))

    # Save the reflections
    _save_reflections(dataset.reflections(),
                      processed_datasets_dir/ dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(dtag=dtag))

    # Get ligand files
    ligand_cif_file, ligand_pdb_file = dataset.ligand_files()
    try_link(ligand_cif_file, processed_datasets_dir / dtag / 'ligand_files' / Path(ligand_cif_file).name)
    try_link(ligand_pdb_file, processed_datasets_dir / dtag / 'ligand_files' / Path(ligand_pdb_file).name)

    # Save the zmap
    zmap = dataset.z_map()
    _save_xmap(zmap, processed_datasets_dir / dtag / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=f"{dtag}"))

    # Save the event maps
    for event_id, event in dataset.events.items():
        event_map = event.event_map()
        _save_xmap(event_map, processed_datasets_dir / dtag / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
            dtag=f"{dtag}",
            event_idx=str(event_id),
            bdc=f"{event.bdc}"
        ))

    # Store the metadata
    with open(processed_datasets_dir/ dtag / 'meta.yaml', 'w') as f:
        yaml.dump(dataset.meta, f)


def _make_event_table(datasets: dict[str, Dataset], path):
    event_rows = []
    for dtag, dataset in datasets.items():
        for event_idx, event in dataset.events.items():
            event_rows.append(event.row)

    event_table = pd.DataFrame(event_rows)

    event_table.to_csv(path, index=False)


def _make_site_table(datasets: dict[str, Dataset], path):
    # Spoof the site table
    site_records = []
    # num_sites = ((j) // 100) + 1
    num_sites = 1+int((len(datasets)-1) / 200)
    print(f"Num sites is: {num_sites}")
    for site_id in np.arange(1, num_sites + 1):
        site_records.append(
            {
                "site_idx": int(site_id),
                "centroid": (0.0, 0.0, 0.0),
                "Name": None,
                "Comment": None
            }
        )
    print(len(site_records))
    site_table = pd.DataFrame(site_records)
    site_table.to_csv(path, index=False)


def make_pandda(path: Path, datasets: dict[str, Dataset]):
    # Make the directories
    analyses_dir = path / "analyses"
    processed_datasets_dir = path / "processed_datasets"
    analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
    analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"

    # Spoof the main directories
    try_make(path)
    try_make(analyses_dir)
    try_make(processed_datasets_dir)

    # Make the dataset directories
    for dtag, dataset in datasets.items():
        _make_dataset_dir(processed_datasets_dir, dtag, dataset)

    # Make the event table
    _make_event_table(datasets, analyse_table_path)

    # Make the site table
    _make_site_table(datasets, analyse_site_table_path)


def get_ligand_array(res):
    poss = []
    for atom in res:
        if atom.element.name != 'H':
            pos = atom.pos
            poss.append([pos.x, pos.y, pos.z])
    return np.array(poss)


def get_centroid(res):
    return np.mean(get_ligand_array(res), axis=0)


def get_ref_event_build(mov_event, mov_panddas_path):
    mov_dtag, mov_x, mov_y, mov_z = mov_event['dtag'], mov_event['x'], mov_event['y'], mov_event['z']
    st_file = mov_panddas_path / 'processed_datasets' / mov_dtag / 'modelled_structures' / PANDDA_MODEL_FILE.format(
        dtag=mov_dtag)
    st = gemmi.read_structure(str(st_file))
    ligands = {}
    for model in st:
        for chain in model:
            for res in chain:
                if (res.name == 'LIG') or (res.name == 'XXX'):
                    ligands[(chain.name, res.seqid.num)] = res

    if len(ligands) == 0:
        return None

    centroids = {idx: get_centroid(res) for idx, res in ligands.items()}

    distances = {idx: np.linalg.norm(centroid - np.array([mov_x, mov_y, mov_z])) for idx, centroid in centroids.items()}
    return ligands[min(ligands, key=lambda _x: distances[_x])]


def get_build_rmsd(mov_build, ref_build):
    ref_build_array = get_ligand_array(ref_build)

    deltas = []
    for mov_atom in mov_build:
        if mov_atom.element.name == 'H':
            continue
        # ref_atom = ref_build[mov_atom.name][0]
        mov_pos = mov_atom.pos
        # ref_pos = ref_atom.pos
        # delta = np.array([ref_pos.x - mov_pos.x, ref_pos.y - mov_pos.y, ref_pos.z - mov_pos.z, ])
        pos = np.array([mov_pos.x, mov_pos.y, mov_pos.z]).reshape((1, 3))
        distances = np.linalg.norm(pos - ref_build_array, axis=1)

        deltas.append(min(distances))
    rmsd = np.sqrt(np.sum(np.square(deltas)) / len(deltas))
    return rmsd


def get_comparator_data():
    records = []
    for mov_panddas_path in PANDDAS_NEW_SCORE_DIR.glob('*'):

        if not mov_panddas_path.is_dir():
            continue

        print(mov_panddas_path.name)

        ref_panddas_path = COMPARATORS_DIR / COMPARATOR_DATASETS[mov_panddas_path.name]

        if not (mov_panddas_path / 'analyses/pandda_analyse_events.csv').exists():
            continue

        mov_event_table = pd.read_csv(mov_panddas_path / 'analyses/pandda_analyse_events.csv')

        ref_event_table = pd.read_csv(ref_panddas_path / 'analyses/pandda_inspect_events.csv')
        ref_high_conf_event_table = ref_event_table[ref_event_table['Ligand Confidence'] == 'High']
        # records = []
        for idx, ref_event in ref_high_conf_event_table.iterrows():
            ref_dtag, ref_x, ref_y, ref_z = ref_event['dtag'], ref_event['x'], ref_event['y'], ref_event['z']
            print(f'\t{ref_dtag}')

            #
            ref_event_build = get_ref_event_build(ref_event, ref_panddas_path)
            ref_centroid = get_centroid(ref_event_build)

            #
            with open(mov_panddas_path / 'processed_datasets' / ref_event['dtag'] / 'processed_dataset.yaml', 'r') as f:
                processed_dataset = yaml.safe_load(f)

            # Get closes events
            distances = {}
            events = {}
            matched = []
            for model_id, model in processed_dataset['Models'].items():
                for event_id, mov_event in model['Events'].items():
                    mov_x, mov_y, mov_z = mov_event['Centroid']
                    distance = np.linalg.norm(np.array([mov_x - ref_x, mov_y - ref_y, mov_z - ref_z, ]))

                    if distance < 6.0:
                        distances[(model_id, event_id)] = distance
                        events[(model_id, event_id)] = mov_event

                        # Get RMSDS to close event builds
                        try:
                            rmsd = get_build_rmsd(gemmi.read_structure(mov_event['Build Path'])[0][0][0],
                                                  ref_event_build, )

                        except Exception as e:
                            print(e)
                            rmsd = None
                        #
                        records.append(
                            {
                                'system': mov_panddas_path.name,
                                'dtag': ref_event['dtag'],
                                'ref_event_id': ref_event['event_idx'],
                                'cent_x': mov_x,
                                'cent_y': mov_y,
                                'cent_z': mov_z,
                                "bdc": mov_event['BDC'],
                                # 'cent_x': ref_centroid[0],
                                # 'cent_y': ref_centroid[1],
                                # 'cent_z': ref_centroid[2],
                                'model_id': model_id,
                                'mov_event_id': event_id,
                                'event_score': mov_event['Score'],
                                'closest_event_distance': np.linalg.norm(
                                    np.array([ref_x, ref_y, ref_z]) - mov_event['Centroid']),
                                'build_score': mov_event['Build Score'],
                                'build_rmsd': rmsd,
                                'centroid_x': ref_centroid[0],
                                'centroid_y': ref_centroid[1],
                                'centroid_z': ref_centroid[2],
                                'spurious': False,
                                'mov_panddas_path': mov_panddas_path

                            }
                        )
                        matched.append((model_id, event_id))

            for model_id, model in processed_dataset['Models'].items():
                for event_id, mov_event in model['Events'].items():
                    key = (model_id, event_id)
                    if key not in matched:
                        mov_x, mov_y, mov_z = mov_event['Centroid']
                        distance = np.linalg.norm(np.array([mov_x - ref_x, mov_y - ref_y, mov_z - ref_z, ]))
                        ligand_dir = mov_panddas_path / 'processed_datasets' / ref_event['dtag'] / 'ligand_file'
                        records.append(
                            {
                                'system': mov_panddas_path.name,
                                'dtag': ref_event['dtag'],
                                'ref_event_id': ref_event['event_idx'],
                                'cent_x': mov_x,
                                'cent_y': mov_y,
                                'cent_z': mov_z,
                                "bdc": mov_event['BDC'],
                                # 'cent_x': ref_centroid[0],
                                # 'cent_y': ref_centroid[1],
                                # 'cent_z': ref_centroid[2],
                                'model_id': model_id,
                                'mov_event_id': event_id,
                                'event_score': mov_event['Score'],
                                'closest_event_distance': np.linalg.norm(
                                    np.array([ref_x, ref_y, ref_z]) - mov_event['Centroid']),
                                'build_score': mov_event['Build Score'],
                                'build_rmsd': None,
                                'centroid_x': ref_centroid[0],
                                'centroid_y': ref_centroid[1],
                                'centroid_z': ref_centroid[2],
                                'spurious': True,
                                'mov_panddas_path': mov_panddas_path,
                                'autobuild_path': mov_event['Build Path'],
                                'ligand_cif_file': ligand_dir / mov_event['Ligand Key'],
                                'ligand_pdb_file': ligand_dir / mov_event['Ligand Key'],

                            }
                        )

    df = pd.DataFrame(records)
    print(df)

    # Filter to the high scoring spurious events
    highest_scoring_build_df = df.loc[df.groupby(['dtag', ])['build_score'].idxmax()]
    spurious_highest_scoring_build_df = highest_scoring_build_df[highest_scoring_build_df['spurious']]
    print(spurious_highest_scoring_build_df)

    # Get the dataset
    datasets = {}
    j = 0
    for _idx, _row in spurious_highest_scoring_build_df.iterrows():
        event_row = {
            "dtag": _row['dtag'],
            "event_idx": 1,
            "1-BDC": _row['bdc'],
            "cluster_size": 1,
            "global_correlation_to_average_map": 0,
            "global_correlation_to_mean_map": 0,
            "local_correlation_to_average_map": 0,
            "local_correlation_to_mean_map": 0,
            "site_idx": 1 + int(j / 200),
            "x": _row['cent_x'],
            "y": _row['cent_y'],
            "z": _row['cent_z'],
            "z_mean": 0.0,
            "z_peak": 0.0,
            "applied_b_factor_scaling": 0.0,
            "high_resolution": 0.0,
            "low_resolution": 0.0,
            "r_free": 0.0,
            "r_work": 0.0,
            "analysed_resolution": 0.0,
            "map_uncertainty": 0.0,
            "analysed": False,
            # "interesting": False,
            "exclude_from_z_map_analysis": False,
            "exclude_from_characterisation": False,
        }
        j += 1

        meta = {
            "dtag": _row['dtag'],
            "model_idx": _row['model_id'],
            "event_idx": _row['mov_event_id'],
            "autobuild_path": _row['autobuild_path'],
            "pandda_path": _row['mov_panddas_path']
        }

        dataset_key = f"{_row['dtag']}_{_row['model_id']}_{_row['mov_event_id']}"

        datasets[dataset_key] = Dataset(
            lambda: _get_structure(_row['mov_panddas_path'], _row['dtag'], ),
            lambda: _get_reflections(_row['mov_panddas_path'], _row['dtag'], ),
            lambda: _get_z_map(_row['mov_panddas_path'], _row['dtag'], _row['model_id']),
            lambda: _get_ligand_files(_row['ligand_cif_file'], _row['ligand_pdb_file']),
            {
                1: Event(
                    lambda: _get_event_map(_row['mov_panddas_path'], _row['dtag'], _row['model_id'], _row['bdc']),
                    round(_row['bdc'], 2),
                    event_row
                )
            },
            meta
        )
    rprint(datasets)

    # Make a annotation pandda
    make_pandda(PANDDAS_REANNOTATION_DIR, datasets)

if __name__ == '__main__':
    fire.Fire(get_comparator_data)