from pathlib import Path
import fire
import yaml
import pandas as pd
import numpy as np
import gemmi

from edanalyzer import constants

annotations = {
    'PTP1B':
        {
            'PTP1B-y1608': "Two credible ligands, of which only one had been modelled"
        }
}

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

NEW_PANDDAS_DIR = Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/test_systems_panddas_new_event_score_4')


def _get_processed_dataset(ref_event, mov_panddas_path):
    with open(mov_panddas_path / 'processed_datasets' / ref_event['dtag'] / 'processed_dataset.yaml', 'r') as f:
        processed_dataset = yaml.safe_load(f)

    return processed_dataset


def match_event(ref_event, mov_event_table):
    ref_dtag, ref_x, ref_y, ref_z = ref_event['dtag'], ref_event['x'], ref_event['y'], ref_event['z']

    distances = {}
    events = {}
    for j, mov_event in mov_event_table[mov_event_table['dtag'] == ref_dtag].iterrows():
        try:
            mov_dtag, mov_x, mov_y, mov_z = mov_event['dtag'], mov_event['x'], mov_event['y'], mov_event['z']
        except:
            return None, None
        if mov_dtag != ref_dtag:
            continue
        distance = np.linalg.norm(np.array([mov_x - ref_x, mov_y - ref_y, mov_z - ref_z, ]))

        distances[j] = distance
        events[j] = mov_event

    if len(distances) != 0:
        idx = min(distances, key=lambda _x: distances[_x])
        return events[idx], distances[idx]

    return None, None


def get_centroid(res):
    poss = []
    for atom in res:
        pos = atom.pos
        poss.append([pos.x, pos.y, pos.z])

    return np.mean(np.array(poss), axis=0)


def get_build(mov_event, mov_panddas_path):
    mov_dtag, mov_x, mov_y, mov_z = mov_event['dtag'], mov_event['x'], mov_event['y'], mov_event['z']
    st_file = mov_panddas_path / 'processed_datasets' / mov_dtag / 'modelled_structures' / constants.PANDDA_MODEL_FILE.format(
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


def get_build_any_model(mov_event):
    build_path = mov_event['Build Path']
    st = gemmi.read_structure(str(build_path))
    return st[0][0][0]


def get_event_score(mov_event, mov_panddas_path):
    with open(mov_panddas_path / 'processed_datasets' / mov_event['dtag'] / 'events.yaml') as f:
        yml = yaml.safe_load(f)

    return yml[mov_event['event_idx']]['Score']


def match_event_all_models(ref_event, processed_dataset, ):
    ref_x, ref_y, ref_z = ref_event['x'], ref_event['y'], ref_event['z']
    distances = {}
    events = {}
    for model_id, model in processed_dataset['Models'].items():
        for event_id, mov_event in model['Events'].items():
            mov_x, mov_y, mov_z = mov_event['Centroid']
            distance = np.linalg.norm(np.array([mov_x - ref_x, mov_y - ref_y, mov_z - ref_z, ]))

            distances[(model_id, event_id)] = distance
            events[(model_id, event_id)] = mov_event

    if len(distances) != 0:
        close_distances = {_k: _v for _k, _v in distances.items() if _v < 6.0}
        if len(close_distances) != 0:
            idx = max(close_distances, key=lambda _x: events[_x]['Score'])
        else:
            idx = min(distances, key=lambda _x: distances[_x])
        return events[idx], distances[idx]

    return None, None


def get_build_score(mov_event, mov_panddas_path):
    with open(mov_panddas_path / 'processed_datasets' / mov_event['dtag'] / 'events.yaml') as f:
        yml = yaml.safe_load(f)

    return yml[mov_event['event_idx']]['Build']['Score']


def get_build_rmsd(mov_build, ref_build):
    deltas = []
    for mov_atom in mov_build:
        ref_atom = ref_build[mov_atom.name][0]
        mov_pos = mov_atom.pos
        ref_pos = ref_atom.pos
        delta = np.array([ref_pos.x - mov_pos.x, ref_pos.y - mov_pos.y, ref_pos.z - mov_pos.z, ])
        deltas.append(delta)
    rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(np.array(deltas), axis=1))) / len(deltas))
    return rmsd


def main(mov_panddas_path, ref_panddas_path):
    mov_panddas_path, ref_panddas_path = Path(mov_panddas_path), Path(ref_panddas_path)
    mov_event_table = pd.read_csv(mov_panddas_path / 'analyses/pandda_analyse_events.csv')
    ref_event_table = pd.read_csv(ref_panddas_path / 'analyses/pandda_inspect_events.csv')
    ref_high_conf_event_table = ref_event_table[ref_event_table['Ligand Confidence'] == 'High']
    records = []
    for idx, ref_event in ref_high_conf_event_table.iterrows():
        ref_dtag, ref_x, ref_y, ref_z = ref_event['dtag'], ref_event['x'], ref_event['y'], ref_event['z']
        print(ref_dtag)

        print(f'Centroid: {round(ref_x, 2)} {round(ref_y, 2)} {round(ref_z, 2)}')

        processed_dataset = _get_processed_dataset(ref_event, mov_panddas_path)

        mov_event, matching_event_distance = match_event(ref_event, mov_event_table)
        if mov_event is None:
            print(f'No match for {ref_event["dtag"]}')
            mov_build = None
            event_score = None

        else:
            event_score = get_event_score(mov_event, mov_panddas_path)
            mov_build = get_build(mov_event, mov_panddas_path)

        closest_event_any_model, closest_event_dist_any_model = match_event_all_models(ref_event, processed_dataset, )

        ref_build = get_build(ref_event, ref_panddas_path)

        if closest_event_any_model is not None:
            mov_build_any_model = get_build_any_model(closest_event_any_model, )
        else:
            mov_build_any_model = None

        if mov_build is None:
            build_score = None
            build_rmsd = None

        else:
            build_score = get_build_score(mov_event, mov_panddas_path)
            print(mov_build)
            print(ref_build)
            try:
                build_rmsd = get_build_rmsd(mov_build, ref_build)
            except:
                build_rmsd = None

        if closest_event_dist_any_model:
            event_score_any_model = closest_event_any_model['Score']
            build_score_any_model = closest_event_any_model['Build Score']
            try:
                build_rmsd_any_model = get_build_rmsd(mov_build_any_model, ref_build)
            except:
                build_rmsd_any_model = None
        else:
            event_score_any_model = None
            build_score_any_model = None
            build_rmsd_any_model = None

        record = {
            'dtag': ref_event['dtag'],
            'event_id': ref_event['event_id'],
            'event_score': event_score,
            'closest_event_distance': matching_event_distance,
            'build_score': build_score,
            'build_rmsd': build_rmsd,
            "delta_any_model": closest_event_dist_any_model,
            'event_score_any_model': event_score_any_model,
            'build_score_any_mode': build_score_any_model,
            'rmsd_any_model': build_rmsd_any_model
        }
        records.append(record)

    pd.DataFrame(records).to_csv(mov_panddas_path / 'reference_comparison.csv')


def all_systems():
    for dataset_dir in NEW_PANDDAS_DIR.glob('*'):
        if not dataset_dir.is_dir():
            continue

        if not (dataset_dir / 'analyses' / 'pandda_analyse_events.csv').exists():
            continue

        comparator_dir = COMPARATOR_DATASETS[dataset_dir.name]
        main(dataset_dir, comparator_dir)
    ...


if __name__ == "__main__":
    fire.Fire(all_systems)
