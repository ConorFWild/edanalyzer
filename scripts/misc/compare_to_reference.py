from pathlib import Path
import fire
import yaml
import pandas as pd
import numpy as np

from edanalyzer import constands


def match_event(ref_event, mov_event_table):
    ref_dtag, ref_x, ref_y, ref_z = ref_event['dtag'], ref_event['x'], ref_event['y'], ref_event['z']

    distances = {}
    events = {}
    for j, mov_event in mov_event_table[mov_event_table['dtag'] == ref_dtag].iterrows():
        mov_dtag, mov_x, mov_y, mov_z = mov_event['dtag'], mov_event['x'], mov_event['y'], mov_event['z']
        if mov_dtag != ref_dtag:
            continue
        distance = np.linalg.norm(np.array([mov_x-ref_x, mov_y-ref_y, mov_z-ref_z,]))

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
    st_file = mov_panddas_path / 'processed_dataset' / mov_dtag / 'modelled_structures'/ constands.PANDDA_MODEL_FILE.format(dtag=dtag)
    st = gemmi.read_structure(str(st_file))
    ligands = {}
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == 'LIG':
                    ligands[(chain.name, res.seqid.num)] = res

    if len(ligands) == 0:
        return None

    centroids = {idx: get_centroid(res) for idx, res in ligands.items()}

    distances = {idx: np.linalg.norm(centroid - np.array(mov_x, mov_y, mov_z)) for idx, centroid in centroids.items()}
    return ligands[min(ligands, key=lambda _x: distances[_x])]

def get_event_score(mov_event, mov_panddas_path):
    with open(mov_panddas_path / 'processed_datasets' / mov_event['dtag'] / 'events.yaml') as f:
        yml = yaml.safe_load(f)

    return yml[mov_event['event_idx']]['Score']

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
        delta = np.array([ref_pos.x-mov_pos.x, ref_pos.y-mov_pos.y, ref_pos.z-mov_pos.z, ])
        deltas.append(delta)
    rmsd = np.sqrt(np.sum(np.square(np.linalg.norm(np.array(deltas), axis=1))) / len(deltas))
    return rmsd

def main(mov_panddas_path, ref_panddas_path):
    mov_panddas_path, ref_panddas_path = Path(mov_panddas_path), Path(ref_panddas_path)
    mov_event_table = pd.read_csv(mov_panddas_path / 'analyses/pandda_analyse_events.csv')
    ref_event_table = pd.read_csv(ref_panddas_path / 'analyses/pandda_inspect_events.csv')
    ref_high_conf_event_table = ref_event_table[ref_event_table['Ligand Confidence'] == 'High']
    records = []
    for ref_event in ref_high_conf_event_table.iterrows():
        mov_event, matching_event_distance = match_event(ref_event, mov_event_table)
        mov_build = get_build(mov_event, mov_panddas_path)
        ref_build = get_build(ref_event, ref_panddas_path)
        event_score = get_event_score(mov_event, mov_panddas_path)
        build_score = get_build_score(mov_event, mov_panddas_path)
        build_rmsd = get_build_rmsd(mov_build, ref_build)

        record = {
            'dtag': ref_event['dtag'],
            'event_score': event_score,
            'closest_event_distance': matching_event_distance,
            'build_score': build_score,
            'build_rmsd': build_rmsd
        }
        records.append(record)

    pd.DataFrame(records).to_csv(mov_panddas_path / 'reference_comparison.csv')


if __name__ == "__main__":
    fire.Fire(main)