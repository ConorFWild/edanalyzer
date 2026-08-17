import time
from pathlib import Path
import dataclasses
from typing import Any
import sqlite3
import os

import fire
import yaml
from rich import print as rprint
import pandas as pd
import numpy as np
import zarr
from rdkit import Chem
from rdkit.Chem import AllChem

from edanalyzer import constants
from edanalyzer.data.database import _get_system_from_dtag
from edanalyzer.shell import PANDDA_JOB_SCRIPT, PANDDA_SUBMIT_COMMAND, submit_script

from edanalyzer.data.event_data import (
    comment_dtype,
    _make_z_map_sample_metadata_table,
    _make_z_map_sample_table,
    _make_ligand_data_table,
    _make_known_hit_pose_table,
    _make_annotation_table,
    _make_comment_table,
    _get_closest_res_from_dataset_dir,
    _get_z_map_sample_from_dataset_dir,
    _get_pose_sample_from_dataset_dir,
    _get_ligand_data_sample_from_dataset_dir,
    _get_annotation_sample_from_dataset_dir,
    _get_z_map_metadata_sample_from_dataset_dir,
    _make_xmap_sample_table,
    _get_xmap_sample_from_dataset_dir,
    _get_most_recent_modelled_structure_from_dataset_dir,
    get_closest_res_from_st_path,
    valid_smiles_dtype,
    ligand_conf_dtype
)

LABXCHEM_DATA_PATH = Path('/dls/labxchem/data')

def try_exists(path):
    try:
        if path.exists():
            return True
        else: 
            return False
    except:
        return False

@dataclasses.dataclass
class Tables:
    z_map_sample_metadata_table: Any
    z_map_sample_table: Any
    xmap_sample_table: Any
    ligand_data_table: Any
    known_hit_pose_table: Any
    annotation_table: Any
    comment_table: Any
    valid_smiles_group: Any
    mol_conf_table: Any

@dataclasses.dataclass
class Idxs:
    z_map_sample_metadata_idx: int 
    idx_pose : int
    idx_ligand_data : int
    annotation_idx : int

    def __repr__(self):
        return f'Idxs: z_map_sample - {self.z_map_sample_metadata_idx}; pose - {self.idx_pose}; ligand_data - {self.idx_ligand_data}; annotation - {self.annotation_idx}'


def process_pandda_event(
        tables, 
        pandda_dir, 
        system, 
        dtag, 
        event_idx, 
        size, 
        high_resolution, 
        conf, 
        initial_x, 
        initial_y, 
        initial_z, 
        comment,
        test_systems,
        idxs,
        st_path=None,
        dry=False
        ):
    
    dataset_dir = pandda_dir / 'processed_datasets' / dtag

    # processed_dataset_yaml = dataset_dir / 'processed_dataset.yaml'
    # with open(processed_dataset_yaml, 'r') as f:
    #     processed_dataset = yaml.safe_load(f)

    # selected_model = processed_dataset['Summary']['Selected Model']


    # events_yaml = dataset_dir / 'events.yaml'
    # if not events_yaml.exists():
    #     print(f'\t\t\tSKIPPING EVENT: No event yaml at {events_yaml}')
    #     return
    # with open(events_yaml, 'r') as f:
    #     events = yaml.safe_load(f)

    # selected_model = processed_dataset['Summary']['Selected Model']

    # event_distances = {}
    # event_centroids = {}
    # for event_num, event in events.items():
    #     event_centroid = event['Centroid']
    #     distance = np.linalg.norm(np.array(event_centroid) - np.array([initial_x, initial_y, initial_z]))
    #     event_distances[event_num] = distance
    #     event_centroids[event_num] = event_centroid

    # if len(event_distances) > 0:
    #     closest_event_id = min(event_centroids, key=lambda _event_num: event_distances[_event_num])
    #     x, y, z = event_centroids[closest_event_id]
    #     rprint(f'\t\t\tClosest event distance is {round(event_distances[closest_event_id], 2)}')


    # else:
    #     rprint(
    #         f'\t\t\tSKIPPING EVENT: Could not match high confidence ligand {dtag} {event_idx} to an initial event!\n'
    #         # f'\t\t\tCheck model in {dataset_dir} is appropriate!\n'
    #         # 'SKIPPING!'
    #     )
    #     return

    x, y, z = initial_x, initial_y, initial_z


    model_dir = dataset_dir / 'modelled_structures'

    # Get the corresponding residue
    try:
        if st_path is None:
            st_path = _get_most_recent_modelled_structure_from_dataset_dir(dataset_dir)

        resid, res, dist = get_closest_res_from_st_path(
            st_path,
            x, y, z
        )
    except Exception as e:
        print(e)
        print('\t\t\tSKIPPING EVENT: Couldn\'t match res! Skipping!')
        return
    if (conf == 'High') & (dist > 6.0):
        rprint(
            f'\t\t\tSKIPPING EVENT: Could not match high confidence ligand {dtag} {event_idx} to a build!\n'
            # f'Check model in {dataset_dir} is appropriate!\n'
            # 'SKIPPING!'
        )
        # raise Exception
        return

    # Get the z map sample
    z_map_sample = _get_z_map_sample_from_dataset_dir(
        dataset_dir,
        x, y, z,
        idxs.z_map_sample_metadata_idx,
    )
    try:
        xmap_sample = _get_xmap_sample_from_dataset_dir(
            dataset_dir,
            x, y, z,
            idxs.z_map_sample_metadata_idx,
        )
    except:
        return 
    if conf == 'High':
        # Get the pose sample
        pose_sample = _get_pose_sample_from_dataset_dir(
            model_dir,
            res,
            x, y, z,
            idxs.idx_pose
        )

    # Get the ligand data
    ligand_data_sample = _get_ligand_data_sample_from_dataset_dir(
        dataset_dir,
        res,
        idxs.idx_ligand_data,
    )
    if not ligand_data_sample:
        rprint(f'\t\t\tSKIPPING EVENT: NO LIGAND DATA! SKIPPING!')
        return

    return True




def process_model_building_dir(model_building_dir, sqlite_path, pandda_dirs, tables, test_systems, known_datasets, dry=False):
    # Get the known good models from sqlite
    try:
        if not sqlite_path.exists():
            print(f'\tSKIPPING SYSYEM: Could not read sqlite table! ')
            return
    except:
        print(f'\tSKIPPING SYSYEM: Could not read sqlite table! ')

        return
    conn = sqlite3.connect(sqlite_path)
    try:
        table = pd.read_sql('select * from mainTable', conn)
    except:
        print(f'\tSKIPPING SYSYEM: Could not read sqlite table!')
        return
    if 'RefinementOutcome' not in table.columns:
        print(f'\tSKIPPING SYSYEM: Could not read sqlite table!')
        return

    good_refinements = table[table['RefinementOutcome'].isin(['4 - CompChem ready', '5 - Deposition ready', '6 - Deposited'])]

    good_refinement_datasets = [x for x in good_refinements['CrystalName'].values]
    print(f'\tGot {len(good_refinement_datasets)} datasets with good refinement outcomes!')

    # Get the PanDDA CSVs
    pandda_tables = {}
    for pandda_dir in pandda_dirs:
        inspect_path = pandda_dir / 'analyses' / 'pandda_inspect_events.csv'
        if not inspect_path.exists():
            # print(f'\tSKIPPING PANDDA: No inspect table at {inspect_path}! ')
            continue
        try:
            pandda_tables[pandda_dir] = pd.read_csv(inspect_path)
        except Exception as e:
            continue
    print(f'\tGot {len(pandda_tables)} valid PanDDAs tables!')

    if len(pandda_tables) == 0:
        print(f'\tSKIPPING SYSTEM: No PanDDA tables!')
        return 

    # Get the idxs
    idxs = Idxs(
        z_map_sample_metadata_idx = len(tables.z_map_sample_metadata_table),
        idx_pose = len(tables.known_hit_pose_table),
        idx_ligand_data = len(tables.ligand_data_table),
        annotation_idx = len(tables.annotation_table)    
    )
    print(f'\t{idxs}')

    # Match each to a high confidence PanDDA event
    new_good_refinement_datasets = [dtag for dtag in good_refinement_datasets if dtag not in known_datasets]
    print(f'\tGot {len(new_good_refinement_datasets)} new datasets with good refinements!')
    if len(new_good_refinement_datasets) == 0:
        print(f'\tSKIPPING SYSTEM: No new good refinement datasets!')
        return 
    datasets_to_reprocess = []
    for dtag in new_good_refinement_datasets:
        if dtag in known_datasets:
            continue

        # Get refined pdb 
        if not isinstance(dtag, str):
            continue
        print(f'\tProcessing dataset: {dtag}')

        refined_pdb_path = model_building_dir / str(dtag) / 'refine.pdb'
        if not refined_pdb_path.exists():
            print(f'\t\tSKIPPING DATASET: No refined pdb path at {refined_pdb_path}')
            continue

        # Check which PanDDA tables have a high conf event
        valid_panddas = [x for x in pandda_tables if len(pandda_tables[x][(pandda_tables[x]['Ligand Confidence'] == 'High') & (pandda_tables[x]['dtag'] == dtag)]) > 0]

        if len(valid_panddas) == 0:
            # print(f'\t\tSKIPPING DATASET: No PanDDAs high conf events for {dtag} in {[x for x in pandda_tables]}')
            print(f'\t\tADDING TO REPROCESS: No PanDDA high conf events')
            datasets_to_reprocess.append(dtag)
            continue

        # Try valid PanDDAs until one produces valid data
        processed = False
        for pandda_dir in valid_panddas:
            # Only need to process one valid PanDDA - skip others!
            if processed:
                continue
            inspect_table = pandda_tables[pandda_dir]
            high_conf_table = inspect_table[(inspect_table['Ligand Confidence'] == 'High') & (inspect_table['dtag'] == dtag)]
            if len(high_conf_table) == 0:
                continue
            dtag_table = inspect_table[inspect_table['dtag'] == dtag]

            # Try events until one works
            for _idx, _row in high_conf_table.iterrows():
                if processed:
                    continue
        
                # Unpack the row information
                try:
                    dtag, event_idx, bdc, conf, viewed, size, high_resolution, comment, x, y, z = (
                        _row['dtag'], 
                        _row['event_idx'], 
                        _row['1-BDC'], 
                        _row[constants.PANDDA_INSPECT_HIT_CONDFIDENCE], 
                        _row[constants.PANDDA_INSPECT_VIEWED], 
                        _row[constants.PANDDA_INSPECT_CLUSTER_SIZE],
                        _row['high_resolution'],
                        _row['Comment'],
                        _row['x'], _row['y'], _row['z']
                    )
                except:
                    continue
        
                system = _get_system_from_dtag(dtag)
        
                # rprint(f'\t\tProcessing event: {dtag} {event_idx} {conf}')
        
                if not viewed:
                    # rprint('\t\t\tNot Viewed! Skipping!')
                    continue

                processed = process_pandda_event(
                    tables, 
                    pandda_dir, 
                    system, 
                    dtag, 
                    event_idx, 
                    size, 
                    high_resolution, 
                    conf, 
                    x, 
                    y, 
                    z, 
                    comment,
                    test_systems,
                    idxs,
                    st_path=refined_pdb_path,
                    dry=dry
                ) 

        if not processed:
            print(f'\t\tADDING TO REPROCESS: No valid PanDDA')
            datasets_to_reprocess.append(dtag)
        
        ...
    return datasets_to_reprocess
        

    ...

def add_valid_smiles(tables: Tables, dry=False):
    df = pd.DataFrame(
        tables.ligand_data_table.get_basic_selection(slice(None), fields=['idx', 'canonical_smiles', ]))

    unique_smiles_series = df['canonical_smiles'].unique()

    valid_smiles_table = pd.DataFrame(
        tables.valid_smiles_group.get_basic_selection(slice(None), fields=['idx', 'valid']))
    print(f'Got {len(valid_smiles_table)} valid smiles')

    smiles_validity = {}
    for j in range(len(valid_smiles_table)):
        ligand_data = df.iloc[j]
        valid = valid_smiles_table.iloc[j]
        if valid['valid']:
            smiles_validity[ligand_data['canonical_smiles']] = True
        else:
            smiles_validity[ligand_data['canonical_smiles']] = False

    print(f'Got {len(smiles_validity)} valid smiles')

    for idx, smiles in enumerate(unique_smiles_series):
        if smiles in smiles_validity:
            continue
        print(f'{idx}/{len(unique_smiles_series)} : {smiles}')
        try:
            m = Chem.MolFromSmiles(smiles)
            m2 = Chem.AddHs(m)
            cids = AllChem.EmbedMultipleConfs(m2, numConfs=10)
            m3 = Chem.RemoveHs(m2)
            embedding = [_conf.GetPositions() for _conf in m3.GetConformers()][0]

            smiles_validity[smiles] = True
        except Exception as e:
            print(e)
            smiles_validity[smiles] = False

    for _idx, _row in df.iterrows():
        if _idx in valid_smiles_table['idx']:
            continue
        smiles = _row['canonical_smiles']
        if smiles_validity[smiles]:
            if not dry:
                tables.valid_smiles_group.append(
                    np.array(
                        [(_idx, True)],
                        dtype=valid_smiles_dtype
                    )
                )
        else:
            if not dry:
                tables.valid_smiles_group.append(
                    np.array(
                        [(_idx, False)],
                        dtype=valid_smiles_dtype
                    )
                )


def add_molecule_conformations(tables: Tables, dry=False):
    mol_conf_table = pd.DataFrame(
        tables.mol_conf_table.get_basic_selection(slice(None), fields=['idx', 'ligand_data_idx', ]))
    mol_conf_idx = len(mol_conf_table)
    
    # for _ligand_data in ligand_data_table:
    z_map_sample_metadata_table = tables.z_map_sample_metadata_table
    
    high_conf = z_map_sample_metadata_table.get_mask_selection(z_map_sample_metadata_table['Confidence'] == ['High'])
    num = len(high_conf)
    for j, _z_map_sample_metadata in enumerate(high_conf):
        print(f'{j} / {num}')
        if _z_map_sample_metadata['Confidence'] != 'High':
            continue

        if _z_map_sample_metadata['ligand_data_idx'] in mol_conf_table['ligand_data_idx']:
            continue
    
        _ligand_data = tables.ligand_data_table[_z_map_sample_metadata['ligand_data_idx']]
    
        m = Chem.MolFromSmiles(_ligand_data['canonical_smiles'])
    
        # 2.c.
        m2 = Chem.AddHs(m)
        cids = AllChem.EmbedMultipleConfs(m2, numConfs=50)
        # print(f'Got {len(cids)} embeddings')
        m3 = Chem.RemoveHs(m2)

    
        # 2.e.2
        for embedding in [_conf.GetPositions() for _conf in m3.GetConformers()]:
            # 2.e.2.a.
            # fragment_smiles = Chem.MolToSmiles(_frag)
            # print(f'Fragment Smiles: {fragment_smiles}')
    
            #
            # print(f'Embedding Size: {embedding.shape[0]}')
            poss = np.zeros((150, 3))
            poss[:embedding.shape[0], :] = embedding[:, :]
            mol_els = np.array(
                [m2.GetAtomWithIdx(_atom_idx).GetAtomicNum() for _atom_idx in [a.GetIdx() for a in m2.GetAtoms()]])
            els = np.zeros(150)
            els[:len(mol_els)] = mol_els[:]
    
            # 2. e. 2. b.
            record = np.array([(
                mol_conf_idx,
                _ligand_data['idx'],
                len(mol_els),
                _ligand_data['canonical_smiles'],
                _ligand_data['canonical_smiles'],
                poss,
                els
            )],
                dtype=ligand_conf_dtype)
            # print(record)
    
            #
            if not dry:
                tables.mol_conf_group.append(record)
            mol_conf_idx += 1


def _make_valid_smiles_table(group):
    table_name = 'valid_smiles'
    if table_name in [x for x in group.keys()]:
        return group[table_name]

    valid_smiles_group = group.create_dataset(
        'valid_smiles',
        shape=(0,),
        chunks=(1,),
        dtype=valid_smiles_dtype
    )
    return valid_smiles_group


def _make_mol_conf_table(group):
    table_name = 'valid_smiles'
    if table_name in [x for x in group.keys()]:
        return group[table_name]
    
    mol_conf_group = group.create_dataset(
            'ligand_confs',
            shape=(0,),
            chunks=(1,),
            dtype=ligand_conf_dtype
        )
    return mol_conf_group



def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_systems = config['test']['test_systems']

    # Open a file in "w"rite mode
    zarr_path = config['collate']['zarr']
    print(f'Updating zarr archive: {zarr_path}')
    root = zarr.open(zarr_path, mode='a')

    # Create the tables in the archive
    # try:
    #     del root['pandda_2']
    # except:
    #     rprint(f'No PanDDA 2 group!')
    if 'pandda_2' in [x for x in root.keys()]:
        pandda_2_group = root['pandda_2']
    else:
        pandda_2_group = root.create_group('pandda_2')

    tables = Tables(
        z_map_sample_metadata_table = _make_z_map_sample_metadata_table(pandda_2_group),
        z_map_sample_table = _make_z_map_sample_table(pandda_2_group),
        xmap_sample_table = _make_xmap_sample_table(pandda_2_group),
        ligand_data_table = _make_ligand_data_table(pandda_2_group),
        known_hit_pose_table = _make_known_hit_pose_table(pandda_2_group),
        annotation_table = _make_annotation_table(pandda_2_group),
        comment_table = _make_comment_table(pandda_2_group),
        valid_smiles_group = _make_valid_smiles_table(pandda_2_group),
        mol_conf_table = _make_mol_conf_table(pandda_2_group),
    )

    # PanDDA 2 events
    rprint(f"Querying events...")
    metadata_table = pd.DataFrame(tables.z_map_sample_metadata_table[:])
    print(f'Got {len(metadata_table)} known events')
    known_datasets = {x for x in metadata_table.loc[metadata_table.index, ['dtag']].values.flatten()}
    print(f'Got {len(known_datasets)} known datasets in zarr archive')
    # print(known_datasets)

    new_pandda_dirs = []
    failed = []

    # Loop over PanDDA directories
    # for pandda_dir in Path('/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/pandda_new_score/panddas_new_score/').glob('*'):
    #     process_pandda_dir(pandda_dir, tables, test_systems, known_datasets)
    for visit_dir in LABXCHEM_DATA_PATH.glob('*'):
        if visit_dir.parts[-1][:2] in ['sw', 'in']:  # SKIP INDUSTRIAL VISITS!
            continue
        for subvisit_dir in visit_dir.glob('*'):
            processing_dir = subvisit_dir / 'processing'
            try:
                if not processing_dir.exists():
                    print(f'\tSKIPPING VISIT: No processing dir!')
                    continue
            except:
                print(f'\tSKIPPING VISIT: No processing dir!')
                continue

            analysis_dir = processing_dir / 'analysis'
            if not analysis_dir.exists():
                print(f'SKIPPING VISIT: No analysis dir!')
                continue

            model_building_dir = analysis_dir / 'model_building'
            if not model_building_dir.exists():
                print(f'\tSKIPPING VISIT: No model building dir!')

            # Get the sqlite path
            sqlite_path = processing_dir / 'database' / 'soakDBDataFile.sqlite'
            try:
                if not sqlite_path.exists():
                    print(f'\tSKIPPING VISIT: No sqlite!')
                    continue
            except:
                print(f'\tSKIPPING VISIT: No sqlite!')
                continue

            # Get the pandda dirs
            pandda_dirs = []
            for potential_pandda_dir in analysis_dir.glob('*'):
                if try_exists(potential_pandda_dir / 'analyses' / 'pandda_inspect_events.csv'):
                    pandda_dirs.append(potential_pandda_dir)
                else:
                    for potential_pandda_dir_2 in potential_pandda_dir.glob("*"):
                        if try_exists(potential_pandda_dir_2 / 'analyses' / 'pandda_inspect_events.csv'):
                            pandda_dirs.append(potential_pandda_dir_2)

            print(f'\tGot {len(pandda_dirs)} PanDDA Dirs: {pandda_dirs}')
            if len(pandda_dirs) == 0:
                print(f'\tSKIPPING VISIT: No PanDDA dirs!')
                continue    

            # Process the model building dir
            datasets_to_reprocess = process_model_building_dir(
                model_building_dir, 
                sqlite_path, 
                pandda_dirs, 
                tables, 
                test_systems,
                known_datasets,
                dry=False
            )
            print(f'\tDatasets to reprocess: {datasets_to_reprocess}')
            # Create the job script
            if datasets_to_reprocess is None:
                continue
            new_pandda_dir = analysis_dir / 'REPROCESSED_pandda_2'
            try:
                os.mkdir(new_pandda_dir)
            except:
                ...

            if not new_pandda_dir.exists():
                print(f'COULD NOT REPROCESS TO: {new_pandda_dir}')
                failed.append(new_pandda_dir)
            job_script = PANDDA_JOB_SCRIPT.format(
                num_cpus=36,
                data_dirs=model_building_dir,
                out_dir=new_pandda_dir,
                only_datasets=",".join(datasets_to_reprocess)
            )
            rprint(f"Job Script")
            rprint(job_script)
            # exit()

            # Create the submission command
            if not (new_pandda_dir / 'pandda2.o').exists():

                submit_script(
                    job_script,
                    new_pandda_dir,
                    script_name=f"pandda2",
                )

            new_pandda_dirs.append(new_pandda_dir)


    print(f'Submitted!')
    for new_pandda_dir in new_pandda_dirs:
        print(new_pandda_dir)

    print(f'Failed!')
    for f in failed:
        print(f)

    # add_valid_smiles(tables, dry=False)
    # add_molecule_conformations(tables, dry=True)

    

    print(f'Generating ligand confs...')
    
    


if __name__ == "__main__":
    fire.Fire(main)
