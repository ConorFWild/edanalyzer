from pathlib import Path

import fire
import yaml
from rich import print as rprint
import pandas as pd
import pony
import joblib
import pickle
import gemmi
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np

from edanalyzer import constants
from edanalyzer.data.database import _parse_inspect_table_row, Event, _get_system_from_dtag
from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM


def _get_inspect_tables(possible_pandda_paths):
    inspect_tables = {}
    for possible_pandda_path in possible_pandda_paths:
        analyse_table_path = possible_pandda_path / "analyses" / "pandda_inspect_events.csv"
        if analyse_table_path.exists():
            try:
                analyse_table = pd.read_csv(analyse_table_path)
                if len(analyse_table[analyse_table[constants.PANDDA_INSPECT_VIEWED] == True]) < 15:
                    continue
                if len(analyse_table[analyse_table[constants.PANDDA_INSPECT_HIT_CONDFIDENCE] == "High"]) < 2:
                    continue
                inspect_tables[possible_pandda_path] = analyse_table
            except Exception as e:
                print(f"\tERROR READING INSPECT TABLE: {analyse_table_path} : {e}")
        else:
            print(f"\tERROR READING INSPECT TABLE : {analyse_table_path} : NO SUCH TABLE!")
    return inspect_tables


def _get_events(
        inspect_tables,
        custom_annotations
):
    with pony.orm.db_session:

        systems = {}
        experiments = {}
        panddas = {}
        annotations = {}
        partitions = {}
        datasets = {}
        events = {}
        # Multiprocess PanDDAs, returning valid events for addition to the
        with joblib.Parallel(n_jobs=-1, verbose=50) as parallel:
            # j = 0
            for pandda_path, inspect_table in inspect_tables.items():
                print(f"### {pandda_path} ")

                # if j > 10:
                #     continue
                # j += 1
                pandda_events: list[Event] = parallel(
                    joblib.delayed(_parse_inspect_table_row)(
                        row,
                        pandda_path
                    )
                    for idx, row
                    in inspect_table.iterrows()
                )
                rprint(
                    f"Got {len(pandda_events)} of which {len([x for x in pandda_events if x is not None])} are not None!")

                for pandda_event in pandda_events:
                    if pandda_event:
                        dtag, event_idx = pandda_event.dtag, pandda_event.event_idx
                        system_name = _get_system_from_dtag(dtag)
                        if not system_name:
                            continue
                        if system_name in systems:
                            system = systems[system_name]
                        else:
                            system = SystemORM(
                                name=system_name,
                                experiments=[],
                                panddas=[],
                                datasets=[],
                            )
                            systems[system_name] = system

                        structure_path = Path(pandda_event.initial_structure).absolute().resolve()
                        dataset_dir_index = [j for j, part in enumerate(structure_path.parts) if part == dtag]
                        dataset_path = Path(*structure_path.parts[:dataset_dir_index[0] + 1])
                        experiment_path = dataset_path.parent
                        if experiment_path in experiments:
                            experiment = experiments[experiment_path]
                        else:
                            experiment = ExperimentORM(
                                path=str(experiment_path),
                                model_dir=str(experiment_path),
                                panddas=[],
                                system=system,
                                datasets=[]
                            )
                            experiments[experiment_path] = experiment

                        if dtag in datasets:
                            dataset = datasets[dtag]
                        else:
                            dataset = DatasetORM(
                                dtag=pandda_event.dtag,
                                path=str(dataset_path),
                                structure=str(Path(pandda_event.initial_structure).absolute().resolve()),
                                reflections=str(Path(pandda_event.initial_reflections).absolute().resolve()),
                                system=system,
                                experiment=experiment,
                                panddas=[]
                            )
                            datasets[dtag] = dataset

                        if pandda_path in panddas:
                            pandda = panddas[pandda_path]
                        else:
                            pandda = PanDDAORM(
                                path=str(pandda_path),  # *
                                events=[],
                                datasets=[dataset, ],
                                system=system,
                                experiment=experiment,
                            )
                            panddas[pandda_path] = pandda

                        if (str(pandda_path), dtag, event_idx) in custom_annotations:
                            print(
                                f"\tUpdating annotation of {(str(pandda_path), dtag, event_idx)} using custom annotation!")
                            _annotation = custom_annotations[(str(pandda_path), dtag, event_idx)]
                        else:
                            _annotation = pandda_event.annotation

                        event = EventORM(
                            dtag=pandda_event.dtag,
                            event_idx=pandda_event.event_idx,
                            x=pandda_event.x,
                            y=pandda_event.y,
                            z=pandda_event.z,
                            bdc=pandda_event.bdc,
                            initial_structure=pandda_event.initial_structure,
                            initial_reflections=pandda_event.initial_reflections,
                            structure=pandda_event.structure,
                            event_map=pandda_event.event_map,
                            z_map=pandda_event.z_map,
                            viewed=pandda_event.viewed,
                            hit_confidence=pandda_event.hit_confidence,
                            ligand=None,
                            dataset=dataset,
                            pandda=pandda,
                            annotations=[],
                            partitions=[]
                        )

                        if pandda_event.ligand:
                            ligand_orm = LigandORM(
                                path=str(pandda_event.ligand.path),
                                smiles=str(pandda_event.ligand.smiles),
                                chain=str(pandda_event.ligand.chain),
                                residue=int(pandda_event.ligand.residue),
                                num_atoms=int(pandda_event.ligand.num_atoms),
                                x=float(pandda_event.ligand.x),
                                y=float(pandda_event.ligand.y),
                                z=float(pandda_event.ligand.z),
                                event=event
                            )
                        else:
                            ligand_orm = None

                        pickled_data_dir = pandda_path / "pickled_data"
                        pandda_done = pandda_path / "pandda.done"
                        statistical_maps = pandda_path / "statistical_maps"
                        pickled_panddas_dir = pandda_path / "pickled_panddas"
                        if pickled_data_dir.exists():
                            source = "pandda_1"
                        elif pandda_done.exists():
                            source = "pandda_1"
                        elif statistical_maps.exists():
                            source = "pandda_1"
                        elif pickled_panddas_dir.exists():
                            source = "pandda_1"
                        else:
                            source = "pandda_2"
                        annotation = AnnotationORM(
                            annotation=_annotation,
                            source=source,
                            event=event
                        )
                        annotations[(pandda_path, pandda_event.dtag, pandda_event.event_idx)] = annotation

                        events[(pandda_path, pandda_event.dtag, pandda_event.event_idx)] = event

        rprint(
            f"Got {len(events)} of which {len([x for x in events.values() if x.hit_confidence == 'High'])} are high confidence!")
        rprint(systems)
        # for event_id, event in events.items():
        #     print(event)

    # return events





def _get_autobuilds(pandda_2_dir):
    processed_datasets_dir = pandda_2_dir / constants.PANDDA_PROCESSED_DATASETS_DIR
    autobuild_dir = pandda_2_dir / "autobuild"
    autobuilds = {}
    for processed_dataset_dir in processed_datasets_dir.glob("*"):
        dtag = processed_dataset_dir.name
        autobuilds[dtag] = {}
        processed_dataset_yaml = processed_dataset_dir / "processed_dataset.yaml"

        if not processed_dataset_yaml.exists():
            continue

        with open(processed_dataset_yaml, 'r') as f:
            data = yaml.safe_load(f)

        selected_model = data['Summary']['Selected Model']
        # selected_model_events = data['Summary']['Selected Model Events']

        for model, model_info in data['Models'].items():
            if model == selected_model:
                selected = True
            else:
                selected = False
            for event_idx, event_info in model_info['Events'].items():
                # if event_idx not in selected_model_events:
                #     continue

                autobuild_file = event_info['Build Path']
                autobuilds[dtag][(model, event_idx,)] = {
                    "build_path": autobuild_file,
                    "build_key": event_info['Ligand Key'],
                    'Score': event_info['Score'],
                    'Size': event_info['Size'],
                    'Local_Strength': event_info['Local Strength'],
                    'RSCC': event_info['RSCC'],
                    'Signal': event_info['Signal'],
                    'Noise': event_info['Noise'],
                    'Signal_Noise': event_info['Signal'] / event_info['Noise'],
                    'X_ligand': event_info['Ligand Centroid'][0],
                    'Y_ligand': event_info['Ligand Centroid'][1],
                    'Z_ligand': event_info['Ligand Centroid'][2],
                    'X': event_info['Centroid'][0],
                    'Y': event_info['Centroid'][1],
                    'Z': event_info['Centroid'][2],
                    'Selected': selected,
                    "BDC": event_info['BDC']
                }

    return autobuilds


def _get_pandda_2_autobuilt_structures(autobuilds):
    autobuilt_structures = {}
    for dtag, dtag_builds in autobuilds.items():
        autobuilt_structures[dtag] = {}
        for build_key, build_info in dtag_builds.items():
            autobuilt_structures[dtag][build_key] = gemmi.read_structure(build_info["build_path"])

    return autobuilt_structures


def get_ligand_cif_graph_matches(cif_path):
    # Open the cif document with gemmi
    cif = gemmi.cif.read(str(cif_path))

    key = "comp_LIG"
    try:
        cif['comp_LIG']
    except:
        key = "data_comp_XXX"

    # Find the relevant atoms loop
    atom_id_loop = list(cif[key].find_loop('_chem_comp_atom.atom_id'))
    atom_type_loop = list(cif[key].find_loop('_chem_comp_atom.type_symbol'))
    # atom_charge_loop = list(cif[key].find_loop('_chem_comp_atom.charge'))

    # Find the bonds loop
    bond_1_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_1'))
    bond_2_id_loop = list(cif[key].find_loop('_chem_comp_bond.atom_id_2'))
    bond_type_loop = list(cif[key].find_loop('_chem_comp_bond.type'))
    aromatic_bond_loop = list(cif[key].find_loop('_chem_comp_bond.aromatic'))

    # Construct the graph nodes
    G = nx.Graph()

    for atom_id, atom_type in zip(atom_id_loop, atom_type_loop):
        if atom_type == "H":
            continue
        G.add_node(atom_id, Z=atom_type)

    # Construct the graph edges
    for atom_id_1, atom_id_2 in zip(bond_1_id_loop, bond_2_id_loop):
        if atom_id_1 not in G:
            continue
        if atom_id_2 not in G:
            continue
        G.add_edge(atom_id_1, atom_id_2)

    # Get the isomorphisms
    GM = iso.GraphMatcher(G, G, node_match=iso.categorical_node_match('Z', 0))

    return [x for x in GM.isomorphisms_iter()]


def get_ligand_graphs(autobuilds, pandda_2_dir):
    ligand_graphs = {}
    for dtag, dtag_builds in autobuilds.items():
        ligand_graphs[dtag] = {}
        for build_key, build_info in dtag_builds.items():
            ligand_key = build_info["build_key"]
            if ligand_key not in ligand_graphs[dtag]:
                try:
                    ligand_graphs[dtag][ligand_key] = get_ligand_cif_graph_matches(
                        pandda_2_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_LIGAND_FILES_DIR / f"{ligand_key}.cif"
                    )
                except:
                    continue

    return ligand_graphs


def get_rmsd(
        known_hit,
        autobuilt_structure,
        known_hit_structure,
        ligand_graph
):
    # Iterate over each isorhpism, then get symmetric distance to the relevant atom
    iso_distances = []
    for isomorphism in ligand_graph:
        # print(isomorphism)
        distances = []
        for atom in known_hit:
            if atom.element.name == "H":
                continue
            model = autobuilt_structure[0]
            chain = model[0]
            res = chain[0]
            try:
                autobuilt_atom = res[isomorphism[atom.name]][0]
            except:
                return None
            sym_clostst_dist = known_hit_structure.cell.find_nearest_image(
                atom.pos,
                autobuilt_atom.pos,
            ).dist()
            distances.append(sym_clostst_dist)
        # print(distances)
        rmsd = np.sqrt(np.mean(np.square(distances)))
        iso_distances.append(rmsd)
    return min(iso_distances)


def _get_builds(pandda_key, test_systems):
    # dfs = {}
    builds = []
    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query_events = pony.orm.select(
            (event, event.annotations, event.pandda, event.pandda.experiment, event.pandda.system) for
            event in EventORM)
        query = pony.orm.select(
            experiment for experiment in ExperimentORM
        )

        # Order experiments from least datasets to most for fast results
        experiment_num_datasets = {
            _experiment.path: len([x for x in Path(_experiment.model_dir).glob("*")])
            for _experiment
            in query
        }
        sorted_experiments = sorted(query, key=lambda _experiment: experiment_num_datasets[_experiment.path])

        for experiment in sorted_experiments:
            experiment_hit_results = [res for res in query_events if
                                      (res[1].annotation) & (experiment.path == res[3].path)]
            experiment_hit_datasets = set(
                [
                    experiment_hit_result[0].dtag
                    for experiment_hit_result
                    in experiment_hit_results
                    if
                    (Path(experiment_hit_result[3].model_dir) / experiment_hit_result[0].dtag / 'refine.pdb').exists()
                ]
            )

            if len(experiment_hit_datasets) == 0:
                print(f"No experiment hit results for {experiment.path}. Skipping!")
                continue

            rprint(f"{experiment.system.name} : {experiment.path} : {experiment_num_datasets[experiment.path]}")
            # continue

            model_building_dir = Path(experiment.model_dir)
            result_dir = model_building_dir / f"../{pandda_key}"
            pandda_dir = result_dir / "pandda"

            if not (pandda_dir / constants.PANDDA_ANALYSIS_DIR / 'pandda_analyse_events.csv').exists():
                print(f"PanDDA either not finished or errored! Skipping!")
                continue

            # Get the known hits structures
            known_hit_structures = _get_known_hit_structures(
                experiment.model_dir,
                experiment_hit_datasets
            )
            print(f"Got {len(known_hit_structures)} known hit structures")

            # Get the known hits
            known_hits = _get_known_hits(known_hit_structures)
            print(f"Got {len(known_hits)} known hits")

            # Get the autobuild structures and their corresponding event info
            autobuilds = _get_autobuilds(pandda_dir)
            print(f"Got {len(autobuilds)} autobuilds")
            autobuilt_structures = _get_pandda_2_autobuilt_structures(autobuilds)
            print(f"Got {len(autobuilt_structures)} autobuilt structures")

            # Get the corresponding cif files
            ligand_graph_matches = get_ligand_graphs(autobuilds, pandda_dir)
            print(f"Got {len(ligand_graph_matches)} ligand graph matches")

            # Get train/test
            if experiment.system.name in test_systems:
                train_test = 'Test'
            else:
                train_test = "Train"

            # For each known hit, for each selected autobuild, graph match and symmtery match and get RMSDs
            records = []
            for dtag, dtag_known_hits in known_hits.items():
                # print(dtag)
                if dtag not in ligand_graph_matches:
                    continue
                ligand_graphs = ligand_graph_matches[dtag]
                if len(ligand_graphs) == 0:
                    continue
                # print(f'\tGot {len(dtag_known_hits)} known hits for dtag')
                if dtag not in autobuilt_structures:
                    continue
                dtag_autobuilt_structures = autobuilt_structures[dtag]
                if len(dtag_autobuilt_structures) == 0:
                    continue
                # print(f"\tGot {len(dtag_autobuilt_structures)} autobuilt structures for dtag ligand")
                if dtag not in autobuilds:
                    continue
                dtag_autobuilds = autobuilds[dtag]
                if len(dtag_autobuilds) == 0:
                    continue
                # print(f"\tGot {len(dtag_autobuilds)} autobuilds for dtag ligand")

                # # Get the autobuilds for the dataset
                for autobuild_key, autobuilt_structure in dtag_autobuilt_structures.items():
                    autobuild = dtag_autobuilds[autobuild_key]
                    rmsds = {}
                    for ligand_key, ligand_graph_automorphisms in ligand_graphs.items():
                        for known_hit_key, known_hit in dtag_known_hits.items():
                            # # Get the RMSD
                            rmsd = get_rmsd(
                                known_hit,
                                autobuilt_structure,
                                known_hit_structures[dtag],
                                ligand_graph_automorphisms
                            )
                            rmsds[(ligand_key, known_hit_key)] = {
                                'experiment_model_dir': str(experiment.model_dir),
                                'pandda_path': str(pandda_dir),
                                "dtag": dtag,
                                "model_idx": autobuild_key[0],
                                "event_idx": autobuild_key[1],
                                "known_hit_key": known_hit_key,
                                # "Autobuild Key": autobuild_key[1],
                                "ligand_key": ligand_key,
                                "rmsd": rmsd,
                                'score': autobuild['Score'],
                                'size': autobuild['Size'],
                                'local_strength': autobuild['Local_Strength'],
                                'rscc': autobuild['RSCC'],
                                'signal': autobuild['Signal'],
                                'noise': autobuild['Noise'],
                                'signal_noise': autobuild['Signal_Noise'],
                                'x_ligand': autobuild['X_ligand'],
                                'y_ligand': autobuild['Y_ligand'],
                                'z_ligand': autobuild['Z_ligand'],
                                'x': autobuild['X'],
                                'y': autobuild['Y'],
                                'z': autobuild['Z'],
                                "build_path": str(autobuild['build_path']),
                                'bdc': autobuild['BDC'],
                                'xmap_path': str(
                                    pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / 'xmap.ccp4'),
                                # 'Mean_Map_Path': str(pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_GROUND_STATE_MAP_TEMPLATE.format(dtag=dtag)),
                                'mean_map_path': str(
                                    pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / 'model_maps' / f'{autobuild_key[0]}_mean.ccp4'),
                                'mtz_path': str(
                                    pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_INITIAL_MTZ_TEMPLATE.format(
                                        dtag=dtag)),
                                # 'Zmap_Path': str(pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / constants.PANDDA_ZMAP_TEMPLATE.format(dtag=dtag)),
                                'zmap_path': str(
                                    pandda_dir / constants.PANDDA_PROCESSED_DATASETS_DIR / dtag / 'model_maps' / f'{autobuild_key[0]}_z.ccp4'
                                ),
                                'train_test': train_test
                            }

                    non_none_rmsds = [_key for _key in rmsds if rmsds[_key]['rmsd'] is not None]
                    if len(non_none_rmsds) == 0:
                        continue
                    selected_known_hit_key = min(non_none_rmsds, key=lambda _key: rmsds[_key]['rmsd'])

                    selected_match = rmsds[selected_known_hit_key]
                    builds.append(
                        AutobuildORM(
                            experiment_model_dir=selected_match['experiment_model_dir'],
                            pandda_path=selected_match['pandda_path'],
                            dtag=selected_match['dtag'],
                            model_idx=selected_match['model_idx'],
                            event_idx=selected_match['event_idx'],
                            known_hit_key=selected_match['known_hit_key'],
                            ligand_key=selected_match['ligand_key'],
                            rmsd=selected_match['rmsd'],
                            score=selected_match['score'],
                            size=selected_match['size'],
                            local_strength=selected_match['local_strength'],
                            rscc=selected_match['rscc'],
                            signal=selected_match['signal'],
                            noise=selected_match['noise'],
                            signal_noise=selected_match['signal_noise'],
                            x_ligand=selected_match['x_ligand'],
                            y_ligand=selected_match['y_ligand'],
                            z_ligand=selected_match['z_ligand'],
                            x=selected_match['x'],
                            y=selected_match['y'],
                            z=selected_match['z'],
                            build_path=selected_match['build_path'],
                            bdc=selected_match['bdc'],
                            xmap_path=selected_match['xmap_path'],
                            mean_map_path=selected_match['mean_map_path'],
                            mtz_path=selected_match['mtz_path'],
                            zmap_path=selected_match['zmap_path'],
                            train_test=selected_match['train_test'],
                        )
                    )
            # print(f"Got {len(records)} rmsds")

            # Get the table of rmsds
    #         df = pd.DataFrame(records)
    #         print(df)
    #         if len(df) != 0:
    #             print(df[df['RMSD'] < 2.5])
    #
    #         dfs[experiment.path] = df
    #
    # table = pd.concat([x for x in dfs.values()], axis=0, ignore_index=True)

    # return builds


def main(config_path):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'rb') as f:
        custom_annotations = pickle.load(f)

    #
    database_path = Path(config['working_directory']) / "database.db"
    try:
        db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
        db.generate_mapping(create_tables=True)
    except Exception as e:
        print(f"Exception setting up database: {e}")

    # Get the possible pandda paths
    possible_pandda_paths = [
        path
        for dataset_pattern
        in config['datasets']
        for path
        in Path('/').glob(dataset_pattern[1:])
        if not any([path.match(exclude_pattern) for exclude_pattern in config['exclude']])

    ]
    rprint(f"Got {len(possible_pandda_paths)} pandda paths!")
    rprint(possible_pandda_paths)

    # Get the pandda event tables
    inspect_tables = _get_inspect_tables(possible_pandda_paths)
    rprint(f"Got {len(inspect_tables)} pandda inspect tables!")

    #

    # Get the database
    # Get the events
    _get_events(
        inspect_tables,
        custom_annotations
    )
    rprint(f"Got events!")

    # Get the builds
    _get_builds(
        config['panddas']['pandda_key'],
        config['test']['test_systems']
    )
    rprint("Got builds!")

    db.disconnect()


if __name__ == "__main__":
    fire.Fire(main)
