import os
import pickle
from pathlib import Path

import yaml
import fire
import pony
from rich import print as rprint

import pandas as pd
import gemmi
import numpy as np

import tables

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer import constants



def _get_event_map(event_map_sample):
    grid = gemmi.FloatGrid(90, 90, 90)
    uc = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    grid.set_unit_cell(uc)

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = event_map_sample['sample'][:, :, :]

    return grid


def _get_model(closest_pose):
    st = gemmi.Structure()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()
    res.name = 'LIG'

    for _pose_row, _element in zip(closest_pose['positions'], closest_pose['elements']):
        pos = gemmi.Position(_pose_row[0], _pose_row[1], _pose_row[2])

        element = gemmi.Element(_element)
        atom = gemmi.Atom()
        atom.name = element.name
        atom.pos = pos
        atom.element = element
        res.add_atom(atom)

    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def try_make(path):
    try:
        os.mkdir(path)
    except Exception as e:
        return


def try_link(source_path, target_path):
    try:
        os.symlink(Path(source_path).resolve(), target_path)
    except Exception as e:
        # print(e)
        return


def _make_test_dataset_psuedo_pandda(
        config_path
):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #
    custom_annotations_path = Path(config['working_directory']) / "custom_annotations.pickle"
    with open(custom_annotations_path, 'rb') as f:
        custom_annotations = pickle.load(f)

    # Open a file in "w"rite mode
    fileh = tables.open_file("output/build_data.h5", mode="r")
    print(fileh)

    # Get the HDF5 root group
    root = fileh.root
    table_mtz_sample = root.mtz_sample
    table_event_map_sample = root.event_map_sample
    table_known_hit_pos_sample = root.known_hit_pose

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Make the directories
    psuedo_pandda_dir = working_dir / "build_annotation_pandda"
    analyses_dir = psuedo_pandda_dir / "analyses"
    processed_datasets_dir = psuedo_pandda_dir / "processed_datasets"
    analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
    inspect_table_path = analyses_dir / "pandda_inspect_events.csv"
    analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"
    inspect_site_table_path = analyses_dir / "pandda_inspect_sites.csv"

    # Spoof the main directories
    try_make(psuedo_pandda_dir)
    try_make(analyses_dir)
    try_make(processed_datasets_dir)

    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = [_x for _x in pony.orm.select(_event for _event in EventORM)]

        # Iterate over the event maps
        event_rows = []
        for event_map_sample in table_event_map_sample.iterrows():
            rprint(event_map_sample['idx'])
            # Get the corresponding poses
            event_map_sample_idx = event_map_sample['idx']
            poses = [x for x in table_known_hit_pos_sample.where(f'event_map_sample_idx == {event_map_sample_idx}')]

            # Get the closest pose
            closest_pose = min(poses, key=lambda _x: _x['rmsd'])
            rprint(f'Closest rmsd is: {closest_pose["rmsd"]}')

            # Get the corresponding event
            rprint(f"Database event idx: {closest_pose['database_event_idx']}")
            event = query[closest_pose['database_event_idx']]
            # rprint()
            rprint(f'Closest event id: {closest_pose["database_event_idx"]}')
            event_id = event.id

            # Make the dataset dir
            dataset_dir = processed_datasets_dir / f'{event_id}'
            try_make(dataset_dir)
            modelled_structures_dir = dataset_dir / "modelled_structures"
            try_make(modelled_structures_dir)

            # Get the model
            st = _get_model(closest_pose)

            # Write the model
            st.write_minimal_pdb(str(dataset_dir / f'{event_id}.pdb'))

            # Get the event map
            event_map = _get_event_map(event_map_sample)

            # Write the event map
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = event_map
            ccp4.update_ccp4_header()
            event_map_path = dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
                dtag=f"{event_id}",
                event_idx="1",
                bdc=f"{event.bdc}"
            )
            ccp4.write_ccp4_map(str(event_map_path))

            # Create the event row
            event_row = {
                "dtag": event_id,
                "event_idx": 1,
                "1-BDC": event.bdc,
                "cluster_size": 1,
                "global_correlation_to_average_map": 0,
                "global_correlation_to_mean_map": 0,
                "local_correlation_to_average_map": 0,
                "local_correlation_to_mean_map": 0,
                "site_idx": 1 + int(event_map_sample_idx / 200),
                "x": 22.5,
                "y": 22.5,
                "z": 22.5,
                "z_mean": 0.0,
                "z_peak": float(closest_pose['rmsd']),
                "applied_b_factor_scaling": 0.0,
                "high_resolution": 0.0,
                "low_resolution": 0.0,
                "r_free": 0.0,
                "r_work": 0.0,
                "analysed_resolution": 0.0,
                "map_uncertainty": 0.0,
                "analysed": False,
                "interesting": False,
                "exclude_from_z_map_analysis": False,
                "exclude_from_characterisation": False,
            }
            event_rows.append(event_row)

            if event_map_sample_idx >5:
                break

        # # Select events and Organise by dtag
        # events = {}
        # for res in query:
        #     if res[2].name == test_partition:
        #         event = res[0]
        #         annotation = res[1]
        #         dtag = event.dtag
        #         event_idx = event.event_idx
        #         if dtag not in events:
        #             events[dtag] = {}
        #
        #         table = inspect_tables[event.pandda.path]
        #         row = table[
        #             (table[constants.PANDDA_INSPECT_DTAG] == dtag)
        #             & (table[constants.PANDDA_INSPECT_EVENT_IDX] == event_idx)
        #             ]
        #
        #         events[dtag][event_idx] = {
        #             'event': event,
        #             'annotation': annotation,
        #             'row': row
        #         }
        # rprint(f"Got {len(events)} events!")

        # Spoof the dataset directories
        # for dtag in events:
        #     dtag_dir = processed_datasets_dir / dtag
        #     try_make(dtag_dir)
        #
        #     for event_idx, event_info in events[dtag].items():
        #         event, annotation = event_info['event'], event_info['annotation']
        #         try_link(event.initial_structure, dtag_dir / Path(event.initial_structure).name)
        #         try_link(event.initial_reflections, dtag_dir / Path(event.initial_reflections).name)
        #         try_link(event.event_map, dtag_dir / Path(event.event_map).name)
        #         if event.structure:
        #             try_link(event.structure, modelled_structures_dir / Path(event.structure).name)

        # Spoof the event table
        rows = []
        j = 0
        # for dtag in events:
        #     for event_idx, event_info in events[dtag].items():
        # for
        #         row = event_info['row']
        #         # row.loc[0, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1
        #         rows.append(row)
        #         # j = j + 1
        #
        # event_table = pd.concat(rows).reset_index()
        event_table = pd.DataFrame(event_rows)
        # for j in range(len(event_table)):
        #     event_table.loc[j, constants.PANDDA_INSPECT_SITE_IDX] = (j // 100) + 1

        # event_table.drop(["index", "Unnamed: 0"], axis=1, inplace=True)
        event_table.to_csv(analyse_table_path, index=False)
        # event_table.to_csv(inspect_table_path, index=False)

        # Spoof the site table
        site_records = []
        # num_sites = ((j) // 100) + 1
        num_sites = event_table['site_idx'].max()
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
        site_table.to_csv(analyse_site_table_path, index=False)
        # site_table.to_csv(inspect_site_table_path, index=False)

    fileh.close()


if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)
