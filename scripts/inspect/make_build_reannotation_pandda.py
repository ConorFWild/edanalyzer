import os
import pickle
import time
from pathlib import Path

import yaml
import fire
import pony
from rich import print as rprint
import zarr

import pandas as pd
import gemmi
import numpy as np

import tables

from edanalyzer.data.database_schema import db, EventORM, DatasetORM, PartitionORM, PanDDAORM, AnnotationORM, SystemORM, \
    ExperimentORM, LigandORM, AutobuildORM
from edanalyzer import constants
from edanalyzer.utils import try_make, try_link


def _get_event_map(event_map_sample):
    grid = gemmi.FloatGrid(90, 90, 90)
    uc = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    grid.set_unit_cell(uc)

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = (event_map_sample['sample'])[:, :, :]

    return grid


def _get_model(closest_pose):
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    st.spacegroup_hm = gemmi.SpaceGroup('P1').xhm()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()
    res.name = 'LIG'
    res.seqid = gemmi.SeqId(1, ' ')

    for _pose_row, _element in zip(closest_pose['positions'], closest_pose['elements']):
        pos = gemmi.Position(_pose_row[0], _pose_row[1], _pose_row[2])
        if _element == 0:
            continue

        element = gemmi.Element(_element)
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.charge = 0
        atom.pos = pos
        atom.element = element
        res.add_atom(atom)

    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st



def _make_test_dataset_psuedo_pandda(
        config_path
):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Open a file in "w"rite mode
    root = zarr.open("output/build_data_v3.zarr", mode="r")
    print(root)

    # Get the HDF5 root group
    # root = fileh.root
    # table_mtz_sample = root.mtz_sample
    table_event_map_sample = root['event_map_sample']
    table_known_hit_pos_sample = root['known_hit_pose']
    table_annotation = root['annotation']

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Make the directories
    psuedo_pandda_dir = working_dir / "build_annotation_pandda_3"
    analyses_dir = psuedo_pandda_dir / "analyses"
    processed_datasets_dir = psuedo_pandda_dir / "processed_datasets"
    analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
    # inspect_table_path = analyses_dir / "pandda_inspect_events.csv"
    analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"
    # inspect_site_table_path = analyses_dir / "pandda_inspect_sites.csv"

    # Spoof the main directories
    try_make(psuedo_pandda_dir)
    try_make(analyses_dir)
    try_make(processed_datasets_dir)

    # Get the idxs of annotated
    # annotated_idxs = table_annotation.cols.event_map_table_idx[:]

    with pony.orm.db_session:
        # partitions = {partition.name: partition for partition in pony.orm.select(p for p in PartitionORM)}
        query = [_x for _x in pony.orm.select(_event for _event in EventORM)]

        # Fetch the
        # Get idxs without annotations
        event_map_idxs = table_event_map_sample['idx']
        close_poses = {_z: {'rmsd': 100} for _z in event_map_idxs }
        begin_get_close_poses = time.time()
        for x in table_known_hit_pos_sample:
            y=x
            # y = x.fetch_all_fields()
            # Skip if processed
            # if y['event_map_sample_idx'] in annotated_idxs:
            #     continue

            close_poses[x['event_map_sample_idx']] = min(
                [y, close_poses[x['event_map_sample_idx']]],
                key=lambda _x: _x['rmsd'])
        finish_get_close_poses = time.time()
        rprint(f'Got {len(close_poses)} close poses in: {finish_get_close_poses-begin_get_close_poses}')

        # Iterate over the event maps
        event_rows = []
        for event_map_sample in table_event_map_sample:
            rprint(event_map_sample['idx'])
            # Get the corresponding poses
            event_map_sample_idx = event_map_sample['idx']
            # if event_map_sample_idx in annotated_idxs:
            #     continue
            database_event_idx = event_map_sample['event_idx']
            # poses = [x.fetch_all_fields() for x in table_known_hit_pos_sample.where(f'event_map_sample_idx == {event_map_sample_idx}')]
            # psuedo_dtag = f"{database_event_idx}_{event_map_sample['res_id'].decode('utf-8')}"
            psuedo_dtag = event_map_sample_idx
            # poses = []
            # for pose in table_known_hit_pos_sample.where(f'event_map_sample_idx == {event_map_sample_idx}'):
            #     poses.append(pose.copy())
            # for pose in poses:
            #     print(f"IDX: {pose['idx']} : {pose['database_event_idx']}")
            # rprint(f"Got {len(poses)} poses")

            # Get the closest pose
            # closest_pose = min(poses, key=lambda _x: _x['rmsd'])
            closest_pose = close_poses[event_map_sample_idx]
            rprint(f'Closest rmsd is: {closest_pose["rmsd"]}')

            # Get the corresponding event
            rprint(f"Database event idx: {database_event_idx}")
            event = query[event_map_sample['event_idx']]
            # rprint()
            # event_id = event.id

            # Make the dataset dir
            dataset_dir = processed_datasets_dir / f'{psuedo_dtag}'
            try_make(dataset_dir)
            modelled_structures_dir = dataset_dir / "modelled_structures"
            try_make(modelled_structures_dir)

            # Get the model
            st = _get_model(closest_pose)

            # Write the model
            st.write_pdb(str(dataset_dir / constants.PANDDA_INITIAL_MODEL_TEMPLATE.format(dtag=f'{psuedo_dtag}')))

            # Get the event map
            event_map = _get_event_map(event_map_sample)

            # Write the event map
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = event_map
            ccp4.update_ccp4_header()
            event_map_path = dataset_dir / constants.PANDDA_EVENT_MAP_TEMPLATE.format(
                dtag=f"{psuedo_dtag}",
                event_idx="1",
                bdc=f"{event.bdc}"
            )
            ccp4.write_ccp4_map(str(event_map_path))

            # Create the event row
            event_row = {
                "dtag": psuedo_dtag,
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
                # "interesting": False,
                "exclude_from_z_map_analysis": False,
                "exclude_from_characterisation": False,
            }
            event_rows.append(event_row)
            if database_event_idx == 10466:
                rprint(event_row)

            # if event_map_sample_idx >200:
            #     break

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

    # fileh.close()


if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)
