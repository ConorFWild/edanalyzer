import os
import pickle
import time
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

    # Open a file in "w"rite mode
    fileh = tables.open_file("output/build_data_v2.h5", mode="r")
    print(fileh)

    # Get the HDF5 root group
    root = fileh.root
    # table_mtz_sample = root.mtz_sample
    table_event_map_sample = root.event_map_sample
    table_known_hit_pos_sample = root.known_hit_pose
    table_annotation = root.annotation

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Make the directories
    psuedo_pandda_dir = working_dir / "build_annotation_pandda_2"
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
        close_poses = {}
        begin_get_close_poses = time.time()
        for x in table_known_hit_pos_sample.where("""rmsd == 0.0"""):
            y = x.fetch_all_fields()
            close_poses[y['event_map_sample_idx']] = y
        finish_get_close_poses = time.time()
        rprint(f'Got {len(close_poses)} close poses in: {finish_get_close_poses-begin_get_close_poses}')

    fileh.close()


if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)
