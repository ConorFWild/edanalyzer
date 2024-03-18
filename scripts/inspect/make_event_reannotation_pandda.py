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





def _make_test_dataset_psuedo_pandda(
        config_path
):
    rprint(f'Running collate_database from config file: {config_path}')
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Open a file in "w"rite mode
    root = zarr.open("output/event_data.zarr", mode="r")
    print(root)

    # Get the HDF5 root group
    # root = fileh.root
    # table_mtz_sample = root.mtz_sample
    table_z_map_sample_metadata = root['z_map_sample_metadata']
    table_z_map_sample = root['z_map_sample']
    table_pose = root['known_hit_pose']
    # table_annotation = root['annotation']

    # Load database
    working_dir = Path(config['working_directory'])
    database_path = working_dir / "database.db"
    db.bind(provider='sqlite', filename=f"{database_path}", create_db=True)
    db.generate_mapping(create_tables=True)

    # Make the directories
    psuedo_pandda_dir = working_dir / "event_annotation_pandda"
    analyses_dir = psuedo_pandda_dir / "analyses"
    processed_datasets_dir = psuedo_pandda_dir / "processed_datasets"
    analyse_table_path = analyses_dir / "pandda_analyse_events.csv"
    analyse_site_table_path = analyses_dir / "pandda_analyse_sites.csv"

    # Spoof the main directories
    try_make(psuedo_pandda_dir)
    try_make(analyses_dir)
    try_make(processed_datasets_dir)

    # Get the idxs of annotated

    with pony.orm.db_session:
        # query = [_x for _x in pony.orm.select(_event for _event in EventORM)]

        # Iterate over the event maps
        event_rows = []
        j = 0
        for z_map_metadata_sample in table_z_map_sample_metadata:

            # Get the corresponding database event and pose
            z_map_metadata_sample_idx = z_map_metadata_sample['idx']
            database_event_idx = z_map_metadata_sample['event_idx']
            closest_pose_idx = z_map_metadata_sample['pose_data_idx']

            if closest_pose_idx == -1:
                continue

            rprint(f'Z map sample idx is: {z_map_metadata_sample_idx}')
            rprint(f"Database event idx: {database_event_idx}")
            rprint(f'Closest pose idx is: {closest_pose_idx}')

            closest_pose = table_pose[closest_pose_idx]
            assert closest_pose['idx'] == closest_pose_idx

            psuedo_dtag = z_map_metadata_sample_idx


            # Get the closest pose

            # Get the corresponding event
            event = EventORM[database_event_idx]
            assert event.id == database_event_idx
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
            z_map_sample = table_z_map_sample[z_map_metadata_sample_idx]
            assert z_map_sample['idx'] == z_map_metadata_sample_idx
            event_map = _get_event_map(z_map_sample)

            # Write the event map
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = event_map
            ccp4.update_ccp4_header()
            z_map_path = dataset_dir / constants.PANDDA_ZMAP_TEMPLATE.format(
                dtag=f"{psuedo_dtag}",
            )
            ccp4.write_ccp4_map(str(z_map_path))

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
                "site_idx": 1 + int(j / 200),
                "x": 22.5,
                "y": 22.5,
                "z": 22.5,
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
                "exclude_from_z_map_analysis": False,
                "exclude_from_characterisation": False,
            }
            event_rows.append(event_row)
            j += 1


        # Spoof the event table

        event_table = pd.DataFrame(event_rows)

        event_table.to_csv(analyse_table_path, index=False)

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



if __name__ == "__main__":
    fire.Fire(_make_test_dataset_psuedo_pandda)
