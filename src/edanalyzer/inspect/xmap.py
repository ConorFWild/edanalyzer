import gemmi
import numpy as np


def _get_event_map(event_map_sample):
    grid = gemmi.FloatGrid(90, 90, 90)
    uc = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    grid.set_unit_cell(uc)

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = (event_map_sample['sample'])[:, :, :]

    return grid

def _write_event_map(event_map, path):
    # Write the event map
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = event_map
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))