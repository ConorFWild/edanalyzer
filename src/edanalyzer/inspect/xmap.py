import gemmi
import numpy as np


def _get_event_map(event_map_sample):
    grid = gemmi.FloatGrid(90, 90, 90)
    uc = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    grid.set_unit_cell(uc)

    grid_array = np.array(grid, copy=False)
    grid_array[:, :, :] = (event_map_sample['sample'])[:, :, :]

    return grid


