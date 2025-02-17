import zarr
import numpy as np
import pandas as pd
from rich import print as rprint
import pickle


if __name__ == '__main__':
    zarr_path = '/dls/data2temp01/labxchem/data/2017/lb18145-17/processing/edanalyzer/output/build_data_augmented_2.zarr'
    root = zarr.open(zarr_path, mode='r')
    rprint(f'Getting deltas')
    deltas = root['delta'].get_basic_selection(slice(None, None), fields='delta')
    rprint('Got Deltas')
    bins = pd.cut(deltas[deltas>0.0], bins=np.linspace(0.0, 6.0, num=61))
    counts = bins.value_counts(dropna=False)
    with open('test.pickle', 'wb') as f:
        pickle.dump(bins, f)
    with open('test.pickle', 'rb') as f:
        df = pickle.load(f)

    rprint('Saved and finished!')