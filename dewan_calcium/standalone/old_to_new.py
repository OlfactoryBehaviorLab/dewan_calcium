import pandas as pd
from pathlib import Path


def convert(path):
    cell_file = list(path.joinpath('PreProcessedData').glob('*CellList.pickle'))[0]
    odor_data = list(path.joinpath('PreProcessedData').glob('*OdorData.pickle'))[0]
    cell_data = list(path.joinpath('CombinedData').glob('*CombinedData.pickle'))[0]

    old_data = pd.read_pickle(cell_data)
    cell_names = pd.read_pickle(cell_file)
    odor_data = pd.read_pickle(odor_data)
    odor_data = [odor[0] for odor in odor_data]

    old_data_df = []
    for cell in old_data:
        odor_data = odor_data[:len(cell)]
        old_data_cell = pd.DataFrame(cell, index=odor_data).T
        old_data_df.append(old_data_cell)

    new_data = pd.concat(old_data_df, axis=1, keys=cell_names, names=['Cells', 'Frames'])

    new_data_path = cell_data.with_stem(f'new-{cell_data.name}')
    new_data.to_pickle(new_data_path)
    return new_data_path


def old_to_new(data_folder: Path):


    if isinstance(data_folder, dict):
        new_paths = {}
        for key in data_folder.keys():
            new_paths[key] = []
            for file in data_folder[key]:
                file = file.parents[1]
                new_path = convert(file)
                new_paths[key].append(new_path)
    else:
        new_paths = []
        for file in data_folder:
            file = file.parents[1]
            new_path = convert(file)
            new_paths.append(new_path)

    return new_paths
