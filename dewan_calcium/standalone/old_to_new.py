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

def convert_time(time_file_path):


    pass

def old_to_new(files_dict: dict):
    for animal_type in files_dict.keys():
        type_files = files_dict[animal_type]

        for project in type_files:
            if not project['old'] and not project['old_time']:
                continue
            if project['old']:
                file_parent = project['file'].parents[1]
                new_path = convert(file_parent)
                project['file'] = new_path
                project['old'] = False

            if project['old_time']:
                time_file_path = project['time']
                new_time_path = convert_time(time_file_path)
                project['file'] = new_time_path
                project['old'] = False

    return files_dict
