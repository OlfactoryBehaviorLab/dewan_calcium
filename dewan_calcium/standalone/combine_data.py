import os
os.environ['ISX'] = '0'

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import get_project_files
import old_to_new

input_dir = Path(r'/mnt/r2d2/2_Inscopix/1_DTT/1_OdorAnalysis/2_Identity')
output_dir_root = Path(r'~/Combined')
MIN_BASELINE_TIME_FRAMES = 20
MIN_POST_TIME_FRAMES = 20
ODOR_TIME_S = 2

def fix_odors(odor_data):
    new_odor_data = []
    for odor in odor_data:
        if '-' in odor:
            print('Dashes present, returning!')
            return odor_data
        try:
            first_val = int(odor[0])
            if first_val == 2:
                new_odor = '-'.join([odor[1:2], odor[2:]])
            else:
                new_odor = '-'.join([odor[0], odor[1:]])
            new_odor_data.append(new_odor)
            continue
        except Exception:
            pass

        new_odor_data.append(odor)

    return new_odor_data


def get_exp_type():
    if 'EPM' in str(input_dir):
        experiment_type = 'EPM'
    elif 'HFvFM' in str(input_dir):
        experiment_type = 'HFvFM'
    elif 'Concentration' in str(input_dir):
        experiment_type = 'Concentration'
    elif 'Identity' in str(input_dir):
        experiment_type = 'Identity'
    else:
        raise RuntimeError('Input folder is not a known experiment type!')

    return experiment_type


def update_cell_names(combined_data):
    """
    Function which will take each cell number, and add it to a string prepended with a 'C.' The dataframe can then be
    indexed by unique cell names instead of an integer. This reflects the nomenclature in the normal processing
    notebooks. This is necessitated by how strings are sorted when dataframes are concatenated.
    Args:
        combined_data (pd.DataFrame): Reference to a Dataframe containing the combined data from n number of
        experiments. This dataframe is indexed by a MultiIndex that is size cells x (number of conditions * # of trials)
        Cells are represented by integers instead of labels of type string.

    Returns:
        Input is passed as a reference and the original object is modified in-place

    """

    string_names = [f'C{i}' for i in range(len(combined_data.columns.levels[0]))]
    combined_data.columns = combined_data.columns.set_levels(string_names, level=0)


def generate_new_numbers(new_cells: int, total: int):
    new_total = new_cells + total
    return list(range(total, new_total))


def strip_insignificant_cells(data, significance_table):
    significance_table = significance_table.set_index(significance_table.columns[0], drop=True)
    drop_mask = (significance_table.loc['MO'] > 0).values | (significance_table.loc['Buzzer'] > 0).values
    cells_to_drop = data.columns.get_level_values(0).unique()[drop_mask].values
    if len(cells_to_drop) > 0:
        data = data.drop(cells_to_drop, axis=1, level=0)

    return data


def strip_multisensory_trials(data):
    trials_to_drop = ['MO', 'Buzzer']
    data = data.drop(trials_to_drop, axis=1, level=1)

    return data


def _drop_trials(cell_data: pd.DataFrame, trials_to_drop: list) -> pd.DataFrame:
    num_rows = cell_data.shape[0]
    rows_to_keep = np.setdiff1d(np.arange(num_rows), trials_to_drop)
    df = cell_data.iloc[rows_to_keep, :]
    return df


def drop_bad_trials(cell_data: pd.DataFrame, trials_to_drop: list) -> pd.DataFrame:
    cell_data = cell_data.T  # Transpose so cells/trials are the index
    grouped_data = cell_data.groupby(level=0, group_keys=False)  # Group by the cells
    grouped_data.apply(lambda df: _drop_trials(df, trials_to_drop)) # Apply _drop_trials to each 'Cell's' Data

    return grouped_data.T  # Flip data back the other direction


def write_to_disk(data, output_dir, file_stem, total_cells, num_animals):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    pickle_path = output_dir.joinpath(f'{file_stem}-combined.pickle')
    total_file = output_dir.joinpath(f"{file_stem}.txt")

    with open(total_file, "w") as out_file:
        out_file.write(f'Num Cells: {total_cells}\n')
        out_file.write(f'Num Animals: {num_animals}')

    print(f'Writing combined data for {file_stem} to disk...')
    data.to_pickle(str(pickle_path), compression={'method': 'xz'})
    print(f'Combined data for {file_stem} successfully written to disk!')


def find_short_trials(time_data, debug=False) -> tuple[dict, list[int]]:
    trial_indices_to_drop = []
    trial_indices = {}

    for i, (name, data) in enumerate(time_data.items()):
        trial_periods = dict.fromkeys(['baseline', 'odor', 'post'], [])
        baseline_indices = data[data < 0]
        odor_indices = data[np.logical_and(0 <= data, data < ODOR_TIME_S)]
        post_indices = data[data >= ODOR_TIME_S]

        baseline_frames = len(baseline_indices)
        post_frames = len(post_indices)

        trial_periods['baseline'] = baseline_indices
        trial_periods['odor'] = odor_indices
        trial_periods['post'] = post_indices

        if baseline_frames < MIN_BASELINE_TIME_FRAMES:
            if debug:
                print(f'Trial {i} w/ odor {name} does not have enough baseline frames!')
            trial_indices_to_drop.append(i)
        elif post_frames < MIN_POST_TIME_FRAMES:
            if debug:
                print(f'Trial {i} w/ odor {name} does not have enough post-time frames ({post_frames})!')
            trial_indices_to_drop.append(i)

        trial_indices[i] = trial_periods

    return trial_indices, trial_indices_to_drop


def combine_data(data_files, filter_significant, class_name=None):
    combined_data = pd.DataFrame()

    total_num_cells = 0
    if class_name:
        desc = f'Processing {class_name} files: '
    else:
        desc = f'Processing files: '

    for file in tqdm(data_files, desc=desc):
        data_file = file['file']
        significance_file = file['sig']
        time_file = file['time']
        name = file['folder'].name

        new_data = pd.read_pickle(str(data_file))
        significance_data = pd.read_excel(str(significance_file))
        time_data = pd.read_pickle(str(time_file))
        new_data = strip_insignificant_cells(new_data, significance_data)
        new_data = strip_multisensory_trials(new_data)
        trial_indices, trial_indices_to_drop = find_short_trials(time_data)

        if len(trial_indices_to_drop) == len(time_data):
            print(f'Skipping {name}, as all trials are marked to be dropped!')
            continue
        elif trial_indices_to_drop:
            print(f'Dropping {trial_indices_to_drop} from {name}!')
            new_data = drop_bad_trials(new_data, trial_indices_to_drop)


        current_cell_names = new_data.columns.get_level_values(0).unique().values # Get all the unique cells in the multiindex
        num_new_cells = len(current_cell_names)
        trial_order = new_data[current_cell_names[0]].columns.values
        fixed_odors = fix_odors(trial_order)
        # Get the order of the trials, all cells in this df share this order, so just use the first cell

        new_numbers = generate_new_numbers(num_new_cells, total_num_cells)
        # Generate new labels for this set of cells
        new_multiindex = pd.MultiIndex.from_product([new_numbers, fixed_odors], sortorder=None, names=['Cells', 'Trials'])
        new_data.columns = new_multiindex
        # Create new multiindex with new cell labels and apply it to the new data

        combined_data = pd.concat([combined_data, new_data], axis=1)

        total_num_cells += num_new_cells

    update_cell_names(combined_data)

    return combined_data, total_num_cells


def combine_and_save(files: dict, exp_type, filter_significant=True, combine_all=False):

    if exp_type == 'EPM':
        print('EPM not implemented yet!')
        return []
        # collected_data = combine_EPM_data(data_files)
    else:
        if combine_all:
            data_files = []
            significance_files = []
            for _type in files.keys():
                data_files = [folder for folder in files[_type] if folder['file'] is not None]

            collected_data, total_cells = combine_data(data_files, filter_significant, class_name=None)
            output_dir = output_dir_root.joinpath(exp_type)
            file_stem = f'{exp_type}'

            write_to_disk(collected_data, output_dir, file_stem, total_cells, len(data_files))

        else:
            for _type in files.keys():
                data_files = [folder for folder in files[_type] if folder['file'] is not None]
                data_files = [folder for folder in data_files if folder['sig'] is not None]
                data_files = [folder for folder in data_files if folder['time'] is not None]

                collected_data, total_cells = combine_data(data_files, filter_significant, _type, )

                output_dir = output_dir_root.joinpath(exp_type)

                file_stem = f'{_type}-{exp_type}'

                write_to_disk(collected_data, output_dir, file_stem, total_cells, len(data_files))


def main():
    animal_types = ['VGAT', 'VGLUT']
    files = {_type: [] for _type in animal_types}

    exp_type = get_exp_type()
    folders = get_project_files.get_folders(input_dir, animal_types)

    for _type in animal_types:
        for folder in tqdm(folders[_type], desc=f'Getting files for {_type} animals'):
            files[_type].append(get_project_files.find_data_files(folder, exp_type))

    files = old_to_new.old_to_new(files)

    combine_and_save(files, exp_type)

def test_main():
    animal_types = ['VGAT', 'VGLUT']
    files = {_type: [] for _type in animal_types}

    exp_type = get_exp_type()
    folders = get_project_files.get_folders(input_dir, animal_types)
    # for _type in animal_types:
    #     for folder in tqdm(folders[_type], desc=f'Getting files for {_type} animals'):
    #         files[_type].append(get_project_files.new_find_data_files(folder, exp_type))

    input_file = '/mnt/r2d2/2_Inscopix/1_DTT/1_OdorAnalysis/1_Concentration/VGLUT/VGLUT-27_RERUN/Analysis/Output/combined/VGLUT27_Conc-8_19_24-combined_data_.pickle'
    file = pd.read_pickle(str(input_file))
    print(file.columns)

if __name__ == "__main__":
    # main()
    test_main()