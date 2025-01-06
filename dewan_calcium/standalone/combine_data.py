import os
os.environ['ISX'] = '0'

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import get_project_files

input_dir = Path(r'C:/Projects/test_data/files_to_combine')
output_dir_root = Path(r'/home/austin/Combined')

MIN_BASELINE_TIME_FRAMES = 20
MIN_POST_TIME_FRAMES = 20
ODOR_TIME_FRAMES = 20
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


def strip_insignificant_cells(data: pd.DataFrame, significance_table: pd.DataFrame) -> (pd.DataFrame, list):
    significance_table = significance_table.set_index(significance_table.columns[0], drop=True)
    columns_to_drop = significance_table.columns[significance_table.sum() == 0].values

    if len(columns_to_drop) > 0:
        data = data.drop(columns_to_drop, level=0, axis=1)

    return data, columns_to_drop


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

    return cell_data.T  # Flip data back the other direction


def _trim_trials(cell_data: pd.DataFrame, trial_indices: dict) -> pd.DataFrame:
    new_df = pd.DataFrame()
    for trial in trial_indices.keys():
        indices = trial_indices[trial]

        baseline = indices['baseline']
        odor = indices['odor']
        post = indices['post']

        trial_data = cell_data.iloc[trial]

        baseline_data = trial_data[baseline]
        odor_data = trial_data[odor]
        post_data = trial_data[post]
        new_row = pd.Series(np.hstack([baseline_data, odor_data, post_data]))
        new_df = pd.concat([new_df, new_row], axis=1)

    new_df = new_df.T
    new_df = new_df.reset_index(drop=True)
    return new_df


def trim_all_trials(cell_data:pd.DataFrame, trial_indices:dict) -> pd.DataFrame:
    trimmed_cell_data = cell_data.T
    grouped_data = trimmed_cell_data.groupby(level=0)
    new_data = grouped_data.apply(lambda df: _trim_trials(df, trial_indices))

    new_data = new_data.T
    new_data.columns = cell_data.columns
    return new_data


def write_to_disk(data, output_dir, file_stem, total_cells, num_animals):
    if not output_dir.exists():
        print(f'Output directory does not exist! Creating {output_dir}')
        output_dir.mkdir(exists_ok=True, parents=True)

    pickle_path = output_dir.joinpath(f'{file_stem}-combined.pickle')
    total_file = output_dir.joinpath(f"{file_stem}.txt")

    with open(total_file, "w") as out_file:
        out_file.write(f'Num Cells: {total_cells}\n')
        out_file.write(f'Num Animals: {num_animals}\n')

    print(f'Writing combined data for {file_stem} to disk...')
    data.to_pickle(str(pickle_path), compression={'method': 'xz'})
    print(f'Combined data for {file_stem} successfully written to disk!')


def find_trials(time_data, debug=False) -> tuple[dict, list[int]]:
    trial_indices_to_drop = []
    trial_indices = {}

    for i, (name, data) in enumerate(time_data.items()):
        trial_periods = {}
        baseline_indices = np.where(data < 0)[0]
        odor_indices = np.where(data >=0)[0]
        post_indices = np.where(data >= ODOR_TIME_S)[0]

        baseline_frames = len(baseline_indices)
        post_frames = len(post_indices)

        if baseline_frames < MIN_BASELINE_TIME_FRAMES:
            if debug:
                print(f'Trial {i} w/ odor {name} does not have enough baseline frames!')
            trial_indices_to_drop.append(i)
        elif post_frames < MIN_POST_TIME_FRAMES:
            if debug:
                print(f'Trial {i} w/ odor {name} does not have enough post-time frames ({post_frames})!')
            trial_indices_to_drop.append(i)
        else:
            trial_periods['baseline'] = baseline_indices[-MIN_BASELINE_TIME_FRAMES:]
            trial_periods['odor'] = odor_indices[:ODOR_TIME_FRAMES]
            trial_periods['post'] = post_indices[:MIN_POST_TIME_FRAMES]
            trial_indices[i] = trial_periods

    return trial_indices, trial_indices_to_drop


def combine_data(data_files, filter_significant=True, strip_multisense=True, trim_trials=True, class_name=None):
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

        cell_data = pd.read_pickle(str(data_file))
        significance_data = pd.read_excel(str(significance_file))
        time_data = pd.read_pickle(str(time_file))
        trial_indices, trial_indices_to_drop = find_trials(time_data)

        if len(trial_indices_to_drop) == len(time_data):
            print(f'Skipping {name}, as all trials are marked to be dropped!')
            continue
        elif trial_indices_to_drop:
            print(f'Dropping {trial_indices_to_drop} from {name}!')
            cell_data = drop_bad_trials(cell_data, trial_indices_to_drop)

        if trim_trials:
            cell_data = trim_all_trials(cell_data, trial_indices)
        if filter_significant:
            cell_data, dropped_cell = strip_insignificant_cells(cell_data, significance_data)
        if strip_multisense:
            cell_data = strip_multisensory_trials(cell_data)

        current_cell_names = cell_data.columns.get_level_values(0).unique().values # Get all the unique cells in the multiindex
        num_new_cells = len(current_cell_names)
        trial_order = cell_data[current_cell_names[0]].columns.values
        fixed_odors = fix_odors(trial_order)
        # Get the order of the trials, all cells in this df share this order, so just use the first cell

        new_numbers = generate_new_numbers(num_new_cells, total_num_cells)
        # Generate new labels for this set of cells
        new_multiindex = pd.MultiIndex.from_product([new_numbers, fixed_odors], sortorder=None, names=['Cells', 'Trials'])
        cell_data.columns = new_multiindex
        # Create new multiindex with new cell labels and apply it to the new data

        combined_data = pd.concat([combined_data, cell_data], axis=1)

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


def new_combine(files: dict, filter_significant=True):
    combined_data = pd.DataFrame()
    total_cells = 0

    for file in tqdm(files.keys()):
        animal_files = files[file]
        name = file
        combined_data_path = animal_files['trace']
        significance_file_path = animal_files['sig']
        odor_file_path = animal_files['odor']
        timestamps_path = animal_files['time']

        try:
            combined_data = pd.read_pickle(combined_data_path, compression={'method': 'xz'})
        except Exception:  # yeah yeah I know; I can't remember how they were saved
            combined_data = pd.read_pickle(combined_data_path)
        try:
            timestamps = pd.read_pickle(timestamps_path, compression={'method': 'xz'})
        except Exception:
            timestamps = pd.read_pickle(timestamps_path)

        significance_table = pd.read_excel(significance_file_path)

        odor_data = pd.read_excel(odor_file_path, header=None, usecols=[0]).values
        odor_data = [odor[0] for odor in odor_data]  # Aha, fixed the off-by-one error
        new_index = pd.MultiIndex.from_product([combined_data.columns.get_level_values(0).unique(), odor_data],
                                              names=['Cells', 'Frames'])
        combined_data.columns = new_index

        trial_indices, trial_indices_to_drop = find_trials(combined_data)

        if len(trial_indices_to_drop) == len(timestamps):
            print(f'Skipping {name}, as all trials are marked to be dropped!')
            continue
        elif trial_indices_to_drop:
            print(f'Dropping {trial_indices_to_drop} from {name}!')
            combined_data = drop_bad_trials(combined_data, trial_indices_to_drop)



def main():
    animal_dirs = list(input_dir.iterdir())
    files = get_project_files.get_test_files(animal_dirs)
    new_combine(files)

if __name__ == "__main__":
    main()