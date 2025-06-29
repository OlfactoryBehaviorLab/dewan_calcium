import os
os.environ['ISX'] = '0'

import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import get_project_files, odors

MIN_BASELINE_TIME_FRAMES = 20
MIN_POST_TIME_FRAMES = 20
ODOR_TIME_FRAMES = 20
ODOR_TIME_S = 2


def update_cell_names(combined_data, significance_table):
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

    if significance_table:
        string_names = [f'C{i}' for i in range(len(significance_table.columns))]
        significance_table.columns = string_names


def generate_new_numbers(new_cells: int, total: int):
    new_total = new_cells + total
    return list(range(total, new_total))


def drop_nonresponsive_cells(data: pd.DataFrame, significance_table: pd.DataFrame) -> (pd.DataFrame, list):
    column_mask = (significance_table != 0).sum() == 0
    columns_to_drop = significance_table.columns[column_mask].tolist()

    if len(columns_to_drop) > 0:
        data = data.drop(columns_to_drop, level=0, axis=1)

    return data, columns_to_drop


def drop_multisense_cells(data, significance_table: pd.DataFrame):

    MO_mask = significance_table.loc['MO'] != 0
    buzzer_mask = significance_table.loc['Buzzer'] != 0

    MO_cells_to_drop = significance_table.columns[MO_mask].tolist()
    buzzer_cells_to_drop = significance_table.columns[buzzer_mask].tolist()

    drop_mask = MO_mask | buzzer_mask
    cells_to_drop = significance_table.columns[drop_mask].values

    if len(cells_to_drop) > 0:
        data = data.drop(cells_to_drop, axis=1, level=0)

    data = data.drop(['MO', 'Buzzer'], axis=1, level=1)

    return data, MO_cells_to_drop, buzzer_cells_to_drop


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


def write_odor(data, sig_table, output_dir_root, file_stem, stats, cells, num_animals):
    output_dir = output_dir_root.joinpath(file_stem)

    global_good_cells, global_total_cells = cells

    if not output_dir.exists():
        print(f'Output directory does not exist! Creating {output_dir}')
        output_dir.mkdir(exist_ok=True, parents=True)

    pickle_path = output_dir.joinpath(f'{file_stem}-combined.pickle')
    excel_path = output_dir.joinpath(f'{file_stem}-combined.xlsx')
    total_file = output_dir.joinpath(f"{file_stem}.txt")

    with open(total_file, "w") as out_file:
        out_file.write(f'Number Total Cells: {global_total_cells}\n')
        out_file.write(f'Number Good Cells: {global_good_cells}\n')
        out_file.write(f'Num Animals: {num_animals}\n')
        out_file.write(f'=============================\n\n')
        for animal in stats.keys():
            animal_stats = stats[animal]
            original_cells = animal_stats['orig_cells']
            good_cells = animal_stats['good_cells']
            dropped_trials = animal_stats['trials']
            dropped_MO = animal_stats['MO']
            dropped_Buzzer = animal_stats['Buzzer']
            dropped_insig = animal_stats['insig']

            if not dropped_trials:
                dropped_trials = 'None'
                num_dropped_trials = 0
            else:
                num_dropped_trials = len(dropped_trials)

            if not dropped_MO:
                dropped_MO = 'None'
                num_dropped_MO = 0
            else:
                num_dropped_MO = len(dropped_MO)

            if not dropped_Buzzer:
                dropped_Buzzer = 'None'
                num_dropped_buzzer = 0
            else:
                num_dropped_buzzer = len(dropped_Buzzer)

            if not dropped_insig:
                dropped_insig = 'None'
                num_dropped_insig = 0
            else:
                num_dropped_insig = len(dropped_insig)

            out_file.write(f'{animal}:\n')
            out_file.write(f'Number Original Cells: {original_cells}\n')
            out_file.write(f'Number Good Cells: {good_cells}\n')
            out_file.write(f'Dropped {num_dropped_trials} Trials: {dropped_trials}\n')
            out_file.write(f'Dropped {num_dropped_MO} MO Cells: {dropped_MO}\n')
            out_file.write(f'Dropped {num_dropped_buzzer} Buzzer Cells: {dropped_Buzzer}\n')
            out_file.write(f'Dropped {num_dropped_insig} Nonresponsive Cells: {dropped_insig}\n')
            out_file.write(f'=============================\n\n')

    print(f'Writing combined data for {file_stem} to disk...')
    data.to_pickle(str(pickle_path), compression={'method': 'xz'})
    print(f'Writing combined significance table for {file_stem} to disk...')
    sig_table.T.to_excel(str(excel_path))
    print(f'Combined data for {file_stem} successfully written to disk!')


def write_hffm(combined_data, total_cells, num_animals, output_dir_root, file_stem):
    output_dir = output_dir_root.joinpath(file_stem)
    output_dir.mkdir(exist_ok=True, parents=True)

    pickle_path = output_dir.joinpath(f'{file_stem}-combined.pickle')
    total_file = output_dir.joinpath(f"{file_stem}.txt")

    print(f'Writing combined data for {file_stem} to disk...')

    with open(total_file, "w") as out_file:
        out_file.write(f'Number Total Animals: {num_animals}\n')
        out_file.write(f'Number Total Cells: {total_cells}\n')

    combined_data.to_pickle(str(pickle_path), compression={'method': 'xz'})
    print(f'Combined data for {file_stem} successfully written to disk!')


def write_epm(combined_data, total_cells, num_animals, output_dir_root, file_stem):
    import pickle

    output_dir = output_dir_root.joinpath(file_stem)
    output_dir.mkdir(exist_ok=True, parents=True)

    pickle_path = output_dir.joinpath(f'{file_stem}-combined.pickle')
    total_file = output_dir.joinpath(f"{file_stem}.txt")

    print(f'Writing combined data for {file_stem} to disk...')
    with open(total_file, "w") as out_file:
        out_file.write(f'Number Total Animals: {num_animals}\n')
        out_file.write(f'Number Total Cells: {total_cells}\n')

    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(combined_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

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


def get_block_maps(block_list, odor_list):
    odor_list_block_number = odor_list.copy()
    for block, odors in block_list.items():
        block_mask = np.isin(odor_list, odors)
        odor_list_block_number[block_mask] = block

    return odor_list_block_number


def combine_HFvFM(files: list):
    combined_data = pd.DataFrame()
    total_cells = 0
    for file in tqdm(files):
        animal_files = files[file]
        combined_data_path = animal_files['file']

        # Load Data
        try:
            cell_data = pd.read_pickle(combined_data_path, compression={'method': 'xz'})
        except Exception:  # yeah yeah I know; I can't remember how they were saved
            cell_data = pd.read_pickle(combined_data_path)

        cell_names = cell_data.columns.get_level_values(0).unique().values  # Get all the unique cells in the multiindex
        num_new_cells = len(cell_names)
        new_numbers = generate_new_numbers(num_new_cells, total_cells)
        trial_order = cell_data[cell_names[0]].columns.values

        # Generate new labels for this set of cells
        new_multiindex = pd.MultiIndex.from_product([new_numbers, trial_order], names=['Cells', 'Trials'])
        cell_data.columns = new_multiindex
        # Create new multiindex with new cell labels and block and apply it to the new data
        combined_data = pd.concat([combined_data, cell_data], axis=1)
        total_cells += num_new_cells

    update_cell_names(combined_data, None)

    return combined_data, total_cells


def combine_odor_data(files: list, cell_class, experiment_type, filter_significant=True, drop_multisense=True, trim_trials=True):
    combined_data = pd.DataFrame()
    combined_significance_table = pd.DataFrame()
    good_cells = 0
    total_cells = 0
    stats = {}

    for file in tqdm(files):
        animal_stats = dict.fromkeys(['trials', 'MO', 'Buzzer', 'insig', 'good_cells', 'orig_cells'], [])
        animal_files = files[file]
        name = file

        # File Paths
        combined_data_path = animal_files['file']
        timestamps_path = animal_files['time']
        significance_file_path = animal_files['sig']
        odor_file_path = animal_files['odor']
        block_file_path = animal_files['block']

        # Load Data
        try:
            cell_data = pd.read_pickle(combined_data_path, compression={'method': 'xz'})
        except Exception:  # yeah yeah I know; I can't remember how they were saved
            cell_data = pd.read_pickle(combined_data_path)
        try:
            timestamps = pd.read_pickle(timestamps_path, compression={'method': 'xz'})
        except Exception:
            timestamps = pd.read_pickle(timestamps_path)


        significance_table = pd.read_excel(significance_file_path, index_col=0, header=[0])
        blocks = pd.read_excel(block_file_path, header=[0])
        odor_data = pd.read_excel(odor_file_path, header=None, usecols=[0]).values
        odor_data = [odor[0] for odor in odor_data]  # Aha, fixed the off-by-one error
        internal_odors = cell_data.columns.get_level_values(1)
        # if needed, reload odors
        odor_data = odor_data[:len(internal_odors)]

        # Count original cells
        orig_cells = cell_data.columns.get_level_values(0).unique()
        num_orig_cells = len(orig_cells)
        animal_stats['orig_cells'] = num_orig_cells

        # # Reset MultiIndex since we may have updated the odor list
        # new_index = pd.MultiIndex.from_product([orig_cells, odor_data], names=['Cells', 'Frames'])
        # cell_data.columns = new_index

        # Get good/bad trials
        trial_indices, trial_indices_to_drop = find_trials(timestamps)

        # Drop trials
        if len(trial_indices_to_drop) == timestamps.shape[1]:
            print(f'Skipping {name}, as all trials are marked to be dropped!')
            continue
        elif trial_indices_to_drop:
            print(f'Dropping {trial_indices_to_drop} from {name}!')
            cell_data = drop_bad_trials(cell_data, trial_indices_to_drop)
            animal_stats['trials'] = trial_indices_to_drop

        if trim_trials:
            cell_data = trim_all_trials(cell_data, trial_indices)
        if drop_multisense:
            cell_data, dropped_MO_cells, dropped_Buzzer_cells = drop_multisense_cells(cell_data, significance_table)
            animal_stats['MO'] = dropped_MO_cells
            animal_stats['Buzzer'] = dropped_Buzzer_cells

            print(f'Dropped {dropped_MO_cells} for having responses to MO!')
            print(f'Dropped {dropped_Buzzer_cells} for having responses Buzzer!')
        if filter_significant:
            cell_data, dropped_insig_cells = drop_nonresponsive_cells(cell_data, significance_table)
            animal_stats['insig'] = dropped_insig_cells
            print(f'Dropped {dropped_insig_cells} for having no significant responses!')

        cell_names = cell_data.columns.get_level_values(0).unique().values  # Get all the unique cells in the multiindex
        num_new_cells = len(cell_names)
        animal_stats['good_cells'] = num_new_cells
        trial_order = cell_data[cell_names[0]].columns.values
        # Get the order of the trials, all cells in this df share this order, so just use the first cell
        block_order = get_block_maps(blocks, trial_order)
        trial_order = odors.normalize_odors(trial_order, experiment_type, cell_class)
        trial_labels = list(zip(trial_order, block_order))
        new_numbers = generate_new_numbers(num_new_cells, good_cells)
        cell_trial_labels = itertools.product(new_numbers, trial_labels)
        cell_trial_tuples = [tuple([item[0]]) + item[1] + tuple([name]) for item in cell_trial_labels]
        # Generate new labels for this set of cells
        new_multiindex = pd.MultiIndex.from_tuples(cell_trial_tuples, sortorder=None, names=['Cells', 'Trials', 'Blocks', 'Animal'])
        cell_data.columns = new_multiindex
        # Create new multiindex with new cell labels and block and apply it to the new data

        combined_data = pd.concat([combined_data, cell_data], axis=1)

        sig_table_odors = significance_table.index.values
        fixed_sig_table_odors = odors.normalize_odors(sig_table_odors, experiment_type, cell_class)
        significance_table.index = fixed_sig_table_odors

        # Includes all cells, INCLUDING any that we dropped above.
        combined_significance_table = pd.concat([combined_significance_table, significance_table], axis=1)

        # update totals
        good_cells += num_new_cells
        total_cells += num_orig_cells

        stats[name] = animal_stats

    update_cell_names(combined_data, combined_significance_table)

    return combined_data, combined_significance_table, stats, (good_cells, total_cells)


def combine_EPM(files: list):
    combined_data = {}
    total_cells = 0
    for file in tqdm(files):
        animal_files = files[file]
        combined_data_path = animal_files['file']

        # Load Data
        try:
            raw_cell_data = pd.read_pickle(combined_data_path, compression={'method': 'xz'})
        except Exception:  # yeah yeah I know; I can't remember how they were saved
            raw_cell_data = pd.read_pickle(combined_data_path)

        uneeded_columns = ['Arms', 'Coordinates', 'Coordinate_Index']
        cell_names = [col for col in raw_cell_data.columns if col not in uneeded_columns]

        num_new_cells = len(cell_names)
        new_numbers = generate_new_numbers(num_new_cells, total_cells)
        new_cell_pairs = zip(cell_names, new_numbers)
        total_cells += num_new_cells
        new_cell_names = ["".join(['C', str(num)]) for num in new_numbers]
        new_cell_pairs = zip(cell_names, new_cell_names)
        arm_location = raw_cell_data['Arms']
        unique_arms = arm_location.unique()

        for original_cell, new_cell in new_cell_pairs:
            dff_per_arm = {}
            for arm in unique_arms:
                arm_mask = arm_location.values == arm
                arm_dff = raw_cell_data.loc[arm_mask, original_cell]
                dff_per_arm[arm] = arm_dff
            combined_data[new_cell] = dff_per_arm


    return combined_data, total_cells


def main():
    animal_types = ['VGLUT', 'VGAT']
    experiment_type = 'EPM'
    input_dir = Path(r'R:\2_Inscopix\1_DTT\3_EPM')
    output_dir_root = Path(r'R:\2_Inscopix\1_DTT\5_Combined\EPM')
    output_dir_root.mkdir(exist_ok=True, parents=True)

    data_files = get_project_files.get_folders(input_dir, experiment_type, animal_types, error=False)

    for _type in animal_types:
        _data_files = data_files[_type]
        stem=f'{experiment_type}-{_type}_Comb'
        num_animals = len(_data_files)

        if experiment_type == 'HFvFM':
            combined_data, total_cells = combine_HFvFM(_data_files)
            write_hffm(combined_data, total_cells, num_animals, output_dir_root, stem)
        elif experiment_type == 'EPM':
            combined_data, total_cells = combine_EPM(_data_files)
            write_epm(combined_data, total_cells, num_animals, output_dir_root, stem)
        else:
            combined_data, combined_significance_table, stats, cells = combine_odor_data(_data_files, _type, experiment_type)
            write_odor(combined_data, combined_significance_table, output_dir_root, stem, stats, cells, num_animals)


if __name__ == "__main__":
    main()
