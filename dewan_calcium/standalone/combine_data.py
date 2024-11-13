import os
os.environ['ISX'] = '0'

from dewan_calcium.standalone.old_to_new import old_to_new

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from . import old_to_new
from . import get_project_files

input_dir = Path(r'R:\2_Inscopix\1_DTT\1_OdorAnalysis\2_Identity')
output_dir_root = Path(r'R:\2_Inscopix\1_DTT\4_Combined\Identity')


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

        new_data = pd.read_pickle(str(data_file))
        significance_data = pd.read_excel(str(significance_file))
        new_data = strip_insignificant_cells(new_data, significance_data)
        new_data = strip_multisensory_trials(new_data)
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

                collected_data, total_cells = combine_data(data_files, filter_significant, _type, )

                output_dir = output_dir_root.joinpath(exp_type)

                file_stem = f'{_type}-{exp_type}'

                write_to_disk(collected_data, output_dir, file_stem, total_cells, len(data_files))


def main():
#     file = {
#         'file': r'R:\2_Inscopix\1_DTT\1_OdorAnalysis\2_Identity\VGAT\VGAT-42\Analysis\Output\combined\VGAT42_ID-8_19_24-combined_data.pickle',
#         'sig':  r'R:\2_Inscopix\1_DTT\1_OdorAnalysis\2_Identity\VGAT\VGAT-42\Analysis\Output\VGAT42_ID-8_19_24-SignificanceTable.xlsx',
#         'old': False
#     }
#
#     files = {
#         'VGAT': [file]
#     }
    # combine_and_save(files, 'Identity')
    #
    # return
    animal_types = ['VGAT', 'VGLUT']
    files = {_type: [] for _type in animal_types}

    exp_type = get_exp_type()
    folders = get_project_files.get_folders(input_dir, animal_types)

    for _type in animal_types:
        for folder in tqdm(folders[_type], desc=f'Getting files for {_type} animals'):
            files[_type].append(get_project_files.new_find_data_files(folder, exp_type))

    files = old_to_new.new_old_to_new(files)

    combine_and_save(files, exp_type)


if __name__ == "__main__":
    main()
