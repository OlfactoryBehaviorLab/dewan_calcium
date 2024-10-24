from pathlib import Path
import pandas as pd
from tqdm import tqdm

input_dir = Path(r'R:\2_Inscopix\1_DTT\2_HFvFM')
output_dir_root = Path(r'R:\2_Inscopix\1_DTT\4_Combined')


def combine_data(data_files: list[Path], exp_type: str):
    combined_data = pd.DataFrame()

    total_num_cells = 0

    for file in tqdm(data_files, desc='Processing files: '):
        new_data = pd.read_pickle(str(file))
        current_cell_names = new_data.columns.levels[0].values  # Get all the unique cells in the multiindex
        num_new_cells = len(current_cell_names)
        trial_order = new_data[current_cell_names[0]].columns.values
        # Get the order of the trials, all cells in this df share this order, so just use the first cell

        new_numbers = generate_new_numbers(num_new_cells, total_num_cells)
        # Generate new labels for this set of cells

        new_multiindex = pd.MultiIndex.from_product([new_numbers, trial_order], sortorder=None, names=['Cells', 'Trials'])
        new_data.columns = new_multiindex
        # Create new multiindex with new cell labels and apply it to the new data

        combined_data = pd.concat([combined_data, new_data], axis=1)

        total_num_cells += num_new_cells

    update_cell_names(combined_data)

    return combined_data


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


def find_data_files(exp_type: str):
    print('Searching for data files, this may take a while...')

    data_files = []
    if exp_type == 'CONC' or exp_type == 'ID':
        data_files = input_dir.glob('*\\Analysis\\Output\\combined\\*combined_data_shift.pickle')
    elif exp_type == 'EPM':
        data_files = input_dir.glob('*\\Analysis\\Output\\pseudotrials\\*pseudotrial_traces.pickle')
    elif exp_type == 'HFvFM':
        data_files = input_dir.glob('*\\Analysis\\Output\\combined\\*combined_data.pickle')

    if not data_files:
        raise FileNotFoundError(f'No data files found in {input_dir}')
    else:
        data_files = list(data_files)
        print(f'Found {len(data_files)} data files.')
        return list(data_files)


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

def main():
    output_dir = []
    combined_data = []

    exp_type = get_exp_type()
    data_files = find_data_files(exp_type)

    if exp_type == 'CONC':
        output_dir = output_dir_root.joinpath('Concentration')
        combined_data = combine_odor_data(data_files)
    elif exp_type == 'ID':
        output_dir = output_dir_root.joinpath('Identity')
        combined_data = combine_odor_data(data_files)
    elif exp_type == 'EPM':
        output_dir = output_dir_root.joinpath('EPM')
    elif exp_type == 'HFvFM':
        output_dir = output_dir_root.joinpath('HFvFM')


    output_path = output_dir.joinpath(f'{exp_type}-combined.pickle')

    if not output_dir.exists():
        output_dir.mkdir()

    pd.to_pickle(combined_data, output_path)


if __name__ == "__main__":
    main()
