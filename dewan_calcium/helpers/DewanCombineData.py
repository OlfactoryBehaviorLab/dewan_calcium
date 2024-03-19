import pickle

import numpy as np
from pathlib import Path


def load_data(path: Path):
    # Function to load raw data from pickled objects
    pickle_in = open(path, 'rb')
    data_in = pickle.load(pickle_in)
    pickle_in.close()
    return data_in


def load_paths(root_path: Path):
    contents = list(root_path.iterdir())

    experiment_folders = [each for each in contents if each.is_dir()]
    # Get folders for each experiment type
    animal_folders = [[animal for animal in folder.iterdir() if '4' not in str(animal)] for folder in experiment_folders]
    # For now, we need to exclude VGAT-4 until it gets fixed

    preprocess_paths = [[path.joinpath('ImagingAnalysis/PreProcessedData') for path in animal_folder] for animal_folder
                        in animal_folders]
    auroc_data_paths = [[path.joinpath('ImagingAnalysis/AUROCData') for path in animal_folder] for animal_folder in
                        animal_folders]

    odor_data_path = []
    for i in range(len(preprocess_paths)):
        odor_data_path.append(list(preprocess_paths[i][0].glob('*OdorData.pickle'))[0])
        # Get one OdorFile for each experiment as all odors are the same

    return preprocess_paths, auroc_data_paths, odor_data_path, experiment_folders


def get_data_file_paths(auroc_data_paths: list[list[Path]]):
    latent_auroc_paths = []
    ontime_auroc_paths = []

    for i in range(len(auroc_data_paths)):  # Loop through each experiment
        ontime_experiment_paths = []
        latent_experiment_paths = []
        experiment_auroc_data_paths = auroc_data_paths[i]  # AUROC Data folders for an experiment

        for each in experiment_auroc_data_paths:  # Loop through each file and find the onTime and latent files
            try:
                files = list(each.iterdir())
                for file in files:
                    file_str = str(file)
                    if 'SignificanceTable' in file_str:
                        if 'onTime' in file_str:
                            ontime_experiment_paths.append(file)
                        elif 'latent' in file_str:
                            latent_experiment_paths.append(file)
            except Exception as e:  # This is fine
                print(f'Error: {str(each)}')
                continue
        ontime_auroc_paths.append(ontime_experiment_paths)
        latent_auroc_paths.append(latent_experiment_paths)

    return ontime_auroc_paths, latent_auroc_paths


def load_raw_data(ontime_auroc_paths: list[list[Path]], latent_auroc_paths: list[list[Path]], odor_data_paths: list[Path]):
    on_time_data = [[load_data(file) for file in files] for files in ontime_auroc_paths]
    latent_data = [[load_data(file) for file in files] for files in latent_auroc_paths]
    odor_data = [np.ravel(load_data(file)) for file in odor_data_paths]   # Flatten Odor Data

    return on_time_data, latent_data, odor_data


def coalesce_cells(dataset: list):
    stacked_data = []
    for i in range(len(dataset)):
        stacked_data.append(np.vstack(dataset[i]))
    return stacked_data


def remove_nonzero_cells(dataset: list):
    filtered_dataset = []

    for experiment in dataset:
        nonzero_rows = np.nonzero(experiment)[0]
        nonzero_row_index = np.unique(nonzero_rows)
        filtered_dataset.append(experiment[nonzero_row_index])

    return filtered_dataset


def sort_by_responses(dataset: list):
    # Hey this computer science concept actually helped here
    sorted_dataset = []  # 0: Excititory, 1: Both, 2:Inhibitory

    for experiment in dataset:  # Each dataset will contain at least two experiments 'CONC' and 'ID'
        excititory_rows = [np.isin(2, row) for row in experiment]  # Rows that contain at least one 2
        inhibitory_rows = [np.isin(1, row) for row in experiment]  # Rows that contain at least one 1
        excit_inhib_row_mask = np.logical_and(excititory_rows, inhibitory_rows)  # Rows that contain at least one 1 AND 2

        removal_mask = np.logical_not(excit_inhib_row_mask)  # Invert the rows that contain both so they can be removed
        excititory_rows_mask = np.logical_and(excititory_rows, removal_mask)  # Remove any rows that appear in both
        inhibitory_rows_mask = np.logical_and(inhibitory_rows, removal_mask)  # Remove any rows that appear in both

        excititory_cell_indexes = np.where(excititory_rows_mask)[0]
        inhibitory_cells_indexes = np.where(inhibitory_rows_mask)[0]
        excit_inhib_cells_indexes = np.where(excit_inhib_row_mask)[0]

        excititory_cells = experiment[excititory_cell_indexes]
        inhibitory_cells = experiment[inhibitory_cells_indexes]
        excit_inhib_cells = experiment[excit_inhib_cells_indexes]

        sorted_dataset.append([excititory_cells, excit_inhib_cells, inhibitory_cells])

    return sorted_dataset


def sort_by_sum(dataset: list):

    sorted_dataset = []

    for experiment in dataset:
        sorted_cell_types = []
        for cell_type in experiment:
            sum_of_rows = np.sum(cell_type, 1)
            new_indexes = np.flip(np.argsort(sum_of_rows, kind='mergesort'))
            sorted_cells = cell_type[new_indexes]
            sorted_cell_types.append(sorted_cells)

        sorted_dataset.append(sorted_cell_types)

    return sorted_dataset


def sort_by_length(dataset: list):
    cells_by_experiment = []
    for i, experiment in enumerate(dataset):  # 0: Conc, 1: ID
        cells_by_type = []
        for j, cell_type in enumerate(experiment):  # 0: Excitatory, 1: Inhibitory, 2: Combo
            cells_by_time = []
            for k, time in enumerate(cell_type):  # 0: On Time, 1: Latent
                responses = np.where(time != 0)[0]
                values, counts = np.unique(responses, return_counts=True)
                new_indexes = np.flip(np.argsort(counts))
                sorted_cells = list(time[new_indexes])
                cells_by_time.append(sorted_cells)
            cells_by_type.append(cells_by_time)
        cells_by_experiment.append(cells_by_type)

    return cells_by_experiment


def get_new_odor_indexes(odor_data):
    new_odor_indexes = []
    new_unique_odors = []
    for each in odor_data:
        unique_odors = np.unique(each)

        buzzer_index = np.where(unique_odors == 'Buzzer')[0]  # Get indexes for buzzer and MO for later
        MO_index = np.where(unique_odors == 'MO')[0]

        odors = np.delete(unique_odors, [buzzer_index, MO_index])  # Remove MO and Buzzer
        odor_components = [odor.split('-') for odor in odors]  # Split odors into components (modifier and ID)

        odor_components = np.reshape(odor_components, (-1, 2))  # Reshape into matrix of N x 2

        unique_IDs = np.unique(odor_components[:, 1])  # Get the unique odor IDs
        sorted_unique_IDs = np.sort(unique_IDs)
        # Sort the odor IDs here so the data_indexes is already in the preferred ID order

        data_indexes = []
        for ID in unique_IDs:
            data_indexes.append(np.where(odor_components[:, 1] == ID)[0])  # Get the indexes binned by ID

        odor_indexes = []
        for indexes in data_indexes:  # Loop through each odor ID
            modifiers = odor_components[indexes, 0]  # Get the modifiers for this specific bin (carbons or ppm)
            new_indexes = np.argsort(modifiers)  # Sort the modifiers in ascending order
            odor_indexes.extend(indexes[new_indexes])  # Reindex the bins by the modifier
            # Since we sorted the bins, and then sorted the odors within each bin, the whole list should be sorted

        odor_indexes = np.append(odor_indexes, [MO_index, buzzer_index])  # Let's put our MO and Buzzer back on the end

        new_odor_indexes.append(odor_indexes)
        new_unique_odors.append(unique_odors[odor_indexes])

    return new_odor_indexes, new_unique_odors


def sort_data_by_odor_indexes(odor_data, new_odor_indexes):
    new_data = []

    for i, experiment in enumerate(odor_data):
        new_indexes = new_odor_indexes[i]
        sorted_rows = []
        for each in experiment:
            row = each[new_indexes]
            sorted_rows.append(row)
        new_data.append(sorted_rows)

    return new_data


def relabel_latent_data(dataset: list):
    latent_data = []

    for experiment in dataset:
        experiment_data = []
        for cell_type in experiment:
            cell_type_data = np.array(cell_type)
            cell_type_data[cell_type_data == 1] = 3
            cell_type_data[cell_type_data == 2] = 4
            experiment_data.append(cell_type_data)
        latent_data.append(experiment_data)

    return latent_data


def combine_data(on_time_data: list, latent_data: list):
    combined_data = []

    for i, experiment in enumerate(on_time_data):
        stacked_cells = []
        for j, cell_type in enumerate(experiment):
            cells = [cell_type, latent_data[i][j]]
            stacked_cells.append(cells)
        combined_data.append(stacked_cells)

    return combined_data


def plot_matrix(data, unique_odors, experiment_names, time):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    mpl.rcParams['font.family'] = 'Arial'
    colormap = ListedColormap(['yellow', 'red', 'green', 'orange', 'cyan'])

    for i, data in enumerate(data):
        collapsed_data = np.vstack(data)
        exp_unique_odors = unique_odors[i]
        num_unique_odors = len(exp_unique_odors)
        num_cells = len(collapsed_data)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(collapsed_data, cmap=colormap, extent=(0, num_unique_odors, num_cells, 0))

        ax.set_yticks(np.arange(num_cells), labels=[])  # Set major ticks but leave them unlabeled
        ax.set_xticks(np.arange(num_unique_odors), labels=[])
        ax.set_yticks(np.arange(0.5, num_cells + 0.5, 1), labels=[], minor=True,
                      fontsize=6)  # Set minor ticks (offset by 0.5) and label them

        ax.set_xticks(np.arange(0.5, num_unique_odors + 0.5, 1), rotation=90, ha='center', labels=exp_unique_odors,
                      minor=True, fontsize=6, fontweight='bold')  # Label the x-axis with the odors
        ax.grid(which='major')  # Show the major grid to make nice little squares

        plt.suptitle(f'{time}, {experiment_names[i]}\n Cell v. Odor Significance Matrix', fontsize=16,
                     fontweight='bold')  # Set the main tit
        plt.tight_layout()
        fig.savefig(f'{time}-{experiment_names[i]}.eps', dpi=200)  # Save as vector image
