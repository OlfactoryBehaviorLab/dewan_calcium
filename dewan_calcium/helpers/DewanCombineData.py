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
    animal_folders = [list(folder.iterdir()) for folder in experiment_folders]
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
    odor_data = [load_data(file) for file in odor_data_paths]

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


def get_new_odor_indexes(odor_data):
    new_odor_indexes = []

    for each in odor_data:
        unique_odors = np.unique(each)

        buzzer_index = np.where(unique_odors == 'Buzzer')[0]  # Get indexes for buzzer and MO for later
        MO_index = np.where(unique_odors == 'MO')[0]

        odors = np.delete(unique_odors, [buzzer_index, MO_index])  # Remove MO and Buzzer
        odor_components = [odor.split('-') for odor in odors]  # Split odors into components (modifier and ID)

        odor_components = np.reshape(odor_components, (-1, 2))  # Reshape into matrix of N x 2

        odor_indexes = np.argsort(odor_components[:, 1])  # Sort by the ID column and return indexes
        odor_indexes = np.append(odor_indexes, [MO_index, buzzer_index]) # Lets put our MO and Buzzer back on the end

        new_odor_indexes.append(odor_indexes)

    return new_odor_indexes


def plot_matrix(data, odor_data, experiment_names):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    mpl.rcParams['font.family'] = 'Arial'
    colormap = ListedColormap(['yellow', 'red', 'green'])

    for i, data in enumerate(data):
        plot_names = ['Concentration', 'Identity']
        time = 'Latent'

        collapsed_data = np.vstack(data)
        odors = odor_data[i]
        unique_odors = np.unique(odors)
        num_unique_odors = len(unique_odors)
        num_cells = len(collapsed_data)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(collapsed_data, cmap=colormap, extent=(0, num_unique_odors, num_cells, 0))

        ax.set_yticks(np.arange(num_cells), labels=[])  # Set major ticks but leave them unlabeled
        ax.set_xticks(np.arange(num_unique_odors), labels=[])
        ax.set_yticks(np.arange(0.5, num_cells + 0.5, 1), labels=[], minor=True,
                      fontsize=6)  # Set minor ticks (offset by 0.5) and label them

        ax.set_xticks(np.arange(0.5, num_unique_odors + 0.5, 1), rotation=90, ha='center', labels=unique_odors,
                      minor=True, fontsize=6, fontweight='bold')  # Label the x-axis with the odors
        ax.grid(which='major')  # Show the major grid to make nice little squares

        plt.suptitle(f'{time}, {experiment_names[i]}\n Cell v. Odor Significance Matrix', fontsize=16,
                     fontweight='bold')  # Set the main tit
        plt.tight_layout()
        fig.savefig(f'{time}-{experiment_names[i]}.pdf', dpi=900)