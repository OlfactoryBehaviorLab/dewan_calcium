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

        excititory_cells = np.where(excititory_rows_mask)[0]
        inhibitory_cells = np.where(inhibitory_rows_mask)[0]
        excit_inhib_cells = np.where(excit_inhib_row_mask)[0]

        sorted_dataset.append([excititory_cells, excit_inhib_cells, inhibitory_cells])

    return sorted_dataset