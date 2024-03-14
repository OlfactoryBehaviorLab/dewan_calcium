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
    for i in range(len(dataset)):
        dataset[i] = np.vstack(dataset[i])

    return dataset


def remove_nonzero_cells(dataset: list):
    filtered_dataset = []

    for experiment in dataset:
        nonzero_rows = np.nonzero(experiment)[0]
        nonzero_row_index = np.unique(nonzero_rows)
        filtered_dataset.append(experiment[nonzero_row_index])

    return filtered_dataset