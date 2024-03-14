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

    return (preprocess_paths, auroc_data_paths, odor_data_path, experiment_folders)
