import glob
import os
import pickle
from isx import make_output_file_path, make_output_file_paths


def createProjectFramework() -> None:
    paths = ['./ImagingAnalysis/RawData/',
             './ImagingAnalysis/PreProcessedData',
             './ImagingAnalysis/AUROCImports',
             './ImagingAnalysis/AUROCData',
             './ImagingAnalysis/CombinedData',
             './ImagingAnalysis/Figures/AUROCPlots/LatentCells',
             './ImagingAnalysis/Figures/AUROCPlots/OnTimeCells',
             './ImagingAnalysis/Figures/AllCellTracePlots/LatentCells',
             './ImagingAnalysis/Figures/AllCellTracePlots/OnTimeCells',
             './ImagingAnalysis/Figures/TrialVariancePlots/OnTimeCells',
             './ImagingAnalysis/Figures/TrialVariancePlots/LatentCells',
             ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def saveDataToDisk(data, name, fileHeader, folder) -> None:
    pickle_out = f'./{folder}/{fileHeader}{name}.pickle'
    output_file = open(pickle_out, 'wb')
    pickle.dump(data, output_file, protocol=-1)
    output_file.close()
    print(f'{fileHeader}{name} has been saved!')


def loadDataFromDisk(name, fileHeader, folder) -> object:
    pickle_in = open(f'./{folder}/{fileHeader}{name}.pickle', 'rb')
    data_in = pickle.load(pickle_in)
    pickle_in.close()
    print(f'{fileHeader}{name} has loaded successfully!')
    return data_in


def makeCellFolder4Plot(cell: str, *Folders: list) -> None:
    path = os.path.join('./ImagingAnalysis/Figures/', generateFolderPath(Folders[0]), f'Cell-{cell}')

    if not os.path.exists(path):
        os.makedirs(path)


def generateFolderPath(*Folders) -> os.path:
    path = ''
    for folder in Folders[0]:
        path = os.path.join(path, folder)

    return path


def get_project_files(directory: os.path) -> (str, list, str):
    gpio_file_path = glob.glob(os.path.join(directory, '*.gpio'))[0]
    gpio_file_name = os.path.basename(gpio_file_path)
    video_base = gpio_file_name[:-5]

    video_files = glob.glob(os.path.join(directory, '*.isxd'))
    video_files = [path for path in video_files if 'gpio' not in path]

    return video_base, video_files, gpio_file_path


def check_files(file_list: list):
    for files in file_list:
        if not os.path.exists(files) or not os.path.getsize(files) > 2048:
            return False
    return True


def make_isx_path(input_files, output_dir, addition='', extention='isxd'):
    if len(input_files) == 1:
        return [make_output_file_path(input_files[0], output_dir, addition, ext=extention)]
    else:
        return make_output_file_paths(input_files, output_dir, addition, ext=extention)


def get_outline_coordinates(override_path=None):
    import json
    import numpy as np
    if override_path is None:
        path = '.\\ImagingAnalysis\\RawData\\Cell_Contours.json'
    else:
        path = override_path

    try:
        json_file = open(path)
    except (FileNotFoundError, IOError):
        print("Error loading Cell_Contours File!")
        return None

    json_data = json.load(json_file)
    keys = np.array(list(json_data.keys()))

    json_file.close()

    return keys, json_data


def generate_deinterleaved_video_paths(video_files, output_directory, efocus_vals):
    paths = []
    for each in efocus_vals:
        focus_paths = make_isx_path(video_files, output_directory, addition=each, extention='isxd')
        paths.extend(focus_paths)

    return paths
