import pickle
import os
from pathlib import Path
from isx import make_output_file_path, make_output_file_paths


def create_project_framework() -> None:
    paths = ['./ImagingAnalysis/RawData/',
             './ImagingAnalysis/PreProcessedData',
             './ImagingAnalysis/AUROCImports',
             './ImagingAnalysis/AUROCData',
             './ImagingAnalysis/CombinedData',
             './ImagingAnalysis/Statistics',
             './ImagingAnalysis/Figures/Statistics',
             './ImagingAnalysis/Figures/AUROCPlots/LatentCells',
             './ImagingAnalysis/Figures/AUROCPlots/OnTimeCells',
             './ImagingAnalysis/Figures/AllCellTracePlots/LatentCells',
             './ImagingAnalysis/Figures/AllCellTracePlots/OnTimeCells',
             './ImagingAnalysis/Figures/TrialVariancePlots/OnTimeCells',
             './ImagingAnalysis/Figures/TrialVariancePlots/LatentCells',
             ]
    paths = [Path(path) for path in paths]

    for path in paths:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


def save_data_to_disk(data, name, fileHeader, folder) -> None:
    file_path = Path(*folder).joinpath(f'{fileHeader}{name}.pickle')

    output_file = open(file_path, 'wb')
    pickle.dump(data, output_file, protocol=-1)
    output_file.close()
    print(f'{fileHeader}{name} has been saved!')


def load_data_from_disk(name, fileHeader, folder) -> object:
    file_path = Path(*folder).joinpath(f'{fileHeader}{name}.pickle')

    pickle_in = open(file_path, 'rb')
    data_in = pickle.load(pickle_in)
    pickle_in.close()
    print(f'{fileHeader}{name} has loaded successfully!')
    return data_in


def make_cell_folder4_plot(cell: str or int, *Folders: list) -> None:
    cell_name = f'Cell-{cell}'
    path = Path('ImagingAnalysis', 'Figures', *Folders, cell_name)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_project_files(directory: str) -> (str, list, Path):
    """
    Takes an input raw data folder and generates a list of video files, the gpio file, and the video base
    (file name minus extension).

    Args:
        directory (string):
            Folder containing the raw data for the project
    Returns:
        tuple (string, list, string)
            0) Video Base (file name minus extension)
            1) List of video files in the folder excluding the GPIO file
            2) Path to the GPIO file
        tuple (None, None, None):
            If one of the file types cannot be found, function returns None
    """
    data_folder = Path(directory)

    try:
        session_json_path = Path(sorted(data_folder.glob('*.json'))[0])
    except IndexError:
        print(f'session.json not found in {data_folder.absolute()}')
        session_json_path = None

    try:
        gpio_file_path = Path(sorted(data_folder.glob('*.gpio'))[0])
        video_base = gpio_file_path.stem
    except IndexError:
        print(f'GPIO File not found in {data_folder.absolute()}')
        gpio_file_path = None

    try:
        video_files = sorted(data_folder.glob('*.isxd'))
        video_files = [str(path) for path in video_files if 'gpio' not in path.name]
        # After processing is run once, an isxd file with gpio in the name is generated.
        # If processing is run again, this will ignore that file and only load the video files
    except IndexError:
        print(f'No video file(s) found in {data_folder.absolute()}')
        video_files = None

    return video_base, video_files, gpio_file_path, session_json_path


def check_files(input_files: list or None, output_files: list or None) -> bool:
    from numpy import hstack
    """
    Check whether each file in a list of files exists

    Args:
        input_files (list of strings or list of lists):
            List of input files to check
        output_files (list of strings or list of lists):
            List of output files to check

    Returns:
        bool:
            False if an input file is missing, or an output already exists
            True if all the input files are present and the output file does not exist
    """
    if input_files is not None:
        input_files = hstack(input_files)
        for infile in input_files:
            if not Path(infile).exists():
                print(f'Input file {infile} is missing!')
                return False

    if output_files is not None:
        output_files = hstack(output_files)
        for outfile in output_files:
            if Path(outfile).exists():
                print(f'Output file {outfile} already exists!')
                return False

    return True


def make_isx_path(input_files: list[str], output_dir: str, addition: str = '', extension: str = 'isxd') \
        -> str or list[str]:
    """
    The Inscopix API has two convenient functions for taking lists of files and adding output directories, extra labels,
    and new extensions. There are times when there is only one video, and other times when we have multiple videos.
    This function runs the appropriate Inscopix function based on the number of video files.

    Args:
        input_files (list of strings):
            Input video files to create paths for
        output_dir (string):
            Folder to place new files in. Each input file will be appended to this path.
        addition (string):
            Addition information to add at the end of the file name.
            Default Value: Empty String ('')
        extension (string):
            File extension for the file path.
            Default Value: 'isxd'


    Returns:
        path(s) (list of strings):
            A list containing the newly generated file paths
    """
    if len(input_files) == 1:
        return [make_output_file_path(input_files[0], output_dir, addition, ext=extension)]
    else:
        return make_output_file_paths(input_files, output_dir, addition, ext=extension)


def get_outline_coordinates(path: os.PathLike):
    import json
    import numpy as np

    try:
        json_file = open(path)
    except (FileNotFoundError, IOError):
        print("Error loading Cell_Contours File!")
        return None

    json_data = json.load(json_file)
    keys = np.array(list(json_data.keys()))

    json_file.close()

    return keys, json_data


def generate_deinterleaved_video_paths(video_files: list[str], output_directory: str, efocus_vals: list[int]) -> list:
    """
    When deinterleaving a video file into its component focal planes, we must pass the de_interleave function a list
    of file paths. The paths have to be in the form [video1_focal1, video1_focal2, video2_focal1, video2_focal2, etc.]
    This function takes a list of video files and focal planes and outputs all the needed file paths.

    Args:
        video_files (list of strings):
            List of the video files to generate paths for.
        output_directory (str):
            Output path to place the new video files into. Video files will be appended to this path.
        efocus_vals (list of ints):
            List of the focal planes used in the experiment

    Returns:
        paths (list of strings):
            A list containing all the generated video_file-focal_plane paths used in the deinterleaving process
    """

    paths = []
    for each in efocus_vals:  # Iterate through each focal plane and generate file paths
        focus_paths = make_isx_path(video_files, output_directory, addition=str(each), extension='isxd')
        paths.extend(focus_paths)

    return paths
