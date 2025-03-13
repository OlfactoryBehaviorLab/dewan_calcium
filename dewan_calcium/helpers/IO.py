import os
import pickle
from pathlib import Path

import pandas as pd

if os.environ['ISX'] == '1':
    from isx import make_output_file_path, make_output_file_paths


def save_data_to_disk(data: pd.DataFrame | object, name: str, file_header: str, folder: Path) -> None:
    file_path = folder.joinpath(f'{file_header}{name}.pickle')

    try:
        if type(data) is pd.DataFrame:
            pd.to_pickle(data, file_path)
        else:
            output_file = open(file_path, 'wb')
            pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.close()

        print(f'{file_header}{name} has been saved!')

    except pickle.PicklingError as e:
        print(f"Unable to pickle {file_path.name}!")
        print(e)


def load_data_from_disk(name: str, file_header: str, folder: Path, xlsx=False) -> object:
    ext = 'pickle'
    if xlsx:
        ext = 'xlsx'

    file_path = folder.joinpath(f'{file_header}{name}.{ext}')

    try:
        if not xlsx:
            data_in = pd.read_pickle(file_path)  # Since we don't know what the data is, we can just load with Pandas
        else:
            data_in = pd.read_excel(file_path, index_col=0)

        print(f'{file_header}{name}.{ext} has loaded successfully!')
        return data_in

    except pickle.UnpicklingError as e:
        print(f"Unable to load {file_path.name}!")
        print(e)
        return None


def make_cell_folder4_plot(cell: str | int, *Folders: list) -> None:
    cell_name = f'Cell-{cell}'
    path = Path('ImagingAnalysis', 'Figures', *Folders, cell_name)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)



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


def save_SVM_output(svm_output_dir: Path, mean_score_df: pd.DataFrame, mean_svm_scores: pd.DataFrame,
                    splits_v_repeat_df: pd.DataFrame, all_confusion_mats: dict,
                    true_labels: list[list[int]], pred_labels: list[list[int]], shuffle=False):

    if shuffle:
        svm_output_dir = svm_output_dir.joinpath('Shuffle')
        if not svm_output_dir.exists():
            svm_output_dir.mkdir(parents=True, exist_ok=True)

    mean_score_df.to_excel(svm_output_dir.joinpath('mean_svm_scores.xlsx'))

    mean_scores_path = svm_output_dir.joinpath('mean_svm_scores.pickle')
    pd.to_pickle(mean_svm_scores, mean_scores_path)
    splits_path = svm_output_dir.joinpath('splits_v_repeat_df.pickle')
    pd.to_pickle(splits_v_repeat_df, splits_path)
    all_confusion_mat_path = svm_output_dir.joinpath('all_confusion_mat.pickle')
    pd.to_pickle(all_confusion_mats, all_confusion_mat_path)
    labels_path = svm_output_dir.joinpath('labels.pickle')
    pd.to_pickle((true_labels, pred_labels), labels_path)

    print('Successfully saved SVM data!')


def verify_input(var_name, input_var, allowed_types, allowed_values=None, allowed_range=None, inclusive=False):
    if type(input_var) not in allowed_types:
        raise TypeError(f'{var_name} has type of {type(input_var)}, but must be of type {allowed_types}')
    if allowed_values and input_var not in allowed_values:
        raise ValueError(f'{var_name} has value of {input_var}, but must be one of {allowed_values}')
    if allowed_range:
        if inclusive:
            if input_var < allowed_range[0] or input_var >= allowed_range[1]:
                raise ValueError(f'{var_name} has value of {input_var}, but must be between {allowed_range[0]} and {allowed_range[1]} inclusive')
        else:
            if input_var < allowed_range[0] or input_var > allowed_range[1]:
                raise ValueError(f'{var_name} has value of {input_var}, but must be between {allowed_range[0]} and {allowed_range[1]} noninclusive')
