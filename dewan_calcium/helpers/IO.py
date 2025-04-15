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


def load_SVM_data(input_dir):

    mean_scores_path = input_dir.joinpath('mean_svm_scores.pickle')
    mean_svm_scores = pd.read_pickle(mean_scores_path)
    splits_path = input_dir.joinpath('splits_v_repeat_df.pickle')
    splits_v_repeat_df = pd.read_pickle(splits_path)
    all_confusion_mat_path = input_dir.joinpath('all_confusion_mat.pickle')
    all_confusion_mats = pd.read_pickle(all_confusion_mat_path)
    labels_path = input_dir.joinpath('labels.pickle')
    (true_labels, pred_labels) = pd.read_pickle(labels_path)

    SVM_data = (mean_svm_scores, splits_v_repeat_df, all_confusion_mats, true_labels, pred_labels)

    print('Successfully loaded SVM output!')

    shuffle_input_dir = input_dir.joinpath('Shuffle')

    shuffled_mean_scores_path = shuffle_input_dir.joinpath('mean_svm_scores.pickle')
    shuffled_mean_svm_scores = pd.read_pickle(shuffled_mean_scores_path)
    shuffled_splits_path = shuffle_input_dir.joinpath('splits_v_repeat_df.pickle')
    shuffled_splits_v_repeat_df = pd.read_pickle(shuffled_splits_path)
    shuffled_all_confusion_mat_path = shuffle_input_dir.joinpath('all_confusion_mat.pickle')
    shuffled_all_confusion_mats = pd.read_pickle(shuffled_all_confusion_mat_path)
    shuffled_labels_path = shuffle_input_dir.joinpath('labels.pickle')
    (shuffled_true_labels, shuffled_pred_labels) = pd.read_pickle(shuffled_labels_path)

    shuffled_SVM_data = (shuffled_mean_svm_scores, shuffled_splits_v_repeat_df, shuffled_all_confusion_mats, shuffled_true_labels, shuffled_pred_labels)
    print('Successfully loaded shuffled SVM output!')

    return SVM_data, shuffled_SVM_data


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


def save_wilcoxon_data(wilcoxon_total, wilcoxon_binned, cell_bins, p_val, corrected_p_val, output_dir):
    _handle = lambda writer: _write_wilcoxon_data(wilcoxon_total, wilcoxon_binned, cell_bins, p_val, corrected_p_val, writer)
    _write_excel(None, output_dir, 'binned_wilcoxon.xlsx', handle=_handle)


def save_auroc_data(significance_table, auroc_values_df, AUROC_data, file_header, output_dir):
    auroc_values_file_name = f'{file_header}AUROCValues.xlsx'
    _write_excel(auroc_values_df, output_dir, auroc_values_file_name, index=False)
    all_return_values_file_name = f'{file_header}AllReturnValues.xlsx'
    _write_excel(AUROC_data, output_dir, all_return_values_file_name)

    significance_table_file_name = f'{file_header}SignificanceTable.xlsx'
    _handle = lambda writer: _write_significance_table(significance_table, writer)
    _write_excel(None, output_dir, significance_table_file_name, handle=_handle)


def save_sorted_significance_table(significance_table, output_dir):
    significance_table_file_name = f'SignificanceTable_sorted.xlsx'
    _handle = lambda writer: _write_significance_table(significance_table, writer)
    _write_excel(None, output_dir, significance_table_file_name, handle=_handle)


def _write_wilcoxon_data(wilcoxon_total, wilcoxon_binned, cell_bins, p_val, corrected_p_val, writer):
    wilcoxon_total.to_excel(writer, sheet_name='Evoked Period')
    wilcoxon_binned.to_excel(writer, sheet_name='Bins')
    cell_bins.to_excel(writer, sheet_name='Significant Bins')

    workbook = writer.book
    total_ws = writer.sheets['Evoked Period']
    binned_ws = writer.sheets['Bins']

    green_bg = workbook.add_format({'bg_color': '#22f229'})
    blue_bg = workbook.add_format({'bg_color': '#22adf2'})
    total_cf = {
        'type': 'cell',
        'criteria': '<',
        'value': p_val,
        'format': green_bg
    }

    corrected_cf = {
        'type': 'cell',
        'criteria': '<',
        'value': corrected_p_val,
        'format': blue_bg
    }

    (max_row, max_col) = wilcoxon_total.shape
    total_ws.conditional_format(1, 1, max_row, max_col, total_cf)
    (max_row, max_col) = wilcoxon_binned.shape
    binned_ws.conditional_format(1, 1, max_row, max_col, corrected_cf)


def _write_significance_table(significance_table, writer):
    significance_table.to_excel(writer, sheet_name='Sig Table')

    workbook = writer.book
    sig_table = writer.sheets['Sig Table']

    green_bg = workbook.add_format({'bg_color': '#22f229'})
    red_bg = workbook.add_format({'bg_color': '#fc0303'})

    excitatory_cf = {
        'type': 'cell',
        'criteria': '=',
        'value': 1,
        'format': green_bg
    }

    inhibitory_cf = {
        'type': 'cell',
        'criteria': '=',
        'value': -1,
        'format': red_bg
    }

    (max_row, max_col) = significance_table.shape
    sig_table.conditional_format(1, 1, max_row, max_col, excitatory_cf)
    sig_table.conditional_format(1, 1, max_row, max_col, inhibitory_cf)


def _write_excel(df, output_dir, file_name, handle=None, **kwargs):
    file_path = output_dir.joinpath(file_name)
    try:
        if not handle:  # Write file directly
            df.to_excel(file_path, **kwargs)
        else:  # Open context manager and execute the handle the user passed in
            with pd.ExcelWriter(file_path, engine='xlsxwriter', **kwargs) as writer:
                handle(writer)
    except PermissionError as e:
        raise PermissionError(f'Cannot write to {file_path} Is it still open?') from e

    print(f'Successfully wrote {file_name} to disk!')