"""
Module containing classifier functions and related helpers
Author: Austin Pauley (pauley@psy.fsu.edu)
Date Created: 10/30/2024

Most of the code in this module is either directly from, or heavily influenced by, code from
The Vincis Lab at Florida State University (https://github.com/vincisLab/thermalGC)
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm, trange


def _run_svm(traces: pd.DataFrame, trial_labels: pd.Series, test_percentage: float = 0.2, num_splits: int = 20):
    """

    Args:
        traces (pd.DataFrame): DataFrame containing preprocessed 1-P calcium transient data. DataFrame is size n x t,
        where each row is a trial and each column is a sample (total samples equal to t).
        trial_labels (pd.Series): Series of labels corresponding to each trial (total trials equal to n)
        test_percentage (float): Percentage of data to use as test set
        num_splits (int): Number of splits to use for cross-validation

    Returns:
        split_scores (np.array): List of floats representing the score for each split
        cm (list[confusion_matrix]): Confusion matrix of all results
        true_labels (list[list[int]]): List containing the list of correct labels for each cross-validation split
        pred_labels (list[list[int]]): List containing the list of predicted labels for each cross-validation split
    """

    svm = LinearSVC(dual='auto', max_iter=10000, random_state=1000)

    true_labels = []
    pred_labels = []
    split_scores = []
    cms = []

    for _ in trange(num_splits):
        train_trials, test_trials, train_labels, test_labels = train_test_split(
            traces, trial_labels, test_size=test_percentage, shuffle=True, stratify=trial_labels)

        svm.fit(train_trials, train_labels)
        svm_score = svm.score(test_trials, test_labels)
        split_scores.append(svm_score)
        true_labels.append(test_labels)
        svm_prediction = svm.predict(test_trials)
        pred_labels.append(svm_prediction)
        cm = confusion_matrix(test_labels, svm_prediction, normalize='true')
        cms.append(cm)

    return split_scores, cms, true_labels, pred_labels


def _decode_single_neuron(cell, combined_data, num_splits, test_percentage):

    cell_data = combined_data[cell].T
    cell_data = cell_data.dropna(axis=1)  # We lose a few trials occasionally due to concatenation
    correct_labels = cell_data.index.to_series(name='correct_labels')

    svm_scores, confusion_mat, y_true, y_pred = _run_svm(cell_data, correct_labels, test_percentage=test_percentage,
                                        num_splits=num_splits)
    svm_score_average = np.mean(svm_scores)

    return svm_score_average, svm_scores, confusion_mat, (y_true, y_pred)


def _get_minimum_trials(combined_data, cell_names, odors):
    min_odor_trials = {}

    for odor in odors:
        odor_trials = []
        for cell in cell_names:
            data = combined_data[cell][odor]
            num_trials = data.shape[1]  # Each column is a trial here
            odor_trials.append(num_trials)
        min_odor_trials[odor] = (np.min(odor_trials))

    return min_odor_trials


def _generate_dataframe_index(odor_mins):
    all_trial_labels = []
    for odor in odor_mins:
        min_odor_trials = odor_mins[odor]
        labels = np.repeat(odor, min_odor_trials).astype(str)
        all_trial_labels.extend(labels)

    return all_trial_labels


def _randomly_sample_trials(z_score_combined_data, combined_data_index, cell_names, trial_labels, odor_mins, window=None):
    data_per_trial = pd.DataFrame()

    progress_desc = 'Randomly Sampling Cells'

    if window:
        progress_desc = progress_desc + f' for window size {window}'

    for cell_i, cell in enumerate(tqdm(cell_names, desc=progress_desc, leave=False, position=2)):
        trial_num = 0
        cell_data = pd.DataFrame()

        if window:
            start_frame, end_frame = window

        # Iterate through each taste and select the appropriate number of trials
        for odor_i, odor in enumerate(trial_labels):
            cell_odor_data = z_score_combined_data[cell][odor].T
            if window:
                cell_odor_data = cell_odor_data.iloc[:, start_frame:end_frame]
            random_trials = cell_odor_data.sample(odor_mins[odor], axis=0)
            cell_data = pd.concat([cell_data, random_trials], ignore_index=True)

        data_per_trial = pd.concat([data_per_trial, cell_data], axis=1)

    data_per_trial.index = combined_data_index
    data_per_trial = data_per_trial.dropna(axis=1)
    data_per_trial.columns = np.arange(0, data_per_trial.shape[1])

    return data_per_trial


def _get_windows(data_size, steps):
    windows_end = np.arange(steps, data_size, steps)
    final_index = data_size - 1

    dangling_frames = final_index % steps
    if dangling_frames > 0:
        windows_end = np.hstack([windows_end, final_index])

    windows_start = windows_end - steps

    return list(zip(windows_start, windows_end))


def _shuffle_index(df):
    rng = np.random.default_rng()
    index = df.index.values
    rng.shuffle(index)
    df.index = index
    return df


def _decode_ensemble(z_scored_combined_data, test_percentage, num_splits, iterator, loop_message, window=False):

    true_labels = {}
    pred_labels = {}
    all_split_scores = {}
    all_confusion_mats = {}
    mean_svm_scores = {}

    cells = np.unique(z_scored_combined_data.columns.get_level_values(0))
    odors = z_scored_combined_data.columns.get_level_values(1).unique().values

    data_size = z_scored_combined_data.shape[0]
    windows = get_windows(data_size, window_size)
    odor_mins = _get_minimum_trials(z_scored_combined_data, cells, odors)
    combined_data_index = _generate_dataframe_index(odor_mins)

    for window in tqdm(windows, desc='Sliding Window Ensemble Decoding', leave=True, position=0):
        cropped_data_per_trial = randomly_sample_trials(z_scored_combined_data,
                                                        combined_data_index, cells, odors, odor_mins, window=window)

        correct_labels = cropped_data_per_trial.index.to_series()

        split_scores, cm, true_label, pred_label = run_svm(cropped_data_per_trial, correct_labels, test_percentage, num_splits)
        avg_score = np.mean(split_scores)
        mean_svm_scores[window] = avg_score

        all_split_scores[window] = split_scores
        all_confusion_mats[window] = cm
        true_labels[window] = true_label  # Record the 'true' taste
        pred_labels[window] = pred_label

    splits_v_repeat_df = pd.DataFrame(all_split_scores)

    return mean_svm_scores, splits_v_repeat_df, all_confusion_mats, (true_labels, pred_labels)


def _shuffle_index(df):
    rng = np.random.default_rng()
    index = df.index.values
    rng.shuffle(index)
    df.index = index
    return df


def shuffle_data(z_scored_combined_data):
    shuffled_data = z_scored_combined_data.T.groupby(level=0, group_keys=False).apply(_shuffle_index)
    shuffled_data.index = pd.MultiIndex.from_tuples(shuffled_data.index, names=['cell', 'odor'])
    shuffled_data = shuffled_data.T

    return shuffled_data
