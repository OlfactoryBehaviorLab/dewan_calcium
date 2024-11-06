"""
Module containing classifier functions and related helpers
Author: Austin Pauley (pauley@psy.fsu.edu)
Date Created: 10/30/2024

Most of the code in this module is either directly from, or heavily influenced by, code from
The Vincis Lab at Florida State University (https://github.com/vincisLab/thermalGC)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange


def run_svm(traces: pd.DataFrame, correct_labels: pd.Series, test_percentage: float = 0.2, num_splits: int =20 ):
    """

    Args:
        traces (pd.DataFrame): DataFrame containing preprocessed 1-P calcium transient data. DataFrame is size n x t,
        where each row is a trial and each column is a sample (total samples equal to t).
        correct_labels (pd.Series): Series of labels corresponding to each trial (total trials equal to n)
        test_percentage (float): Percentage of data to use as test set
        num_splits (int): Number of splits to use for cross-validation

    Returns:
        split_scores (np.array): List of floats representing the score for each split
        cm (confusion_matrix): Confusion matrix of all results

    """

    num_samples = traces.shape[0] * (1 - test_percentage)  # Number of trials
    num_features = traces.shape[1]  # Number of data points
    use_dual_param = (num_samples < num_features)
    # Documentation recommends using dual_param mode if samples < features

    svm = LinearSVC(dual=use_dual_param, max_iter=10000, random_state=1000)

    true_label = np.array([], dtype=int)
    pred_label = np.array([], dtype=int)
    split_scores = []

    # for _ in trange(num_splits, desc='Running SVM splits: '):
    for _ in range(num_splits):

        train_trials, test_trials, train_labels, test_labels = train_test_split(
            traces, correct_labels, test_size=test_percentage, shuffle=True, stratify=correct_labels)

        svm.fit(train_trials, train_labels)
        svm_score = svm.score(test_trials, test_labels)
        split_scores.append(svm_score)

        true_label = np.concatenate((true_label, test_labels))

        svm_prediction = svm.predict(test_trials)

        pred_label = np.concatenate((pred_label, svm_prediction))

    cm = confusion_matrix(true_label, pred_label, normalize='true')

    return split_scores, cm, true_label, pred_label


def _decode_single_neuron(cell, combined_data, num_splits, test_percentage):
    cell_data = combined_data[cell].T
    cell_data = cell_data.dropna(axis=1)  # We lose a few trials occasionally due to concatenation
    correct_labels = cell_data.index.to_series(name='correct_labels')

    svm_scores, confusion_mat, y_true, y_pred = run_svm(cell_data, correct_labels, test_percentage=test_percentage,
                                        num_splits=num_splits)
    svm_score_average = np.mean(svm_scores)

    return svm_score_average, svm_scores, confusion_mat, (y_true, y_pred)


def single_neuron_decoding(combined_data: pd.DataFrame, test_percentage=0.2, num_splits=20):
    cell_names = np.unique(combined_data.columns.get_level_values(0))
    num_labels = len(np.unique(combined_data.columns.get_level_values(1)))
    num_cells = len(cell_names)

    scores = np.zeros(shape=(num_cells, num_splits))  # num_cells x num_splits array to combine the SVM scores
    all_confusion_mats = np.zeros(shape=(num_cells, num_labels, num_labels))
    mean_svm_scores = []

    for cell_i, cell in enumerate(tqdm(cell_names, desc='Running SVM per single neuron: ')):
        svm_score_avg, svm_scores, confusion_mat, y_vals = decode_single_neuron(cell, combined_data, num_splits,
                              #TODO: send y_vals back in the return of this function
                            # Can now be used with the avg ensemble decoder

        mean_svm_scores.append(svm_score_avg)
        scores[cell_i, :] = svm_scores
        all_confusion_mats[cell_i, :, :] = confusion_mat

    # Make the dataframes to return
    mean_score_dict = {'Cell': cell_names, 'Overall SVM Score': mean_svm_scores}
    mean_score_df = pd.DataFrame(mean_score_dict)

    split_score_df = pd.DataFrame(scores, columns=range(num_splits))
    split_score_df.index = cell_names

    return mean_score_df, split_score_df, all_confusion_mats


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


def randomly_sample_trials(z_score_combined_data, combined_data_index, cell_names, trial_labels, odor_mins):
    data_per_trial = pd.DataFrame()

    for cell_i, cell in enumerate(tqdm(cell_names, desc='Randomly Sampling Cell:')):
        trial_num = 0
        cell_data = pd.DataFrame()
        # Iterate through each taste and select the appropriate number of trials
        for odor_i, odor in enumerate(trial_labels):
            cell_odor_data = z_score_combined_data[cell][odor].T
            random_trials = cell_odor_data.sample(odor_mins[odor], axis=0)
            cell_data = pd.concat([cell_data, random_trials], ignore_index=True)

        data_per_trial = pd.concat([data_per_trial, cell_data], axis=1)

    data_per_trial.index = combined_data_index
    data_per_trial = data_per_trial.dropna(axis=1)

    return data_per_trial


def ensemble_decoding(z_scored_combined_data, ensemble_averaging=False,
                      n_repeats=50, test_percentage=.2, num_splits=20, class_labels=None):
    data_len = z_scored_combined_data.shape[1]
    cells = np.unique(z_scored_combined_data.columns.get_level_values(0))
    num_cells = len(cells)
    odors = z_scored_combined_data.columns.get_level_values(1).unique().values

    # If ensemble_averaging is set to True, then we just average the individual SVM scores for each neuron.
    # This is just single_neuron_decoding again
    if ensemble_averaging:

        mean_score_df, split_score_df, all_confusion_mats, all_y_vals = single_neuron_decoding(
            z_scored_combined_data, test_percentage=test_percentage, num_splits=num_splits)

        return mean_score_df, split_score_df, all_confusion_mats, all_y_vals

    else:
        # Set up arrays for recording results
        true_labels = np.array([], dtype=int)
        pred_labels = np.array([], dtype=int)
        all_split_scores = []
        all_confusion_mats = []
        mean_svm_scores = []

        # Figure out the minimum number of trials a neuron has for each taste
        odor_mins = _get_minimum_trials(z_scored_combined_data, cells, odors)
        combined_data_index = _generate_dataframe_index(odor_mins)

        # Repeat sampling of trials
        for repeat_i in trange(n_repeats, desc='Running SVM repeats: '):

            # X is the concatenated dataframe. Each row will represent a trial, and the neural spike trains will be stacked horizonatally.
            data_per_trial = randomly_sample_trials(z_scored_combined_data, combined_data_index, cells, odors, odor_mins)
            correct_labels = data_per_trial.index.to_series(name='correct_labels')

            split_scores, cm, true_label, pred_label = run_svm(data_per_trial, correct_labels, test_percentage, num_splits)
            avg_score = np.mean(split_scores)
            mean_svm_scores.append(avg_score)

            all_split_scores.append(split_scores)
            all_confusion_mats.append(cm)
            true_labels = np.concatenate((true_labels, true_label))  # Record the 'true' taste
            pred_labels = np.concatenate((pred_labels, pred_label))  # Record the predicted taste


        splits_v_repeat_df = pd.DataFrame(all_split_scores, columns=range(num_splits))

        return mean_svm_scores, splits_v_repeat_df, all_confusion_mats, (true_labels, pred_labels)
