"""
Module containing classifier functions and related helpers
Author: Austin Pauley (pauley@psy.fsu.edu)
Date Created: 10/30/2024

Most of the code in this module is either directly from, or heavily influenced by, code from
The Vincis Lab at Florida State University (https://github.com/vincisLab/thermalGC)

CorrelationClassifier is heavy influenced and based on the Neural Decoding Toolbox by Ethan Meyers (emeyers@mit.edu)
More information can be found at <https://readout.info/>
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm, trange


from dewan_calcium import plotting


class CorrelationClassifier:
    """
    Classifier that computes 'template' vectors for each unique label. Predictions are made by computing the pearson correlation coefficient
    between the test sample and each template vector. The label from the template-test pair with the largest correlation coefficient
    is chosen as the predicted label. In the event of a tie, a random label is chosen from the tied results.

    This class is heavily influenced by and based on the max_correlation_coefficient_CL class from
    the Neural Decoding Toolbox by Ethan Meyers (emeyers@mit.edu). Original license found below:

    %     This code is part of the Neural Decoding Toolbox.
    %     Copyright (C) 2011 by Ethan Meyers (emeyers@mit.edu)
    %
    %     This program is free software: you can redistribute it and/or modify
    %     it under the terms of the GNU General Public License as published by
    %     the Free Software Foundation, either version 3 of the License, or
    %     (at your option) any later version.
    %
    %     This program is distributed in the hope that it will be useful,
    %     but WITHOUT ANY WARRANTY; without even the implied warranty of
    %     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %     GNU General Public License for more details.
    %
    %     You should have received a copy of the GNU General Public License
    %     along with this program.  If not, see <http://www.gnu.org/licenses/>.

    """
    entropy_seed = []
    current_entropy = []
    rng = None

    labels = []
    templates = {}
    predicted_labels = []


    def __init__(self, labels=None, entropy_seed=None, rng=None):
        if rng is None:
            self._init_rng(entropy_seed)
        else:
            self.rng = rng

        if labels:
            self.labels=labels


    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        if not self.labels:
            self.labels = y_train.unique()

        for label in self.labels:
            _mean_vector = x_train.loc[label].mean(axis=0)  # get one mean vector
            self.templates[label] = _mean_vector


    def predict(self, x_test):
        vectorized_predict_class = np.vectorize(self._predict_class)
        self.predicted_labels = vectorized_predict_class(x_test)


    def score(self, y_test):
        if not self.predicted_labels:
            raise ValueError('Predictions must be run on the test dataset before the classifier score can be calculated')

        correct_results = y_test == self.predicted_labels
        num_correct = correct_results.sum()
        percent_correct = round(num_correct / len(y_test))
        cm = confusion_matrix(y_test, self.predicted_labels, labels=self.labels)

        return percent_correct, cm


    def _predict_class(self, sample):
        sample_corr_coeff = pd.Series(index=self.labels)
        for label in self.labels:
            template_vector = self.templates[label]
            correlation_coeff, _ = round(pearsonr(sample, template_vector), 3)  # for our sanity, round to 3 places
            sample_corr_coeff[label] = correlation_coeff

        max_corr_coeff = sample_corr_coeff.sort_values(ascending=False)  # Largest at the top
        largest_corr_coeff = max_corr_coeff.iloc[0]

        shared_max_coeff = (max_corr_coeff == largest_corr_coeff)
        # Check to see if any other labels share the largest (by sorting) correlation coefficient
        # If there are multiple, we now have to choose.
        if shared_max_coeff.sum() > 1:
            # we now need to ensure there is only one; for simplicity, we will randomly choose
            possible_labels = max_corr_coeff.index[shared_max_coeff]
            predicted_class = self.rng.choice(possible_labels, 1)
        else:
            predicted_class = max_corr_coeff.index[-1]

        return predicted_class, sample_corr_coeff


    def _init_rng(self, entropy_seed=None):
        seed_generator = np.random.SeedSequence(entropy=entropy_seed)
        self.rng = np.random.default_rng(seed_generator)
        self.current_entropy = seed_generator.entropy


def _run_svm(traces: pd.DataFrame, trial_labels: pd.Series, test_percentage: float = 0.2, num_splits: int = 20, class_labels: list = None):
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

    #for _ in trange(num_splits, desc="Running bootstrapped cross-validation: ", position=2, leave=True):
    for _ in range(num_splits):
        train_trials, test_trials, train_labels, test_labels = train_test_split(
            traces, trial_labels, test_size=test_percentage, shuffle=True)

        svm.fit(train_trials, train_labels)
        svm_score = svm.score(test_trials, test_labels)
        split_scores.append(svm_score)
        true_labels.append(test_labels)
        svm_prediction = svm.predict(test_trials)
        pred_labels.append(svm_prediction)
        cm = confusion_matrix(test_labels, svm_prediction, labels=class_labels, normalize='true')
        cms.append(cm)

    return split_scores, cms, true_labels, pred_labels


def _decode_single_neuron(cell, combined_data, num_splits, test_percentage, class_labels=None):

    cell_data = combined_data[cell].T
    cell_data = cell_data.dropna(axis=1)  # We lose a few trials occasionally due to concatenation
    correct_labels = cell_data.index.to_series(name='correct_labels')

    svm_scores, confusion_mat, y_true, y_pred = _run_svm(cell_data, correct_labels, test_percentage=test_percentage,
                                        num_splits=num_splits, class_labels=class_labels)
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
        progress_desc = progress_desc + f' for window size {int(window[0]), int(window[1])}'

    # for cell_i, cell in enumerate(tqdm(cell_names, desc=progress_desc, position=1, leave=True)):
    for cell_i, cell in enumerate(cell_names):
        cell_data = pd.DataFrame()

        # Iterate through each taste and select the appropriate number of trials
        for odor_i, odor in enumerate(trial_labels):
            cell_odor_data = z_score_combined_data[cell][odor].T
            if window:
                start_frame, end_frame = window
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


def _decode_ensemble(z_scored_combined_data, test_percentage, num_splits, iterator, loop_message, window=False, class_labels=None):

    true_labels = {}
    pred_labels = {}
    all_split_scores = {}
    all_confusion_mats = {}
    mean_svm_scores = {}

    cells = np.unique(z_scored_combined_data.columns.get_level_values(0))
    #odors = z_scored_combined_data.columns.get_level_values(1).unique().values
    odors = class_labels

    odor_mins = _get_minimum_trials(z_scored_combined_data, cells, odors)
    combined_data_index = _generate_dataframe_index(odor_mins)

    for iterator in tqdm(iterator, desc=loop_message, leave=True, position=0):

        if window:
            _window = iterator
        else:
            _window = None

        cropped_data_per_trial = _randomly_sample_trials(z_scored_combined_data, combined_data_index, cells, odors,
                                                         odor_mins, window=_window)

        correct_labels = cropped_data_per_trial.index.to_series()

        split_scores, cm, true_label, pred_label = _run_svm(cropped_data_per_trial, correct_labels, test_percentage,
                                                            num_splits,  class_labels=class_labels)
        avg_score = np.mean(split_scores)

        mean_svm_scores[iterator] = avg_score
        all_split_scores[iterator] = split_scores
        all_confusion_mats[iterator] = cm
        true_labels[iterator] = true_label  # Record the 'true' odorants
        pred_labels[iterator] = pred_label # Record the predicted odorants

    splits_v_repeat_df = pd.DataFrame(all_split_scores)

    return mean_svm_scores, splits_v_repeat_df, all_confusion_mats, (true_labels, pred_labels)


def single_neuron_decoding(combined_data: pd.DataFrame, test_percentage=0.2, num_splits=20, class_labels=None):

    cell_names = np.unique(combined_data.columns.get_level_values(0))
    num_labels = len(np.unique(combined_data.columns.get_level_values(1)))
    num_cells = len(cell_names)

    scores = np.zeros(shape=(num_cells, num_splits))  # num_cells x num_splits array to combine the SVM scores
    all_confusion_mats = np.zeros(shape=(num_cells, num_labels, num_labels))
    mean_svm_scores = []
    all_y_vals = {}
    for cell_i, cell in enumerate(tqdm(cell_names, desc='Running SVM per single neuron: ')):
        svm_score_avg, svm_scores, confusion_mat, y_vals = _decode_single_neuron(
            cell, combined_data, num_splits, test_percentage, class_labels=class_labels)
            # Can now be used with the avg ensemble decoder

        mean_svm_scores.append(svm_score_avg)
        scores[cell_i, :] = svm_scores
        all_confusion_mats[cell_i, :, :] = confusion_mat
        all_y_vals[cell] = y_vals

    # Make the dataframes to return
    mean_score_dict = {'Cell': cell_names, 'Overall SVM Score': mean_svm_scores}
    mean_score_df = pd.DataFrame(mean_score_dict)

    split_score_df = pd.DataFrame(scores, columns=range(num_splits))
    split_score_df.index = cell_names

    return mean_score_df, split_score_df, all_confusion_mats, all_y_vals


def _per_cell_generator(combined_data_dff, bins):
    for bin in bins:
        yield combined_data_dff[bin]

def ensemble_decoding(z_scored_combined_data, n_repeats=50, test_percentage=.2, num_splits=20, class_labels=None):
    iterator = np.arange(n_repeats)
    loop_message = 'Running Repeat SVM Ensemble Decoding: '

    return _decode_ensemble(z_scored_combined_data, test_percentage, num_splits, iterator, loop_message,
                            class_labels=class_labels)


def sliding_window_ensemble_decoding(z_scored_combined_data, window_size=2, test_percentage=.2, num_splits=20,
                                     class_labels=None):
    data_size = z_scored_combined_data.shape[0]
    windows = _get_windows(data_size, window_size)
    loop_message = 'Running Sliding Window SVM Decoding: '

    return _decode_ensemble(z_scored_combined_data, test_percentage, num_splits, windows, loop_message,
                            window=True, class_labels=class_labels)


def shuffle_data(z_scored_combined_data):
    names = z_scored_combined_data.columns.names

    shuffled_data = z_scored_combined_data.T.groupby(level=0, group_keys=False).apply(_shuffle_index)
    shuffled_data.index = pd.MultiIndex.from_tuples(shuffled_data.index, names=names)
    shuffled_data = shuffled_data.T

    return shuffled_data


def postprocess(mean_svm_scores, num_cells, window=None):
    mean_score_df = pd.DataFrame(mean_svm_scores, index=[0])
    mean_score_df.insert(0, column='num_cells', value=num_cells)
    if window:
        mean_score_df.insert(0, column='window_size', value=window)

    return mean_score_df


def preprocess_for_plotting(mean_svm_scores, splits_v_repeat_df):
    # Unpack Values
    mean_performance = [mean_svm_scores[key] for key in mean_svm_scores]

    # Calculate confidence interval scalar
    sqrt = np.sqrt(splits_v_repeat_df.shape[0])
    std_devs = splits_v_repeat_df.T.std(axis=1)
    CI_scalar = 2.576 * (std_devs / sqrt)

    # Calculate all CI values
    CI_min = np.subtract(mean_performance, CI_scalar)
    CI_max = np.add(mean_performance, CI_scalar)

    return mean_performance, CI_min, CI_max


def save_svm_data(mean_performance, shuffle_mean_performance, index, CI, shuffle_CI, svm_output_dir):
    CI_min, CI_max = CI
    shuffle_CI_min, shuffle_CI_max = shuffle_CI

    output_path = svm_output_dir.joinpath('SVM_Performance_Stats.xlsx')

    svm_df = pd.DataFrame(
        np.vstack([mean_performance, CI_min, CI_max, shuffle_mean_performance, shuffle_CI_min, shuffle_CI_max]).T,
        index=index,
        columns=['Mean SVM Performance', '99% CI Min', '99% CI Max', 'Shuffled SVM Performance', 'Shuffled 99% CI Min',
                 'Shuffled 99% CI Max'])
    svm_df.to_excel(output_path)


def average_CM(all_confusion_mats, windows):
    window_averaged_cms = {}

    for window in windows:
        window_cm = all_confusion_mats[window]
        avg_cm = np.mean(window_cm, axis=0)
        window_averaged_cms[window] = avg_cm

    return window_averaged_cms


def save_and_plot_CM(window_averaged_cms, cm_window, window_name, windows, labels, new_norm, cm_data_save_dir, cm_figure_save_dir):
    odor_cm = []
    start_idx, end_idx = cm_window
    for window in windows:
        if window[0] >= start_idx and window[1] <= end_idx:
            odor_cm.append(window_averaged_cms[window])

    average_odor_cm = np.mean(odor_cm, axis=0)

    average_odor_cm_df = pd.DataFrame(average_odor_cm, columns=labels, index=labels)

    title_text = ' '.join(window_name.split('_'))
    title_with_index = f'{title_text} ({start_idx}-{end_idx})'
    df_save_path = cm_data_save_dir.joinpath(f'{title_with_index}.xlsx')
    fig_save_path = cm_figure_save_dir.joinpath(f'{title_with_index}.pdf')

    fig, ax = plotting.plot_avg_cm(labels, average_odor_cm, new_norm, fig_save_path, title_with_index)

    average_odor_cm_df.to_excel(df_save_path, index=True)

    return fig, ax


def add_odor_class(combined_data, odor_classes):
    new_columns = []
    odors = combined_data.columns.get_level_values(1)
    for i, orig_tuple in enumerate(combined_data.columns.values):
        odor_class = odor_classes[odors[i]]
        new_columns.append(orig_tuple + tuple([odor_class]))
    new_index = pd.MultiIndex.from_tuples(new_columns, names=['Cells', 'Odor', 'Block', 'Animal', 'Class'])
    combined_data.columns = new_index
    original_columns = combined_data.columns

    return combined_data, original_columns