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


def decode_single_neuron(cell, combined_data, num_splits, test_percentage):
    cell_data = combined_data[cell].T
    correct_labels = cell_data.index.to_series(name='correct_labels')
    cell_data = cell_data.dropna(axis=1)  # We lose a few trials occasionally due to concatenation

    svm_scores, confusion_mat, y_true, y_pred = run_svm(cell_data, correct_labels, test_percentage=test_percentage,
                                        num_splits=num_splits)
    svm_score_average = np.mean(svm_scores)

    return svm_score_average, svm_scores, confusion_mat, (y_true, y_pred)



def ensemble_decoding(combined_data, ensemble_averaging=False,
                      n_trial_pairings=50, test_percentage=.2, num_splits=20, class_labels=None):
    data_len = combined_data.shape[1]
    cells = np.unique(combined_data.columns.get_level_values(0))
    num_cells = len(cells)
    unique_odorants = combined_data.columns.get_level_values(1).unique().values

    # If ensemble_averaging is set to True, then we just average the individual SVM scores for each neuron.
    if ensemble_averaging:

        mean_score_df, split_score_df, all_confusion_mats = single_neuron_decoding(combined_data, test_percentage=test_percentage, num_splits=num_splits)
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

        if plot_confusion_matrix:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=class_labels,
                normalize='true',
                im_kw={'vmin': 0, 'vmax': 1}
            )
            if cm_plot_title == None:
                cm_plot_title = f'Ensemble Averaged across {nNeurons} Neurons'
            disp.ax_.set_title(cm_plot_title)
            disp.figure_.set_size_inches(10, 10)
            plt.show()

        # Set up the dataframes to return
        mean_score_dict = {'Neuron': f'Average of {nNeurons} Neurons', ' Overall SVM Score': np.mean(mean_SVM_scores)}
        mean_score_df = pd.DataFrame(mean_score_dict, index=[0])

        split_score_df = pd.DataFrame(splits_SVM_scores, columns=range(num_splits))
        split_score_df.insert(0, 'Neuron', cells)

        return mean_score_df, split_score_df

    else:

        # Figure out the minimum number of trials a neuron has for each taste
        taste_mins = []

        for taste in tastes:
            taste_trials = []
            for neuron in cells:
                ntdf = dataFrame[(dataFrame['Neuron'] == neuron) & (dataFrame['Taste'] == taste)]
                taste_trials.append(ntdf.shape[0])
            taste_mins.append(min(taste_trials))

        # Set up arrays for recording results
        y_true = np.array([], dtype=int)
        y_pred = np.array([], dtype=int)
        split_scores = []
        pairing_split_scores = np.zeros(shape=(n_trial_pairings, num_splits))

        # Repeat sampling of trials
        for T in range(n_trial_pairings):

            y = np.repeat(tastes, taste_mins)
            # X is the concatenated dataframe. Each row will represent a trial, and the neural spike trains will be stacked horizonatally.
            X = np.zeros(shape=(len(y), nNeurons * data_len))

            # Iterate through each neuron
            for nindex, neuron in enumerate(cells):
                trialindex = 0
                # Iterate through each taste and select the appropriate number of trials
                for tindex, taste in enumerate(tastes):
                    ntdf = dataFrame[(dataFrame['Neuron'] == neuron) & (dataFrame['Taste'] == taste)]
                    trials = np.array(ntdf['Trial'])
                    selected_trials = np.random.choice(trials, taste_mins[tindex], replace=False)
                    for trial in selected_trials:
                        # Put it into X
                        X[trialindex, (nindex * data_len):(nindex * data_len) + data_len] = np.array(
                            ntdf[ntdf['Trial'] == trial].iloc[:, start_index:])
                        trialindex += 1

            # Now we have our concatenated data and labels. Let's run SVM.

            n_samples = X.shape[0] * (1 - test_size)  # Number of spike trains in the training set
            n_features = X.shape[1]  # Number of time points in the spike trains
            dual_param = (n_samples < n_features)

            # Define the SVM model
            model_SVM = LinearSVC(dual=dual_param, max_iter=10000, random_state=651)

            for j in range(num_splits):  # Use several splits of training and testing sets for robustness

                X_train, X_test, y_train, y_test = train_test_split(X, y,  # This function is from sklearn
                                                                    test_size=test_size,
                                                                    # Default: 2/3 of data to train and 1/3 to test
                                                                    shuffle=True,
                                                                    stratify=y)  # Sample from each taste

                model_SVM.fit(X_train, y_train)  # Re-fit the classifier with the training set
                s_score = model_SVM.score(X_test, y_test)  # Fit the testing set
                split_scores.append(s_score)  # record score
                pairing_split_scores[T, j] = s_score

                if plot_confusion_matrix:
                    y_true = np.concatenate((y_true, y_test))  # Record the 'true' taste
                    y_pred = np.concatenate((y_pred, model_SVM.predict(X_test)))  # Record the predicted taste

        # Plot the confusion matrix if applicable
        if plot_confusion_matrix:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=class_labels,
                normalize='true',
                im_kw={'vmin': 0, 'vmax': 1}
            )

            if cm_plot_title == None:
                cm_plot_title = f'Ensemble of {nNeurons} Neurons'
            disp.ax_.set_title(cm_plot_title)
            disp.figure_.set_size_inches(10, 10)
            plt.show()

        # Make the dataframe to return
        mean_SVM_dict = {'Neuron': f'Ensemble of {nNeurons} Neurons', 'Overall SVM Score': np.mean(split_scores)}
        mean_SVM_df = pd.DataFrame(mean_SVM_dict, index=[0])

        splits_SVM_df = pd.DataFrame(pairing_split_scores, columns=range(num_splits))
        splits_SVM_df.insert(0, 'Trial Pairing', range(n_trial_pairings))

        return mean_SVM_df, splits_SVM_df
