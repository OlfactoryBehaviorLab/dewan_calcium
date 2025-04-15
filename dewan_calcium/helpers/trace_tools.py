### Dewan Trace Tools Helper Functions
### Shared functions that collect or manipulate trace data
### Austin Pauley: Dewan Lab, Florida State University, 2024
from typing import Union

import numpy as np
import pandas as pd


def collect_trial_data(odor_df: pd.DataFrame, time_df: pd.DataFrame, duration: int, baseline=None):
    """

    Args:
        odor_df: Pandas DataFrame containing all the trials for a specific odor
        time_df: Pandas DataFrame containing all the timestamps for the trials for a specific odor
        evoked_duration: int reflecting the amount of time that the "response" period envelops

    Returns:

    """

    evoked_data = []
    evoked_indices = []

    if baseline is not None:
        lower_index = baseline
        upper_index = 0
    else:
        lower_index = 0
        upper_index = duration

    for trial_index, (_, data) in enumerate(odor_df.items()):
        trial_timestamps = time_df.iloc[:, trial_index]
        evoked_trial_indices = trial_timestamps[trial_timestamps.between(lower_index, upper_index, 'both')].index
        evoked_trial_data = data[evoked_trial_indices]
        evoked_data.append(evoked_trial_data)

        evoked_indices.append((evoked_trial_indices[0], evoked_trial_indices[-1]))

    evoked_data = pd.DataFrame(evoked_data)
    evoked_data = evoked_data.dropna(axis=1)
    return evoked_data, evoked_indices


def get_evoked_baseline_means(odor_df, timestamps_df, response_duration: int, baseline_duration: Union[int, None]):
    baseline_means = []

    evoked_data, _ = collect_trial_data(odor_df, timestamps_df, response_duration)
    evoked_means = evoked_data.mean(axis=1)

    if baseline_duration is not None:
        baseline_data, _ = collect_trial_data(odor_df, timestamps_df, response_duration, baseline_duration)
        baseline_means = baseline_data.mean(axis=1)

    return baseline_means, evoked_means


def average_odor_responses(odor_df: pd.DataFrame, odor_timestamps:pd.DataFrame, response_duration: int) -> float:
    _, evoked_means = get_evoked_baseline_means(odor_df, odor_timestamps, response_duration, None)
    average_response = evoked_means.mean()

    return average_response


def average_trial_data(baseline_data: list, response_data: list) -> tuple:
    baseline_vector = []
    evoked_vector = []

    for trial in range(len(baseline_data)):
        response_mean = np.mean(response_data[trial])
        evoked_vector = np.append(evoked_vector, response_mean)
        baseline_mean = np.mean(baseline_data[trial])
        baseline_vector = np.append(baseline_vector, baseline_mean)

    return baseline_vector, evoked_vector


def truncate_data(data1: list, data2: list) -> tuple:
    data1_minima = [np.min(len(row)) for row in data1]
    data2_minima = [np.min(len(row)) for row in data2]
    row_minimum = int(min(min(data1_minima), min(data2_minima)))
    data1 = [row[:row_minimum] for row in data1]
    data2 = [row[:row_minimum] for row in data2]

    return data1, data2


def _calc_dff(trial_series: pd.Series, baseline_frames: int):
    f0 = np.mean(trial_series.iloc[0:baseline_frames])
    df = np.subtract(trial_series, f0)
    dff = np.divide(df, f0)
    return dff


def _baseline_avg_dff(odor_df: pd.DataFrame, FV_timestamps: pd.DataFrame, baseline_frames: int):
    # baseline_frames = odor_df.iloc[:, baseline_timestamps]
    odor = odor_df.index.get_level_values(1).unique()
    baseline_frames, _ = collect_trial_data(odor_df.T, FV_timestamps[odor], None, baseline_frames)
    f0 = baseline_frames.mean(axis=1).mean()
    diff_df = odor_df.subtract(f0)
    div_df = diff_df.divide(np.abs(f0))
    return div_df


def dff(combined_data: pd.DataFrame, FV_timestamps: pd.DataFrame, num_baseline_frames: int):
    dff_combined = pd.DataFrame()
    groupby_cell = combined_data.T.groupby(level=0, group_keys=False)
    for cell, cell_df in groupby_cell:
        groupby_odor = cell_df.groupby(level=1, group_keys=False).apply(
            lambda x: _baseline_avg_dff(x, FV_timestamps, num_baseline_frames)
        )
        dff_combined = pd.concat([dff_combined, groupby_odor.T], axis=1)

    return dff_combined


def dff_magnitude_means(combined_data_dff, labels, BASELINE_FRAMES, ODOR_FRAMES, subset: int = 0):
    cell_medians = {}
    cell_means = {}
    nonzero_cell_medians = {}
    nonzero_cell_means = {}
    cell_odor_orders = []

    cells = combined_data_dff.groupby('Cells')

    for cell_name, cell_df in cells:
        odor_medians = []
        odor_means = []
        nonzero_odor_medians = []
        nonzero_odor_means = []
        odor_order = []

        odors = cell_df.groupby('Odor')

        for i, (odor_name, odor_data) in enumerate(odors):
            odor_trial_means = odor_data.iloc[:, BASELINE_FRAMES: BASELINE_FRAMES + ODOR_FRAMES].mean(
                axis=1)  # odor evoked means for each trial
            baseline_mean = odor_data.iloc[:, :BASELINE_FRAMES].mean(axis=1)  # baseline means for each trial
            diff = odor_trial_means.subtract(baseline_mean, axis=0)  # subtract the baseline from odor activity

            _mean = diff.mean()
            _median = diff.median()

            nonzero_odor_means.append(_mean)
            nonzero_odor_medians.append(_median)

            # What happens if you zero the values BEFORE taking the mean?
            if _mean < 0:
                _mean = 0
            if _median < 0:
                _median = 0

            odor_means.append(_mean)
            odor_medians.append(_median)
            odor_order.append(odor_name)

        cell_means[cell_name] = odor_means
        cell_medians[cell_name] = odor_medians
        nonzero_cell_medians[cell_name] = nonzero_odor_medians
        nonzero_cell_means[cell_name] = nonzero_odor_means
        cell_odor_orders.append(odor_order)

    odors = cell_odor_orders[0]

    cell_means = pd.DataFrame(cell_means, index=odors)
    cell_medians = pd.DataFrame(cell_medians, index=odors)
    nonzero_cell_medians = pd.DataFrame(nonzero_cell_medians, index=odors)
    nonzero_cell_means = pd.DataFrame(nonzero_cell_means, index=odors)
    cell_means = cell_means.loc[labels]
    cell_medians = cell_medians.loc[labels]

    nonzero_cell_order = nonzero_cell_means.median().sort_values(ascending=False).index
    nonzero_cell_means = nonzero_cell_means.loc[labels]
    nonzero_cell_means = nonzero_cell_means[nonzero_cell_order]
    nonzero_cell_medians = nonzero_cell_medians.loc[labels]
    nonzero_cell_medians = nonzero_cell_medians[nonzero_cell_order]

    if subset > 0:
        nonzero_cell_means = nonzero_cell_means.iloc[:, np.r_[0:subset, -subset:0]]
        nonzero_cell_medians = nonzero_cell_medians.iloc[:, np.r_[0:subset, -subset:0]]

    return cell_means, cell_medians, nonzero_cell_means, nonzero_cell_medians
