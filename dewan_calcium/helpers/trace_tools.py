### Dewan Trace Tools Helper Functions
### Shared functions that collect or manipulate trace data
### Austin Pauley: Dewan Lab, Florida State University, 2024

import numpy as np
import pandas as pd


def collect_trial_data(odor_df: pd.DataFrame, time_df: pd.DataFrame, evoked_duration: int):
    """

    Args:
        odor_df: Pandas DataFrame containing all the trials for a specific odor
        time_df: Pandas DataFrame containing all the timestamps for the trials for a specific odor
        evoked_duration: int reflecting the amount of time that the "response" period envelops

    Returns:

    """

    evoked_data = []
    evoked_indices = []

    for trial_index, (_, data) in enumerate(odor_df.items()):
        trial_timestamps = time_df.iloc[:, trial_index]

        evoked_trial_indices = trial_timestamps[trial_timestamps.between(0, evoked_duration, 'both')].index
        evoked_trial_data = data[evoked_trial_indices]
        evoked_data.append(evoked_trial_data)

        evoked_indices.append((evoked_trial_indices[0], evoked_trial_indices[-1]))

    evoked_data = pd.DataFrame(evoked_data)
    return evoked_data, evoked_indices


def get_evoked_baseline_means(odor_df, timestamps_df, response_duration: int, latent: bool = False):
    baseline_data, evoked_data, _, _ = collect_trial_data(odor_df, timestamps_df, response_duration, latent)

    baseline_means = baseline_data.mean(axis=1)
    evoked_means = evoked_data.mean(axis=1)

    return baseline_means, evoked_means


def average_odor_responses(odor_df: pd.DataFrame, odor_timestamps:pd.DataFrame, response_duration: int) -> float:
    baseline_means, evoked_means = get_evoked_baseline_means(odor_df, odor_timestamps, response_duration)
    diff = evoked_means - baseline_means
    average_response = diff.mean()

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


def _baseline_avg_dff(odor_df: pd.DataFrame, baseline_frames: int):
    baseline_frames = odor_df.iloc[:, :baseline_frames-5]
    f0 = baseline_frames.mean(axis=1).mean()
    diff_df = odor_df.subtract(f0)
    div_df = diff_df.divide(f0)
    return div_df


def dff(combined_data: pd.DataFrame, num_baseline_frames: int):
    dff_combined = pd.DataFrame()
    groupby_cell = combined_data.T.groupby(level=0, group_keys=False)
    for cell, cell_df in groupby_cell:
        new_cell_df = pd.DataFrame()
        groupby_odor = cell_df.groupby(level=1, group_keys=False)
        for odor_name, odor_df in groupby_odor:
            baseline_frames = odor_df.iloc[:, :num_baseline_frames]
            f0 = baseline_frames.mean(axis=1).mean()
            diff_df = odor_df.subtract(f0)

            div_df = diff_df.divide(f0)
            new_cell_df = pd.concat([new_cell_df, div_df.T], axis=1)
        dff_combined = pd.concat([dff_combined, new_cell_df], axis=1)


        # dff_combined = pd.concat([dff_combined, groupby_odor.T], axis=1)

    return dff_combined
