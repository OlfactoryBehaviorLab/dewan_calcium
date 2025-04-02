### Dewan Trace Tools Helper Functions
### Shared functions that collect or manipulate trace data
### Austin Pauley: Dewan Lab, Florida State University, 2024

import numpy as np
import pandas as pd


def new_collect_trial_data(odor_df: pd.DataFrame, time_df: pd.DataFrame, response_duration: int, latent: bool = False):
    """

    Args:
        odor_df: Pandas DataFrame containing all the trials for a specific odor
        time_df: Pandas DataFrame containing all the timestamps for the trials for a specific odor
        response_duration: int reflecting the amount of time that the "response" period envelops
        latent: True if we're looking for latent responses, response start time is offset by response_duration

    Returns:

    """

    baseline_data = []
    evoked_data = []

    baseline_indices = []
    evoked_indices = []

    for trial_index, (_, data) in enumerate(odor_df.items()):
        evoke_start_time = 0
        evoke_end_time = response_duration
        if latent:
            evoke_start_time += response_duration
            evoke_end_time += response_duration

        trial_timestamps = time_df.iloc[:, trial_index]

        baseline_trial_indices = trial_timestamps[trial_timestamps.between(-response_duration, 0, 'both')].index
        baseline_trial_data = data[baseline_trial_indices]

        baseline_data.append(baseline_trial_data)
        baseline_indices.append((baseline_trial_indices[0], baseline_trial_indices[-1]))

        evoked_trial_indices = trial_timestamps[trial_timestamps.between(evoke_start_time, evoke_end_time, 'both')].index
        evoked_trial_data = data[evoked_trial_indices]
        evoked_data.append(evoked_trial_data)

        evoked_indices.append((evoked_trial_indices[0], evoked_trial_indices[-1]))

    baseline_data = pd.DataFrame(baseline_data)
    evoked_data = pd.DataFrame(evoked_data)
    return baseline_data, evoked_data, baseline_indices, evoked_indices


def get_evoked_baseline_means(odor_df, timestamps_df, response_duration: int, latent: bool = False):
    baseline_data, evoked_data, _, _ = new_collect_trial_data(odor_df, timestamps_df, response_duration, latent)

    baseline_means = baseline_data.mean(axis=1)
    evoked_means = evoked_data.mean(axis=1)

    return baseline_means, evoked_means


def average_odor_responses(odor_df: pd.DataFrame, odor_timestamps:pd.DataFrame, response_duration: int) -> float:
    baseline_means, evoked_means = get_evoked_baseline_means(odor_df, odor_timestamps, response_duration)
    diff = evoked_means - baseline_means
    average_response = diff.mean()

    return average_response


def collect_trial_data(data_input, return_values = None,
                       latent_cells_only: bool = False) -> tuple:
    baseline_data = []
    evoked_data = []
    baseline_start_indexes = []
    baseline_end_indexes = []
    evoked_start_indexes = []
    evoked_end_indexes = []

    for trial in data_input.current_odor_trials:  # For each odor
        time_array = data_input.unix_time_array[trial, :] # Get times for trial
        trial_data = data_input.Data[data_input.cell_index, trial, :]  # Get data for cell x trial combo
        fv_on_time = float(data_input.FV_Data[data_input.FV_on_index[trial], 0])
        fv_on_index = len(np.nonzero(time_array < fv_on_time)[0])
        baseline_start_index = len(np.nonzero(time_array < (fv_on_time - data_input.baseline_duration))[0])
        baseline_end_index = fv_on_index - 1

        baseline_trial_data = trial_data[baseline_start_index: baseline_end_index]
        baseline_data.append(baseline_trial_data)

        if latent_cells_only:
            evoked_start_index = len(np.nonzero(time_array < (fv_on_time + data_input.response_duration))[0])
            evoked_end_index = len(np.nonzero(time_array < (time_array[evoked_start_index]
                                                              + data_input.response_duration))[0])
        else:
            evoked_start_index = fv_on_index
            evoked_end_index = len(np.nonzero(time_array < (fv_on_time + data_input.response_duration))[0])

        evoked_trial_data = trial_data[evoked_start_index: evoked_end_index]
        evoked_data.append(evoked_trial_data)

        baseline_start_indexes.append(baseline_start_index)
        baseline_end_indexes.append(baseline_end_index)
        evoked_start_indexes.append(evoked_start_index)
        evoked_end_indexes.append(evoked_end_index)

    if return_values is not None:
        return_values.baseline_start_indexes.append(baseline_start_indexes)
        return_values.baseline_end_indexes.append(baseline_end_indexes)
        return_values.evoked_start_indexes.append(evoked_start_indexes)
        return_values.evoked_end_indexes.append(evoked_end_indexes)

    return baseline_data, evoked_data


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
    baseline_frames = odor_df.iloc[:, :baseline_frames]
    f0 = baseline_frames.mean().mean()
    odor_df = odor_df.subtract(f0)
    odor_df = odor_df.divide(f0)
    return odor_df


def dff(combined_data: pd.DataFrame, baseline_frames: int):
    dff_combined = pd.DataFrame()
    groupby_cell = combined_data.T.groupby(level=0, group_keys=False)
    for cell, cell_df in groupby_cell:
        groupby_odor = cell_df.groupby(level=1, group_keys=False).apply(lambda x: _baseline_avg_dff(x, baseline_frames))
        dff_combined = pd.concat([dff_combined, groupby_odor.T], axis=1)

    return dff_combined
