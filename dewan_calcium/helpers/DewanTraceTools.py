### Dewan Trace Tools Helper Functions
### Shared functions that collect or manipulate trace data
### Austin Pauley: Dewan Lab, Florida State University, 2024

import numpy as np

from .DewanDataStore import AUROCdataStore, AUROCReturn # Import for typedef


def collect_trial_data(data_input: AUROCdataStore, return_values: AUROCReturn = None,
                       latent_cells_only: bool = False) -> tuple:
    baseline_data = []
    evoked_data = []
    baseline_start_indexes = []
    baseline_end_indexes = []
    evoked_start_indexes = []
    evoked_end_indexes = []

    for trial in data_input.current_odor_trials:
        time_array = data_input.unix_time_array[trial, :]
        trial_data = data_input.Data[data_input.cell_index, trial, :]
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