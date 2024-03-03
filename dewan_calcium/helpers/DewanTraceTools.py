### Dewan Trace Tools Helper Functions
### Shared functions that collect or manipulate trace data
### Austin Pauley: Dewan Lab, Florida State University, 2024

import numpy as np

from DewanDataStore import AUROCdataStore, AUROCReturn # Import for typedef


def collect_trial_data(dataInput: AUROCdataStore, returnValues: AUROCReturn = None,
                       latentCellsOnly: bool = False) -> tuple:
    baseline_data = []
    evoked_data = []
    baseline_start_indexes = []
    baseline_end_indexes = []
    evoked_start_indexes = []
    evoked_end_indexes = []

    for trial in dataInput.current_odor_trials:
        time_array = dataInput.unix_time_array[trial, :]
        trial_data = dataInput.Data[dataInput.cell_index, trial, :]
        fv_on_time = float(dataInput.FV_Data[dataInput.FV_on_index[trial], 0])
        fv_on_index = len(np.nonzero(time_array < fv_on_time)[0])
        baseline_start_index = len(np.nonzero(time_array < (fv_on_time - dataInput.baseline_duration))[0])
        baseline_end_index = fv_on_index - 1

        baseline_trial_data = trial_data[baseline_start_index: baseline_end_index]
        baseline_data.append(baseline_trial_data)

        if latentCellsOnly:
            evoked_start_index = len(np.nonzero(time_array < (fv_on_time + dataInput.response_duration))[0])
            evoked_end_index = len(np.nonzero(time_array < (time_array[evoked_start_index]
                                                              + dataInput.response_duration))[0])
        else:
            evoked_start_index = fv_on_index
            evoked_end_index = len(np.nonzero(time_array < (fv_on_time + dataInput.response_duration))[0])

        evoked_trial_data = trial_data[evoked_start_index: evoked_end_index]
        evoked_data.append(evoked_trial_data)

        baseline_start_indexes.append(baseline_start_index)
        baseline_end_indexes.append(baseline_end_index)
        evoked_start_indexes.append(evoked_start_index)
        evoked_end_indexes.append(evoked_end_index)

    if returnValues is not None:
        returnValues.baseline_start_indexes.append(baseline_start_indexes)
        returnValues.baseline_end_indexes.append(baseline_end_indexes)
        returnValues.evoked_start_indexes.append(evoked_start_indexes)
        returnValues.evoked_end_indexes.append(evoked_end_indexes)

    return baseline_data, evoked_data

def averageTrialData(baselineData: list, responseData: list) -> tuple:
    baseline_vector = []
    evoked_vector = []

    for trial in range(len(baselineData)):
        response_mean = np.mean(responseData[trial])
        evoked_vector = np.append(evoked_vector, response_mean)
        baselineMean = np.mean(baselineData[trial])
        baseline_vector = np.append(baseline_vector, baselineMean)

    return baseline_vector, evoked_vector