"""
AUROC Analysis for Cells v. Odor Presentations.

Dewan Lab, Florida State University
Original Script: Roberto Vincis' Lab, FSU
Modified for Dewan Lab Use: S. Caton, A. Pauley, A. Dewan
December, 2022
"""

import itertools
from multiprocessing import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map
import numpy as np
from sklearn.model_selection import train_test_split

from .Helpers import DewanDataStore
from .Helpers import DewanSlidingProbability
from . import DewanPlotting


def compute_percentile(auroc, auroc_shuffle) -> float:
    return np.sum(auroc_shuffle < auroc) / auroc_shuffle.size


def collect_trial_data(dataInput: DewanDataStore.AUROCdataStore, returnValues: DewanDataStore.AUROCReturn = None,
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


def generateShuffledDistribution(all_vector: np.ndarray, vect_base1: np.ndarray) -> np.ndarray:
    shuffled_auroc = []

    for _ in itertools.repeat(None, 1000):  # Repeat 1000 times, faster than range()

        split_1, split_2 = train_test_split(all_vector, test_size=len(vect_base1))

        shuffle_max = max(split_1)
        shuffle_min = min(split_1)
        increments = 100

        baseline_shuffle = DewanSlidingProbability.sliding_probability(
            split_1, shuffle_min, shuffle_max, increments)

        baseline_shuffle.reverse()
        baseline_shuffle.insert(0, 0.0)
        baseline_shuffle.insert(len(split_1) + 1, 1.0)

        evoked_shuffle = DewanSlidingProbability.sliding_probability(
            split_2, shuffle_min, shuffle_max, increments)

        evoked_shuffle.reverse()
        evoked_shuffle.insert(0, 0.0)
        evoked_shuffle.insert(len(evoked_shuffle) + 1, 1.0)

        shuffled_auroc.append(np.trapz(evoked_shuffle, baseline_shuffle, axis=-1))
    return np.array(shuffled_auroc)


def allOdorsPerCell(data_input: DewanDataStore.AUROCdataStore, latentCellsOnly: bool,
                    cellNum: int) -> DewanDataStore.AUROCReturn:

    data_input = data_input.makeCopy()
    data_input.update_cell(cellNum)
    return_values = DewanDataStore.AUROCReturn()

    for odor_iterator in range(data_input.num_unique_odors):
        data_input.update_odor(odor_iterator)

        baseline_data, evoked_data = collect_trial_data(data_input, return_values, latentCellsOnly)
        baseline_means, evoked_means = averageTrialData(baseline_data, evoked_data)
        max_baseline_val = max(baseline_means)
        min_baseline_val = min(baseline_means)

        increments = 100

        baseline_prob = DewanSlidingProbability.sliding_probability(
            baseline_means, min_baseline_val, max_baseline_val, increments)

        baseline_prob.reverse()
        baseline_prob.insert(0, 0.0)
        baseline_prob.insert(len(baseline_prob) + 1, 1.0)

        evoked_prob = DewanSlidingProbability.sliding_probability(
            evoked_means, min_baseline_val, max_baseline_val, increments)

        evoked_prob.reverse()
        evoked_prob.insert(0, 0.0)
        evoked_prob.insert(len(baseline_prob) + 1, 1.0)

        auroc_value = np.trapz(evoked_prob, baseline_prob, axis=-1)

        all_means_vector = np.concatenate((baseline_means, evoked_means))

        auroc_shuffle = generateShuffledDistribution(all_means_vector, baseline_means)

        lower_bound = np.percentile(auroc_shuffle, [1])
        upper_bound = np.percentile(auroc_shuffle, [99])

        return_values.all_lower_bounds.append(lower_bound)
        return_values.all_upper_bounds.append(upper_bound)
        return_values.auroc_values.append(auroc_value)
        return_values.percentiles.append(compute_percentile(auroc_value, auroc_shuffle))  # 0 - 1

        if auroc_value > upper_bound:
            return_values.response_chart.append(2)
            if data_input.do_plot:
                DewanPlotting.plot_auroc_distributions(data_input.file_header, auroc_shuffle, auroc_value, upper_bound, lower_bound, data_input.Cell_List,
                                                       cellNum, data_input.unique_odors, odor_iterator, latentCellsOnly)
        elif auroc_value < lower_bound:
            return_values.response_chart.append(1)
            if data_input.do_plot:
                DewanPlotting.plot_auroc_distributions(data_input.file_header, auroc_shuffle, auroc_value, upper_bound, lower_bound, data_input.Cell_List,
                                                       cellNum, data_input.unique_odors, odor_iterator, latentCellsOnly)
        else:
            return_values.response_chart.append(0)

    return return_values


def AUROC(data_input: DewanDataStore.AUROCdataStore, latent_cells_only: bool) -> list:
    workers = Pool()
    partial_function = partial(allOdorsPerCell, data_input, latent_cells_only)
    return_values = process_map(partial_function, range(data_input.number_cells), desc='AUROC Progress: ')
    workers.close()
    workers.join()

    return return_values
