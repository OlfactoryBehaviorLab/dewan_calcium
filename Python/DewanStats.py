from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy import stats
import itertools

from . import DewanAUROC
from .Helpers import DewanDataStore


def sparseness(iterable, means) -> float:
    """

    The actual calculations for lifetime and population sparseness are the same.
    The only difference is the nature of the arguments passed in. This serves
    as a function to do the mathematical calculations. The arguments are
    constructed elsewhere in the module:
    lifetimeSparseness() and popSparseness()

    Args:
        iterable (list or np.array):
            Cells or Odors to iterate over.
        means (list or np.array):
            List of means corresponding to each respective pairing in the iterable.

    Returns:
        sparseness (numpy.float64):
            Returns a float corresponding to the calculated sparseness. Whether this is
            lifetimeSparseness or populationSparseness depends on the arguments. Regardless,
            this function should not be called standalone.

    """

    upper_value = np.sum(means / iterable) ** 2
    lower_value = np.sum((means ** 2) / iterable)

    # if lower_value == 0:
    #     return 0
    #     # If all the means are zero, then it was clearly inhibitory and "didn't respond" should fix this elsewhere

    sparseness_val = (1 - (upper_value / lower_value))
    denominator = (1 - (1 / iterable))

    sparseness_val = sparseness_val / denominator

    return sparseness_val


def popSparseness(dataInput: DewanDataStore.AUROCdataStore, significanceTable: ndarray) -> tuple:
    population_sparseness = []

    mineral_oil_index = np.nonzero(dataInput.unique_odors == 'MO')[0]
    odor_indexes = np.nonzero(dataInput.unique_odors != 'MO')[0]
    non_mo_cells = np.nonzero(significanceTable[:, mineral_oil_index] == 0)[0]
    odor_significance_table = np.transpose(significanceTable)
    inhibitory_responses = [np.nonzero(row == 1)[0] for row in odor_significance_table[:, non_mo_cells]]

    for i, odor in enumerate(odor_indexes):
        dataInput.update_odor(odor)

        odor_data = []
        inhibitory_response_list = inhibitory_responses[i]
        for j, cell in enumerate(non_mo_cells):

            if len(inhibitory_response_list) > 0 and j in inhibitory_response_list:
                odor_data.append(0)
                continue
                # Inhibitory odor responses are set to zero

            dataInput.update_cell(cell)

            difference_of_means = returnDifferenceOfMeans(dataInput)

            odor_data.append(difference_of_means)
            # Add this odor's mean to the list of means

        odor_data = np.array(odor_data)
        sparseness_value = sparseness(len(non_mo_cells), odor_data)
        population_sparseness.append(sparseness_value)

    return np.array(population_sparseness), np.array(odor_indexes)


def lifetimeSparseness(dataInput: DewanDataStore.AUROCdataStore, significanceTable: ndarray) -> tuple:
    cells_lifetime_sparseness = []

    mineral_oil_index = np.nonzero(dataInput.unique_odors == 'MO')[0]
    odor_indexes = np.nonzero(dataInput.unique_odors != 'MO')[0]

    non_mo_cells = np.nonzero(significanceTable[:, mineral_oil_index] == 0)[0]
    # Only keep cells that are not responsive to MO
    # responsive_cells = non_mo_cells[np.any(significanceTable[non_mo_cells], axis=1)]
    # single_response_cells = np.nonzero(np.sum(significanceTable[responsive_cells], axis=1) <= 2)[0]
    inhibitory_responses = [np.nonzero(row == 1)[0] for row in significanceTable[non_mo_cells, :]]
    # Find where the inhibitory responses are, we need to set them to zero

    for i, cell in enumerate(non_mo_cells):

        dataInput.update_cell(cell)
        inhibitory_response_list = inhibitory_responses[i]
        cell_data = []
        for j, odor in enumerate(odor_indexes):

            if len(inhibitory_response_list) > 0 and j in inhibitory_response_list:
                cell_data.append(0)
                # Set all inhibitory responses to zero and skip to next odor
                continue

            dataInput.update_odor(odor)

            difference_of_means = returnDifferenceOfMeans(dataInput)
            cell_data.append(difference_of_means)
            # Add this odor's mean to the list of means

        cell_data = np.array(cell_data)

        sparseness_value = sparseness((dataInput.num_unique_odors - 1), cell_data)
        cells_lifetime_sparseness.append(sparseness_value)

    return np.array(cells_lifetime_sparseness), np.array(non_mo_cells)


def truncateData(data1, data2) -> tuple:
    data1_minima = [np.min(len(row)) for row in data1]
    data2_minima = [np.min(len(row)) for row in data2]
    row_minimum = int(min(min(data1_minima), min(data2_minima)))
    data1 = [row[:row_minimum] for row in data1]
    data2 = [row[:row_minimum] for row in data2]

    return data1, data2


def returnDifferenceOfMeans(dataInput: DewanDataStore.AUROCdataStore) -> float:
    baseline_data, evoked_data = DewanAUROC.collect_trial_data(dataInput, False)

    baseline_data, evoked_data = truncateData(baseline_data, evoked_data)
    # Sometimes the frame numbers don't line up between trials
    # We will find the shortest row in the evoked and baseline data, and set all rows to be that length
    # Occasionally will lose one datapoint from each row if min(row) == 39

    evoke_mean = np.mean(np.hstack(evoked_data))
    baseline_mean = np.mean(np.hstack(baseline_data))

    difference = evoke_mean - baseline_mean

    return difference


def crossTrialConsistency(dataInput: DewanDataStore.AUROCdataStore, significanceTable: np.array, latentCells: bool):
    mineral_oil_index = np.nonzero(dataInput.unique_odors == 'MO')[0]
    responsive_cell_list_index = np.nonzero(significanceTable[:, mineral_oil_index] == 0)[0]
    # Only keep cells that are not responsive to MO

    correlation_coefficient_matrix = []

    for cell in responsive_cell_list_index:
        odor_indexes = np.nonzero(dataInput.unique_odors != 'MO')[0]
        # Only keep odors that are not MO

        dataInput.update_cell(cell)

        cell_correlation_coefficients = []

        for odor in odor_indexes:
            dataInput.update_odor(odor)
            baseline_data, evoked_data = DewanAUROC.collect_trial_data(dataInput, False)
            baseline_mean, evoked_trials = truncateData(baseline_data, evoked_data)

            baseline_mean = np.mean(baseline_mean)

            odor_trials = np.subtract(evoked_trials, baseline_mean)

            mean_cc = spearmanCorrelation(odor_trials)

            cell_correlation_coefficients.append(mean_cc)

        correlation_coefficient_matrix.append(cell_correlation_coefficients)

    return correlation_coefficient_matrix


def generateCorrelationPairs(numTrials):
    return [pair for pair in itertools.combinations(range(numTrials), r=2)]
    # We <3 list comprehension


def spearmanCorrelation(trials):
    pairs2correlate = generateCorrelationPairs(len(trials))
    pairwise_correlation_coefficients = []

    for pair in pairs2correlate:
        trialA = trials[pair[0]]
        trialB = trials[pair[1]]
        CC = stats.spearmanr(trialA, trialB)
        pairwise_correlation_coefficients.append(CC)

    mean_cc = np.mean(pairwise_correlation_coefficients)
    return mean_cc
