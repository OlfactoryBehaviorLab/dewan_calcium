import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, pairwise

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


def popSparseness(zeroed_trial_averaged_responses_matrix: pd.DataFrame, significant_ontime_cells: list, unique_odors)\
        -> pd.DataFrame:
    num_odors = len(zeroed_trial_averaged_responses_matrix.iloc[0])
    population_sparseness_values = []

    for i in range(num_odors):
        population_sparseness = sparseness(len(significant_ontime_cells),
                                           zeroed_trial_averaged_responses_matrix.iloc[:, i])
        population_sparseness_values.append(population_sparseness)

    population_sparse_DF = pd.DataFrame(population_sparseness_values, columns=['Population Sparseness'],
                                        index=unique_odors)

    return population_sparse_DF


def lifetimeSparseness(zeroed_trial_averaged_responses_matrix: pd.DataFrame, significant_ontime_cells: list) \
        -> pd.DataFrame:
    num_odors = len(zeroed_trial_averaged_responses_matrix.iloc[0])

    lifetime_sparseness_values = []
    cell_row_names = []
    for i, cell in enumerate(significant_ontime_cells):
        lifetime_sparseness = sparseness(num_odors, zeroed_trial_averaged_responses_matrix.iloc[i])
        lifetime_sparseness_values.append(lifetime_sparseness)
        cell_row_names.append(f'Cell {cell + 1}')

    lifetime_sparse_DF = pd.DataFrame(lifetime_sparseness_values, columns=['Lifetime Sparseness'], index=cell_row_names)

    return lifetime_sparse_DF


def truncate_data(data1, data2) -> tuple:
    data1_minima = [np.min(len(row)) for row in data1]
    data2_minima = [np.min(len(row)) for row in data2]
    row_minimum = int(min(min(data1_minima), min(data2_minima)))
    data1 = [row[:row_minimum] for row in data1]
    data2 = [row[:row_minimum] for row in data2]

    return data1, data2


def generate_correlation_pairs(numTrials):
    return [pair for pair in itertools.combinations(range(numTrials), r=2)]
    # We <3 list comprehension


def trial_averaged_odor_responses(stats_data: DewanDataStore.AUROCdataStore, significant_ontime_cells: list):
    trial_averaged_responses_matrix = []

    for cell in significant_ontime_cells:
        # Loop through the significant cells
        stats_data.update_cell(cell)  # Update the datastore to grab the correct data

        trial_averaged_responses = []

        for odor in range(len(stats_data.unique_odors)):  # Loop through all odors
            stats_data.update_odor(odor)
            baseline_data, evoked_data = DewanAUROC.collect_trial_data(stats_data, None, False)
            # Get cell-odor data for all trials, no returns, on time cells only
            baseline_data, evoked_data = truncate_data(baseline_data, evoked_data)  # Make all rows the same length

            baseline_mean = np.mean(baseline_data)  # Get the average baseline for all trials
            evoked_data = np.subtract(evoked_data, baseline_mean)  # Baseline shift all the evoked data
            average_response = np.mean(evoked_data)  # Average the baseline-shifted responses for all trials

            trial_averaged_responses.append(average_response)

        trial_averaged_responses_matrix.append(trial_averaged_responses)
    return trial_averaged_responses_matrix


def calculate_pairwise_distances(trial_averaged_responses_matrix: list, unique_odors: list):
    scaler = StandardScaler()
    scaled_trial_averaged_responses = scaler.fit_transform(trial_averaged_responses_matrix)

    odor_trial_averaged_responses_matrix = pd.DataFrame(np.transpose(scaled_trial_averaged_responses))
    odor_trial_averaged_responses_matrix.set_index(unique_odors, inplace=True)

    # 8A.3: Calculate several different correlation coefficients
    cell_pairwise_distances = pairwise_distances(scaled_trial_averaged_responses, metric='correlation')
    odor_pairwise_distances = pairwise_distances(odor_trial_averaged_responses_matrix, metric='correlation')

    return odor_pairwise_distances, cell_pairwise_distances


def cell_v_correlation(Centroids, cell_pairwise_distances):
    distance_matrix = calculate_spatial_distance(Centroids)

    # 8B.2: Pair the spatial distance with its associated correlation coefficient
    distance_v_correlation_pairs = np.stack((distance_matrix, cell_pairwise_distances), axis=-1)

    unique_distance_v_correlation_pairs = []
    for i, cell in enumerate(distance_v_correlation_pairs[:-1]):
        #8B.3: Select half of the n x n matrix since the bottom half is just a reflection
        unique_distance_v_correlation_pairs.extend(cell[i + 1:])

    unique_distance_v_correlation_pairs = np.vstack(unique_distance_v_correlation_pairs)  # Combine individual rows into one large contiguous array

    ## Alternative way to do the above that is vectorized instead of looped
    # distance_v_correlation = np.vstack(distance_v_correlation)
    # distance_v_correlation = pd.DataFrame(np.round(distance_v_correlation, 6))
    # distance_v_correlation.drop_duplicates(inplace=True)
    # distance_v_correlation = distance_v_correlation.iloc[1:]

    return unique_distance_v_correlation_pairs

def calculate_spatial_distance(Centroids):
    return pairwise.euclidean_distances(Centroids.values, Centroids.values)
