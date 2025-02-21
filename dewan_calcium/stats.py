import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, pairwise


def sparseness(iterable: int, means: np.array) -> float:
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
    if lower_value <= 0:
        return np.nan
    # If all the means are zero, then it was only inhibitory and "didn't respond" in context of lifetime/population sparseness
    # return nan where division by zero would occur
        

    sparseness_val = (1 - (upper_value / lower_value))
    denominator = (1 - (1 / iterable))

    sparseness_val = sparseness_val / denominator


    return sparseness_val


def generate_correlation_pairs(number_trials: int) -> list:
    return [pair for pair in itertools.combinations(range(number_trials), r=2)]
    # We <3 list comprehension


def calculate_pairwise_distances(trial_averaged_responses_matrix: list, unique_odors: list) -> tuple:
    scaler = StandardScaler()
    scaled_trial_averaged_responses = scaler.fit_transform(trial_averaged_responses_matrix)

    odor_trial_averaged_responses_matrix = pd.DataFrame(np.transpose(scaled_trial_averaged_responses))
    odor_trial_averaged_responses_matrix.set_index(unique_odors, inplace=True)

    # 8A.3: Calculate several different correlation coefficients
    cell_pairwise_distances = pairwise_distances(scaled_trial_averaged_responses, metric='correlation')
    odor_pairwise_distances = pairwise_distances(odor_trial_averaged_responses_matrix, metric='correlation')

    return odor_pairwise_distances, cell_pairwise_distances


def cell_v_correlation(centroids, cell_pairwise_distances) -> list:
    distance_matrix = calculate_spatial_distance(centroids)

    # Pair the spatial distance with its associated correlation coefficient
    distance_v_correlation_pairs = np.stack((distance_matrix, cell_pairwise_distances), axis=-1)

    unique_distance_v_correlation_pairs = []
    for i, cell in enumerate(distance_v_correlation_pairs[:-1]):
        # Select half of the n x n matrix since the bottom half is just a reflection
        unique_distance_v_correlation_pairs.extend(cell[i + 1:])

    unique_distance_v_correlation_pairs = np.vstack(unique_distance_v_correlation_pairs)
    # Combine individual rows into one large contiguous array

    # Alternative way to do the above that is vectorized instead of looped
    # distance_v_correlation = np.vstack(distance_v_correlation)
    # distance_v_correlation = pd.DataFrame(np.round(distance_v_correlation, 6))
    # distance_v_correlation.drop_duplicates(inplace=True)
    # distance_v_correlation = distance_v_correlation.iloc[1:]

    return unique_distance_v_correlation_pairs


def calculate_spatial_distance(centroids: pd.DataFrame) -> list:
    return pairwise.euclidean_distances(centroids.values, centroids.values)
