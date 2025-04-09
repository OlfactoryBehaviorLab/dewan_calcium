import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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


def calculate_pairwise_distances(trial_averaged_responses_matrix: pd.DataFrame) -> tuple:
    cells = trial_averaged_responses_matrix.columns
    odors = trial_averaged_responses_matrix.index

    scaler = StandardScaler()
    scaled_trial_averaged_responses = scaler.fit_transform(trial_averaged_responses_matrix)
    cell_trial_averaged_responses_matrix = scaled_trial_averaged_responses.T

    # 8A.3: Calculate several different correlation coefficients
    cell_pairwise_distances = pairwise_distances(cell_trial_averaged_responses_matrix, metric='correlation')
    odor_pairwise_distances = pairwise_distances(scaled_trial_averaged_responses, metric='correlation')

    cell_pairwise_distances = pd.DataFrame(cell_pairwise_distances, index=cells, columns=cells)
    odor_pairwise_distances = pd.DataFrame(odor_pairwise_distances, index=odors, columns=odors)

    return odor_pairwise_distances, cell_pairwise_distances


def calculate_spatial_distance(centroids: pd.DataFrame) -> list:
    return pairwise.euclidean_distances(centroids.values, centroids.values)


def _calc_wilcoxon(baseline_periods, evoked_periods, bin_pairs):
    wilcoxon_results = []
    baseline_means = np.mean(baseline_periods, axis=1)
    for _bin in bin_pairs:
        evoked_periods_bin = evoked_periods[:, _bin[0]:_bin[1]]
        evoked_means = np.mean(evoked_periods_bin, axis=1)
        bin_wilcoxon = wilcoxon(evoked_means, baseline_means).pvalue
        wilcoxon_results.append(bin_wilcoxon)

    return pd.Series(wilcoxon_results)


def binned_wilcoxon(odor_df, bin_pairs, baseline_duration, evoked_duration):
    baseline_values = odor_df.iloc[:, :baseline_duration].values
    evoked_values = odor_df.iloc[:, baseline_duration: (evoked_duration + baseline_duration + 1 )].values

    return _calc_wilcoxon(baseline_values, evoked_values, bin_pairs)
