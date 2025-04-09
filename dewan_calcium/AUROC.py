"""
AUROC Analysis for Cells v. Odor Presentations.

Dewan Lab, Florida State University
Original Script: Roberto Vincis' Lab, FSU
Modified for Dewan Lab Use: S. Caton, A. Pauley, A. Dewan
December 2022
"""

import itertools
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from functools import partial

from numba import njit
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

# Import from local modules
from .helpers import trace_tools, sliding_prob
# from .libs import sliding_prob_numba

NUM_SHUFFLES = 1000

@njit()
def compute_percentile(auroc, auroc_shuffle) -> float:
    return np.sum(auroc_shuffle < auroc) / auroc_shuffle.size


@njit()
def compute_auc(means_1, means_2) -> float:
    max_baseline_val = max(means_1)
    min_baseline_val = min(means_1)

    baseline_prob = sliding_prob.sliding_probability(means_1, min_baseline_val, max_baseline_val)
    baseline_prob = sliding_prob.prep_probabilities(np.array(baseline_prob))

    evoked_prob = sliding_prob.sliding_probability(means_2, min_baseline_val, max_baseline_val)
    evoked_prob = sliding_prob.prep_probabilities(np.array(evoked_prob))

    auroc_value = np.trapz(evoked_prob, baseline_prob)
    # Calculate AUC values using the trapezoid method

    return auroc_value


def shuffled_distribution(all_vector: pd.DataFrame, test_data_size: int) -> np.ndarray:
    shuffled_auroc = []

    for _ in itertools.repeat(None, NUM_SHUFFLES):  # Repeat 1000 times, faster than range()

        split_1, split_2 = train_test_split(all_vector, test_size=test_data_size)
        # Split all the data into two randomized pools

        shuffled_auroc_value = compute_auc(split_1.values, split_2.values)
        # Get the AUC between the two pools

        shuffled_auroc.append(shuffled_auroc_value)
        # Save that AUC value

    return np.array(shuffled_auroc)


def _pseudotrial_auroc(pseudotrial_groups):
    group_1, group_2 = pseudotrial_groups
    cell, g1_data = group_1
    _, g2_data = group_2

    try:

        auroc_value = compute_auc(g1_data, g2_data)

        # # # GET SHUFFLED DISTRIBUTION # # #
        all_means = pd.concat((g1_data, g2_data), ignore_index=True)
        auroc_shuffle = shuffled_distribution(all_means, len(g1_data))
        bounds = np.percentile(auroc_shuffle, [1, 99])
        lower_bound, upper_bound = bounds

        if auroc_value >= upper_bound:
            significance = 1
        elif auroc_value <= lower_bound:
            significance = -1
        else:
            significance = 0

        cell_data = {
            'name': cell,
            'auroc': auroc_value,
            'lb': lower_bound,
            'ub': upper_bound,
            'shuffle': auroc_shuffle,
            'significance': significance
        }

        return cell_data
    except ValueError:  # Yes, this is not the proper way to do this
        return {'name': cell, 'error': 'error'}


def _EPM_generator(group1, group2):
    g1_iterable = group1.items()
    g2_iterable = group2.items()

    return zip(g1_iterable, g2_iterable)


def _HFFM_generator(group1, group2):
    g1_iterable = group1.iterrows()
    g2_iterable = group2.iterrows()

    return zip(g1_iterable, g2_iterable)


def pooled_EPM_auroc(pseudotrial_means, groups, num_workers=8):
    group1, group2 = _prep_EPM_data(pseudotrial_means, groups)

    EPM_iterable = _EPM_generator(group1, group2)

    return_dicts = process_map(_pseudotrial_auroc, EPM_iterable, max_workers=num_workers,
                               desc="Calculating auROC Statistics for EPM Cells", total=len(group1.columns))

    return return_dicts


def pooled_HFFM_auroc(pseudotrial_means, groups, num_workers=8):
    group1, group2 = _prep_HFFM_data(pseudotrial_means, groups)
    HFFM_iterable = _HFFM_generator(group1, group2)

    return_dicts = process_map(_pseudotrial_auroc, HFFM_iterable, max_workers=num_workers,
                               desc="Calculating auROC Statistics for HF_FM Cells", total=len(group1.index))

    return return_dicts


def _prep_HFFM_data(means, groups):
    group1, group2 = groups
    group1_data = [means[arm] for arm in group1]
    group2_data = [means[arm] for arm in group2]
    group1_data = pd.concat(group1_data, axis=1)
    group2_data = pd.concat(group2_data, axis=1)

    return group1_data, group2_data


def _prep_EPM_data(means, groups):
    group1, group2 = groups
    group1_data = [means[arm] for arm in group1]
    group2_data = [means[arm] for arm in group2]
    group1_data = pd.concat(group1_data)
    group2_data = pd.concat(group2_data)

    group1_data = group1_data.reset_index(drop=True)
    group2_data = group2_data.reset_index(drop=True)

    return group1_data, group2_data


def odor_auroc(FV_timestamps: pd.DataFrame, baseline_duration: int,
              cell_data: tuple) -> dict:

    all_bounds = []
    auroc_values = []
    all_percentiles = []
    significance_matrix = []
    all_indices = []
    shuffles = []

    # # # Unpack the Input # # #
    cell_name, trace_data = cell_data
    cell_df = trace_data.T[cell_name]  # Transpose data so (Cells, Odor) is the columns, and the enter one level
    odor_list = cell_df.columns.unique()

    for odor in odor_list:
        odor_df = cell_df[odor]  # Get traces for each odor type, this should be 10-12 long
        odor_timestamps = FV_timestamps[odor]
        baseline_data, evoked_data, baseline_indices, evoked_indices = (
            trace_tools.new_collect_trial_data(odor_df, odor_timestamps, baseline_duration, latent))

        baseline_means = baseline_data.mean(axis=1)
        evoked_means = evoked_data.mean(axis=1)
        auroc_value = compute_auc(baseline_means.values, evoked_means.values)

        # # # GET SHUFFLED DISTRIBUTION # # #
        all_means = pd.concat((baseline_means, evoked_means), ignore_index=True)
        auroc_shuffle = shuffled_distribution(all_means, len(baseline_means))
        bounds = np.percentile(auroc_shuffle, [1, 99])

        lower_bound, upper_bound = bounds

        # # # Output Data # # #
        if auroc_value > upper_bound:
            significance_matrix.append(2)  # Positive evoked response
        elif auroc_value < lower_bound:
            significance_matrix.append(1)  # Negative evoked response
        else:
            significance_matrix.append(0)  # No response

        all_bounds.append(bounds)
        auroc_values.append(auroc_value)
        all_percentiles.append(compute_percentile(auroc_value, auroc_shuffle))
        indices = (baseline_indices, evoked_indices)
        all_indices.append(indices)
        shuffles.append(auroc_shuffle)

    return_dict = {
        'auroc_values': auroc_values,
        'significance_chart': significance_matrix,
        'bounds': all_bounds,
        'percentiles': all_percentiles,
        'indices': all_indices,
        'shuffles': shuffles
    }

    return return_dict


def pooled_odor_auroc(combined_data_dff: pd.DataFrame, FV_timestamps: pd.DataFrame, evoked_duration,
                     num_workers: int = 8):
    print('Starting auROC!')
    iterable = combined_data_dff.T.groupby(level=0)
    # Level 0 is the cells; groupby() works on indexes, so we need to transpose it
    # since we ordered the data as columns with (Cell Name, Odor Name)
    auroc_partial_function = partial(odor_auroc, FV_timestamps, evoked_duration)
    return_dicts = process_map(auroc_partial_function, iterable, max_workers=num_workers)

    print("auROC Processing Finished!")

    return return_dicts
