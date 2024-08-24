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

from functools import partial
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

# Import from local modules
from .helpers import sliding_prob, trace_tools

NUM_SHUFFLES = 100


def compute_percentile(auroc, auroc_shuffle) -> float:
    return np.sum(auroc_shuffle < auroc) / auroc_shuffle.size


def compute_auc(group_1, group_2) -> float:
    max_baseline_val = max(group_1)
    min_baseline_val = min(group_1)

    baseline_prob = sliding_prob.sliding_probability(group_1, min_baseline_val, max_baseline_val)
    baseline_prob = sliding_prob.prep_probabilities(baseline_prob)

    evoked_prob = sliding_prob.sliding_probability(group_2, min_baseline_val, max_baseline_val)
    evoked_prob = sliding_prob.prep_probabilities(evoked_prob)

    auroc_value = np.trapz(evoked_prob, baseline_prob, axis=-1)
    # Calculate AUC values using the trapezoid method

    return auroc_value


def shuffled_distribution(all_vector: pd.DataFrame, test_data_size: int) -> np.ndarray:
    shuffled_auroc = []

    for _ in itertools.repeat(None, NUM_SHUFFLES):  # Repeat 1000 times, faster than range()

        split_1, split_2 = train_test_split(all_vector, test_size=test_data_size)
        # Split all the data into two randomized pools

        shuffled_auroc_value = compute_auc(split_1, split_2)
        # Get the AUC between the two pools

        shuffled_auroc.append(shuffled_auroc_value)
        # Save that AUC value

    return np.array(shuffled_auroc)


def EPM_auroc(pseudotrial_means, groups, cell_names):

    auroc_values = {}

    group_1, group_2 = prep_EPM_data(pseudotrial_means, groups)

    for cell in tqdm(cell_names):
        group_1_cell = group_1[cell]
        group_2_cell = group_2[cell]

        auroc_value = compute_auc(group_1_cell, group_2_cell)

        # # # GET SHUFFLED DISTRIBUTION # # #
        all_means = pd.concat((group_1_cell, group_2_cell), ignore_index=True)
        auroc_shuffle = shuffled_distribution(all_means, len(group_1_cell))
        bounds = np.percentile(auroc_shuffle, [1, 99])
        lower_bound, upper_bound = bounds

        cell_data = {
            'auroc': auroc_value,
            'lb': lower_bound,
            'ub': upper_bound,
            'shuffle': auroc_shuffle
        }
        auroc_values[cell] = cell_data

    return auroc_values

def prep_EPM_data(means, groups):
    group1, group2 = groups
    group1_data = [means[arm] for arm in group1]
    group2_data = [means[arm] for arm in group2]
    group1_data = pd.concat(group1_data)
    group2_data = pd.concat(group2_data)

    return group1_data, group2_data


def odor_auroc(FV_timestamps: pd.DataFrame, baseline_duration: int,
               latent: bool, cell_data: tuple) -> dict:

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

        auroc_value = compute_auc(baseline_means, evoked_means)

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


def pooled_odor_auroc(combined_data_shift: pd.DataFrame, FV_timestamps: pd.DataFrame, baseline_duration: int,
                     num_workers: int = 8, latent_cells_only: bool = False):
    # if latent_cells_only:
    #     auroc_type = 'Latent'
    # else:
    #     auroc_type = 'On Time'

    # print(f"Begin {auroc_type} AUROC Processing with {num_workers} processes!")

    iterable = combined_data_shift.T.groupby(level=0)
    # Level 0 is the cells; groupby() works on indexes, so we need to transpose it
    # since we ordered the data as columns with (Cell Name, Odor Name)
    auroc_partial_function = partial(odor_auroc, FV_timestamps, baseline_duration, latent_cells_only)
    return_dicts = process_map(auroc_partial_function, iterable, max_workers=num_workers)

    print("AUROC Processing Finished!")

    return return_dicts
