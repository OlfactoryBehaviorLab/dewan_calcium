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

from multiprocessing import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

# Import from local modules
from .helpers import data_stores, sliding_prob, trace_tools
from . import DewanPlotting

NUM_SHUFFLES = 1000


def compute_percentile(auroc, auroc_shuffle) -> float:
    return np.sum(auroc_shuffle < auroc) / auroc_shuffle.size


def compute_auc(means_1, means_2) -> float:
    max_baseline_val = max(means_1)
    min_baseline_val = min(means_1)

    baseline_prob = sliding_prob.sliding_probability(means_1, min_baseline_val, max_baseline_val)
    baseline_prob = sliding_prob.prep_probabilities(baseline_prob)

    evoked_prob = sliding_prob.sliding_probability(means_2, min_baseline_val, max_baseline_val)
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


def run_auroc(data_input: data_stores.AUROCdataStore, latent_cells: bool, cell_number: int) -> data_stores.AUROCReturn:

    significant = True  # Does this particular cell-odor pair show a significant response; True by default

    data_input = data_input.makeCopy()  # Get a local copy of the data for this process
    data_input.update_cell(cell_number)  # Update the local copy with which cell we're computing significance for
    return_values = data_stores.AUROCReturn()  # Create an empty AUROCReturn object to store the return values in

    for odor_iterator in range(data_input.num_unique_odors):

        data_input.update_odor(odor_iterator)
        # Update the current odor that we are computing significance for

        baseline_data, evoked_data = trace_tools.collect_trial_data(data_input, return_values, latent_cells)
        # Get the raw df/F values for this cell-odor combination
        baseline_means, evoked_means = trace_tools.average_trial_data(baseline_data, evoked_data)
        # average the baseline and trial data

        auroc_value = compute_auc(baseline_means, evoked_means)
        # compute the actual AUROC value for our trace data

        all_means_vector = np.concatenate((baseline_means, evoked_means))
        auroc_shuffle = shuffled_distribution(all_means_vector, baseline_means)
        # Put the baseline and evoked means together and create a shuffled distribution of all the data

        lower_bound = np.percentile(auroc_shuffle, [1])
        upper_bound = np.percentile(auroc_shuffle, [99])
        # Get the 1st and 99th percentile values of the 'random' shuffled data

        return_values.all_lower_bounds.append(lower_bound)
        return_values.all_upper_bounds.append(upper_bound)
        return_values.auroc_values.append(auroc_value)
        return_values.percentiles.append(compute_percentile(auroc_value, auroc_shuffle))
        # 0 - 1; Calculate where each auroc value falls on the shuffled distribution

        if auroc_value > upper_bound:
            return_values.response_chart.append(2)  # Positive evoked response
        elif auroc_value < lower_bound:
            return_values.response_chart.append(1)  # Negative evoked response
        else:
            return_values.response_chart.append(0)  # No response
            significant = False

        if significant and data_input.do_plot:
            # Check that both the response was significant and we want to plot the distributions
            DewanPlotting.plot_auroc_distributions(data_input.file_header, auroc_shuffle, auroc_value, upper_bound,
                                                   lower_bound, data_input.Cell_List, cell_number,
                                                   data_input.unique_odors, odor_iterator, latent_cells)

    return return_values


def new_run_auroc(cell_df: pd.DataFrame, FV_timestamps: pd.DataFrame, odor_list: list,
                  baseline_duration: int, latent: bool = False) -> data_stores.AUROCReturn:
    all_bounds = []
    auroc_values = []
    all_percentiles = []
    response_chart = []

    significant = True

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
            response_chart.append(2)  # Positive evoked response
        elif auroc_value < lower_bound:
            response_chart.append(1)  # Negative evoked response
        else:
            response_chart.append(0)  # No response
            significant = False

        all_bounds.append(bounds)
        auroc_values.append(auroc_value)
        all_percentiles.append(compute_percentile(auroc_value, auroc_shuffle))
        
    return data_stores.AUROCReturn()


def pooled_auroc(data_input: data_stores.AUROCdataStore, num_workers: int = 8, latent_cells_only: bool = False) -> list:
    if latent_cells_only:
        auroc_type = 'Latent'
    else:
        auroc_type = 'On Time'

    print(f"Begin {auroc_type} AUROC Processing with {num_workers} processes!")

    workers = Pool()
    partial_function = partial(run_auroc, data_input, latent_cells_only)
    return_values = process_map(partial_function, range(data_input.number_cells), max_workers=num_workers,
                                desc='AUROC Progress: ')
    # TQDM wrapper for concurrent features
    workers.close()
    workers.join()

    print("AUROC Processing Finished!")

    return return_values
