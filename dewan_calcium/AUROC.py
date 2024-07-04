"""
AUROC Analysis for Cells v. Odor Presentations.

Dewan Lab, Florida State University
Original Script: Roberto Vincis' Lab, FSU
Modified for Dewan Lab Use: S. Caton, A. Pauley, A. Dewan
December, 2022
"""

import itertools
import numpy as np

from multiprocessing import Pool
from functools import partial
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

# Import from local modules
from .helpers import data_stores, DewanSlidingProbability, DewanTraceTools
from . import DewanPlotting


def compute_percentile(auroc, auroc_shuffle) -> float:
    return np.sum(auroc_shuffle < auroc) / auroc_shuffle.size


def shuffled_distribution(all_vector: np.ndarray, vect_base1: np.ndarray) -> np.ndarray:
    shuffled_auroc = []

    for _ in itertools.repeat(None, 1000):  # Repeat 1000 times, faster than range()

        split_1, split_2 = train_test_split(all_vector, test_size=len(vect_base1)) # Split the data into two randomized pools

        shuffle_max = max(split_1)
        shuffle_min = min(split_1)
        increments = 100

        baseline_shuffle = DewanSlidingProbability.sliding_probability(split_1, shuffle_min, shuffle_max, increments)

        baseline_shuffle.reverse()
        baseline_shuffle.insert(0, 0.0)
        baseline_shuffle.insert(len(split_1) + 1, 1.0)

        evoked_shuffle = DewanSlidingProbability.sliding_probability(split_2, shuffle_min, shuffle_max, increments)

        evoked_shuffle.reverse()
        evoked_shuffle.insert(0, 0.0)
        evoked_shuffle.insert(len(evoked_shuffle) + 1, 1.0)

        shuffled_auroc_values = np.trapz(evoked_shuffle, baseline_shuffle, axis=-1) # Calculate AUROC values of this iteration of the shuffled distribution

        shuffled_auroc.append(shuffled_auroc_values)

    return np.array(shuffled_auroc)


def run_auroc(data_input: DewanDataStore.AUROCdataStore, latent_cells: bool,
                    cell_number: int) -> DewanDataStore.AUROCReturn:

    significant = False; # Does this particular cell-odor pair show a significant response

    data_input = data_input.makeCopy() # Get a local copy of the data for this process
    data_input.update_cell(cell_number) # Update the local copy with which cell we're computing significance for
    return_values = DewanDataStore.AUROCReturn() # Create an empty AUROCReturn object to store the return values in

    for odor_iterator in range(data_input.num_unique_odors): # Iterate over each odor
        data_input.update_odor(odor_iterator) # Update the current odor that we are computing significance for

        baseline_data, evoked_data = DewanTraceTools.collect_trial_data(data_input, return_values, latent_cells) # Get the raw df/F values for this cell-odor combination
        baseline_means, evoked_means = DewanTraceTools.average_trial_data(baseline_data, evoked_data) # average the baseline and trial data
        max_baseline_val = max(baseline_means)
        min_baseline_val = min(baseline_means)

        increments = 100

        baseline_prob = DewanSlidingProbability.sliding_probability(baseline_means, min_baseline_val, max_baseline_val, increments)

        baseline_prob.reverse()
        baseline_prob.insert(0, 0.0) # Insert 0 at the beginning of the probabilities as the lowest possible value
        baseline_prob.insert(len(baseline_prob) + 1, 1.0)  # Insert 1 at the end of the probabilities as the highest possible value

        evoked_prob = DewanSlidingProbability.sliding_probability(evoked_means, min_baseline_val, max_baseline_val, increments)

        evoked_prob.reverse()
        evoked_prob.insert(0, 0.0)
        evoked_prob.insert(len(baseline_prob) + 1, 1.0)

        auroc_value = np.trapz(evoked_prob, baseline_prob, axis=-1) # Calculate AUROC values using the trapezoid method

        all_means_vector = np.concatenate((baseline_means, evoked_means)) # Put the baseline and evoked means together and create a shuffled distribution
        auroc_shuffle = shuffled_distribution(all_means_vector, baseline_means)

        lower_bound = np.percentile(auroc_shuffle, [1]) # Get the 1st and 99th percentile values of the 'random' shuffled data
        upper_bound = np.percentile(auroc_shuffle, [99])

        return_values.all_lower_bounds.append(lower_bound)
        return_values.all_upper_bounds.append(upper_bound)
        return_values.auroc_values.append(auroc_value)
        return_values.percentiles.append(compute_percentile(auroc_value, auroc_shuffle))  # 0 - 1; Calculate where each auroc value falls on the shuffled distribution

        
        if auroc_value > upper_bound:
            return_values.response_chart.append(2) # Positive evoked response
        elif auroc_value < lower_bound:
            return_values.response_chart.append(1) # Negative evoked response
        else:
            return_values.response_chart.append(0) # No response

        if significant and data_input.do_plot: # Check that both the response was significant and we want to plot the distributions
            DewanPlotting.plot_auroc_distributions(data_input.file_header, auroc_shuffle, auroc_value, upper_bound, lower_bound, data_input.Cell_List,
                                                       cell_number, data_input.unique_odors, odor_iterator, latent_cells)

    return return_values


def pooled_auroc(data_input: DewanDataStore.AUROCdataStore, num_workers: int=8, latent_cells_only: bool=False) -> list:
    auroc_type = []
    
    if not latent_cells_only:
        auroc_type = 'On Time'
    else:
        auroc_type = 'Latent'

    print(f"Begin {auroc_type} AUROC Processing with {num_workers} processes!")

    workers = Pool()
    partial_function = partial(run_auroc, data_input, latent_cells_only)
    return_values = process_map(partial_function, range(data_input.number_cells), max_workers = num_workers, desc='AUROC Progress: ')
    # TQDM wrapper for concurrent features
    workers.close()
    workers.join()

    print("AUROC Processing Finished!")

    return return_values

