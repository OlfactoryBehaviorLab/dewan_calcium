"""
Dewan Lab Sliding Probability for AUROC.
Original Author: Roberto Vincis' Lab @ Florida State University, Aug 10, 2022
Modified for use by Dewan Lab: ACP and AKD, December 2, 2022

Computes sliding probability for a data set based on max, min and increment
values provided as arguments
"""
import numpy as np

SLID_PROB_INC = 100


def sliding_probability(data, start_range: int, end_range: int) -> np.array:
    prob_vector = []
    bin_steps = (end_range - start_range) / SLID_PROB_INC
    bins = np.arange(start_range, end_range, bin_steps)

    for current_bin in bins:
        prob_vector.append(sum(np.ravel(data > current_bin)) / len(np.ravel(data)))

    return prob_vector


def prep_probabilities(data: np.array):

    data.reverse()
    data.insert(0, 0.0)
    data.insert(len(data) + 1, 1.0)

    return data
