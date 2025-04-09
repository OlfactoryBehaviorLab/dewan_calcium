"""
Dewan Lab Sliding Probability for AUROC.
Original Author: Roberto Vincis' Lab @ Florida State University, Aug 10, 2022
Modified for use by Dewan Lab: ACP and AKD, December 2, 2022

Computes sliding probability for a data set based on max, min and increment
values provided as arguments
"""
import numpy as np
# from numba.pycc import CC
from numba import njit

# cc=CC('sliding_prob_numba')


SLID_PROB_INC = 100

# @cc.export('sliding_prob', '(f8[:], i4, i4)')
# @njit
def sliding_probability(data, start_range: int, end_range: int) -> np.array:
    prob_vector = []
    bin_steps = (end_range - start_range) / SLID_PROB_INC
    bins = np.arange(start_range, end_range, bin_steps)

    for current_bin in bins:
        prob_vector.append(sum(np.ravel(data > current_bin)) / len(np.ravel(data)))

    return prob_vector

# @cc.export('prep_prob', '(f8[:],)')
def prep_probabilities(data: np.array):

    data.reverse()
    data.insert(0, 0.0)
    data.insert(len(data) + 1, 1.0)

    return data

if __name__ == '__main__':
    cc.compile()