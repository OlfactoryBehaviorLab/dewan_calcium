"""
Helper Functions for Dewan Lab Image Analysis
Authors: A. Pauley, S. Caton, A. Dewan
November 2022

Framework for the mtlb_intersect function provided by user6655984 on stackoverflow
https://stackoverflow.com/questions/45637778/how-to-find-intersect-indexes-and-values-in-python
Function adapted for Dewan Lab on 11/22/2022 A. Pauley and S. Caton

"""

import numpy as np


def findGoodTrials(TraceData, TimeStamps, good_criterion):
    number_of_timestamps = len(TimeStamps)
    data_length = len(TraceData.index.values)
    number_good_trials = -1
    cell_trace_idx = np.zeros((number_of_timestamps, 1))
    good_trials = np.zeros((number_of_timestamps, 1))

    for i, TIMESTAMP in enumerate(TimeStamps):
        list_of_time_points = np.nonzero(TraceData.index.values.astype(float) < TIMESTAMP)[0]
        # Number of time points less than the current one
        # Gives current index - 1
        current_position = len(list_of_time_points)  # The index before valve switched on/off
        if 0 < current_position < data_length:  # Not interested in very first or very last time point
            one_before_current_point = float(TraceData.index.values[current_position])
            # Time stamp one before the current position
            one_post_current_point = float(TraceData.index.values[current_position + 1])
            # Time point one after the current position
            if ((TIMESTAMP - one_before_current_point) < good_criterion and
                    (one_post_current_point - TIMESTAMP) < good_criterion):
                # Check to make sure that the time point before and after is within the goodCriterion
                number_good_trials += 1
                cell_trace_idx[number_good_trials] = current_position
                good_trials[number_good_trials] = i

    cell_trace_idx = cell_trace_idx[:number_good_trials + 1]
    good_trials = good_trials[:number_good_trials + 1]
    return cell_trace_idx, good_trials


def intersect_matlab(a, b):
    list1, index1 = np.unique(a, return_index=True)
    list2, index2 = np.unique(b, return_index=True)
    list1_list2 = np.concatenate((list1, list2))
    list1_list2.sort()

    all_unique_items, num_appearances = np.unique(list1_list2, return_counts=True)

    shared_items = all_unique_items[num_appearances > 1]
    a_indexes = index1[np.isin(list1, shared_items)]
    b_indexes = index2[np.isin(list2, shared_items)]

    return shared_items, a_indexes, b_indexes

# Credit to user6655984 on stackoverflow
# https://stackoverflow.com/questions/45637778/how-to-find-intersect-indexes-and-values-in-python
# Modified 11/22/2022 ACP SAC, DewanLab
