"""
Module containing classifier functions and related helpers
Author: Austin Pauley (pauley@psy.fsu.edu)
Date Created: 10/30/2024

Most of the code in this module is either directly from, or heavily influenced by, code from
The Vincis Lab at Florida State University (https://github.com/vincisLab/thermalGC)
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


def run_svm(traces: pd.DataFrame, correct_labels: pd.Series, test_percentage: float = 0.2, num_splits: int =20 ):
    """

    Args:
        traces (pd.DataFrame): DataFrame containing preprocessed 1-P calcium transient data. DataFrame is size n x t,
        where each row is a trial and each column is a sample (total samples equal to t).
        correct_labels (pd.Series): Series of labels corresponding to each trial (total trials equal to n)
        test_percentage (float): Percentage of data to use as test set
        num_splits (int): Number of splits to use for cross-validation

    Returns:

    """

    num_samples = traces.shape[0] * (1 - test_percentage)  # Number of trials
    num_features = traces.shape[1]  # Number of data points
    use_dual_param = (num_samples < num_features)
    # Documentation recommends using dual_param mode if samples < features

    svm = LinearSVC(dual=use_dual_param, max_iter=10000, random_state=1000)

    pass
