import numpy as np
import pandas as pd
from scipy import signal
from oasis.functions import deconvolve  # install using conda install to avoid having to build
from tqdm import tqdm


def z_score_data(trace_data: pd.DataFrame, cell_names) -> pd.DataFrame:
    # Function is given a Cells x Trials array
    # Zscores each trial and then returns the array

    from scipy.stats import zscore

    z_scored_data = pd.DataFrame()

    for cell in cell_names:

        fluorescence_values = trace_data[cell].values
        z_score = zscore(fluorescence_values)

        z_score = pd.Series(z_score, name=cell)
        z_scored_data = pd.concat((z_scored_data, z_score), axis=1)

    return z_scored_data


def find_peaks(smoothed_data: pd.DataFrame, cell_names, framerate: int, peak_args: dict) -> dict:
    width_time = peak_args['decay']
    inter_spike_time = peak_args['ISI']
    peak_height = peak_args['height']

    peak_width_distance = framerate * (width_time / 1000)
    inter_transient_distance = framerate * (inter_spike_time / 1000)

    transient_indexes = dict()

    for name, trace in tqdm(smoothed_data[cell_names].items(), desc="Find Transient Indexes: ", total=len(cell_names)):
        peaks = signal.find_peaks(trace, height=peak_height, width=peak_width_distance,
                                  distance=inter_transient_distance)
        peaks = peaks[0]  # Return only the indexes (x locations) of the peaks
        transient_indexes[name] = peaks

    return transient_indexes


def calc_smoothing_params(endoscope_framerate=10, decay_time_s=0.4, rise_time_s=0.08):
    """

    Args:
        endoscope_framerate: Frame rate in seconds of the micro-endoscope (10Hz)
        decay_time_s: Time in seconds for the decay of 10 action potentials (0.4 for gcamp6f)
        rise_time_s: Time in seconds for the rise to peak of 10 action potentials (0.08 for gcamp6f)

    Returns:
        g1: kernel component 1
        g2: kernel component 2

    """
    decay_param = np.exp(-1 / (decay_time_s * endoscope_framerate))
    rise_param = np.exp(-1 / (rise_time_s * endoscope_framerate))

    g1 = round(decay_param + rise_param, 5)
    g2 = round(-decay_time_s * rise_param, 5)

    return g1, g2


def _run_deconv(trace, g1, g2):
    import warnings

    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)

    deconv_data = deconvolve(trace, (g1, g2))
    smoothed_trace = deconv_data[0]

    return smoothed_trace


def smooth_data(smoothing_kernel, trace_data) -> dict:

    cell_smoothed_traces = {}

    name, cell_data = trace_data
    cell_data = cell_data.T
    g1, g2 = smoothing_kernel

    for trial in cell_data.columns:
        _, trial_name = trial
        trace = cell_data[trial].values

        nan_vals = np.where(np.isnan(trace))[0]

        if len(nan_vals) > 0:
            trace = trace[:nan_vals[0]]

        smoothed_trace = _run_deconv(trace, g1, g2)
        cell_smoothed_traces[trial_name] = smoothed_trace

    cell_smoothed_traces['name'] = name

    return cell_smoothed_traces


def pooled_deconvolution(combined_data, smoothing_kernel, workers=8):
    from functools import partial
    from tqdm.contrib.concurrent import process_map

    iterable = combined_data.T.groupby(level=0)
    partial_function = partial(smooth_data, smoothing_kernel)

    return_dicts = process_map(partial_function, iterable, max_workers=workers)

    return _repackage_return(return_dicts)

def _repackage_return(return_dicts):
    new_return_dicts = {}

    for cell in return_dicts:
        cell_name = cell['name']
        cell.pop('name', None)
        new_return_dicts[cell_name] = cell

    return new_return_dicts