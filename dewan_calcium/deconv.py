import numpy as np
from scipy import signal
from oasis.functions import deconvolve  # install using conda install to avoid having to build

import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else: import tqdm


def z_score_data(smoothed_data: dict, cell_names) -> dict:
    # Function is given a Cells x Trials array
    # Z-scores each trial and then returns the array

    from scipy.stats import zscore

    z_scored_data = dict()

    for cell in cell_names:
        cell_data_zscore = dict()
        cell_data = smoothed_data[cell]
        for trial in cell_data.keys():
            trial_dff = cell_data[trial]
            z_score = zscore(trial_dff)
            cell_data_zscore[trial] = z_score

        z_scored_data[cell] = cell_data_zscore

    return z_scored_data


def find_peaks(smoothed_data: dict, cell_names, ENDOSCOPE_FRAMERATE, INTER_SPIKE_INTERVAL, PEAK_MIN_DUR_S, height=1) -> dict:


    peak_width = ENDOSCOPE_FRAMERATE * PEAK_MIN_DUR_S
    inter_transient_distance = ENDOSCOPE_FRAMERATE * INTER_SPIKE_INTERVAL

    transient_indexes = dict()

    for cell_name in tqdm(cell_names, desc="Find Transient Indexes: "):
        cell_data = smoothed_data[cell_name]
        trial_indexes = dict()
        for trial in cell_data.keys():
            trace_data = cell_data[trial]
            peaks = signal.find_peaks(trace_data, height=height, width=peak_width,
                                      distance=inter_transient_distance)
            peaks = peaks[0]  # Return only the indexes (x locations) of the peaks
            trial_indexes[trial] = peaks
        transient_indexes[cell_name] = trial_indexes

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