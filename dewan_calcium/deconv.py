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


def smooth_data(calc_kernel, trace_data) -> tuple[str, np.ndarray]:
    import warnings

    name, trace = trace_data
    trace = trace.values

    g1, g2 = calc_kernel

    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)

    deconv_data = deconvolve(trace, (g1, g2))
    smoothed_trace = deconv_data[0]

    return name, smoothed_trace
