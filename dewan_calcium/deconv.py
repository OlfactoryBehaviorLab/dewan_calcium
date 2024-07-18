import numpy as np
import pandas as pd
from scipy import signal
from oasis.functions import deconvolve  # install using conda install to avoid having to build


def find_peaks(smoothed_data: np.ndarray, framerate: int, peak_args: dict) -> list:
    width_time = peak_args['decay']
    distance_time = peak_args['distance']
    peak_height = peak_args['height']
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

    z_scored_data = []

    for cell in data.columns:

        fluorescence_values = data[cell].values

        combined_data = np.hstack(fluorescence_values)
        z_score_combined = zscore(combined_data)

        z_scored_data.append(z_score_combined)

    return z_scored_data


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
