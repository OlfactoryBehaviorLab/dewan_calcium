import numpy as np
import pandas as pd
from scipy import signal
from oasis.functions import deconvolve  # install using conda install to avoid having to build
from pathlib import Path


def deconvolve_traces(trace_data: pd.DataFrame, framerate: int, peak_args: dict):
    denoised_data = []
    peaks = []

    scaled_data = z_score_data(trace_data)

    try:
        denoised_data = multithread_smoothing(scaled_data, num_workers=8)
    except Exception as e:
        denoised_data.append(np.NaN)

    for cell in scaled_data:

        cell_data = np.hstack(cell)

        try:
            new_trace = smooth_data(cell_data)
            cell_peaks = find_peaks(new_trace, framerate, peak_args)
        except UnboundLocalError:
            new_trace = np.NAN
            cell_peaks = np.NAN

        denoised_data.append(new_trace)
        peaks.append(cell_peaks)

    return denoised_data, peaks


def find_peaks(data: np.ndarray, framerate: int, peak_args: dict) -> np.ndarray:
    width_time = peak_args['width']
    distance_time = peak_args['distance']
    peak_height = peak_args['height']

    peak_width = (framerate * width_time) / 1000
    peak_distance = (framerate * distance_time) / 1000

    peaks = signal.find_peaks(data, height=peak_height, width=peak_width, distance=peak_distance)
    peaks = peaks[0]  # Return only the indexes (x locations) of the peaks
    return peaks



def z_score_data(data: pd.DataFrame) -> list:
    # Function is given a Cells x Trials array
    # Zscores each trial and then returns the array

    from scipy.stats import zscore

    z_scored_data = []

    for cell in data.columns:

        fluorescence_values = data[cell].values

        combined_data = np.hstack(fluorescence_values)
        z_score_combined = zscore(combined_data)

        # z_scored_cell = np.split(z_score_combined, num_trials)

        z_scored_data.append(z_score_combined)

    return z_scored_data


def calc_smoothing_params(framerate, decay_time, rise_time):

    decay_param = np.exp(-1 / (decay_time * framerate))
    rise_param = np.exp(-1 / (rise_time * framerate))

    g1 = round(decay_param + rise_param, 5)
    g2 = round(-decay_time * rise_param, 5)

    return g1, g2


def smooth_data(trace, calc_kernel) -> np.ndarray:
    import warnings

    #g1, g2 = calc_smoothing_params(10, .4, 0.08)
    g1, g2 = calc_kernel

    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)

    deconv_data = deconvolve(trace, (g1, g2))

    smoothed_trace = deconv_data[0]

    return smoothed_trace


def multithread_smoothing(zscored_data: list, framerate: int, peak_args: dict, num_workers: int = 4) -> list[tuple]:
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import ExitStack
    from functools import partial
    from tqdm.notebook import tqdm

    iterator = zscored_data

    rise_time = peak_args['rise'] / 1000
    decay_time = peak_args['decay'] / 1000

    calc_kernel = calc_smoothing_params(framerate, decay_time, rise_time)

    num_traces = len(iterator)

    traces = []

    print("Begin smoothing of trace data...")


    with ExitStack() as stack:
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=num_workers))
        progress_bar = stack.enter_context(tqdm(position=0, total=num_traces))

        smooth_func = partial(smooth_data, calc_kernel=calc_kernel)

        for trace in pool.map(smooth_func, iterator):
            progress_bar.update(1)
            traces.append(trace)

    print("Trace smoothing completed!")

    return traces
