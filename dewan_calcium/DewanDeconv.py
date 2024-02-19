import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing
from oasis.functions import deconvolve
from pathlib import Path


def deconvolve_traces(trace_file: Path, framerate: int, peak_args: dict) -> tuple[pd.DataFrame, np.ndarray]:
    data = load_transients(trace_file)
    scaled_data = z_score_data(data)

    denoised_data = []
    peaks = []
    for each in data.columns:
        new_trace = scaled_data[each].values
        new_trace = smooth_data(new_trace)
        denoised_data.append(new_trace)
        found_peaks = find_peaks(new_trace, framerate, peak_args)
        peaks.append(found_peaks)

    denoised_data = pd.DataFrame(denoised_data)
    peaks = np.ndarray(peaks)

    return denoised_data, peaks


def find_peaks(data: np.ndarray, framerate: int, peak_args: dict) -> np.ndarray:
    width_scalar = peak_args['width']
    distance_scalar = peak_args['distance']
    peak_height = peak_args['height']

    peak_width = framerate * width_scalar
    peak_distance = framerate * distance_scalar

    peaks = signal.find_peaks(data, height=peak_height, width=peak_width, distance=peak_distance)

    return peaks



def load_transients(path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, header=0, index_col=0)
        data = data[1:]  # Remove first row
    except FileNotFoundError:
        raise f'Error, cannot find the data file: {path}'
    return data


def z_score_data(data: pd.DataFrame) -> pd.DataFrame:
    scaler = preprocessing.StandardScaler()
    zscore_trace = scaler.fit_transform(data)
    return zscore_trace


def smooth_data(trace: np.ndarray) -> np.ndarray:
    deconv_data = deconvolve(trace, g=(None, None), optimize_g=5, penalty=1, max_iter=5)

    smoothed_trace = deconv_data[0]

    return smoothed_trace


def multithread_smoothing():
    # Placeholder
    pass
