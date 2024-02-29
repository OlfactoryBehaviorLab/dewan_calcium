import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing
from oasis.functions import deconvolve # install using conda install to avoid having to build
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
        data = pd.read_csv(path, header=0, index_col=0, dtype=object)
        data = data[1:]  # Remove first row
        
        cols = [column[1:] for column in data.columns] # Remove leading space from each column name
        data.columns = cols

        data = data.astype(np.float32) # Cast all numbers from object -> np.float32

        data.interpolate(inplace=True) # Fill all NaN with a linearly interpolated value

    except FileNotFoundError:
        raise f'Error, cannot find the data file: {path}'
    return data


def z_score_data(data: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import zscore

    keys = data.keys() # Get each column name
    z_scored_data = dict.fromkeys(keys)

    for each in keys: # Loop through columns
        col = data[each]
        z_scored = zscore(col) # Zscore column and replace it
        
        z_scored_data[each] = z_scored
        
    z_scored_data = pd.DataFrame(z_scored_data, index=data.index)

    return z_scored_data


def smooth_data(trace: np.ndarray) -> np.ndarray:
    deconv_data = deconvolve(trace, g=(None, None), optimize_g=5, penalty=1, max_iter=5)

    smoothed_trace = deconv_data[0]

    return smoothed_trace


def multithread_smoothing(zscored_data, max_workers=8):
    from concurrent.futures import ProcessPoolExecutor

    results = []

    with ProcessPoolExecutor(max_workers) as pool:
        results = pool.map(smooth_data, zscored_data.T.values)

    return results