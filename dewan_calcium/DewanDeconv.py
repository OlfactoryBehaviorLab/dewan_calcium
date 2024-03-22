import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing
from oasis.functions import deconvolve # install using conda install to avoid having to build
from pathlib import Path


def deconvolve_traces(trace_data: list, framerate: int, peak_args: dict):
    #data = load_transients(trace_file)
    scaled_data = z_score_data(trace_data)

    denoised_data = []
    peaks = []
    for cell in scaled_data:
        new_cell_traces = []
        cell_peaks = []

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
    # Function is given a Cells x Trials array
    # Zscores each trial and then returns the array

    from scipy.stats import zscore

    z_scored_data = []

    for cell in data:
        z_scored_cell = []
        num_trials = len(cell)
        combined_data = np.hstack(cell)
        z_score_combined = zscore(combined_data)

        z_scored_cell = np.split(z_score_combined, num_trials)

        z_scored_data.append(z_scored_cell)

    return z_scored_data


def smooth_data(trace: tuple) -> np.ndarray:
    import warnings
    #trace_name, trace_data = trace # Unpack tuple
    #trace_data = trace_data.values
    #print(f'Smoothing trace: {trace_name}')


    g1 = round(np.exp(-1/(.4*20)) + np.exp(-1/(.08*20)), 5)
    g2 = round(-np.exp(-1/(.4*20)) * np.exp(-1/(.08*20)), 5)
    
    warnings.simplefilter("ignore", category=UserWarning)
    deconv_data = deconvolve(trace, (g1, g2))

    smoothed_trace = deconv_data[0]

    #print(f'Smoothing complete for: {trace_name}!')


    return smoothed_trace


def multithread_smoothing(zscored_data: pd.DataFrame, num_workers: int = 4) -> list[tuple]:
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import ExitStack
    from tqdm.notebook import tqdm

    iterator = zscored_data.items() # List of tuples
    num_traces = zscored_data.shape[1] # Num of cells
    traces = []
    print("Begin smoothing of trace data...")

    with ExitStack() as stack:
        pool = stack.enter_context(ProcessPoolExecutor(max_workers = num_workers))
        progress_bar = stack.enter_context(tqdm(position=0, total=num_traces))

        for trace in pool.map(smooth_data, iterator):
            progress_bar.update(1)
            traces.append(trace)

    print("Trace smoothing completed!")

    return traces
