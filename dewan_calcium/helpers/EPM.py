import sys
from typing import Union
import os
import shapely
from matplotlib import pyplot as plt
from roipoly import MultiRoi
import pandas as pd
import numpy as np

from shapely import Polygon, Point, intersection, symmetric_difference_all
from sklearn.metrics.pairwise import paired_distances
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def subsample_pseudotrials(pseudotrials: dict, NUM_PSEUDOTRIALS: int, seed: Union[None, np.random.SeedSequence]):
    rng_generator = np.random.default_rng(seed)

    subsampled_pseudotrials = {}

    for trial, trial_pseudotrials in pseudotrials.items():
        subsample = rng_generator.choice(trial_pseudotrials, NUM_PSEUDOTRIALS, replace=False, shuffle=False)
        subsampled_pseudotrials[trial] = subsample

    del rng_generator
    return subsampled_pseudotrials


def get_pseudotrials(arm_indexes, all_transitions, pseudotrial_len_s, endoscope_framerate) -> tuple:

    trials_per_arm = {
        'open1': [],
        'open2': [],
        'closed1': [],
        'closed2': []
    }

    trial_stats = {
        'good_trials': 0,
        'bad_trials': 0,
        'all_times': [],
        'good_times': [],
        'bad_times': [],
    }

    for i in arm_indexes.index:
        # If we are on the last item, stop
        if i == len(all_transitions) - 1:
            break
        # Index 0 is always skipped b/c this is where the animal was placed to start the experiment
        if i == 0:
            continue

        new_region = all_transitions.iloc[i]
        new_region_start = new_region['Region_Start']  # First frame of new region
        new_region_end = new_region['Region_End']  # Last frame of new region
        new_region_name = new_region['Location']  # Where the animal is at

        # deltaFrames / framerate = time in seconds
        time_in_arm_f = (new_region_end - new_region_start)
        time_in_arm_s = time_in_arm_f / endoscope_framerate
        trial_stats['all_times'].append(time_in_arm_s)

        if time_in_arm_s >= pseudotrial_len_s:
            # If a trial meets our length criteria, save it
            # Each visit will have these properties
            region_visit = {
                'time_s': time_in_arm_s,
                'time_f': time_in_arm_f,
                'start': new_region_start,
                'end': new_region_end,
            }

            trials_per_arm[new_region_name].append(region_visit)
            trial_stats['good_times'].append(time_in_arm_s)
            trial_stats['good_trials'] += 1

        else:
            trial_stats['bad_times'].append(time_in_arm_s)
            trial_stats['bad_trials'] += 1

        trial_stats['PSEUDOTRIAL_LEN_S'] = pseudotrial_len_s

    return trials_per_arm, trial_stats


def new_get_pseudotrials(auroc_data, num_pseudotrials, pseudotrial_len_s, endoscope_framerate):
    import math
    pseudotrials = {}
    pseudotrial_len_f = pseudotrial_len_s * endoscope_framerate

    for cell_name, cell_data in auroc_data.items():
        cell_pseudotrials = {}
        for trial_name, trial_data in cell_data.items():
            remainder = len(trial_data) % pseudotrial_len_f
            _data = trial_data[:-remainder] # Trim data so it equally divides
            _pseudotrials = int(len(_data) / pseudotrial_len_f)
            # Number of pseudotrial_len_f length rows to divide data into
            if _pseudotrials < num_pseudotrials:
                # If any of the pseudotrial types don't have enough pseudotrials, we toss that cell
                print(f'Cell {cell_name} has an insufficient number of {trial_name} pseudotrials. Discarding!')
                break

            cell_pseudotrials[trial_name] = np.reshape(_data, (_pseudotrials, -1))
            # split data into the previously calculated pseudotrials
        else:
            pseudotrials[cell_name] = cell_pseudotrials

    return pseudotrials


def segment_by_arm(trimmed_trace_data):
    uneeded_columns = ['Arms', 'Coordinates', 'Coordinate_Index']
    needed_columns = [col for col in trimmed_trace_data.columns if col not in uneeded_columns]

    arm_location = trimmed_trace_data['Arms']
    unique_arms = arm_location.unique()
    dff_per_cell = {}

    for cell in needed_columns:
        dff_per_arm = {}
        for arm in unique_arms:
            arm_mask = arm_location.values == arm
            arm_dff = trimmed_trace_data.loc[arm_mask, needed_columns]
            dff_per_arm[arm] = arm_dff
        dff_per_cell[cell] = dff_per_arm

    return dff_per_cell


def calc_pseudotrial_stats(pseudotrials: dict, trial_stats_dict: dict) -> dict:

    arm_count = {}

    median_time = round(np.mean(trial_stats_dict['all_times']), 2)
    median_good_time = round(np.mean(trial_stats_dict['good_times']), 2)
    median_bad_time = round(np.mean(trial_stats_dict['bad_times']), 2)

    num_good_trials = trial_stats_dict['good_trials']
    num_bad_trials = trial_stats_dict['bad_trials']
    pseudotrial_len = trial_stats_dict['PSEUDOTRIAL_LEN_S']

    num_open_trials = 0
    num_closed_trials = 0

    for arm in pseudotrials:
        current_pseudotrials = pseudotrials[arm]
        num_pseudotrials = len(current_pseudotrials)
        if 'open' in arm:
            num_open_trials += num_pseudotrials
        elif 'closed' in arm:
            num_closed_trials += num_pseudotrials
        arm_count[arm] = num_pseudotrials

    stats = {
        'median_time': median_time,
        'median_good_time': median_good_time,
        'median_bad_time': median_bad_time,
        'num_good_trials': num_good_trials,
        'num_bad_trials': num_bad_trials,
        'num_open_trials': num_open_trials,
        'num_closed_trials': num_closed_trials,
        'PSEUDOTRIAL_LEN_S': pseudotrial_len,
        'arm_count': arm_count
    }

    return stats


def print_pseudotrial_stats(pseudotrial_stats: dict) -> None:

    median_time = pseudotrial_stats['median_time']
    median_good_time = pseudotrial_stats['median_good_time']
    median_bad_time = pseudotrial_stats['median_bad_time']
    num_good_trials = pseudotrial_stats['num_good_trials']
    num_bad_trials = pseudotrial_stats['num_bad_trials']
    num_open_trials = pseudotrial_stats['num_open_trials']
    num_closed_trials = pseudotrial_stats['num_closed_trials']
    arm_counts = pseudotrial_stats['arm_count']
    PSEUDOTRIAL_LEN_S = pseudotrial_stats['PSEUDOTRIAL_LEN_S']

    print('=== Stats for Pseudotrials ===')
    print(f'Number of trials with time of stay >= {PSEUDOTRIAL_LEN_S}s (Good Trials): {num_good_trials}')
    print(f'Number of trials with time of stay < {PSEUDOTRIAL_LEN_S}s (Bad Trials): {num_bad_trials}')
    print('=== Stats for Visits ===')
    print(f'Mean visit time (all trials): {median_time}s')
    print(f'Mean visit time (good trials): {median_good_time}s')
    print(f'Mean visit time (bad trials): {median_bad_time}s\n')

    print('-' * 30)

    for arm in arm_counts:
        num_pseudotrials = arm_counts[arm]
        print(f'The arm {arm} has {num_pseudotrials} trials!')

    print('\nTotals:')
    print(f'Number of open arm trials: {num_open_trials}')
    print(f'Number of closed arm trials: {num_closed_trials}')


def save_pseudotrial_stats(pseudotrial_stats: dict, project_folder) -> None:
    arms = pseudotrial_stats.pop('arm_count')

    pseudotrial_df = pd.DataFrame(pseudotrial_stats, columns=list(pseudotrial_stats.keys()), index=[0])

    for arm in arms:
        pseudotrial_df.insert(pseudotrial_df.shape[1], arm, arms[arm])

    output_dir = project_folder.analysis_dir.output_dir.path
    excel_file_path = output_dir.joinpath('pseudotrial_stats.xlsx')

    pseudotrial_df.to_excel(excel_file_path)


def find_region_transitions(animal_locations) -> tuple:
    # Find locations where the location transitions/changes e.g. [..., open1, open1, center, ...]
    # The transition from open1 -> center is a  transition

    transition_locations = {
        'Region_Start': [],
        'Region_End': [],
        'Location': [],
    }

    for i, location in enumerate(tqdm(animal_locations)):
        # If we are at the last item, stop
        if i == len(animal_locations) - 1:
            break
        # If we are still in the same location, continue
        if location == animal_locations.iloc[i + 1]:
            continue
        else:  # This will mark the last frame of the region
            if len(transition_locations['Region_Start']) == 0:
                start_time = 0
            else:
                start_time = transition_locations['Region_End'][-1] + 1
                # A region starts 1 frame after the last region ends

            transition_locations['Region_Start'].append(start_time)
            transition_locations['Region_End'].append(i)
            transition_locations['Location'].append(location)

    transition_locations = pd.DataFrame(transition_locations)
    # Find all "centers" as we only care about "center" -> {arm} transitions
    arm_indexes = transition_locations[transition_locations['Location'] != 'center']

    return transition_locations, arm_indexes


def find_index_bins(indices) -> np.array:
    num_indices = len(indices)
    bins = []
    temp_bin = []

    for i in range(num_indices):
        i1 = indices[i]

        if not temp_bin:
            temp_bin.append(i1)

        if i == (num_indices - 1):
            temp_bin.append(i1)
            bins.append(temp_bin)
            continue
        else:
            i2 = indices[i + 1]
            diff = i2 - i1

            if diff == 1:
                continue
            elif diff > 1:
                temp_bin.append(i1)
                bins.append(temp_bin)
                temp_bin = []

    bins = np.array(bins)
    return bins


def replace_the_void(coordinate_locations, region_indexes, void_index_bins) -> tuple:
    for index_bin in void_index_bins:
        bin_start, bin_end = index_bin

        if bin_start == 0:  # If the beginning of the bin is 0, we want the first value after the end of the bin
            replacement_index = bin_end + 1
        else:
            replacement_index = bin_start - 1

        if bin_end + 1 < len(coordinate_locations):
            # If not at the last value, we need to add one to account for slices no being inclusive
            bin_end = bin_end + 1

        replacement_value = coordinate_locations[replacement_index]
        replacement_index = region_indexes[replacement_index]

        if bin_start == bin_end:  # One value to replace
            coordinate_locations[bin_start] = replacement_value
            region_indexes[bin_start] = replacement_index
        else:  # replace the whole range
            coordinate_locations[bin_start:bin_end] = replacement_value
            region_indexes[bin_start:bin_end] = replacement_index

    return coordinate_locations, region_indexes


def get_arm_rois(image) -> np.array:
    fig, ax = plt.subplots()
    ax.imshow(image)
    arms = MultiRoi(fig, ax, ['Open', 'Closed'])

    arm_coordinates = []

    for name, ROI in arms.rois.items():
        coordinates = ROI.get_roi_coordinates()
        arm_coordinates.append(coordinates)

    arm_coordinates = np.round(arm_coordinates, 2)

    return arm_coordinates


def display_roi_instructions() -> None:
    from IPython.display import display, HTML

    # Define the HTML content for the alert box
    alert_html = """
    <div class="alert alert-block alert-info">
        <b>Mark ROIs:</b> The next step is to mark the regions of the open and closed arm! </br>
        Read the following instructions and then run the next cell!</br>
        </br>
        
        To mark the region, click on all the corners of the ROI. After marking the last corner,
         right-click anywhere to confirm!
         
        </br></br>
        <h4> Instructions </h4>
        <ol>
        <li> Click 'New ROI' and outline the open arm</li>
        <li> Click 'New ROI' again and now outline the closed arm </li>
        <li> Click 'Finish' to close the window </li>
        </ol>
        
        </br>
        
        <b>If the displayed image does not show the appropriate ROIs, run the cell again and relabel the ROIS. </b>
    </div>
    """

    # Display the HTML content
    display(HTML(alert_html))


def get_region_polygons(arm_coordinates) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_arm_coordinates = [tuple(each) for each in arm_coordinates[0]]  # Convert the coordinates into tuples
    closed_arm_coordinates = [tuple(each) for each in arm_coordinates[1]]

    open_arm_polygon = Polygon(open_arm_coordinates)
    closed_arm_polygon = Polygon(closed_arm_coordinates)
    center_polygon = intersection(open_arm_polygon, closed_arm_polygon)
    # The center area is the intersection of the two regions

    # Split the entire open (or closed) arm into its two constituent arms ignoring the center
    open_arm_1, closed_arm_1, open_arm_2, closed_arm_2 = (
        symmetric_difference_all([open_arm_polygon, closed_arm_polygon]).geoms)

    # Starting in the top left and going clockwise, O1, C1, O2, C2

    names = ['open1', 'open2', 'closed1', 'closed2', 'center']
    polygons = [open_arm_1, open_arm_2, closed_arm_1, closed_arm_2, center_polygon]

    dimensions = [approximate_rectangle_dimensions(polygon) for polygon in polygons]
    widths, lengths = zip(*dimensions)
    individual_data = zip(polygons, widths, lengths)

    original_names = ['open_arm', 'closed_arm', 'center']
    original_shapes = [open_arm_polygon, closed_arm_polygon, center_polygon]
    original_shape_dimensions = [approximate_rectangle_dimensions(polygon) for polygon in original_shapes]
    original_widths, original_lengths = zip(*original_shape_dimensions)
    original_data = zip(original_names, original_shapes, original_widths, original_lengths)

    individual_polygons = pd.DataFrame(individual_data, index=names, columns=['Shape', 'Width', 'Length'])
    original_polygons = pd.DataFrame(original_data, index=original_names, columns=['Name', 'Shape', 'Width', 'Length'])

    return individual_polygons, original_polygons


def approximate_rectangle_dimensions(polygon: shapely.Polygon) -> tuple[float, float]:
    """
    Small function to estimate the length of an approximately rectangular polygon. Using perimeter and area, the
    length and width of a rectangle can be derived. When the formulae for area and perimeter are combined and solved for
    length, we get a quadratic where 0 = 2*l^2 - P*l + 2A Solving the two roots of this quadratic will give both
    the length and width of the rectangle. The longer root is length, the shorter width.
    Args:
        polygon (shapely.Polygon): Shapely polygon containing the four corners of the rectangle

    Returns:
        width (float): approximate width of the rectangle
        length (float): approximate length of the rectangle

    """
    area, perimeter = polygon.area, polygon.length
    polynomial = np.polynomial.Polynomial([2 * area, -perimeter, 2])
    #  Coefficients are ordered as C, B, A for a quadratic function
    roots = polynomial.roots()
    width, length = np.sort(roots)

    width, length = round(width, 4), round(length, 4)

    return width, length


def generate_activity_heatmap(coordinates, spike_indexes, cell_names, image_shape: tuple) -> tuple:
    image_x, image_y, _ = image_shape

    combined_spike_heatmap = np.zeros((image_x, image_y))
    cell_heatmaps = dict()
    for cell in cell_names:
        cell_indexes = spike_indexes[cell]
        cell_heatmap = np.zeros((image_x, image_y))
        for index in cell_indexes:
            x, y = coordinates[index]
            combined_spike_heatmap[y][x] += 1
            cell_heatmap[y][x] += 1

        cell_heatmaps[cell] = cell_heatmap

    return combined_spike_heatmap, cell_heatmaps


def generate_position_lines(coordinates, threshold=70) -> list:
    from shapely import LineString
    line_coordinates = []
    for i in range(len(coordinates) - 1):
        current_coord = coordinates[i]
        next_coord = coordinates[i + 1]
        line_coords = [current_coord, next_coord]
        line = LineString(line_coords)
        if line.length > threshold:
            # Skip any very long lines; they're probably artifact, or our mice can teleport...
            continue

        line_coordinates.append(line_coords)

    return line_coordinates


def get_regions(animal_coordinates: pd.Series, individual_regions: pd.DataFrame) -> tuple[list, list]:
    location_name = []
    location_index = []
    names = individual_regions.index.values
    polygons = individual_regions['Shape']

    for coordinate in animal_coordinates.values:
        point = Point(coordinate)
        temp_name = []
        temp_index = []
        for index in range(len(names)):
            name = names[index]
            polygon = polygons.iloc[index]

            if point.within(polygon):
                temp_name = name
                temp_index = index
                break

        if not temp_name:
            temp_name = 'The_Void'
            temp_index = -1

        location_name.append(temp_name)
        location_index.append(temp_index)

    location_name = np.array(location_name)
    location_index = np.array(location_index)
    return location_name, location_index


def get_distances(individual_regions: pd.DataFrame, coordinate_pairs: list) -> np.array:
    distances = []

    for pair in coordinate_pairs:
        coordinate, region_index = pair

        current_polygon = individual_regions.iloc[region_index]['Shape']
        open_arm_1_polygon = individual_regions.loc['open1']['Shape']  # O1 is always first
        center_polygon = individual_regions.loc['center']['Shape']  # Center is always last
        current_point = Point(coordinate)

        if region_index == -1:  # If point is in 'the_void'
            distances.append(-1)
            continue
        elif region_index == 4:  # If point is in the center
            shared_border_center = intersection(open_arm_1_polygon, center_polygon).centroid
            # Since the center is part of the open region, we're just going to measure distance from arm 1
            distance = current_point.distance(shared_border_center)
            distances.append(distance)
        else:  # If point is in any other arm
            shared_border_center = intersection(current_polygon, center_polygon).centroid
            # Find the shared border between the center and the current arm
            distance = current_point.distance(shared_border_center)
            # Distance between the current coordinate and center of our shared border
            distances.append(distance)

    return np.array(distances)


def normalize_distance(individual_regions, coordinate_locations, distances, is_percent=False) -> np.ndarray:
    """
    This function takes the animals position as calculated by get_distances and normalizes each distance to the length
    of the currently occupied arm.

    Args:
        individual_regions (pd.DataFrame): Dataframe containing information about the five arms of the EPM
        coordinate_locations (np.ndarray): Ndarray containing the arm each position/distance is located in
        distances (np.ndarray): Ndarray containing the position along the arm for each frame
        is_percent (bool) (default: False): Optional argument to convert output into a percentage
    Returns: new_distance (np.ndarray): Ndarray containing positions that have been normalized to the length of their
    respective arms
    """

    lengths = individual_regions.loc[coordinate_locations]['Length'].values
    new_distances = np.divide(distances, lengths)
    new_distances = np.round(new_distances, 4)

    if is_percent:
        new_distances = np.multiply(new_distances, 100)

    return new_distances


def interpolate_DLC_coordinates(coordinates, percentile=95, threshold=None) -> tuple[float, int, np.array]:
    """
    Function that finds points where adjacent coordinates are separated by an euclidian distance greater than some
    threshold. If the distance >= the threshold, the later coordinate is replaced with the former. This effectively
    "freezes" the animal in place in case the DeepLabCut tracking isn't perfect.

    Args:
        coordinates (list[list]: [X, Y] coordinate pairs
        percentile (int): Percentile to calculate the threshold of what is a "jump" from the data
        threshold (np.number): Manually set the threshold of what a "jump" is considered

    Returns:
        threshold (np.number): Threshold used to define a "jump"
        coordinates (list[list]): List of new coordinates with jumps replaced with the last coordinate before the jump

    """
    coordinates_1 = coordinates[:-1]
    coordinates_2 = coordinates[1:]

    distances = paired_distances(coordinates_1, coordinates_2)

    if threshold is None:
        threshold = np.percentile(distances, [percentile])

    jump_indexes = np.where(distances >= threshold)[0] + 1  # We offset by one to match the original list

    for index in jump_indexes:
        coordinates[index] = coordinates[index - 1]

    coordinates = np.array(coordinates)

    return threshold, len(jump_indexes), coordinates


def get_true_bin_index(led_bins: np.ndarray, num_total_frames: int) -> list:
    """
    Once we have multiple LED bins, we must find which one represents the LED turning on to mark the start of the
    experiment. We first find the largest bin, as the "true" bin will be larger than any other bin. However,
    it is possible that the LED is turned on a second time to mark the end of the experiment.

        If this is the case, there are two possible outcomes: 1) The "true" bin is larger than the "end" bin. If the
        largest bin occurs in the first half of the video, it is the "true" bin and returned. 2) The "end" bin is
        larger than the "true" bin. If the largest bin occurs in the second half of the video, it is the "end" bin.
        The "true" bin must be the second-largest bin which is returned.

        Note: To determine if the bin is in the first or second half, the start of the bin is compared to the
        midpoint (1/2 the number of frames in the video).

    Args: led_bins ([n x 2] ndarray): A ndarray containing n [1 x 2] rows. Each row contains a start index followed
    by an end index. num_total_frames (int): Number of frames in the video

    Returns:
        true_bin ([1 x 2] list): A [1 x 2] list containing the start and end index of the "true" LED bin

    """
    led_bin_sizes = np.subtract(led_bins[:, 1], led_bins[:, 0])  # Size of each bin
    sorted_bins = np.argsort(led_bin_sizes)  # Sort size of bins from smallest -> largest

    true_bin = sorted_bins[-1]

    if led_bins[true_bin][0] > (num_total_frames / 2):
        true_bin = sorted_bins[-2]

    return list(led_bins[true_bin])
