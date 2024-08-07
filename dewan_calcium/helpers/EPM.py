import shapely
from matplotlib import pyplot as plt
from roipoly import MultiRoi
# Must be installed from GitHub as the PyPi version is obsolete; pip install git+https://github.com/jdoepfert/roipoly.py
import pandas as pd
import numpy as np

from shapely import Polygon, Point, prepare, intersection, symmetric_difference_all
from sklearn.metrics.pairwise import paired_distances


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


def replace_the_void(coordinate_locations, region_indexes, void_index_bins):
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


def get_arm_rois(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    arms = MultiRoi(fig, ax, ['Open', 'Closed'])

    arm_coordinates = []

    for name, ROI in arms.rois.items():
        coordinates = ROI.get_roi_coordinates()
        arm_coordinates.append(coordinates)

    arm_coordinates = np.round(arm_coordinates, 2)

    return arm_coordinates


def display_roi_instructions():
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
    individual_data = zip(names, polygons, widths, lengths)

    original_names = ['open_arm', 'closed_arm', 'center']
    original_shapes = [open_arm_polygon, closed_arm_polygon, center_polygon]
    original_data = zip(original_names, original_shapes)

    individual_polygons = pd.DataFrame(individual_data, columns=['Name', 'Shape', 'Width', 'Length'])
    original_polygons = pd.DataFrame(original_data, columns=['Name', 'Shape'])

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
    polynomial = np.polynomial.Polynomial([2*area, -perimeter, 2])
    #  Coefficients are ordered as C, B, A for a quadratic function
    roots = polynomial.roots()
    width, length = np.sort(roots)

    width, length = round(width, 4), round(length, 4)

    return width, length


def generate_activity_heatmap(coordinates, spike_indexes, cell_names, image_shape: tuple):
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


def generate_position_lines(coordinates, threshold=70):
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


def get_regions(animal_coordinates: pd.Series, individual_regions: dict):
    location_name = []
    location_index = []
    names = individual_regions['Name']
    polygons = individual_regions['Polygon']

    for coordinate in animal_coordinates.values:
        point = Point(coordinate)
        temp_name = []
        temp_index = []
        for index in range(len(individual_regions['Name'])):
            name = names[index]
            polygon = polygons[index]

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


def get_distances(individual_regions: dict, coordinate_pairs: list):
    distances = []

    for pair in coordinate_pairs:
        coordinate, region_index = pair

        current_polygon = individual_regions['Polygon'][region_index]
        open_arm_1_polygon = individual_regions['Polygon'][0] # O1 is always first
        center_polygon = individual_regions['Polygon'][-1]  # Center is always last
        current_point = Point(coordinate)

        if region_index == -1:  # If point is in 'the_void'
            distances.append(-1)
            continue
        elif region_index == 4:  # If point is in the center
            shared_border_center = intersection(open_arm_1_polygon, center_polygon).centroid
            # Since the center is part of the open region, we're just going to measure distance from arm 1
            distance = current_point.distance(shared_border_center)
            distances.append(distance)
        else:  # If point is in a closed arm
            shared_border_center = intersection(current_polygon, center_polygon).centroid
            # Find the shared border between the center and the current arm
            distance = current_point.distance(shared_border_center)
            # Distance between the current coordinate and center of our shared border
            distances.append(distance)

    return distances


def interpolate_DLC_coordinates(coordinates, percentile=95, threshold=None):
    """
    Function that finds points where adjacent coordinates are separated by an euclidian distance greater than some
    threshold. If the distance >= the threshold, the later coordinate is replaced with the former. This effectively
    "freezes" the animal in place in case the Deep Lab Cut tracking isn't perfect.

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
        coordinates[index] = coordinates[index-1]

    coordinates = np.array(coordinates)

    return threshold, coordinates
