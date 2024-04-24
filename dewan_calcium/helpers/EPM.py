from matplotlib import pyplot as plt
from roipoly import MultiRoi
# Must be installed from GitHub as the PyPi version is obsolete; pip install git+https://github.com/jdoepfert/roipoly.py
import pandas as pd
import numpy as np

from shapely import Polygon, Point, prepare, intersection, symmetric_difference_all


def find_led_start(points: pd.DataFrame) -> list:
    indexes = points.index[points['led_p'] > 0.98].values
    bins = []
    temp_bin = []
    for i in range(len(indexes) - 1):
        i1 = indexes[i]
        i2 = indexes[i + 1]

        diff = i2 - i1
        if not temp_bin:
            temp_bin.append(i1)
        else:
            if i == len(indexes) - 2:  # End is len - 1, and then -1 again for the zero indexes
                temp_bin.append(i2)
                bins.append(temp_bin)
                temp_bin = []
            elif diff == 1:
                continue
            elif diff > 1:
                temp_bin.append(i1)
                bins.append(temp_bin)
                temp_bin = []

    return bins


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


def get_region_polygons(arm_coordinates):

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

    individual_polygons = {
        'Name': ['open1', 'open2', 'closed1', 'closed2', 'center'],
        'Polygon': [open_arm_1, open_arm_2, closed_arm_1, closed_arm_2, center_polygon]
    }
    original_polygons = {
        'Name': ['open_arm', 'closed_arm', 'center'],
        'Polygon': [open_arm_polygon, closed_arm_polygon, center_polygon]
    }

    for polygon in individual_polygons['Polygon']:
        prepare(polygon)

    return individual_polygons, original_polygons


def get_coordinate_region(animal_coordinates: pd.Series, individual_regions: dict):
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

    return location_name, location_index


def get_distances(individual_regions: dict, coordinate_pairs:list):
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
