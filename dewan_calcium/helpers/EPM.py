from matplotlib import pyplot as plt
from roipoly import MultiRoi
import pandas as pd
import numpy as np


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
        
        To mark the region, left click on one corner to start the region. </br>
        Left click on two more corners to mark the other two corners of the region. </br>
        To close the region, right click on the original point.
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
    import collections.abc
    collections.Iterable = collections.abc.Iterable  # Fix for python 3.10 depreciation of collections.Iterable
    from pypex import Polygon

    open_polygon = Polygon(arm_coordinates[0])
    closed_polygon = Polygon(arm_coordinates[1])
    center_polygon = open_polygon.intersection(closed_polygon)

    return open_polygon, closed_polygon, center_polygon