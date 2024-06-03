import sys
import numpy as np
import qdarktheme

from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QDialog, QApplication, QListWidgetItem, QSizePolicy, QCheckBox
from PySide6.QtCore import Qt, QSize, QCoreApplication
from sklearn.preprocessing import MinMaxScaler

from dewan_calcium.helpers import DewanManualCurationWidget


def manual_curation_gui(all_cell_props, cell_data, max_projection_image):
    qdarktheme.enable_hi_dpi()

    app = QCoreApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    qdarktheme.setup_theme('dark')

    gui = DewanManualCurationWidget.ManualCurationUI(max_projection_image)

    cell_keys = all_cell_props['Name']
    cell_list = [int(cell[1:]) for cell in cell_keys]

    populate_cell_selection_list(gui, cell_list)

    cell_traces = generate_cell_traces(cell_keys, cell_list, cell_data)  # ignore column one since its just time
    populate_traces(gui, cell_traces)
    gui.cell_labels = cell_list

    app.aboutToQuit.connect(lambda: cleanup(gui))
    #app.aboutToQuit.connect(app.deleteLater())

    gui.show()
    gui.activateWindow()  # Bring window to front

    if gui.exec() == QDialog.Accepted:
        return_val = gui.cells_2_keep
        del gui
        del app
        return return_val


def cleanup(gui):
    gui.Error.close()
    gui.confirmation.close()
    gui.close()


def generate_cell_traces(cell_keys, cell_list, cell_data):
    from matplotlib import font_manager
    traces = []

    for i, key in enumerate(cell_keys):
        cell_name = cell_list[i]

        data = cell_data[key].values
        y_max_label = max(data)
        y_min_label = min(data)

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data.reshape(-1, 1)).ravel()

        x = np.arange(len(data))

        trace = DewanManualCurationWidget.CellTrace()  # Create trace container
        axes = trace.axes
        arial_bold = font_manager.FontProperties(family='arial', weight='bold', size=14)

        axes.set_ylabel(f'Cell {cell_name}', fontproperties=arial_bold, rotation=0, va='center', ha='center')
        axes.plot(x, data, color='k')

        axes.tick_params(axis='both', which='both', left=False, bottom=False)
        axes.set_xticks([], labels=[])

        offset = x[-1] * 0.01
        axes.set_xlim([-offset, (x[-1] + offset)])
        axes.set_ylim([-0.05, 1.05])
        axes.set_yticks([0, 1], labels=[round(y_min_label, 4), round(y_max_label, 4)])  # Display max/min dF/F value
        axes.get_yaxis().set_label_coords(-.1, .5)  # Align all the things
        axes.yaxis.tick_right()  # Move max/min to other side

        traces.append(trace)

    return traces


def populate_traces(gui, cell_trace_list: list[DewanManualCurationWidget.CellTrace]):
    for i, each in enumerate(cell_trace_list):
        each.setMinimumSize(QSize(0, each.get_width_height()[1]))
        each.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum))
        each.installEventFilter(each.installEventFilter(gui))
        each.setObjectName(f'graph{i}')
        gui.scroll_area_vertical_layout.addWidget(each)


def populate_cell_selection_list(gui: DewanManualCurationWidget.ManualCurationUI, cell_list):
    for each in cell_list:
        item = QListWidgetItem(str(each))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        gui.cell_view_selector.addItem(item)

        item2 = item.clone()
        gui.cell_list.addItem(item2)


def generate_max_projection(image_path: Path, all_cell_props, cell_outlines, return_raw_image=False,
                            is_downsampled=False, downsample_factor=4, brightness=1.5, contrast=1.5,
                            font_size=20, text_color='magenta', outline_color='gold', outline_width=2):
    import cv2
    from PIL import ImageDraw, ImageFont, ImageQt, ImageEnhance

    font = ImageFont.truetype('arial.ttf', font_size)  # Font size defaults to 12 but can be changed

    max_projection_path = str(image_path)
    image = cv2.imread(max_projection_path)

    if image is None:
        print("Error, could not load image!")
        return None

    # For some reason PIL won't load the image, so we do a little trickery to make it work
    image = np.array(image)
    image = Image.fromarray(image)
    # Computer, Enhance Image
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    centroids = np.stack((all_cell_props['CentroidX'].values, all_cell_props['CentroidY'].values), axis=-1)

    drawer = ImageDraw.Draw(image)  # Give the computer a crayon

    cell_keys = all_cell_props['Name']  # Cell Keys are the cell Names

    for i, each in enumerate(cell_keys):
        if is_downsampled:
            points = cell_outlines[each][0]
            centroid = (centroids[i][0], centroids[i][1])
        else:  # If using full-frame image we need to scale our points up by the downsample factor
            points = np.multiply(cell_outlines[each][0], downsample_factor)
            centroid = np.multiply((centroids[i][0], centroids[i][1]), downsample_factor)

        points = [tuple(x) for x in points]
        drawer.polygon(points, outline=outline_color, width=outline_width)
        drawer.text(centroid, str(int(each[1:])), anchor='mm', fill=text_color, font=font)
        # Drop the C from CXX, convert to an INT to drop any leading zeros, convert back to string for drawing function

    if return_raw_image:
        return image
    else:
        q_image = ImageQt.ImageQt(image)
        # Some voodoo to change the image format so Qt likes it
        return q_image


def save_image(image: Image, folder: list) -> None:
    folder = Path(*folder)
    save_path = folder.joinpath('Max_Projection_contours.tiff')
    image.save(save_path)

