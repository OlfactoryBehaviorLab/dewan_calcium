import sys
from PySide6.QtWidgets import QDialog, QApplication, QListWidgetItem
from PySide6.QtCore import Qt, QSize, QCoreApplication
from PySide6.QtGui import QPixmap
from Python.Helpers import DewanManualCurationWidget
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import qdarktheme


def manual_curation_gui(cell_list, cell_data, max_projection_image):
    qdarktheme.enable_hi_dpi()

    app = QCoreApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    qdarktheme.setup_theme('dark')

    gui = DewanManualCurationWidget.ManualCurationUI(max_projection_image)
    populate_cell_selection_list(gui, cell_list)
    cell_traces = generate_cell_traces(cell_list, cell_data)  # ignore column one since its just time
    populate_traces(gui, cell_traces)
    gui.cell_labels = cell_list

    app.aboutToQuit.connect(lambda: cleanup(gui))
    #app.aboutToQuit.connect(app.deleteLater())

    gui.show()

    if gui.exec() == QDialog.Accepted:
        return_val = gui.cells_2_keep
        del gui
        del app
        return return_val


def cleanup(gui):
    gui.Error.close()
    gui.confirmation.close()
    gui.close()


def generate_cell_traces(cell_list, cell_data):
    from matplotlib import font_manager
    traces = []
    for each in cell_list:
        data = np.array(cell_data.iloc[:, (each+1)])  # Skip item 1
        y_max = np.max(data)
        y_min = np.min(data)

        # data = np.divide(data, y_max)  # Normalize

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data.reshape(-1, 1))

        x = np.arange(len(data))

        trace = DewanManualCurationWidget.CellTrace()  # Create trace container
        axes = trace.axes

        arial_bold = font_manager.FontProperties(family='arial', weight='bold', size=14)

        axes.set_ylabel(f'Cell {each}', fontproperties=arial_bold, rotation=0, va='center', ha='center')
        axes.plot(x, data, color='k')

        axes.tick_params(axis='both', which='both', left=False, bottom=False)
        axes.set_xticks([], labels=[])

        offset = x[-1] * 0.01
        axes.set_xlim([-offset, (x[-1] + offset)])
        axes.set_ylim([-0.05, 1.05])
        axes.set_yticks([0, 1], labels=[round(y_min), round(y_max)])  # Display max/min dF/F value
        axes.get_yaxis().set_label_coords(-.1, .5)  # Align all the things
        axes.yaxis.tick_right()  # Move max/min to other side

        traces.append(trace)

    return traces


def populate_traces(gui, cell_trace_list):
    for each in cell_trace_list:
        each.setMinimumSize(QSize(0, each.get_width_height()[1]))
        gui.scroll_area_vertical_layout.addWidget(each)


def populate_cell_selection_list(gui: DewanManualCurationWidget.ManualCurationUI, cell_list):
    for each in cell_list:
        item = QListWidgetItem(str(each))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        gui.cell_list.addItem(item)


def generate_max_projection(AllCellProps, CellKeys, CellOutlines, MaxProjectionImage=None, scale_val=3):
    import cv2
    from PIL import Image, ImageDraw, ImageFont, ImageQt

    font = ImageFont.truetype('arial.ttf', 12)

    if MaxProjectionImage is None:
        MaxProjection = '.\\ImagingAnalysis\\RawData\\Max_Projection.tiff'
    else:
        MaxProjection = MaxProjectionImage

    image = cv2.imread(MaxProjection)

    if image is None:
        print("Error, could not load image!")
        return None

    # For some reason PIL won't load the image, so we do a little trickery to make it work
    image = np.array(image) * 2
    # Computer, Enhance Brightness by a factor of two
    image = Image.fromarray(image)
    centroids = np.stack((AllCellProps['CentroidX'].values, AllCellProps['CentroidY'].values), axis=-1)

    drawer = ImageDraw.Draw(image)  # Give the computer a crayon

    for i, each in enumerate(CellKeys):
        points = CellOutlines[each][0]
        points = [tuple(x) for x in points]
        centroid = (centroids[i][0], centroids[i][1])
        drawer.polygon(points, outline='yellow', width=1)
        drawer.text(centroid, str(each[1:]), font=font)

    new_size = (int(image.size[0] * scale_val), int(image.size[1] * scale_val))
    image = image.resize(new_size, Image.LANCZOS)

    q_image = ImageQt.ImageQt(image)
    # Some voodoo to change the image format so Qt likes it

    return q_image



