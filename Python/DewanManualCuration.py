import sys
from PySide6.QtWidgets import QApplication, QMessageBox, QListWidgetItem
from PySide6.QtCore import Qt, QSize
from Python.Helpers import DewanManualCurationWidget

import numpy as np
import qdarktheme


def manual_curation_gui(cell_list, cell_data, max_projection):
    qdarktheme.enable_hi_dpi()

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    qdarktheme.setup_theme('dark')

    widget = DewanManualCurationWidget.ManualCurationUI()
    populate_cell_list(widget, cell_list)
    cell_traces = generate_cell_traces(cell_list, cell_data)
    populate_traces(widget, cell_traces)

    widget.show()

    app.aboutToQuit.connect(app.deleteLater)
    app.exec()


def generate_cell_traces(cell_list, cell_data):
    from matplotlib import font_manager
    traces = []
    for each in cell_list:  # Skip item 1
        data = cell_data[:, each]
        y_max = np.max(data)
        y_min = np.min(data)

        data = np.divide(data, y_max)  # Normalize

        x = np.arange(len(data))

        trace = DewanManualCurationWidget.CellTrace()  # Create trace container
        axes = trace.axes

        arial_bold = font_manager.FontProperties(family='arial', weight='bold', size=14)

        axes.set_ylabel(f'Cell {cell_list[each]}', fontproperties=arial_bold, rotation=0, va='center', ha='center')
        axes.plot(x, data, color='k')

        axes.tick_params(axis='both', which='both', left=False, bottom=False)
        axes.set_xticks([], labels=[])

        offset = x[-1] * 0.01
        axes.set_xlim([-offset, (x[-1] + offset)])

        axes.set_yticks([0, 1], labels=[round(y_min), round(y_max)])  # Display max/min dF/F value
        axes.get_yaxis().set_label_coords(-.1, .5)  # Align all the htings
        axes.yaxis.tick_right()  # Move max/min to other side

        traces.append(trace)

    return traces


def populate_traces(gui, cell_trace_list):
    for each in cell_trace_list:
        each.setMinimumSize(QSize(0, each.get_width_height()[1]))
        gui.scroll_area_vertical_layout.addWidget(each)


def populate_cell_list(gui: DewanManualCurationWidget.ManualCurationUI, cell_list):
    for each in cell_list:
        item = QListWidgetItem(str(each))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        gui.cell_list.addItem(item)


def get_checked_items(gui):
    cells_2_keep = []
    for list_item in range(gui.cell_list.count()):
        if gui.cell_list.item(list_item).checkState() == Qt.Checked:
            cells_2_keep.append(list_item)

    if len(cells_2_keep) == 0:
        bruh = DewanManualCurationWidget.Bruh(gui)
    #else:
        #are_you_sure(gui, cells_2_keep)


