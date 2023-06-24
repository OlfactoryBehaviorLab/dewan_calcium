import sys
from PySide6.QtWidgets import QApplication, QWidget, QListWidgetItem
from PySide6.QtCore import Qt, QSize
from Python.Helpers import DewanManualCurationWidget

import numpy as np
import qdarktheme


def manual_curation_gui(cell_list, cell_data, cell_proc, max_projection):
    qdarktheme.enable_hi_dpi()

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    qdarktheme.setup_theme('dark')



    widget = DewanManualCurationWidget.ManualCurationUI()
    populate_cells(widget, cell_list)
    cell_traces = generate_cell_traces(cell_list, cell_data)
    populate_traces(widget, cell_traces)
    widget.show()

    app.aboutToQuit.connect(app.deleteLater)
    app.exec()


def generate_cell_traces(cell_list, cell_data):
    import matplotlib.pyplot as plt

    traces = []
    for each in range(1, len(cell_list)): #Skip item 1
        data = cell_data[:, each]
        data = np.divide(data, np.max(data))
        x = np.arange(len(data))
        trace = DewanManualCurationWidget.CellTrace()
        axes = trace.axes
       # axes.set(ylabel=f'Cell {cell_list[each]}')
        axes.set_ylabel(f'Cell {cell_list[each]}')
        plt.setp(axes.yaxis.label, rotation=90)

        axes.plot(x, data, color='k')
        axes.tick_params(axis='both', which='both', left=False, bottom=False)
        axes.set_xticks([], labels=[])
        axes.set_yticks([], labels=[])

        traces.append(trace)

    return traces


def populate_traces(gui, cell_trace_list):
    for each in cell_trace_list:
        each.setMinimumSize(QSize(0, each.get_width_height()[1]))
        gui.scroll_area_vertical_layout.addWidget(each)

def populate_cells(gui: DewanManualCurationWidget.ManualCurationUI, cell_list):
    for each in cell_list:
        item = QListWidgetItem(str(each))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        gui.cell_list.addItem(item)


if __name__ == "__main__":
    manual_curation_gui(np.arange(25), None, None, None)