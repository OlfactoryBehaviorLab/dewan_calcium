import sys
from PySide6.QtWidgets import QDialog, QApplication, QListWidgetItem
from PySide6.QtCore import Qt, QSize, QCoreApplication
from Python.Helpers import DewanManualCurationWidget
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import qdarktheme


def manual_curation_gui(cell_list, cell_data):
    qdarktheme.enable_hi_dpi()

    app = QCoreApplication.instance()
    print(app)
    if not app:
        app = QApplication(sys.argv)

    qdarktheme.setup_theme('dark')

    gui = DewanManualCurationWidget.ManualCurationUI()
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
    for each in cell_list:  # Skip item 1
        data = cell_data[:, each]
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
        item.setCheckState(Qt.CheckState.Unchecked)
        gui.cell_list.addItem(item)



