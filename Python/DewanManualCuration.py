import sys
from PySide6.QtWidgets import QApplication, QWidget, QListWidgetItem
from PySide6.QtCore import Qt
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
    widget.show()

    app.aboutToQuit.connect(app.deleteLater)
    app.exec()


def generate_cell_traces(cell_list, cell_data):
    for each in cell_list:
        pass

    pass


def populate_cells(gui: DewanManualCurationWidget.ManualCurationUI, cell_list):
    for each in cell_list:
        item = QListWidgetItem(str(each))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        gui.cell_list.addItem(item)
    #gui.cell_list.addItems(cell_list)
    pass


if __name__ == "__main__":
    manual_curation_gui(np.arange(25), None, None, None)