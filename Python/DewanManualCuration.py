import sys
from PySide6.QtWidgets import QApplication, QWidget

from Python.Helpers import DewanManualCurationWidget



def manual_curation_gui(cell_list, cell_proc, max_projection):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    widget = DewanManualCurationWidget.ManualCurationUI()
    widget.show()

    app.aboutToQuit.connect(app.deleteLater)
    app.exec()


if __name__ == "__main__":
    manual_curation_gui(None, None, None)