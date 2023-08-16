from PySide6.QtCore import (QRect, QSize, Qt)
from PySide6.QtGui import (QFont, QPixmap)
from PySide6.QtWidgets import (QDialog, QFrame, QListWidgetItem, QGridLayout,
                               QGroupBox, QHBoxLayout, QListWidget, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout,
                               QWidget, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class CellTrace(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=1.1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(CellTrace, self).__init__(fig)


class Error(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.font = None
        self.h_layout = None
        self.button = None
        self.message = None
        self.pixmap = None
        self.bruh_image = None
        self.v_layout = None
        self.error_gui = None
        self.gui = parent
        self.setup()

    @staticmethod
    def def_font(self):
        font = QFont()
        font.setBold(True)
        font.setFamily('Arial')
        font.setPointSize(20)

        return font

    def setup(self):
        self.error_gui = QWidget()
        self.error_gui.setWindowTitle('You sure about that?')
        self.v_layout = QVBoxLayout(self.error_gui)
        self.error_gui.setLayout(self.v_layout)
        self.bruh_image = QLabel()
        self.pixmap = QPixmap('.\\Python\\Resources\\bruh.png')
        self.bruh_image.setPixmap(self.pixmap)
        self.v_layout.addWidget(self.bruh_image)
        self.h_layout = QHBoxLayout(self.error_gui)
        self.message = QLabel()
        self.font = self.def_font(self)
        self.message.setText("You selected no cells, are you sure about that?")
        self.message.setLayout(QHBoxLayout(self.error_gui))
        self.message.setFont(self.font)
        self.message.setAlignment(Qt.AlignCenter)
        self.v_layout.addWidget(self.message)
        self.button = QPushButton()
        self.button.setText("Close")
        self.button.clicked.connect(self.close)
        self.v_layout.addWidget(self.button)

    def closeEvent(self, event):
        self.parent().setEnabled(True)
        self.error_gui.hide()
        event.accept()


class Confirmation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cancel = None
        self.accept = None
        self.button_layout = None
        self.font = None
        self.label = None
        self.gui = None
        self.v_layout = None
        self.cell_list = None
        self.cells_2_keep = None
        self.setup()

    def setup(self):
        self.gui = QWidget()
        self.gui.setWindowTitle("Confirm Cell Selection")
        self.v_layout = QVBoxLayout(self.gui)
        self.gui.setLayout(self.v_layout)

        self.label = QLabel()
        self.font = Error.def_font(self)
        self.label.setFont(self.font)
        self.label.setText("Please confirm the selected cells!")
        self.label.setLayout(QHBoxLayout())
        self.label.setAlignment(Qt.AlignCenter)

        self.v_layout.addWidget(self.label)

        self.cell_list = QListWidget(self.gui)
        self.v_layout.addWidget(self.cell_list)

        self.button_layout = QHBoxLayout(self.gui)

        self.accept = QPushButton()
        self.accept.setText("Accept")
        self.accept.setLayout(self.button_layout)
        self.accept.clicked.connect(self.accept_action)
        # self.button_layout.addWidget(self.accept)

        self.cancel = QPushButton()
        self.cancel.setText("Cancel")
        self.cancel.setLayout(self.button_layout)
        self.cancel.clicked.connect(self.gui.close)
        # self.button_layout.addWidget(self.cancel)

        self.v_layout.addWidget(self.accept)
        self.v_layout.addWidget(self.cancel)

    def closeEvent(self, e):
        self.parent().setEnabled(True)
        e.accept()

    def populate_confirmation_list(self):
        cell_labels = self.parent().cell_labels
        for each in self.cells_2_keep:
            cell_name = cell_labels[each]
            item = QListWidgetItem(str(cell_name))  # Add one so the cell numbers match up
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.cell_list.addItem(item)

    def accept_action(self):
        self.parent().cells_2_keep = self.cells_2_keep
        self.gui.close()
        self.parent().close()
        self.parent().accept()


class ManualCurationUI(QDialog):

    def __init__(self, max_projection_image):

        super().__init__()
        self.max_projection_image = max_projection_image
        self.export_selection_button = None
        self.scroll_area_vertical_layout = None
        self.scroll_area_contents = None
        self.cell_trace_scroll_area = None
        self.cell_traces_grid_layout = None
        self.cell_traces_group = None
        self.horizontal_div = None
        self.max_projection_view = None
        self.max_projection_vertical_layout = None
        self.max_projection_group = None
        self.select_none_button = None
        self.select_all_button = None
        self.cell_list_control_horizontal = None
        self.cell_labels = None
        self.cell_list = None
        self.cell_list_vertical_layout = None
        self.cell_list_group = None
        self.cell_layout_horizontal = None
        self.verticalLayout = None
        self.font1 = None
        self.font = None
        self.size_policy = None
        self.gui = None
        self.cells_2_keep = None
        self.Error = None
        self.confirmation = None
        self.setupUi(self)

    def setupUi(self, manual_curation_window):
        self.Error = Error(self)
        self.confirmation = Confirmation(self)
        self.size_policy = self.def_size_policy(manual_curation_window)
        self.font1 = self.def_font_1(self)
        self.font = self.def_font()

        manual_curation_window.setObjectName(u"manual_curation_window")
        manual_curation_window.resize(900, 500)
        manual_curation_window.setWindowTitle("Manual Curation")

        manual_curation_window.setSizePolicy(self.size_policy)
        manual_curation_window.setMinimumSize(QSize(600, 600))
        manual_curation_window.setFont(self.font)

        self.gui = manual_curation_window

        self.verticalLayout = QVBoxLayout(manual_curation_window)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.cell_layout_horizontal = QHBoxLayout()
        self.cell_layout_horizontal.setObjectName(u"cell_layout_horizontal")
        self.cell_list_group = QGroupBox(manual_curation_window)
        self.cell_list_group.setTitle("Cells")
        self.cell_list_group.setObjectName(u"cell_list_group")
        self.size_policy.setHeightForWidth(self.cell_list_group.sizePolicy().hasHeightForWidth())
        self.cell_list_group.setSizePolicy(self.size_policy)
        self.cell_list_group.setMaximumSize(QSize(250, 16777215))
        self.cell_list_group.setFont(self.font)
        self.cell_list_group.setAlignment(Qt.AlignCenter)
        self.cell_list_vertical_layout = QVBoxLayout(self.cell_list_group)
        self.cell_list_vertical_layout.setObjectName(u"cell_list_vertical_layout")
        self.cell_list = QListWidget(self.cell_list_group)
        self.cell_list.setObjectName(u"cell_list")
        self.cell_list.setMaximumSize(QSize(250, 16777215))

        self.cell_list_vertical_layout.addWidget(self.cell_list)

        self.cell_list_control_horizontal = QHBoxLayout()
        self.cell_list_control_horizontal.setObjectName(u"cell_list_control_horizontal")
        self.select_all_button = QPushButton(self.cell_list_group)
        self.select_all_button.setObjectName(u"select_all_button")
        self.select_all_button.setText(u"Select All")
        self.select_all_button.setFont(self.font1)
        self.select_all_button.clicked.connect(self.select_all)
        self.select_all_button.setMinimumSize(100, 20)

        self.cell_list_control_horizontal.addWidget(self.select_all_button)

        self.select_none_button = QPushButton(self.cell_list_group)
        self.select_none_button.setObjectName(u"select_none_button")
        self.select_none_button.setText(u"Select None")
        self.select_none_button.setFont(self.font1)
        self.select_none_button.clicked.connect(self.deselect_all)
        self.select_none_button.setMinimumSize(100, 20)
        self.cell_list_control_horizontal.addWidget(self.select_none_button)
        self.cell_list_vertical_layout.addLayout(self.cell_list_control_horizontal)

        self.export_selection_button = QPushButton(self.cell_list_group)
        self.export_selection_button.setObjectName(u'export_selection_button')
        self.export_selection_button.setText(u'Export Selected Cells')
        self.export_selection_button.setFont(self.font1)
        self.export_selection_button.clicked.connect(lambda: self.get_checked_items())
        self.export_selection_button.setMinimumSize(150, 20)
        self.cell_list_vertical_layout.addWidget(self.export_selection_button)

        self.cell_layout_horizontal.addWidget(self.cell_list_group)

        self.max_projection_group = QGroupBox(manual_curation_window)
        self.max_projection_group.setObjectName(u"max_projection_group")
        self.max_projection_group.setTitle(u"Maximum Projection")
        self.max_projection_group.setAlignment(Qt.AlignCenter)
        self.max_projection_vertical_layout = QVBoxLayout(self.max_projection_group)
        self.max_projection_vertical_layout.setObjectName(u"max_projection_vertical_layout")
        # self.max_projection_view = QGraphicsView(self.max_projection_group)

        self.max_projection_view = QLabel(self)

        pixmap = QPixmap(self.max_projection_image)
        self.max_projection_view.setPixmap(pixmap.scaled(pixmap.size(), Qt.KeepAspectRatio))

        self.max_projection_vertical_layout.addWidget(self.max_projection_view)

        self.cell_layout_horizontal.addWidget(self.max_projection_group)

        self.verticalLayout.addLayout(self.cell_layout_horizontal)

        self.horizontal_div = QFrame(manual_curation_window)
        self.horizontal_div.setObjectName(u"horizontal_div")

        self.verticalLayout.addWidget(self.horizontal_div)

        self.cell_traces_group = QGroupBox(manual_curation_window)
        self.cell_traces_group.setObjectName(u"cell_traces_group")
        self.cell_traces_group.setTitle(u"Cell Traces")
        self.cell_traces_group.setMinimumSize(QSize(0, 110))
        self.cell_traces_group.setAlignment(Qt.AlignCenter)

        self.cell_traces_grid_layout = QGridLayout(self.cell_traces_group)
        self.cell_traces_grid_layout.setObjectName(u"cell_traces_grid_layout")

        self.cell_trace_scroll_area = QScrollArea(self.cell_traces_group)
        self.cell_trace_scroll_area.setObjectName(u"cell_trace_scroll_area")
        self.cell_trace_scroll_area.setMinimumSize(QSize(0, 110))
        self.cell_trace_scroll_area.setWidgetResizable(True)
        self.scroll_area_contents = QWidget()
        self.scroll_area_contents.setObjectName(u"scroll_area_contents")
        self.scroll_area_contents.setGeometry(QRect(0, 0, 858, 320))
        self.scroll_area_contents.setAutoFillBackground(True)
        self.scroll_area_vertical_layout = QVBoxLayout(self.scroll_area_contents)
        self.scroll_area_vertical_layout.setObjectName(u"scroll_area_vertical_layout")

        self.cell_trace_scroll_area.setWidget(self.scroll_area_contents)
        self.cell_traces_grid_layout.addWidget(self.cell_trace_scroll_area, 0, 0, 1, 1)

        self.verticalLayout.addWidget(self.cell_traces_group)

    def get_checked_items(self):
        cells_2_keep = []  # Indexes of checked cells
        for list_item in range(self.cell_list.count()):
            if self.cell_list.item(list_item).checkState() == Qt.Checked:
                cells_2_keep.append(list_item)
        if len(cells_2_keep) == 0:
            self.setEnabled(False)
            self.Error.error_gui.show()
        else:
            self.setEnabled(False)
            self.confirmation.cells_2_keep = cells_2_keep
            self.confirmation.populate_confirmation_list()
            self.confirmation.gui.show()

    def select_all(self):
        num_items = self.cell_list.count()
        for i in range(num_items):
            self.cell_list.item(i).setCheckState(Qt.CheckState.Checked)

    def deselect_all(self):
        num_items = self.cell_list.count()
        for i in range(num_items):
            self.cell_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    @staticmethod
    def def_size_policy(manual_curation_window):
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(manual_curation_window.sizePolicy().hasHeightForWidth())
        return sizePolicy

    @staticmethod
    def def_font():
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(12)
        font.setBold(True)
        return font

    @staticmethod
    def def_font_1(self):
        self.font1 = QFont()
        self.font1.setFamilies([u"Arial"])
        self.font1.setPointSize(10)
        self.font1.setBold(True)
        return self.font1

