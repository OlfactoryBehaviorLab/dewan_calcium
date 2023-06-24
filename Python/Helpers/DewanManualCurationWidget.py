from PySide6.QtCore import (QRect, QSize, Qt)
from PySide6.QtGui import (QFont, QPixmap)
from PySide6.QtWidgets import (QFrame, QGraphicsView, QGridLayout,
                               QGroupBox, QHBoxLayout, QListWidget, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                               QWidget, QLabel)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


class CellTrace(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(CellTrace, self).__init__(fig)


class ManualCurationUI(QWidget):

    def __init__(self):

        super().__init__()

        self.cell_trace_graphic_2 = None
        self.cell_trace_graphic_3 = None
        self.cell_trace_graphic_1 = None
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
        self.cell_list = None
        self.cell_list_vertical_layout = None
        self.cell_list_group = None
        self.cell_layout_horizontal = None
        self.verticalLayout = None
        self.font1 = None
        self.font = None
        self.size_policy = None

        self.setupUi(self)

    def setupUi(self, manual_curation_window):
        self.size_policy = self.def_size_policy(self, manual_curation_window)
        self.font1 = self.def_font_1(self)
        self.font = self.def_font(self)

        manual_curation_window.setObjectName(u"manual_curation_window")
        manual_curation_window.resize(900, 500)
        manual_curation_window.setWindowTitle("Manual Curation")

        manual_curation_window.setSizePolicy(self.size_policy)
        manual_curation_window.setMinimumSize(QSize(600, 0))
        manual_curation_window.setFont(self.font)
        # manual_curation_window.setFrameShape(QFrame.NoFrame)

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
        self.select_all_button.setMinimumSize(100,20)

        self.cell_list_control_horizontal.addWidget(self.select_all_button)

        self.select_none_button = QPushButton(self.cell_list_group)
        self.select_none_button.setObjectName(u"select_none_button")
        self.select_none_button.setText(u"Select None")
        self.select_none_button.setFont(self.font1)
        self.select_none_button.clicked.connect(self.deselect_all)
        self.select_none_button.setMinimumSize(100,20)
        self.cell_list_control_horizontal.addWidget(self.select_none_button)

        self.cell_list_vertical_layout.addLayout(self.cell_list_control_horizontal)

        self.cell_layout_horizontal.addWidget(self.cell_list_group)

        self.max_projection_group = QGroupBox(manual_curation_window)
        self.max_projection_group.setObjectName(u"max_projection_group")
        self.max_projection_group.setTitle(u"Maximum Projection")
        self.max_projection_group.setAlignment(Qt.AlignCenter)
        self.max_projection_vertical_layout = QVBoxLayout(self.max_projection_group)
        self.max_projection_vertical_layout.setObjectName(u"max_projection_vertical_layout")
        # self.max_projection_view = QGraphicsView(self.max_projection_group)

        self.max_projection_view = QLabel(self)
        pixmap = QPixmap('..\\maxprojection.jpg')
        self.max_projection_view.setPixmap(pixmap.scaled(self.frameSize(), Qt.KeepAspectRatio))
        # self.max_projection_view.setObjectName(u"max_projection_view")
        # self.max_projection_view.setMinimumSize(QSize(pixmap.width()/4, pixmap.height())/4)
        # self.max_projection_view.setScaledContents(True)
        # self.max_projection_view.setFrameShape(QFrame.NoFrame)

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
        self.cell_traces_group.setFlat(False)
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

    def get_all_list_items(self):
        num_items = self.cell_list.count()
        list_items = []

        for each in range(num_items):
            list_items.append(self.cell_list.item(each))

        return list_items

    def select_all(self):
        num_items = self.cell_list.count()
        for i in range(num_items):
            self.cell_list.item(i).setCheckState(Qt.CheckState.Checked)

    def deselect_all(self):
        num_items = self.cell_list.count()
        for i in range(num_items):
            self.cell_list.item(i).setCheckState(Qt.CheckState.Unchecked)


    @staticmethod
    def def_size_policy(self, manual_curation_window):
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(manual_curation_window.sizePolicy().hasHeightForWidth())
        return sizePolicy

    @staticmethod
    def def_font(self):
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
