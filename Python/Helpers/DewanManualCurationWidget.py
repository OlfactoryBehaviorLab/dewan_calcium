from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QSize, Qt)
from PySide6.QtGui import (QFont)
from PySide6.QtWidgets import (QFrame, QGraphicsView, QGridLayout,
                               QGroupBox, QHBoxLayout, QListWidget, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                               QWidget)


class Ui_manual_curation_window(object):

    def __init__(self):
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
        manual_curation_window.setFrameShape(QFrame.NoFrame)

        self.verticalLayout = QVBoxLayout(manual_curation_window)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.cell_layout_horizontal = QHBoxLayout()
        self.cell_layout_horizontal.setObjectName(u"cell_layout_horizontal")
        self.cell_list_group = QGroupBox(manual_curation_window)
        self.cell_list_group.setTitle("Cells")
        self.cell_list_group.setObjectName(u"cell_list_group")
        self.size_policy.setHeightForWidth(self.cell_list_group.sizePolicy().hasHeightForWidth())
        self.cell_list_group.setSizePolicy(self.size_policy)
        self.cell_list_group.setMaximumSize(QSize(160, 16777215))
        self.cell_list_group.setFont(self.font)
        self.cell_list_group.setAlignment(Qt.AlignCenter)
        self.cell_list_vertical_layout = QVBoxLayout(self.cell_list_group)
        self.cell_list_vertical_layout.setObjectName(u"cell_list_vertical_layout")
        self.cell_list = QListWidget(self.cell_list_group)
        self.cell_list.setObjectName(u"cell_list")
        self.cell_list.setMaximumSize(QSize(150, 16777215))

        self.cell_list_vertical_layout.addWidget(self.cell_list)

        self.cell_list_control_horizontal = QHBoxLayout()
        self.cell_list_control_horizontal.setObjectName(u"cell_list_control_horizontal")
        self.select_all_button = QPushButton(self.cell_list_group)
        self.select_all_button.setObjectName(u"select_all_button")
        self.select_all_button.setText(u"Select All")


        self.select_all_button.setFont(self.font1)

        self.cell_list_control_horizontal.addWidget(self.select_all_button)

        self.select_none_button = QPushButton(self.cell_list_group)
        self.select_none_button.setObjectName(u"select_none_button")
        self.select_none_button.setText(u"Select None")
        self.select_none_button.setFont(self.font1)

        self.cell_list_control_horizontal.addWidget(self.select_none_button)

        self.cell_list_vertical_layout.addLayout(self.cell_list_control_horizontal)

        self.cell_layout_horizontal.addWidget(self.cell_list_group)

        self.max_projection_group = QGroupBox(manual_curation_window)
        self.max_projection_group.setObjectName(u"max_projection_group")
        self.max_projection_group.setTitle(u"Maximum Projection")
        self.max_projection_group.setAlignment(Qt.AlignCenter)
        self.max_projection_vertical_layout = QVBoxLayout(self.max_projection_group)
        self.max_projection_vertical_layout.setObjectName(u"max_projection_vertical_layout")
        self.max_projection_view = QGraphicsView(self.max_projection_group)
        self.max_projection_view.setObjectName(u"max_projection_view")
        self.max_projection_view.setMinimumSize(QSize(400, 300))
        self.max_projection_view.setFrameShape(QFrame.NoFrame)

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
        self.cell_trace_graphic_1 = QGraphicsView(self.scroll_area_contents)
        self.cell_trace_graphic_1.setObjectName(u"cell_trace_graphic_1")
        self.cell_trace_graphic_1.setMinimumSize(QSize(0, 100))

        self.scroll_area_vertical_layout.addWidget(self.cell_trace_graphic_1)

        self.cell_trace_graphic_3 = QGraphicsView(self.scroll_area_contents)
        self.cell_trace_graphic_3.setObjectName(u"cell_trace_graphic_3")
        self.cell_trace_graphic_3.setMinimumSize(QSize(0, 100))

        self.scroll_area_vertical_layout.addWidget(self.cell_trace_graphic_3)

        self.cell_trace_graphic_2 = QGraphicsView(self.scroll_area_contents)
        self.cell_trace_graphic_2.setObjectName(u"cell_trace_graphic_2")
        self.cell_trace_graphic_2.setMinimumSize(QSize(0, 100))

        self.scroll_area_vertical_layout.addWidget(self.cell_trace_graphic_2)

        self.cell_trace_scroll_area.setWidget(self.scroll_area_contents)

        self.cell_traces_grid_layout.addWidget(self.cell_trace_scroll_area, 0, 0, 1, 1)

        self.verticalLayout.addWidget(self.cell_traces_group)

        self.retranslateUi(manual_curation_window)

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
