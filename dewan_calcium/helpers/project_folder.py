import os
from PySide6.QtWidgets import QFileDialog
from pathlib import Path


class ProjectFolder:
    def __init__(self, project_dir=None, root_dir=None, select_dir=False):
        self.path = None
        self.search_root_dir = None

        self.raw_data_dir: RawDataDir = None
        self.inscopix_dir: InscopixDir = None
        self.analysis_dir: AnalysisDir = None

        self._set_project_dir(project_dir, select_dir)  # Allow the user to select/supply the folder
        self._set_root_dir(root_dir)
        self._create_subfolders()  # Create or aquire folders


        #  Empty file/folder paths
        #self.setup_folder()
        #self.get_project_files()
    def get_data(self):
        self.raw_data_dir._get_files()
        self.inscopix_dir._get_files()
        self.inscopix_dir._get_files()


    def _set_root_dir(self, root_dir):
        cwd = Path(os.getcwd())
        if root_dir is None or root_dir == '.':
            self.search_root_dir = cwd
        else:
            user_root_dir = Path(root_dir)

            if not user_root_dir.exists():
                print(f"User-supplied search root path \'{str(user_root_dir)}\' does not exist! Setting root path to CWD {str(cwd)}!")
                self.search_root_dir = cwd
            else:
                self.search_root_dir = user_root_dir

    def _set_project_dir(self, project_dir, select_dir):
        if project_dir is None and select_dir == True:
            # For backwards compatability with manual curation
            selected_dir = self.select_project_folder()
            if len(selected_dir) > 0:
                self.path = Path(returned_folder[0])
            else:
                raise FileNotFoundError(f'No project folder selected!')
        elif project_dir is not None:
            user_project_dir = Path(project_dir)
            if not user_project_dir.exists():
                raise FileNotFoundError(f'User-supplied project folder \'{str(user_project_dir)}\' does not exist')
            else:
                self.path = user_project_dir
        elif project_dir is None or project_dir == '.':
            self.path = self.search_root_dir


    def _create_subfolders(self):
        self.raw_data_dir = RawDataDir(self)
        self.inscopix_dir = InscopixDir(self)
        self.analysis_dir = AnalysisDir(self)


    def old_get_project_files(self):
        max_projection_path = list(self.inscopix_path.glob('*HD*MAX_PROJ*.tiff'))
        cell_trace_data_path = list(self.inscopix_path.glob('*TRACES*.csv'))
        cell_props_path = list(self.inscopix_path.glob('*props*.csv'))
        cell_contours_path = list(self.inscopix_path.glob('*CONTOURS*.json'))

        if len(max_projection_path) == 0:
            raise FileNotFoundError(f'Max Projection image not found!')
        elif len(cell_trace_data_path) == 0:
            raise FileNotFoundError(f'Cell Trace data not found!')
        elif len(cell_props_path) == 0:
            raise FileNotFoundError(f'Cell Props data not found!')
        elif len(cell_contours_path) == 0:
            raise FileNotFoundError(f'Cell Contour data not found!')
        else:
            self.max_projection_path = max_projection_path[0]
            self.cell_trace_data_path = cell_trace_data_path[0]
            self.cell_props_path = cell_props_path[0]
            self.cell_contours_path = cell_contours_path[0]

    def old_select_project_folder(self) -> list[str]:
        file_names = []

        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Select Project Directory:")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setDirectory(self.search_root_dir)

        if file_dialog.exec():
            file_names = file_dialog.selectedFiles()
        return file_names

    #  Dunder Methods  #
    def __str__(self):
        return f'Project folder: {str(self.path)}'

    def __repr__(self):
        description = f'{type(self).__name__}(root_directory={self.search_root_dir}, \n \
                project_directory={self.project_folder}, \n \
                inscopix_directory={self.inscopix_path}, \n \
                max_projection_path={self.max_projection_path}, \n \
                cell_trace_data_path={self.cell_trace_data_path}, \n \
                cell_props_path={self.cell_props_path}, \n \
                cell_contours_path={self.cell_contours_path})'
        return description


class Dir:
    def __init__(self, project_folder, name):
        self.parent = project_folder
        self.path_stem = project_folder.path
        self.name = name
        self.path = self.path_stem.joinpath(name)
        self.new_dir = False

    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)
            self.new_dir = True

    def _check_file_not_found(self, file_list, filename: str):
        if not len(file_list) > 0:
            print(f"{{{filename}}} not found in {self.path}")
            return False
        return True

    def __str__(self):
        return f'{self.parent.__str__()}\n{self.name} Directory: {self.path}'



class RawDataDir(Dir):
    def __init__(self, project_folder: ProjectFolder, name='Raw_Data'):
        super().__init__(project_folder, name)

        # Raw inscopix files
        self.session_json_path = None
        self.raw_GPIO_path = None
        self.raw_recordings = []

        # Raw experiment files
        self.exp_h5_path = None
        self.odorlist_path = None

        self._create()

        if not self.new_dir:
            self._get_files()


    def _get_files(self):
        json_file = list(self.path.glob('*session*.json'))
        raw_GPIO = list(self.path.glob('*GPIO*.csv'))
        raw_recordings = list(self.path.glob('*.isxd'))
        h5_file = list(self.path.glob('*.h5'))
        odor_list = list(self.path.glob('*.xlsx'))

        if self._check_file_not_found(json_file, 'session.json'):
            self.session_json_path = json_file[0]
        if self._check_file_not_found(raw_GPIO, 'Raw GPIO'):
            self.raw_GPIO_path = raw_GPIO[0]
        if self._check_file_not_found(raw_recordings, 'Raw Recordings'):
            self.raw_recordings = raw_recordings  # If there are multiple recordings, we want them all
        if self._check_file_not_found(h5_file, 'H5 File'):
            self.exp_h5_path = h5_file[0]
        if self._check_file_not_found(odor_list, 'Odor List'):
            self.odorlist_path = odor_list[0]


class InscopixDir(Dir):
    def __init__(self, project_folder: ProjectFolder, name='Inscopix'):
        super().__init__(project_folder, name)
        # Processed Inscopix Files
        self.cell_images_dir = self.path.joinpath('Cell_Images')
        self.interim_file_dir = self.path.joinpath('Interim_Files')
        self.cell_trace_path = None
        self.GPIO_path = None
        self.max_projection_path = None
        self.contours_path = None
        self.props_path = None

        self._create()
        self._create_subdirs()

        if not self.new_dir:
            self._get_files()

    def _create_subdirs(self):
        if not self.cell_images_dir.exists():
            self.cell_images_dir.mkdir()
        if not self.interim_file_dir.exists():
            self.interim_file_dir.mkdir()

    def _get_files(self):
        cell_trace_file = list(self.path.glob('*TRACES*.csv'))
        GPIO_file = list(self.path.glob('*GPIO*.csv'))
        max_projection_file = list(self.path.glob('*HD*MAX*.tiff'))
        cell_contours = list(self.path.glob('*CONTOURS*.json'))
        cell_props = list(self.path.glob('*PROPS*.csv'))

        if self._check_file_not_found(cell_trace_file, 'Cell Traces'):
            self.cell_trace_path = cell_trace_file[0]
        if self._check_file_not_found(GPIO_file, 'GPIO File'):
            self.GPIO_path = GPIO_file[0]
        if self._check_file_not_found(max_projection_file, 'HD Max Projection'):
            self.max_projection_path = max_projection_file[0]
        if self._check_file_not_found(cell_contours, 'Cell Contours'):
            self.contours_path = cell_contours[0]
        if self._check_file_not_found(cell_props, 'Cell Props'):
            self.props_path = cell_props[0]


class AnalysisDir(Dir):
    def __init__(self, project_folder: ProjectFolder, name='Analysis'):
        super().__init__(project_folder, name)

        self.figures_dir = self.path.joinpath('Figures')
        self.preprocess_dir = self.path.joinpath('Preprocessed')
        self.output_dir = self.path.joinpath('Output')

        self._create()
        self._create_subdirs()

        if not self.new_dir:
            self._get_files()

    def _create_subdirs(self):
        if not self.figures_dir.exists():
            self.figures_dir.mkdir()
        if not self.preprocess_dir.exists():
            self.preprocess_dir.mkdir()
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _get_files(self):
        pass
