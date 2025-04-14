import os
from pathlib import Path


class ProjectFolder:
    def __init__(self, project_type, project_dir=None, root_dir=None, combined=False, select_dir=False, existing_app=None,
                 suppress_filenotfound=False):
        self.project_type = project_type
        self.combined = combined
        self.path = None
        self.search_root_dir = None
        self.suppress_filenotfound = suppress_filenotfound
        self.subdirs = {}

        self.raw_data_dir: RawDataDir = None
        self.inscopix_dir: InscopixDir = None
        self.analysis_dir: AnalysisDir = None

        self._set_root_dir(root_dir)
        self._set_project_dir(project_dir, select_dir, existing_app)  # Allow the user to select/supply the folder
        self._create_subfolders()  # Create or acquire folders

    def get_data(self):
        if self.raw_data_dir:
            self.raw_data_dir._get_files()
        if self.inscopix_dir and not self.combined:
            self.inscopix_dir._get_files()
        if self.analysis_dir:
            self.analysis_dir._get_files()

    def _set_root_dir(self, root_dir):
        cwd = Path(os.getcwd())
        if root_dir is None or root_dir == '.':
            self.search_root_dir = cwd
        else:
            user_root_dir = Path(root_dir)

            if not user_root_dir.exists():
                print(
                    f"User-supplied search root path \'{str(user_root_dir)}\' does not exist! Setting root path to CWD {str(cwd)}!")
                self.search_root_dir = cwd
            else:
                self.search_root_dir = user_root_dir

    def _set_project_dir(self, project_dir, select_dir, app):
        if project_dir is None:
            if select_dir:
                # For backwards compatability with manual curation
                selected_dir = self._folder_selection(existing_app=app)
                if len(selected_dir) > 0:
                    self.path = Path(selected_dir[0])
                else:
                    raise FileNotFoundError('No project folder selected!')
            else:
                self.path = self.search_root_dir
        elif project_dir is not None:
            if project_dir == '.':
                self.path = self.search_root_dir
            else:
                user_project_dir = Path(project_dir)
                if not user_project_dir.exists():
                    raise FileNotFoundError(f'User-supplied project folder \'{str(user_project_dir)}\' does not exist')
                else:
                    self.path = user_project_dir

    def _create_subfolders(self):
        if not self.combined:
            self.inscopix_dir = InscopixDir(self)

        self.raw_data_dir = RawDataDir(self)
        self.analysis_dir = AnalysisDir(self)

    def _folder_selection(self, existing_app) -> list:
        from PySide6.QtWidgets import QFileDialog, QApplication
        if existing_app is None:
            # Project Folder can be launched standalone, or as part of a package with an existing QApplication
            # Safest to check if an app exists before we create a new one
            app = QApplication.instance()
            if not app:
                app = QApplication([])
        else:
            app = existing_app

        file_names = []

        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Select Project Directory:")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setDirectory(str(self.search_root_dir))

        if file_dialog.exec():
            file_names = file_dialog.selectedFiles()

        if not existing_app:
            # Only close the app if we spawned it
            app.exit()

        return file_names

    #  Dunder Methods  #
    def __getattr__(self, item):
        if item in self.subdirs.keys():
            return self.subdirs[item]
        else:
            raise AttributeError(item)

    def __str__(self):
        return f'Project folder: {str(self.path)}'

    def __repr__(self):
        description = f'{type(self).__name__}(root_directory={self.search_root_dir}, project_directory={self.path}'
        return description


class Dir:
    def __init__(self, parent_folder, name, get_subdirs=False):
        self.parent = parent_folder
        self.path_stem = parent_folder.path
        self.name = name
        self.path = self.path_stem.joinpath(name)

        self.subdirs = {}

        self._new_dir = False

        self._create()
        self.parent.subdirs[self.name] = self
        if get_subdirs:
            self._get_subdirs()


    def joinpath(self, addition):
        return self.path.joinpath(addition)


    def _get_subdirs(self):
        for _dir in self.path.iterdir():
            if _dir.is_dir():
                self.subdirs[_dir.name] = Dir(self, _dir.name)

    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)
            self.new_dir = True

    def _check_file_not_found(self, file_list, filename: str):
        if not len(file_list) > 0:
            if not self.parent.suppress_filenotfound:
                print(f"{{{filename}}} not found in {self.path}")
            return False
        return True

    def _get_files(self):
        if not self._new_dir:
            return self.path.glob('*')

    def subdir(self, name):
        if name in self.subdirs.keys():
            return self.subdirs[name]
        else:
            _subdir = Dir(self, name)
            self.subdirs[name] = _subdir
            return _subdir

    def __getattr__(self, item):
        if item in self.subdirs.keys():
            return self.subdirs[item]
        else:
            raise AttributeError(item)

    def __str__(self):
        return f'{self.path}'

    def __repr__(self):
        return f'{type(self).__name__}({str(self)})\nParent(s): {self.parent.__repr__()}'


class FigureDir(Dir):
    def __init__(self, parent_folder, name='Figures'):
        super().__init__(parent_folder, name)

        # This is a sub folder in a sub folder
        if not self.parent.parent.combined:
            self.auroc_dir = Dir(self, 'auroc', get_subdirs=True)
            self.traces_dir = Dir(self, 'traces', get_subdirs=True)
            # self.scatter_dir = Dir(self, 'scatter')
        else:
            self.svm_dir = Dir(self, 'svm')
            self.cm_dir = Dir(self, 'cm')


class RawDataDir(Dir):
    def __init__(self, parent_folder: ProjectFolder, name='Raw_Data'):
        super().__init__(parent_folder, name)

        # Raw inscopix files
        self.session_json_path = None
        self.raw_GPIO_path = None
        self.raw_recordings = []

        # Raw experiment files
        self.exp_h5_path = None
        self.odorlist_path = None

        # Raw DLC Files
        self.labeled_video_path = None
        self.points_h5_path = None

        if self.parent.combined:
            self.combined_data_path = None
            self.combined_sig_table_path = None

        if not self._new_dir:
            self._get_files()

    def _get_files(self) -> None:
        if not self.parent.combined:
            if self.parent.project_type == 'MAN':
                return
            elif self.parent.project_type == 'ODOR':
                odor_list = list(self.path.glob('*.xlsx'))
                odor_list = [path for path in odor_list if 'Blocks' not in path.name]
                if self._check_file_not_found(odor_list, 'Odor List'):
                    self.odorlist_path = odor_list[0]
            elif self.parent.project_type == 'EPM':
                points_h5_file = list(self.path.glob('*DLC*.h5'))
                labeled_video = list(self.path.glob('*DLC*labeled*'))  # Usually mp4 files, but this is more flexible

                if self._check_file_not_found(points_h5_file, 'Points H5 File'):
                    self.points_h5_path = points_h5_file[0]
                if self._check_file_not_found(labeled_video, 'Labeled Video'):
                    self.labeled_video_path = labeled_video[0]
            elif self.parent.project_type == 'HFvFM':
                pass  # There's nothing here yet
            elif self.parent.project_type == 'ISX':
                raw_GPIO = list(self.path.glob('*.gpio'))
                raw_recordings = list(self.path.glob('*.isxd'))
                exp_h5_file = list(self.path.glob('*mouse*.h5'))

                if self._check_file_not_found(raw_GPIO, 'Raw GPIO'):
                    self.raw_GPIO_path = raw_GPIO[0]
                if self._check_file_not_found(raw_recordings, 'Raw Recordings'):
                    self.raw_recordings = raw_recordings  # If there are multiple recordings, we want them all
                if self._check_file_not_found(exp_h5_file, 'Experiment H5 File'):
                    self.exp_h5_path = exp_h5_file[0]
            else:
                raise ValueError((f'{{{self.parent.project_type}}} is not a valid project type. '
                                  f'Please select from the following list: [\'ODOR\', \'EPM\', \'HFvFM\']'))

            # General Files for all non-combined project types
            json_file = list(self.path.glob('*session*.json'))
            if self._check_file_not_found(json_file, 'session.json'):
                self.session_json_path = json_file[0]
        else:
            combined_data_path = list(self.path.glob('*combined*.pickle'))
            sig_table_path = list(self.path.glob('*combined*.xlsx'))
            if self._check_file_not_found(combined_data_path, 'Combined Data'):
                if len(combined_data_path) > 0:
                    self.combined_data_path = combined_data_path
                else:
                    self.combined_data_path = combined_data_path[0]
            if self._check_file_not_found(sig_table_path, 'Significance Table'):
                if len(combined_data_path) > 0:
                    self.combined_sig_table_path = sig_table_path
                else:
                    self.combined_sig_table_path = sig_table_path[0]


class InscopixDir(Dir):
    def __init__(self, parent_folder: ProjectFolder, name='Inscopix'):
        super().__init__(parent_folder, name)
        # Processed Inscopix Files
        self.cell_images_dir = Dir(self, 'cell_images')
        self.interim_file_dir = Dir(self, 'interim_files')
        self.cell_trace_path = None
        self.GPIO_path = None
        self.max_projection_path = None
        self.contours_path = None
        self.props_path = None

        if not self._new_dir:
            self._get_files()

    def _get_files(self):

        if not self.parent.project_type == 'ISX' and not self.parent.combined:  # These files dont exist if in the ISX notebook
            cell_trace_file = list(self.path.glob('*TRACES*.csv'))
            GPIO_file = list(self.path.glob('*GPIO*.csv'))
            max_projection_file = list(self.path.glob('*HD*MAX*.tiff'))
            cell_contours = list(self.path.glob('*CONTOURS*.json'))
            cell_props = list(self.path.glob('*props*.csv'))

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
    def __init__(self, parent_folder: ProjectFolder, name='Analysis'):
        super().__init__(parent_folder, name)

        #  Main Directories
        self.figures_dir = FigureDir(self)
        self.output_dir = Dir(self, 'Output')

        if not self.parent.combined:
            self.preprocess_dir = Dir(self, 'Preprocessed')
            #  Output Subdirectories
            self.combined_dir = Dir(self.output_dir, 'combined')

        if not self._new_dir:
            self._get_files()

    def _get_files(self):
        pass
