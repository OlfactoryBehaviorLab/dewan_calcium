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
        self._create_subfolders()  # Create or acquire folders


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
                self.path = Path(selected_dir[0])
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

    #  Dunder Methods  #
    def __str__(self):
        return f'Project folder: {str(self.path)}'

    def __repr__(self):
        description = f'{type(self).__name__}(root_directory={self.search_root_dir}, \n \
                project_directory={self.path} \n '
        return description


class Dir:
    def __init__(self, parent_folder, name):
        self.parent = parent_folder
        self.path_stem = parent_folder.path
        self.name = name
        self.path = self.path_stem.joinpath(name)

        self._new_dir = False

        self._create()

    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)
            self.new_dir = True

    def _check_file_not_found(self, file_list, filename: str):
        if not len(file_list) > 0:
            print(f"{{{filename}}} not found in {self.path}")
            return False
        return True

    def _get_files(self):
        if not self._new_dir:
            return self.path.glob('*')

    def __str__(self):
        return f'Directory: {self.path}'

    def __repr__(self):
        return f'{self.name}: {self.parent.__repr__()}\n{str(self)}'


class FigureDir(Dir):
    def __init__(self, parent_folder, name='Figures'):
        super().__init__(parent_folder, name)

        self.auroc_dir = Dir(self, 'auroc')

        # There shouldn't be anything placed in these folders, so we'll just mark them private
        self._traces_dir = Dir(self, 'traces')
        self._scatter_dir = Dir(self, 'scatter')

        self.ontime_traces_dir = Dir(self._traces_dir, 'ontime_traces')
        self.latent_traces_dir = Dir(self._traces_dir, 'latent_traces')
        self.ontime_trial_scatter_dir = Dir(self._scatter_dir, 'ontime_trial_scatter')
        self.latent_trial_scatter_dir = Dir(self._scatter_dir, 'latent_trial_scatter')


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

        if not self._new_dir:
            self._get_files()

    def _get_files(self):
        json_file = list(self.path.glob('*session*.json'))
        raw_GPIO = list(self.path.glob('*.gpio'))
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
    def __init__(self, parent_folder: ProjectFolder, name='Analysis'):
        super().__init__(parent_folder, name)

        #  Main Directories
        self.figures_dir = FigureDir(self)
        self.preprocess_dir = Dir(self, 'Preprocessed')
        self.output_dir = Dir(self, 'Output')

        #  Figure Subdirectories
        self.auroc_figures_dir = Dir(self.figures_dir, 'auroc')

        #  Output Subdirectories
        self.combined_directory = Dir(self.output_dir, 'combined')

        if not self._new_dir:
            self._get_files()


    def _get_files(self):
        pass
