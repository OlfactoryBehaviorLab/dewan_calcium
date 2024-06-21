import os
from PySide6.QtWidgets import QFileDialog
from pathlib import Path


class ProjectFolder:
    def __init__(self, root_dir=None, project_dir=None, select_dir=False):

        self._set_root_dir(root_dir)

        self.project_dir = None
        self.project_folder = None

        self.raw_data_dir = None
        self.inscopix_dir = None
        self.analysis_dir = None
        self._create_subfolders()  # Create folders if they do not exist


        #  Empty file/folder paths

        #self.setup_folder()
        #self.get_project_files()

    def _set_root_dir(self, root_dir):
        cwd = Path(os.getcwd())
        if root_dir is None or root_dir == '.':
            self.root_dir = cwd
        else:
            user_root_dir = Path(root_dir)

            if not user_root_dir.exists():
                print(f"User-supplied root path \'{str(user_root_dir)}\' does not exist! Setting root path to CWD {str(cwd)}!")
                self.root_dir = cwd
            else:
                self.root_dir = user_root_dir



    def _create_subfolders(self):
        self.raw_data_dir = RawDataDir(self)
        self.inscopix_dir = InscopixDir(self)
        self.analysis_dir = AnalysisDir(self)


    def setup_folder(self):

        #  Check if the root directory exists  #
        root_directory = Path(self.root_dir)



        #  Check if the user-supplied path exists  #
        if self.project_dir is not None:
            temp_folder = Path(self.project_dir)

            if not temp_folder.exists():
                raise FileNotFoundError(f'User-supplied project folder {str(temp_folder)} does not exist')
            else:
                self.project_folder = temp_folder
        else:
            #  Get Project Folder from Selector  #
            returned_folder = self.select_project_folder()
            if len(returned_folder) > 0:
                project_folder = returned_folder[0]
                self.project_folder = Path(project_folder)
            else:
                raise FileNotFoundError(f'No project folder selected!')

        inscopix_path = self.project_folder.joinpath(*['InscopixProcessing', 'DataAnalysis'])

        if not inscopix_path.exists():
            raise FileNotFoundError(f'Inscopix data folder {str(inscopix_path)} does not exist!\n'
                                    f' Has data processing been run?')
        else:
            self.inscopix_path = inscopix_path

    def get_project_files(self):
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

    def select_project_folder(self) -> list[str]:
        file_names = []

        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Select Project Directory:")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setDirectory(self.root_dir)

        if file_dialog.exec():
            file_names = file_dialog.selectedFiles()
        return file_names

    #  Dunder Methods  #
    def __str__(self):
        return f'Project folder: {self.project_folder}'

    def __repr__(self):
        description = f'{type(self).__name__}(root_directory={self.root_dir}, \n \
                project_directory={self.project_folder}, \n \
                inscopix_directory={self.inscopix_path}, \n \
                max_projection_path={self.max_projection_path}, \n \
                cell_trace_data_path={self.cell_trace_data_path}, \n \
                cell_props_path={self.cell_props_path}, \n \
                cell_contours_path={self.cell_contours_path})'
        return description


class RawDataDir:
    def __init__(self, project_folder: ProjectFolder):
        self.root_dir: Path = project_folder.root_dir
        self.path = self.root_dir.joinpath('Raw_Data')
        # Raw inscopix files
        self.session_json_path = None
        self.raw_GPIO_path = None
        self.raw_recordings = []

        # Raw experiment files
        self.exp_h5_path = None
        self.odorlist_path = None

        exists = self._create()

        if exists:
            self._get_files()

    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)
            return False
        else:
            return True

    def _get_files(self):



        pass


class InscopixDir:
    def __init__(self, project_folder: ProjectFolder):
        self.root_dir = project_folder.root_dir
        self.path = self.root_dir.joinpath('Inscopix')
        # Processed Inscopix Files
        self.image_dir = None
        self.interim_file_dir = None
        self.cell_trace_path = None
        self.GPIO_path = None
        self.max_projection_path = None
        self.contours_path = None
        self.props_path = None

        self._create()


    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)


class AnalysisDir:
    def __init__(self, project_folder: ProjectFolder):
        self.root_dir = project_folder.root_dir
        self.path = self.root_dir.joinpath('Analysis')
        self.figures_dir = None
        self.preprocess_dir = None

        self._create()


    def _create(self):
        if not self.path.exists():
            self.path.mkdir(exist_ok=True)

    def _get_files(self):
        pass
