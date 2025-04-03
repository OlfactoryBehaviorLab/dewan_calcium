import os
os.environ['ISX'] = '0'
from dewan_calcium_git.dewan_calcium.helpers.project_folder import ProjectFolder
from dewan_calcium_git.dewan_calcium.helpers import parse_json, IO
from dewan_calcium_git.dewan_calcium import AUROC

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

def main():
    animal = "VGLUT20-ID"
    date = "1-9-2025"
    pre_trial_time = 3.5  # Imaging time before the final valve opens
    post_trial_time = 3.5  # Imaging time after final valve closes

    # Configurables for AUROC
    baseline_duration = 2  # number of seconds before the FV turns on
    response_duration = 2  # number of seconds after the FV turns off

    # Create Project Folder to Gather and Hold all the File Paths

    project_folder = ProjectFolder('ODOR', project_dir=r'R:\2_Inscopix\1_DTT\1_OdorAnalysis\2_Identity\VGLUT\VGLUT-20')
    file_header = animal + '-' + date + '-'
    # Get settings from imaging session and display them for the user

    gain, LED_power, endoscope_framerate, focal_planes = parse_json.get_session_settings(
        project_folder.raw_data_dir.session_json_path)

    print(f'Recording Gain: {gain}')
    print(f'LED Power: {LED_power}')
    print(f'Endoscope Framerate: {endoscope_framerate}')
    print(f'Focal Plane(s): {focal_planes}')

    folder = project_folder.analysis_dir.combined_dir.path
    combined_data_shift = IO.load_data_from_disk('combined_data_shift', file_header, folder)

    folder = project_folder.analysis_dir.preprocess_dir.path
    FV_data = IO.load_data_from_disk('FV_data', file_header, folder)
    FV_indexes = IO.load_data_from_disk('FV_indexes', file_header, folder)
    unix_timestamps = IO.load_data_from_disk('unix_timestamps', file_header, folder)
    FV_timestamps = IO.load_data_from_disk('FV_timestamps', file_header, folder)
    odor_data = IO.load_data_from_disk('odor_data', file_header, folder)
    odor_list = IO.load_data_from_disk('odor_list', file_header, folder)
    curated_cell_props = IO.load_data_from_disk('curated_cell_props', file_header, folder)
    cell_names = curated_cell_props['Name']

    trimmed_cells = cell_names[:10]
    combined_data_shift = combined_data_shift[trimmed_cells]

    # STEP 6A.1: RUN AUROC FOR ON-TIME CELLS
    # Note: On time cells are those that respond during the stimulus window (0s-2s)
    # on_time_AUROC_return = AUROC.pooled_odor_auroc(combined_data_shift, FV_timestamps, baseline_duration, 8, False) # This takes a long time!
    # # STEP 6A.2: RUN AUROC FOR LATENT CELLS
    # Note: Latent cells are those that respond immediately after the stimulus window (2s-4s)
    # latent_AUROC_return = AUROC.pooled_odor_auroc(combined_data_shift, FV_timestamps, baseline_duration, 8, True) # This takes a long time!

    cell_grouped_data = combined_data_shift.T.groupby(level=0)

    for cell in tqdm(cell_grouped_data):
        _ = AUROC.odor_auroc(FV_timestamps, baseline_duration, False, cell)

if __name__ == '__main__':
    main()