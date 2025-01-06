from pathlib import Path

import os
os.environ['ISX'] = '0'


def get_folders(input_dir: Path, animal_type: list):
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist!')

    folders_per_type = {_type:[] for _type in animal_type}

    for _dir in input_dir.iterdir():
        for _type in animal_type:
            if _type in str(_dir):
                type_dir = list(_dir.iterdir())
                folders_per_type[_type].extend(type_dir)

    return folders_per_type


def find_data_files(animal_dir: Path, exp_type: str, error: bool = False):
    # return_dict = {'old': False, 'time_old': False}
    return_dict = dict.fromkeys(['file', 'sig', 'time', 'folder'], [])
    return_dict['folder'] = animal_dir
    if exp_type == 'Concentration' or exp_type == 'Identity':
        data_file = animal_dir.glob('Analysis/Output/combined/*combined_data_shift.pickle')
        new_old_data = animal_dir.glob('ImagingAnalysis/CombinedData/new*CombinedData*.pickle')
        new_FV_timestamps = animal_dir.glob('Analysis/Preprocessed/*FV_timestamps*.pickle')
        new_old_FV_timestamps = animal_dir.glob('ImagingAnalysis/AUROCImports/new*FVTimeMap*.pickle')
        significance_data_old = animal_dir.glob('*SignificanceTable.xlsx')
        significance_data_new = animal_dir.glob('Analysis/Output/*SignificanceTable.xlsx')
    else:
        raise ValueError(f'{exp_type} not implemented!')

    data_file = list(data_file)
    new_old_data = list(new_old_data)

    new_FV_timestamps_file = list(new_FV_timestamps)
    new_old_FV_timestamps = list(new_old_FV_timestamps)

    significance_data_old = list(significance_data_old)
    significance_data_new = list(significance_data_new)

    if new_old_data:
        return_dict['file'] = new_old_data[0]
    elif data_file:
        return_dict['file'] = data_file[0]
    else:
        if error:
            raise FileNotFoundError(f'There are no combined data files in {animal_dir}!')
        else:
            return_dict['file'] = None

    if significance_data_old:
        return_dict['sig'] = significance_data_old[0]
    elif significance_data_new:
        return_dict['sig'] = significance_data_new[0]
    else:
        if error:
            raise FileNotFoundError(f'Cannot find the significance chart in {animal_dir}')
        else:
            return_dict['sig'] = None

    if new_FV_timestamps_file:
        return_dict['time'] = new_FV_timestamps_file[0]
    elif new_old_FV_timestamps:
        return_dict['time'] = new_old_FV_timestamps[0]
    else:
        if error:
            raise FileNotFoundError(f'Cannot find the FV timestamps in {animal_dir}')
        else:
            return_dict['time'] = None

    return return_dict


def get_test_files(animal_dirs: list[Path]):
    file_map = {}
    for dir in animal_dirs:
        trace_data = dir.glob('*combined*.pickle')
        sig_table = dir.glob('*Significance*.xlsx')
        odor_data = dir.glob('*OdorData.xlsx')
        FV_times = dir.glob('*FV_timestamps.pickle')

        dir_dict = {
            'trace': list(trace_data)[0],
            'sig': list(sig_table)[0],
            'odor': list(odor_data)[0],
            'time': list(FV_times)[0]
        }
        file_map[dir.name] = dir_dict

    return file_map

