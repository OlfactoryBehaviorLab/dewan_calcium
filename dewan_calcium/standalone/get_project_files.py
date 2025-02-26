import os
os.environ['ISX'] = '0'

from pathlib import Path


def get_folders(input_dir: Path, exp_type, animal_type: list, error:bool = False) -> dict:
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist!')

    data_files = dict.fromkeys(animal_type, {})

    for _dir in input_dir.iterdir():
        for _type in animal_type:
            if _type in str(_dir):
                data_files[_type] = {}
                type_dir = list(_dir.iterdir())
                for animal in type_dir:
                    animal_files = _find_data_files(animal, exp_type, error=error)
                    if not animal_files:
                        continue

                    data_files[_type][animal.name] = {}
                    data_files[_type][animal.name] = animal_files

    return data_files


def _find_data_files(animal_dir: Path, exp_type: str, error: bool = False):

    return_none = False

    if exp_type == 'Concentration' or exp_type == 'Identity':
        return_dict = dict.fromkeys(['file', 'sig', 'time', 'odor', 'folder'], [])
        return_dict['folder'] = animal_dir

        data_file = animal_dir.glob('Analysis/Output/combined/*combined_data_shift.pickle')
        FV_timestamps = animal_dir.glob('Analysis/Preprocessed/*FV_timestamps*.pickle')
        significance_file = animal_dir.glob('Analysis/Output/*SignificanceTable.xlsx')
        odor_data_file = animal_dir.glob('Raw_Data/*.xlsx')
    else:
        raise ValueError(f'{exp_type} not implemented!')

    data_file = list(data_file)
    FV_timestamps_file = list(FV_timestamps)
    significance_file = list(significance_file)
    odor_data_file = list(odor_data_file)

    if data_file:
        return_dict['file'] = data_file[0]
    else:
        if error:
           print(f'There are no combined data files in {animal_dir}!')
        else:
            return_none = True

    if FV_timestamps_file:
        return_dict['time'] = FV_timestamps_file[0]
    else:
        if error:
            print(f'Cannot find the FV timestamps in {animal_dir}')
        else:
            return_none = True

    if significance_file:
        return_dict['sig'] = significance_file[0]
    else:
        if error:
           print(f'Cannot find the significance chart in {animal_dir}')
        else:
            return_none = True
    if odor_data_file:
        return_dict['odor'] = odor_data_file[0]
    else:
        if error:
            print(f'Cannot find the odor data in {animal_dir}')
        else:
            return_none = True

    if return_none:
        return None
    else:
        return return_dict

