from pathlib import Path

SKIP = ['.', 'z']

def get_folders(input_dir: Path, exp_type, animal_type: list, error:bool = False) -> dict:
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist!')

    data_files = {key: {} for key in animal_type}

    for _dir in input_dir.iterdir(): # Loop through the root directory
        for _type in animal_type: # Find the folder for our animal type [VGAT/VGLUT/OXTR/etc.]
            if _type in str(_dir):
                data_files[_type] = {}
                type_dir = list(_dir.iterdir()) # List all the animals within a class
                for animal in type_dir:
                    if animal.name[0] in SKIP:
                        continue # Skip any .* or z directories
                    print(f'Searching {animal}...')
                    animal_files = _find_data_files(animal, exp_type, error=error)
                    if not animal_files:
                        continue

                    data_files[_type][animal.name] = animal_files

    return data_files


def _check_file(path, name):
    if path:
        return path[0]
    else:
        print(f'Unable to find {name}!')
        return None


def _find_data_files(animal_dir: Path, exp_type: str, error: bool = False):
    return_dict = {}

    if exp_type == 'Concentration' or exp_type == 'Identity':
        return_dict['folder'] = animal_dir

        data_file = animal_dir.glob('Analysis/Output/combined/*combined_data_dff.pickle')
        FV_timestamps = animal_dir.glob('Analysis/Preprocessed/*FV_timestamps*.pickle')
        significance_file = animal_dir.glob('Analysis/Output/*SignificanceTable.xlsx')
        odor_data_file = animal_dir.glob('Raw_Data/*List*.xlsx')
        block_data_file = animal_dir.glob('Raw_Data/*Blocks*.xlsx')
    else:
        raise ValueError(f'{exp_type} not implemented!')

    data_file = list(data_file)
    FV_timestamps_file = list(FV_timestamps)
    significance_file = list(significance_file)
    odor_data_file = list(odor_data_file)
    block_data_file = list(block_data_file)

    return_dict['file'] = _check_file(data_file, 'combined data file')
    return_dict['time'] = _check_file(FV_timestamps_file, 'FV timestamps file')
    return_dict['sig'] = _check_file(significance_file, 'significance matrix file')
    return_dict['odor'] = _check_file(odor_data_file, 'odor data file file')
    return_dict['block'] = _check_file(block_data_file, 'odor-block mapping file')

    results = [True if return_dict[key] is None else False for key in return_dict.keys()]

    if sum(results) > 0:
        return None
    else:
        return return_dict
