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

def new_find_data_files(animal_dir: Path, exp_type: str, error: bool = False):
    return_dict = {'old': False}

    data_file = []
    old_data_file = []
    new_old_data = []

    significance_data_old = []
    significance_data_new = []


    if exp_type == 'Concentration' or exp_type == 'Identity':
        data_file = animal_dir.glob('Analysis/Output/combined/*combined_data_shift.pickle')
        old_data_file = animal_dir.glob('ImagingAnalysis/CombinedData/*CombinedData.pickle')
        new_old_data = animal_dir.glob('ImagingAnalysis/CombinedData/new*CombinedData*.pickle')

        significance_data_old = animal_dir.glob('*SignificanceTable.xlsx')
        significance_data_new = animal_dir.glob('Analysis/Output/*SignificanceTable.xlsx')
    # elif exp_type == 'EPM':
    #     data_files = animal_dir.rglob(r'*/Analysis/Output/pseudotrials/*pseudotrial_traces.pickle')
    # elif exp_type == 'HFvFM':
    #     data_files = animal_dir.rglob(r'*/Analysis/Output/combined/*combined_data.pickle')
    else:
        raise ValueError(f'{exp_type} not implemented!')

    data_file = list(data_file)
    old_data_file = list(old_data_file)
    new_old_data = list(new_old_data)
    significance_data_old = list(significance_data_old)
    significance_data_new = list(significance_data_new)

    if new_old_data:
        return_dict['file'] = new_old_data[0]
    elif old_data_file:
        return_dict['file'] = old_data_file[0]
        return_dict['old'] = True
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

    return return_dict
