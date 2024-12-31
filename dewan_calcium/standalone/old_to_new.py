import pandas as pd
from pathlib import Path

input_dir = Path(r'R:/2_Inscopix/1_DTT/1_OdorAnalysis/2_Identity/VGAT')

CONC_ODOR_MAP = {
    '25ONE': '10-5ONE', '25ONE1': '1-5ONE', '25ONE100': '100-5ONE', '25ONE1000': '1000-5ONE',
    '5AL1': '1-5AL', '5AL10': '10-5AL', '5AL100': '100-5AL', '5AL1000': '1000-5AL',
    '5AMINE1': '1-5AMINE', '5AMINE10': '10-5AMINE', '5AMINE100': '100-5AMINE', '5AMINE1000': '1000-5AMINE',
    '5ATE1': '1-5ATE', '5ATE10': '10-5ATE', '5ATE100': '100-5ATE', '5ATE1000': '1000-5ATE',
    '5OL1': '1-5OL', '5OL10': '10-5OL', '5OL100': '100-5OL', '5OL1000': '1000-5OL',
    'Buzzer': "Buzzer",
    'MO': "MO"
}

ID_ODOR_MAP = {
    '25ONE': '5-ONE', '6ONE': '6-ONE', '7ONE': '7-ONE', '8ONE': '8-ONE',
    '5AL': '5-AL', '6AL': '6-AL', '7AL': '7-AL', '8AL': '8-AL',
    '5AMINE': '5-AMINE', '6AMINE': '6-AMINE', '7AMINE': '7-AMINE', '8AMINE': '8-AMINE',
    '5ATE': '5-ATE', '6ATE': '6-ATE', '7ATE': '7-ATE', '8ATE': '8-ATE',
    '5OL': '5-OL', '6OL': '6-OL', '7OL': '7-OL', '8OL': '8-OL',
    'Buzzer': "Buzzer",
    'MO': "MO"
}


def convert_combined_data(path):
    cell_file = list(path.joinpath('PreProcessedData').glob('*CellList.pickle'))[0]
    odor_data = list(path.joinpath('PreProcessedData').glob('*OdorData.pickle'))[0]
    cell_data = list(path.joinpath('CombinedData').glob('*CombinedData.pickle'))[0]

    old_data = pd.read_pickle(cell_data)
    cell_names = pd.read_pickle(cell_file)
    odor_data = pd.read_pickle(odor_data)
    odor_data = [odor[0] for odor in odor_data]

    old_data_df = []
    for cell in old_data:
        odor_data = odor_data[:len(cell)]
        old_data_cell = pd.DataFrame(cell, index=odor_data).T
        old_data_df.append(old_data_cell)

    new_data = pd.concat(old_data_df, axis=1, keys=cell_names, names=['Cells', 'Frames'])

    new_data_path = cell_data.with_stem(f'new-{cell_data.name}')
    new_data.to_pickle(new_data_path)
    return new_data_path


def convert_old_times(animal_dir):
    old_FV_timestamps = list(animal_dir.glob('ImagingAnalysis/AUROCImports/*FVTimeMap*.pickle'))
    old_odor_data = list(animal_dir.glob('ImagingAnalysis/PreProcessedData/*odordata*.pickle'))

    if not old_FV_timestamps or not old_odor_data:
        print(f'Files missing from {animal_dir}, skipping...')
        return

    old_FV_timestamps = old_FV_timestamps[0]
    old_odor_data = old_odor_data[0]

    timestamps = pd.read_pickle(old_FV_timestamps)
    odor_data = pd.read_pickle(old_odor_data)
    try:
        odor_data = [ID_ODOR_MAP[odor[0]] for odor in odor_data]
    except KeyError:
        odor_data = [odor[0] for odor in odor_data]

    new_timestamps = pd.DataFrame(timestamps, index=odor_data[:len(timestamps)]).T
    new_path = old_FV_timestamps.with_stem(f'new{old_FV_timestamps.stem}')
    pd.to_pickle(new_timestamps, new_path)
    print(f'Saved {new_path}')


def main():
    print('Starting!')
    for _dir in input_dir.iterdir():
        convert_old_times(_dir)

if __name__ == "__main__":
    main()
