from pathlib import Path
import numpy as np
import pandas as pd

folder = Path('R:\\2_Inscopix\\1_DTT\\1_OdorAnalysis\\2_Identity')


def find_data_files():
    print('Searching for data files, this may take a while...')
    data_files = folder.glob('*\\Analysis\\Output\\combined\\*combined_data_shift.pickle')
    if not data_files:
        print('No data files found.')
        return None
    else:
        data_files = list(data_files)
        print(f'Found {len(data_files)} data files.')
        return list(data_files)


def main():
    data_files = find_data_files()
    combined_data = pd.DataFrame()
    for file in data_files:
        new_data = pd.read_pickle(file)
        cells = len(np.unique([tup[0] for tup in new_data.columns]))
        print(cells)
        combined_data = pd.concat([combined_data, pd.read_pickle(file)], axis=1)
    print(len(np.unique([tup[0] for tup in combined_data.columns])))


if __name__ == "__main__":
    main()