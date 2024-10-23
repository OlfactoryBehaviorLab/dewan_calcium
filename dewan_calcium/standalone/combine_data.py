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



if __name__ == "__main__":
    main()