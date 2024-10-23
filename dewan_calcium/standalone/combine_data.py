from pathlib import Path
import numpy as np
import pandas as pd

folder = Path('R:\2_Inscopix\1_DTT\1_OdorAnalysis\1_Concentration')

def main():
    data_files = folder.rglob('*combined_data_shift.pickle')
    print(data_files)



if __name__ == "__main__":
    main()