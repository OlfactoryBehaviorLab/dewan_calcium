import os
import pickle


def createProjectFramework() -> None:
    paths = ['./RawData/',
             './PreProcessedData',
             './AUROCImports',
             './AUROCData',
             './CombinedData',
             './Figures/AUROCPlots/LatentCells',
             './Figures/AUROCPlots/OnTimeCells',
             './Figures/AllCellTracePlots/LatentCells',
             './Figures/AllCellTracePlots/OnTimeCells',
             './Figures/TrialVariancePlots/OnTimeCells',
             './Figures/TrialVariancePlots/LatentCells',
             ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def saveDataToDisk(data, name, fileHeader, folder) -> None:
    pickle_out = f'./{folder}/{fileHeader}{name}.pickle'
    output_file = open(pickle_out, 'wb')
    pickle.dump(data, output_file, protocol=-1)
    output_file.close()
    print(f'{fileHeader}{name} has been saved!')


def loadDataFromDisk(name, fileHeader, folder) -> object:
    pickle_in = open(f'./{folder}/{fileHeader}{name}.pickle', 'rb')
    data_in = pickle.load(pickle_in)
    pickle_in.close()
    print(f'{fileHeader}{name} has loaded successfully!')
    return data_in


def makeCellFolder4Plot(cell: str, *Folders: list) -> None:
    path = os.path.join('./Figures/', generateFolderPath(Folders[0]), f'Cell-{cell}')

    if not os.path.exists(path):
        os.makedirs(path)


def generateFolderPath(*Folders) -> os.path:
    path = ''
    for folder in Folders[0]:
        path = os.path.join(path, folder)

    return path
