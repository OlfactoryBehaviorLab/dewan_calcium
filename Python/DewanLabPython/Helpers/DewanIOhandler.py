import glob
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


def get_video_base(data_directory: os.path) -> str:
    gpio_file_path = glob.glob(os.path.join(data_directory, '*.gpio'))[0]
    gpio_file_name = os.path.basename(gpio_file_path)
    video_base = gpio_file_name[:-5]

    return video_base

def get_video_paths(video_base: str, video_directory: os.path) -> list:
    video_files = glob.glob(os.path.join(video_directory, '*.isxd'))
    return video_files
