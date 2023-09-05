"""
Created on Sun Dec  4 19:59:13 2022

@author: A. Pauley, Dewan Lab
"""
import os
import numpy as np

import cycler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from functools import partial
from matplotlib.colors import ListedColormap
from multiprocessing import Pool

from .Helpers import DewanIOhandler
from .Helpers import DewanDataStore
from . import DewanAUROC


# noinspection DuplicatedCode

def generateColorMap(numColors: int):
    color_map = cm.get_cmap('rainbow')
    indices = (np.linspace(0, 1, numColors))  # % color_map.N
    return cycler.cycler('color', color_map(indices))


def plotOdorTracesPerCell(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool, cell: int) -> None:
    if latentCellsOnly:
        folder = 'LatentCells'
    else:
        folder = 'OnTimeCells'

    folders = ['AllCellTracePlots', folder]

    inputData = inputData.makeCopy()
    inputData.update_cell(cell)
    cell_name = inputData.current_cell_name

    DewanIOhandler.makeCellFolder4Plot(str(cell_name), folders)

    for index, odor in enumerate(inputData.unique_odors):
        inputData.update_odor(index)

        colormap = generateColorMap(len(inputData.current_odor_trials))
        plt.rcParams['axes.prop_cycle'] = colormap

        odor_name = odor

        auroc_percentile = np.around(inputData.percentiles[cell][index], 4)

        lower_bound = 0.01  # Set upper and lower boundaries for the percentile of the AUROC value in the shuffled
        # AUROC distribution.
        upper_bound = 0.99
        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])
        plt.suptitle(f'Cell:{cell_name} Odor:{odor_name}', fontsize=14)
        ax1.set_title('Cell Traces', fontsize=10)
        ax1.set(xlabel="Time ((s) since FV On)", ylabel="Signal")

        y_min = []
        y_max = []
        data4average = []
        x_vals = []

        for trial in inputData.current_odor_trials:
            data2plot = inputData.CombinedDataArray[cell, trial, :]
            data4average.append(data2plot)
            x_vals = inputData.FV_time_map[trial, :]
            min_val = np.min(data2plot)
            y_min.append(min_val)
            max_val = np.max(data2plot)
            y_max.append(max_val)
            ax1.plot(x_vals, data2plot, linewidth=0.5)

        data4average = np.mean(data4average, axis=0)
        ax1.plot(x_vals, data4average, "k", linewidth=1.5)

        ax1.text(0.015, 0.02, f'AUROC Percentile: {str(auroc_percentile)}',
                 transform=ax1.transAxes, fontsize='x-small', style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

        if auroc_percentile > upper_bound or auroc_percentile < lower_bound:
            ax1.text(0.015, 0.965, 'Significant!', transform=ax1.transAxes,
                     fontsize='x-small', style='italic',
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})

        y_min = min(y_min)
        y_max = max(y_max)
        x_min = min(inputData.FV_time_map[0, :])
        x_max = max(inputData.FV_time_map[0, :])

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min - (0.01 * y_min), y_max + (0.01 * y_max)])

        ax1.set_xticks(np.arange(-3, 6), labels=np.arange(-3, 6))

        rectangle_y_min = y_min - 30
        rectangle_y_max = y_max - y_min + 60

        # fv_rectangle = mpatches.Rectangle((0, rectangle_y_min), 2, rectangle_y_max,
        #                                  alpha=0.3, facecolor='red')

        baselineXStart = inputData.FV_time_map[0, min(inputData.baseline_start_indexes[cell][index])]
        baselineXEnd = inputData.FV_time_map[0, max(inputData.baseline_end_indexes[cell][index])]
        evokedXStart = inputData.FV_time_map[0, min(inputData.evoked_start_indexes[cell][index])]
        evokedXEnd = inputData.FV_time_map[0, max(inputData.evoked_end_indexes[cell][index])]

        baseline_rectangle = mpatches.Rectangle((baselineXStart, rectangle_y_min), (baselineXEnd - baselineXStart),
                                                rectangle_y_max, alpha=0.3, facecolor='blue')

        evoked_rectangle = mpatches.Rectangle((evokedXStart, rectangle_y_min), (evokedXEnd - evokedXStart),
                                              rectangle_y_max,
                                              alpha=0.3, facecolor='green')

        # ax1.add_patch(fv_rectangle)
        ax1.add_patch(baseline_rectangle)
        ax1.add_patch(evoked_rectangle)

        plotEvokedAndBaselineMeans(inputData, ax2, colormap)

        path = DewanIOhandler.generateFolderPath(
            ['.', 'ImagingAnalysis', 'Figures', 'AllCellTracePlots', folder, f'Cell-{cell_name}'])

        filename = f'{inputData.file_header}Cell{cell_name}-{odor_name}-CellTrace.png'

        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f'{path}/{filename}', dpi=800)
        plt.close()


def plotEvokedAndBaselineMeans(inputData: DewanDataStore.PlottingDataStore, ax2: plt.axis, colormap: cycler):
    baseline_means, evoked_means = DewanAUROC.averageTrialData(*DewanAUROC.collect_trial_data(inputData))

    x_val = [[1], [2]]
    x_vals = np.tile(x_val, (1, len(baseline_means)))
    ax2.set_title('Baseline v. Evoked Means', fontsize=10)

    ax2.plot(x_vals, (baseline_means, evoked_means), '-o', linewidth=2)

    ax2.set_xticks([1, 2], labels=['Baseline', 'Evoked'], rotation=45, ha='right', )
    ax2.yaxis.tick_right()

    y_min = np.min((baseline_means, evoked_means))
    y_max = np.max((baseline_means, evoked_means))

    ax2.set_ylim([y_min - (0.05 * y_min), y_max + (0.05 * y_max)])
    ax2.set_xlim([0.8, 2.2])

    baseline_mean = np.mean(baseline_means)
    evoked_mean = np.mean(evoked_means)

    ax2.plot(x_val, (baseline_mean, evoked_mean), '--ok', linewidth=3)


def plotAllCells(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool) -> None:
    workers = Pool()

    partial_function = partial(plotOdorTracesPerCell, inputData, latentCellsOnly)

    workers.map(partial_function, range(inputData.number_cells))
    # workers.map(partial_function, range(0, 1))
    workers.close()
    workers.join()


def plotSignificantCells(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool,
                         SignificantCells: np.array) -> None:
    workers = Pool()

    partial_function = partial(plotOdorTracesPerCell, inputData, latentCellsOnly)

    workers.close()
    workers.join()


def plotCellvOdorMatricies(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool) -> None:
    if latentCellsOnly:
        folder = 'LatentCells'
        title = 'Latent'
    else:
        folder = 'OnTimeCells'
        title = 'OnTime'

    fig, ax = plt.subplots()
    unique_odor_labels = inputData.unique_odors
    ax.set_facecolor((1, 1, 1))
    colormap = ListedColormap(['yellow', 'red', 'green'])
    ax.imshow(inputData.significance_table, cmap=colormap,
              extent=(0, inputData.num_unique_odors, inputData.number_cells, 0))
    ax.set_yticks(np.arange(inputData.number_cells), labels=[])
    ax.set_xticks(np.arange(inputData.num_unique_odors), labels=[])
    ax.set_xticks(np.arange(0.5, inputData.num_unique_odors + 0.5, 1), labels=unique_odor_labels, minor=True)
    ax.set_yticks(np.arange(0.5, inputData.number_cells + 0.5, 1), labels=inputData.Cell_List, minor=True)

    ax.tick_params(axis='both', which='both', left=False, bottom=False)

    plt.setp(ax.get_xminorticklabels(), rotation=90, ha="center")

    ax.set_title(f'{title} Cells v. Odors AUROC Values')
    plt.xlabel("Odors")
    plt.ylabel("Cells")

    ax.set_xlim([0, inputData.num_unique_odors])
    ax.set_ylim([0, inputData.number_cells])
    red_patch = mpatches.Patch(color='red', label='Negative AUROC')
    green_patch = mpatches.Patch(color='green', label='Positive AUROC')
    yellow_patch = mpatches.Patch(color='yellow', label='No Significance')
    ax.legend(handles=[red_patch, green_patch, yellow_patch], loc='upper right', bbox_to_anchor=(1.5, 1.0),
              borderaxespad=0)

    plt.grid(which='major')

    fig.tight_layout()

    folders = DewanIOhandler.generateFolderPath(['.', 'ImagingAnalysis', 'Figures', 'AUROCPlots', folder])
    filename = f'{inputData.file_header}{title}AllCellsvOdorsAUROC.png'
    plt.savefig(f'{folders}/{filename}', dpi=1200)
    plt.close()


def plotAuroc(fileHeader, auroc_shuffle, auroc, ub, lb, CellList, cellNum, uniqueOdors, odori, latentCellsOnly) -> None:
    cell_name = str(CellList[cellNum])
    odor_name = str(uniqueOdors[odori])

    fig, ax = plt.subplots()
    plt.hist(auroc_shuffle, bins=10)
    plt.axvline(x=ub, color='b'), plt.axvline(x=lb, color='b'), plt.axvline(x=auroc, color='r')
    plt.title(f'Cell:{cell_name} {odor_name}')

    if latentCellsOnly:
        sub_folder = 'LatentCells'
    else:
        sub_folder = 'OnTimeCells'

    folder = DewanIOhandler.generateFolderPath(['.', 'ImagingAnalysis', 'Figures', 'AUROCPlots', f'{sub_folder}'])
    filename = f'{fileHeader}Cell{cell_name}-{odor_name}.png'
    plt.savefig(f'{folder}/{filename}')
    plt.close()


def plotTrialsPerPairing(inputData: DewanDataStore.AUROCdataStore, SignificanceTable: np.array,
                         latentCells: bool) -> None:
    responsive_cells_truth_table = np.any(SignificanceTable, axis=1)  # Find all rows that are not all zeros
    responsive_cell_list_index = np.nonzero(responsive_cells_truth_table)[0]  # Only keep cells that are not

    if latentCells:
        folder = 'LatentCells'
    else:
        folder = 'OnTimeCells'

    folders = ['TrialVariancePlots', folder]

    for cell in responsive_cell_list_index:
        responsive_odor_indexes = np.nonzero(SignificanceTable[cell] > 0)[0]

        inputData.update_cell(cell)
        DewanIOhandler.makeCellFolder4Plot(inputData.current_cell_name, folders)

        for odor in responsive_odor_indexes:
            inputData.update_odor(odor)

            baseline_data, evok_data = DewanAUROC.collect_trial_data(inputData, None, latentCells)

            baseline_mean = np.mean(np.hstack(baseline_data))

            truncated_evok_data = truncateData(evok_data)

            baseline_corrected_evok_data = truncated_evok_data - baseline_mean

            verticalScatterPlot(baseline_corrected_evok_data, inputData, folders)


def truncateData(data):
    row_len = np.min([len(row) for row in data])
    # print(row_len)
    for i, row in enumerate(data):
        data[i] = row[:row_len]

    return data


def verticalScatterPlot(data2Plot: list, dataInput: DewanDataStore.AUROCdataStore, folders):
    # x = len(data2Plot)
    width = 0.6
    dotSize = 10
    fig, ax = plt.subplots()
    for i, y in enumerate(data2Plot):
        x = np.ones(len(y)) * (i + 1) + (np.random.rand(len(y)) * width - width / 2.)
        ax.scatter(x, y, s=dotSize)
    plt.title(f'Trials for Cell: {str(dataInput.current_cell_name)} v. {dataInput.odor_name}')
    plt.ylabel("Signal (Evoked - Baseline")
    plt.xlabel("Trial")
    ax.set_xticks(range(1, len(data2Plot) + 1))

    folders_path = DewanIOhandler.generateFolderPath(folders)
    path = os.path.join('./ImagingAnalysis/Figures', folders_path, f'Cell-{dataInput.current_cell_name}',
                        f'{dataInput.odor_name}-TrialTraces.png')
    plt.savefig(path, dpi=800)
    plt.close()
