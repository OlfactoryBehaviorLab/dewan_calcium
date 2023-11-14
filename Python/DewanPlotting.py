"""
Created on Sun Dec  4 19:59:13 2022

@author: A. Pauley, Dewan Lab
"""
import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from functools import partial
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from multiprocessing import Pool
from pathlib import Path
from .Helpers import DewanIOhandler
from .Helpers import DewanDataStore
from . import DewanAUROC


# noinspection DuplicatedCode

def generate_color_map(numColors: int):
    color_map = cm.get_cmap('rainbow')
    indices = (np.linspace(0, 1, numColors))  # % color_map.N
    return cycler.cycler('color', color_map(indices))


def plot_cell_odor_traces(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool,
                          plotAll: bool, cell: int) -> None:

    if latentCellsOnly:
        folder = 'LatentCells'
    else:
        folder = 'OnTimeCells'

    folders = ['AllCellTracePlots', folder]

    inputData = inputData.makeCopy()
    inputData.update_cell(cell)
    cell_name = inputData.current_cell_name

    if plotAll:
        odor_indexes = np.arange(inputData.num_unique_odors)
    else:
        odor_indexes = np.nonzero(inputData.significance_table[cell])[0]

    DewanIOhandler.make_cell_folder4_plot(str(cell_name), *folders)

    for index in odor_indexes:

        odor = inputData.unique_odors[index]

        inputData.update_odor(index)

        colormap = generate_color_map(len(inputData.current_odor_trials))
        plt.rcParams['axes.prop_cycle'] = colormap

        odor_name = odor

        auroc_percentile = inputData.percentiles[cell][index]

        lower_bound = inputData.lower_bounds[cell][index]
        upper_bound = inputData.upper_bounds[cell][index]

        # lower_bound = 0.01  # Set upper and lower boundaries for the percentile of the AUROC value in the shuffled
        # # AUROC distribution.
        # upper_bound = 0.99

        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])
        plt.suptitle(f'Cell:{cell_name} Odor:{odor_name}', fontsize=14)
        ax1.set_title('Cell Traces', fontsize=10)
        ax1.set(xlabel="Time ((s) since FV On)", ylabel="Signal")

        y_min = []
        y_max = []
        data4average = []
        x_vals = []

        for trial in inputData.current_odor_trials:
            data2plot = inputData.Data[cell, trial, :]
            data4average.append(data2plot)
            x_vals = inputData.FV_time_map[trial, :]
            min_val = np.min(data2plot)
            y_min.append(min_val)
            max_val = np.max(data2plot)
            y_max.append(max_val)
            ax1.plot(x_vals, data2plot, linewidth=0.5)

        data4average = np.mean(data4average, axis=0)
        ax1.plot(x_vals, data4average, "k", linewidth=1.5)

        ax1.text(0.015, 0.02, f'AUROC Percentile: {str(auroc_percentile*100)}',
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

        plot_evoked_baseline_means(inputData, ax2, colormap)

        filename = f'{inputData.file_header}Cell{cell_name}-{odor_name}-CellTrace.pdf'

        path = Path(
            '.', 'ImagingAnalysis', 'Figures', 'AllCellTracePlots', folder, f'Cell-{cell_name}', filename)

        plt.subplots_adjust(bottom=0.15)

        plt.savefig(path, dpi=800)
        plt.close()


def plot_evoked_baseline_means(inputData: DewanDataStore.PlottingDataStore, ax2: plt.axis, colormap: cycler):
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


def plot_cells(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool = False, plotAll: bool = False) -> None:
    thread_pool = Pool()

    partial_function = partial(plot_cell_odor_traces, inputData, latentCellsOnly, plotAll)

    if plotAll:
        cells = range(inputData.number_cells)
    else:
        cells = np.unique(np.nonzero(inputData.significance_table > 0)[0])
        # Get only the cells that had some type of significant response

    thread_pool.map(partial_function, cells)
    thread_pool.close()
    thread_pool.join()


def plot_significance_matricies(inputData: DewanDataStore.PlottingDataStore, latentCellsOnly: bool = False) -> None:
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

    fig.tight_layout(pad=0.2)

    folders = Path(*['.', 'ImagingAnalysis', 'Figures', 'AUROCPlots', folder])
    filename = f'{inputData.file_header}{title}AllCellsvOdorsAUROC.svg'
    plt.savefig(folders.joinpath(filename), dpi=1200, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_auroc_distributions(fileHeader, auroc_shuffle, auroc, ub, lb, CellList, cellNum, uniqueOdors, odori,
                             latentCellsOnly: bool = False) -> None:
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

    folder = Path(*['.', 'ImagingAnalysis', 'Figures', 'AUROCPlots', f'{sub_folder}'])
    filename = f'{fileHeader}Cell{cell_name}-{odor_name}.pdf'
    plt.savefig(folder.joinpath(filename))
    plt.close()


def plot_trial_variances(inputData: DewanDataStore.AUROCdataStore, SignificanceTable: np.array,
                         latentCells: bool = False) -> None:
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
        DewanIOhandler.make_cell_folder4_plot(inputData.current_cell_name, *folders)

        for odor in responsive_odor_indexes:
            inputData.update_odor(odor)

            baseline_data, evok_data = DewanAUROC.collect_trial_data(inputData, None, latentCells)

            baseline_mean = np.mean(np.hstack(baseline_data))

            truncated_evok_data = truncate_data(evok_data)

            baseline_corrected_evok_data = truncated_evok_data - baseline_mean

            vertical_scatter_plot(baseline_corrected_evok_data, inputData, *folders)


def truncate_data(data):
    row_len = np.min([len(row) for row in data])
    # print(row_len)
    for i, row in enumerate(data):
        data[i] = row[:row_len]

    return data


def vertical_scatter_plot(data2Plot: list, dataInput: DewanDataStore.AUROCdataStore, *folders):
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

    path = Path('ImagingAnalysis', 'Figures', *folders, f'Cell-{dataInput.current_cell_name}',
                f'{dataInput.odor_name}-TrialTraces.pdf')
    plt.savefig(path, dpi=800)
    plt.close()


def pairwise_correlation_distances(odor_pairwise_distances, cell_pairwise_distances, unique_odors):
    inferno = plt.colormaps['inferno']
    new_cmap = LinearSegmentedColormap.from_list('new_magma', inferno(np.linspace(0.2, 1.2, 128)))

    fig, ax = plt.subplots(1, 2)
    fig.tight_layout()
    fig.set_figwidth(15)
    fig.suptitle('Pairwise Correlation Distance (1-r)')

    odor_pdist = ax[0].matshow(odor_pairwise_distances, cmap=new_cmap)
    ax[0].tick_params(axis='x', labelrotation=50, bottom=False)
    ax[0].set_xticks(np.arange(len(unique_odors)))
    ax[0].set_yticks(np.arange(len(unique_odors)))
    ax[0].set_xticklabels(unique_odors, ha='left')
    ax[0].set_yticklabels(unique_odors)

    pdist = ax[1].matshow(cell_pairwise_distances, cmap=new_cmap)
    ax[1].set_xticks(np.arange(len(cell_pairwise_distances[0])))
    ax[1].set_yticks(np.arange(len(cell_pairwise_distances[0])))

    ax[0].set_title("Odor v. Odor")
    ax[1].set_title("Cell v. Cell")

    fig.colorbar(odor_pdist, ax=ax[0], shrink=0.8)
    fig.colorbar(pdist, ax=ax[1], shrink=0.8)

    fig.tight_layout()

    path = Path('ImagingAnalysis', 'Figures', 'Statistics', 'correlations.pdf')
    fig.savefig(path, dpi=800)


def plot_distance_v_correlation(unique_distance_v_correlation):
    x = unique_distance_v_correlation[:, 0]
    y = unique_distance_v_correlation[:, 1]

    # 8C.1: Quick linear regression
    m, b = np.polyfit(x, y, deg=1)
    reg_x = np.arange(np.max(x))
    reg_y = np.add(np.multiply(m, reg_x), b)

    # 8C.2: Plot regression line, formula, and data
    fig, ax = plt.subplots()
    ax.plot(reg_x, reg_y)
    ax.scatter(x, y)
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Pairwise Signal Correlation")
    plt.title("Activity vs. Spatial Distance")
    formula = ax.text(0, np.max(y), f'y={np.round(m, 4)}x + {np.round(b, 4)}')

