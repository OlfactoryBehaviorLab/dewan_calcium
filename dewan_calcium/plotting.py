import concurrent.futures
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
elif 'IPython' in sys.modules:
    from tqdm import tqdm

from .helpers import IO, trace_tools
from .helpers.project_folder import ProjectFolder

mpl.rcParams['font.family'] = 'Arial'

AXIS_PAD = 0.05  # PERCENT


def plotting_data_generator(cell_names, combined_data_shift, AUROC_data, significance_matrix):
    for cell in cell_names:
        yield cell, combined_data_shift[cell], AUROC_data[cell], significance_matrix[cell]


def genminmax(data: list[pd.Series], pad: float = 0):
    all_values = []
    for _, values in data:
        all_values.append(values)

    data_min, data_max = np.min(all_values), np.max(all_values)

    if pad > 0:
        if data_min > 0:
            data_min *= (1 - pad)
        else:
            data_min *= (1 + pad)
        data_max *= (1 + pad)

    return data_min, data_max


def generate_color_map(numColors: int):
    color_map = cm.get_cmap('rainbow')
    indices = (np.linspace(0, 1, numColors))  # % color_map.N
    return cycler.cycler('color', color_map(indices))


def truncate_data(data):
    row_len = np.min([len(row) for row in data])
    # print(row_len)
    for i, row in enumerate(data):
        data[i] = row[:row_len]

    return data


def _plot_odor_traces(FV_data: pd.DataFrame,
                      odor_list: pd.Series, response_duration: int, save_path: Path, latent: bool,
                      all_cells: bool, cell_data: tuple):

    cell_name, cell_df, auroc_data, significance_table = cell_data
    baseline_start = 0
    baseline_end = -response_duration
    evoked_start = 0
    evoked_end = response_duration

    if latent:
        evoked_start = response_duration
        evoked_end = response_duration * 2

    for odor in odor_list:
        significant = False
        significance_val = significance_table[odor]

        if all_cells is False and significance_val == 0:
            # If were not plotting everything and the cell is not significant; skip
            continue

        if significance_val > 0:  # Tag the significant graphs
            significant = True

        odor_data = cell_df[odor]
        odor_times = FV_data[odor]

        timestamps = list(odor_times.items())
        trial_data = list(odor_data.items())
        x_min, x_max = genminmax(timestamps, 0.05)
        y_min, y_max = genminmax(trial_data, 0.05)
        auroc_stats = auroc_data.loc[odor]
        percentile = auroc_stats['percentiles']
        lower_bound, upper_bound = auroc_stats['bounds']

        baseline_means, evoked_means = trace_tools.get_evoked_baseline_means(odor_data, odor_times,
                                                                             response_duration, latent)

        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])
        plt.suptitle(f'Cell: {cell_name} Odor: {odor}', fontsize=14)
        ax1.set_title('Cell Traces', fontsize=10)
        ax1.set(xlabel="Time ((s) since FV On)", ylabel="Signal")

        plot_data = zip(timestamps, trial_data)

        x_vals = []  # We want to expose the last set of x_vals for the average to use
        for x_vals, y_vals in plot_data:
            _, x_vals = x_vals  # Items() returns a name, we don't need it
            _, y_vals = y_vals

            ax1.plot(x_vals, y_vals, linewidth=0.5)

        avg_data = odor_data.mean(axis=1)
        ax1.plot(x_vals, avg_data, "k", linewidth=1.5)

        ax1.set_xticks(np.arange(-3, 6), labels=np.arange(-3, 6))
        ax1.text(0.015, 0.02, f'AUROC Percentile: {str(percentile * 100)}',
                 transform=ax1.transAxes, fontsize='x-small', style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

        if significant:
            ax1.text(0.015, 0.965, 'Significant!', transform=ax1.transAxes,
                     fontsize='x-small', style='italic',
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})

        baseline_rectangle = mpatches.Rectangle((baseline_start, y_min),
                                                (baseline_end - baseline_start),
                                                y_max, alpha=0.3, facecolor='blue')

        evoked_rectangle = mpatches.Rectangle((evoked_start, y_min), (evoked_end - evoked_start),
                                              y_max,
                                              alpha=0.3, facecolor='green')

        ax1.add_patch(baseline_rectangle)
        ax1.add_patch(evoked_rectangle)

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        plot_evoked_baseline_means(ax2, baseline_means, evoked_means)

        fig_name = f'{cell_name}-{odor}.pdf'
        fig_save_path = save_path.path.joinpath(fig_name)
        fig.savefig(fig_save_path, dpi=300)


def new_pooled_cell_plotting(combined_data_shift, AUROC_data: pd.DataFrame, significance_matrix: pd.DataFrame,
                             FV_data: pd.DataFrame, cell_names, odor_list: pd.Series, response_duration: int,
                             project_folder: ProjectFolder, latent: bool = False, all_cells: bool = False,
                             num_workers: int = None):
    save_path = project_folder.analysis_dir.figures_dir.ontime_traces_dir
    plot_type = 'On Time'
    if latent:
        save_path = project_folder.analysis_dir.figures_dir.latent_traces_dir
        plot_type = 'Latent'

    data_iterator = plotting_data_generator(cell_names, combined_data_shift, AUROC_data, significance_matrix)
    plot_function = partial(_plot_odor_traces,
                            FV_data, odor_list, response_duration, save_path, latent, all_cells)

    with tqdm(desc=f"Plotting {plot_type} Cells: ", total=len(cell_names)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executer:
            for _ in executer.map(plot_function, data_iterator):
                pbar.update()


def plot_cell_odor_traces(input_data, latent_cells_only: bool,
                          plot_all_cells: bool, cell_number: int) -> None:
    if latent_cells_only:
        folder = 'LatentCells'
    else:
        folder = 'OnTimeCells'

    folders = ['AllCellTracePlots', folder]

    input_data = input_data.makeCopy()
    input_data.update_cell(cell_number)
    cell_name = input_data.current_cell_name

    if plot_all_cells:
        odor_indexes = np.arange(input_data.num_unique_odors)
    else:
        odor_indexes = np.nonzero(input_data.significance_table[cell_number])[0]

    IO.make_cell_folder4_plot(str(cell_name), *folders)

    for index in odor_indexes:

        odor = input_data.unique_odors[index]

        input_data.update_odor(index)

        colormap = generate_color_map(len(input_data.current_odor_trials))
        plt.rcParams['axes.prop_cycle'] = colormap

        odor_name = odor

        auroc_percentile = input_data.percentiles[cell_number][index]

        lower_bound = input_data.lower_bounds[cell_number][index]
        upper_bound = input_data.upper_bounds[cell_number][index]

        _, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])
        plt.suptitle(f'Cell:{cell_name} Odor:{odor_name}', fontsize=14)
        ax1.set_title('Cell Traces', fontsize=10)
        ax1.set(xlabel="Time ((s) since FV On)", ylabel="Signal")

        y_min = []
        y_max = []
        data_4_average = []
        x_vals = []

        for trial in input_data.current_odor_trials:
            data_2_plot = input_data.Data[cell_number, trial, :]
            data_4_average.append(data_2_plot)

            x_vals = input_data.FV_time_map[trial, :]

            min_y_val = np.min(data_2_plot)
            max_y_val = np.max(data_2_plot)
            y_min.append(min_y_val)
            y_max.append(max_y_val)

            ax1.plot(x_vals, data_2_plot, linewidth=0.5)

        data_4_average = np.mean(data_4_average, axis=0)
        ax1.plot(x_vals, data_4_average, "k", linewidth=1.5)

        ax1.text(0.015, 0.02, f'AUROC Percentile: {str(auroc_percentile*100)}',
                 transform=ax1.transAxes, fontsize='x-small', style='italic',
                 bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

        if auroc_percentile > upper_bound or auroc_percentile < lower_bound:
            ax1.text(0.015, 0.965, 'Significant!', transform=ax1.transAxes,
                     fontsize='x-small', style='italic',
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})

        y_min = min(y_min)
        y_max = max(y_max)
        x_min = min(input_data.FV_time_map[0, :])
        x_max = max(input_data.FV_time_map[0, :])

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min - (0.01 * y_min), y_max + (0.01 * y_max)])

        ax1.set_xticks(np.arange(-4, 6), labels=np.arange(-4, 7))

        rectangle_y_min = y_min - 30
        rectangle_y_max = y_max - y_min + 60

        # fv_rectangle = mpatches.Rectangle((0, rectangle_y_min), 2, rectangle_y_max,
        #                                  alpha=0.3, facecolor='red')

        baseline_x_start = input_data.FV_time_map[0, min(input_data.baseline_start_indexes[cell_number][index])]
        baseline_x_end = input_data.FV_time_map[0, max(input_data.baseline_end_indexes[cell_number][index])]
        evoked_x_start = input_data.FV_time_map[0, min(input_data.evoked_start_indexes[cell_number][index])]
        evoked_y_start = input_data.FV_time_map[0, max(input_data.evoked_end_indexes[cell_number][index])]

        baseline_rectangle = mpatches.Rectangle((baseline_x_start, rectangle_y_min),
                                                (baseline_x_end - baseline_x_start),
                                                rectangle_y_max, alpha=0.3, facecolor='blue')

        evoked_rectangle = mpatches.Rectangle((evoked_x_start, rectangle_y_min), (evoked_y_start - evoked_x_start),
                                              rectangle_y_max,
                                              alpha=0.3, facecolor='green')

        # ax1.add_patch(fv_rectangle)
        ax1.add_patch(baseline_rectangle)
        ax1.add_patch(evoked_rectangle)

        plot_evoked_baseline_means(input_data, ax2, colormap)

        filename = f'{input_data.file_header}Cell{cell_name}-{odor_name}-CellTrace.pdf'

        path = Path(
            '.', 'ImagingAnalysis', 'Figures', 'AllCellTracePlots', folder, f'Cell-{cell_name}', filename)

        plt.subplots_adjust(bottom=0.15)

        plt.savefig(path, dpi=800)
        # plt.close()


def plot_evoked_baseline_means(ax2: plt.Axes, baseline_means, evoked_means):
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


def _plot_auroc_distribution(shuffle_dist, auroc_value, bounds, cell_name, odor_name) -> plt.figure:
    upper_bound, lower_bound = bounds

    fig, ax = plt.subplots()
    ax.hist(shuffle_dist, bins=10)
    ax.axvline(x=upper_bound, color='b'), ax.axvline(x=lower_bound, color='b')
    ax.axvline(x=auroc_value, color='r')
    fig.suptitle(f'{cell_name} x {odor_name}')
    plt.close(fig)

    return fig


def plot_auroc_distributions(auroc_data, odor_data, project_folder: ProjectFolder,
                             latent_cells_only: bool = False, plot_all: bool = False) -> None:
    # Plot AUROC Histograms if Desired
    unique_odors = odor_data.unique()

    folder = project_folder.analysis_dir.figures_dir.ontime_auroc_dir.path
    if latent_cells_only:
        folder = project_folder.analysis_dir.figures_dir.latent_auroc_dir.path

    for cell in auroc_data.columns.levels[0]:
        cell_df = auroc_data[cell]
        cell_significance_data = cell_df['significance_chart']
        auroc_values = cell_df['auroc_values']
        ul_bounds = cell_df['bounds']
        shuffles = cell_df['shuffles']

        for i, significance_value in enumerate(cell_significance_data):
            if significance_value != 0 | plot_all:
                odor_name = unique_odors[i]
                shuffle = shuffles.iloc[i]
                auroc_val = auroc_values.iloc[i]
                bounds = ul_bounds.iloc[i]
                auroc_fig = _plot_auroc_distribution(shuffle, auroc_val, bounds, cell, odor_name)

                filename = f'{cell}-{odor_name}.pdf'
                filepath = folder.joinpath(filename)
                auroc_fig.savefig(filepath, dpi=300)


def plot_trial_variances(input_data, significance_table: np.array,
                         latentCells: bool = False) -> None:
    responsive_cells_truth_table = np.any(significance_table, axis=1)  # Find all rows that are not all zeros
    responsive_cell_list_index = np.nonzero(responsive_cells_truth_table)[0]  # Only keep cells that are not

    if latentCells:
        folder = 'LatentCells'
    else:
        folder = 'OnTimeCells'

    folders = ['TrialVariancePlots', folder]

    for cell in responsive_cell_list_index:
        responsive_odor_indexes = np.nonzero(significance_table[cell] > 0)[0]

        input_data.update_cell(cell)
        IO.make_cell_folder4_plot(input_data.current_cell_name, *folders)

        for odor in responsive_odor_indexes:
            input_data.update_odor(odor)

            baseline_data, evok_data = trace_tools.collect_trial_data(input_data, None, latentCells)

            baseline_mean = np.mean(np.hstack(baseline_data))

            truncated_evok_data = truncate_data(evok_data)

            baseline_corrected_evok_data = truncated_evok_data - baseline_mean

            vertical_scatter_plot(baseline_corrected_evok_data, input_data, *folders)


def vertical_scatter_plot(data_2_plot: list, data_input, *folders):
    # x = len(data2Plot)
    width = 0.6
    dotSize = 10
    fig, ax = plt.subplots()
    for i, y in enumerate(data_2_plot):
        x = np.ones(len(y)) * (i + 1) + (np.random.rand(len(y)) * width - width / 2.)
        ax.scatter(x, y, s=dotSize)
    plt.title(f'Trials for Cell: {str(data_input.current_cell_name)} v. {data_input.odor_name}')
    plt.ylabel("Signal (Evoked - Baseline")
    plt.xlabel("Trial")
    ax.set_xticks(range(1, len(data_2_plot) + 1))

    path = Path('ImagingAnalysis', 'Figures', *folders, f'Cell-{data_input.current_cell_name}',
                f'{data_input.odor_name}-TrialTraces.pdf')
    plt.savefig(path, dpi=800)
    plt.close()


def pairwise_correlation_distances(odor_pairwise_distances, cell_pairwise_distances, cells, unique_odors):
    fontdict = {
        'weight': 'bold'
    }

    inferno = plt.colormaps['inferno']
    new_cmap = LinearSegmentedColormap.from_list('new_magma', inferno(np.linspace(0.2, 1.2, 128)))
    height = len(cell_pairwise_distances[0]) * 0.4
    fig, ax = plt.subplots(1, 2, figsize=(7, height))
    plt.subplots_adjust(bottom=0.1, top=0.9)

    odor_pdist = ax[0].matshow(odor_pairwise_distances, cmap=new_cmap)
    ax[0].tick_params(axis='x', labelrotation=50)
    ax[0].set_xticks(np.arange(len(unique_odors)))
    ax[0].set_yticks(np.arange(len(unique_odors)))
    ax[0].set_xticklabels(unique_odors, fontdict=fontdict, fontsize=5.5, ha='left')
    ax[0].set_yticklabels(unique_odors, fontdict=fontdict, fontsize=6)

    pdist = ax[1].matshow(cell_pairwise_distances, cmap=new_cmap)
    ax[1].tick_params(axis='x', labelrotation=50)

    ax[1].set_xticks(np.arange(len(cell_pairwise_distances[0])))
    ax[1].set_yticks(np.arange(len(cell_pairwise_distances[0])))
    ax[1].set_xticklabels(cells, fontdict=fontdict, fontsize=8, ha='left')
    ax[1].set_yticklabels(cells, fontdict=fontdict, fontsize=8)

    ax[0].set_title("Odor v. Odor", y=-0.25)
    ax[1].set_title("Cell v. Cell", y=-0.25)

    fig.colorbar(odor_pdist, ax=ax[0], shrink=0.4)
    fig.colorbar(pdist, ax=ax[1], shrink=0.4)

    fig.tight_layout()
    fig.suptitle('Pairwise Correlation Distance (1-r)', fontweight='bold', fontsize=18)

    path = Path('ImagingAnalysis', 'Figures', 'Statistics', 'correlations.pdf')
    fig.savefig(path, dpi=800)


def plot_distance_v_correlation(unique_distance_v_correlation):
    x = unique_distance_v_correlation[:, 0]
    y = unique_distance_v_correlation[:, 1]

    # Quick linear regression
    m, b = np.polyfit(x, y, deg=1)
    reg_x = np.arange(np.max(x))
    reg_y = np.add(np.multiply(m, reg_x), b)

    # Plot regression line, formula, and data
    fig, ax = plt.subplots()
    ax.plot(reg_x, reg_y)
    ax.scatter(x, y)
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Pairwise Signal Correlation")
    plt.title("Activity vs. Spatial Distance")
    _ = ax.text(0, np.max(y), f'y={np.round(m, 4)}x + {np.round(b, 4)}')
