import concurrent.futures
from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
elif 'IPython' in sys.modules:
    from tqdm import tqdm

from .helpers import IO, trace_tools
from .helpers.project_folder import ProjectFolder, Dir

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


# def generate_color_map(numColors: int):
#     color_map = cm.get_cmap('rainbow')
#     indices = (np.linspace(0, 1, numColors))  # % color_map.N
#     return cycler.cycler('color', color_map(indices))


def _plot_evoked_baseline_means(ax2: plt.Axes, baseline_means, evoked_means):
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


def _plot_odor_traces(FV_data: pd.DataFrame, odor_list: pd.Series, response_duration: int, save_path: Dir, latent: bool,
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
        #  lower_bound, upper_bound = auroc_stats['bounds']

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
                                                baseline_end - baseline_start,
                                                y_max, alpha=0.3, facecolor='blue')

        evoked_rectangle = mpatches.Rectangle((evoked_start, y_min), (evoked_end - evoked_start),
                                              y_max,
                                              alpha=0.3, facecolor='green')

        ax1.add_patch(baseline_rectangle)
        ax1.add_patch(evoked_rectangle)

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        _plot_evoked_baseline_means(ax2, baseline_means, evoked_means)

        plt.subplots_adjust(bottom=0.15)

        fig_name = f'{cell_name}-{odor}.pdf'
        fig_save_path = save_path.path.joinpath(fig_name)
        fig.savefig(fig_save_path, dpi=300)


def pooled_cell_plotting(combined_data_shift, AUROC_data: pd.DataFrame, significance_matrix: pd.DataFrame,
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


def _plot_auroc_distribution(AUROC_data, odor_list, save_path, plot_all, cell_name):

    cell_df = AUROC_data[cell_name]
    cell_significance_data = cell_df['significance_chart']
    auroc_values = cell_df['auroc_values']
    ul_bounds = cell_df['bounds']
    shuffles = cell_df['shuffles']

    for i, significance_value in enumerate(cell_significance_data):
        if significance_value != 0 | plot_all:
            odor_name = odor_list[i]
            shuffle = shuffles.iloc[i]
            auroc_val = auroc_values.iloc[i]
            bounds = ul_bounds.iloc[i]
            upper_bound, lower_bound = bounds

            fig, ax = plt.subplots()
            ax.hist(shuffle, bins=10)
            ax.axvline(x=upper_bound, color='b'), ax.axvline(x=lower_bound, color='b')
            ax.axvline(x=auroc_val, color='r')
            fig.suptitle(f'{cell_name} x {odor_name}')
            plt.close(fig)

            filename = f'{cell_name}-{odor_name}.pdf'
            filepath = save_path.path.joinpath(filename)
            fig.savefig(filepath, dpi=300)


def pooled_auroc_distributions(AUROC_data, cell_names, odor_list, project_folder: ProjectFolder,
                               latent: bool = False, all_cells: bool = False, num_workers: int = None):

    save_path = project_folder.analysis_dir.figures_dir.ontime_auroc_dir
    plot_type = 'On Time'
    if latent:
        save_path = project_folder.analysis_dir.figures_dir.latent_auroc_dir
        plot_type = 'Latent'

    plot_function = partial(_plot_auroc_distribution, AUROC_data, odor_list, save_path, all_cells)

    with tqdm(desc=f"Plotting {plot_type} AUROC Distributions: ", total=len(cell_names)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executer:
            for _ in executer.map(plot_function, cell_names):
                pbar.update()


def plot_animal_track(line_coordinates, background_img, project_folder):

    lc = LineCollection(line_coordinates, colors='dimgrey', alpha=0.8, linewidths=0.1)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    _ = ax.imshow(background_img)
    _ = ax.add_collection(lc)

    plt.suptitle(f'Animal Track')

    file_name = f'animal_track.pdf'
    file_path = project_folder.analysis_dir.figures_dir.path.joinpath(file_name)

    plt.tight_layout()

    fig.savefig(file_path, dpi=900)
    plt.close(fig)


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

            # truncated_evok_data = truncate_data(evok_data)

            # baseline_corrected_evok_data = truncated_evok_data - baseline_mean

            # vertical_scatter_plot(baseline_corrected_evok_data, input_data, *folders)


def pooled_trial_variances():
    # TODO: Setup trial variance plotting like other functions
    pass


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


def plot_epm_roi(polygons: pd.DataFrame, image):
    fig, ax = plt.subplots()  # New Graph
    ax.set_axis_off()

    original_polygons = polygons['Shape']

    open_coordinates, closed_coordinates, center_coordinates = get_polygon_coordinates(original_polygons)

    open_poly = Polygon(open_coordinates, alpha=0.2, color='r')
    closed_poly = Polygon(closed_coordinates, alpha=0.2, color='b')
    center_poly = Polygon(center_coordinates, alpha=0.3, color=(0, 1, 0))

    patches = PatchCollection([open_poly, closed_poly, center_poly], match_original=True)

    ax.add_collection(patches)
    ax.legend([open_poly, closed_poly, center_poly], ['Open', 'Closed', 'Center'], loc='lower center', framealpha=1)
    _ = ax.imshow(image)

    return fig, ax


def get_polygon_coordinates(polygons):
    coordinates = []

    for polygon in polygons:
        coordinates.append(list(polygon.exterior.coords)[:-1])  # Drop last point as it is a duplicate of the first

    return coordinates


def plot_EPM_auroc_histograms(AUROC_results, project_folder):
    auroc_vals = [AUROC_results[cell]['auroc'] for cell in AUROC_results]
    auroc_vals = np.array(auroc_vals)
    direction_indexes = 2 * (auroc_vals - 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(direction_indexes, bins=20)
    ax[1].hist(auroc_vals, color='r', bins=20)
    minmax = [min(direction_indexes), max(direction_indexes)]
    minmax2 = [min(auroc_vals), max(auroc_vals)]

    ax[0].set_xticks(np.linspace(-1, 1, 21))
    ax[1].set_xticks(np.linspace(0, 1, 21))

    ax[0].set_xlim(minmax)
    ax[1].set_xlim(minmax2)

    ax[0].tick_params(axis='x', labelrotation=50)
    ax[1].tick_params(axis='x', labelrotation=50)

    ax[0].set_title('Direction Indices')
    ax[1].set_title('auROC Values')
    plt.tight_layout()

    fig_dir = project_folder.analysis_dir.figures_dir.subdir('AUROC')
    fig_path = fig_dir.joinpath('auROC_distribution.pdf')
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)


def plot_epm_shuffles(AUROC_results, cell_names, project_folder):
    for cell in tqdm(cell_names):
        try:
            results = AUROC_results[cell]
            ub = results['ub']
            lb = results['lb']
            auroc = results['auroc']
            shuffle = results['shuffle']
            significance = results['significance']

            relaxed_bounds = np.percentile(shuffle, [5, 95])
            lb_r, ub_r = relaxed_bounds

            fig, ax = plt.subplots()
            ax.hist(shuffle, color='gray', bins=10)
            ax.axvline(x=ub, color='blue')
            ax.axvline(x=lb, color='blue')
            ax.axvline(x=ub_r, color='green')
            ax.axvline(x=lb_r, color='green')
            ax.axvline(x=auroc, color='red')

            if significance in (-1, 1):
                ax.set_xlabel('Significant!')

            fig.suptitle(f'{cell} - {round(auroc, 3)}')

            save_dir = project_folder.analysis_dir.figures_dir.subdir('AUROC')
            file_path = save_dir.joinpath(f'{cell}.pdf')
            fig.savefig(file_path, dpi=600)
            plt.close(fig)
        except Exception as e:  # yes, this is bad; its okay
            print(f'Error plotting {cell}')
            print(e)
