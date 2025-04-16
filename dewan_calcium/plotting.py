import concurrent
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap

from numba import njit
from sklearn import metrics
from tqdm.contrib.concurrent import process_map

from .helpers import IO, trace_tools
from .helpers.project_folder import ProjectFolder

mpl.rcParams['font.family'] = 'Arial'
AXIS_PAD = 0.05  # PERCENT


def plotting_data_generator(combined_data_dff, AUROC_data, wilcoxon_total):
    for cell, cell_data in combined_data_dff.T.groupby('Cells'):
        yield cell, cell_data, AUROC_data[cell], wilcoxon_total[cell]

@njit()
def genminmax(data: np.array, pad: float = 0):
    all_values = np.ravel(data)
    data_min, data_max = np.min(all_values), np.max(all_values)

    if pad > 0:
        if data_min > 0:
            data_min *= (1 - pad)
        else:
            data_min *= (1 + pad)
        data_max *= (1 + pad)

    return data_min, data_max

def _plot_evoked_MO_means(ax3: plt.Axes, MO_means, evoked_means):

    x_vals_mo = [1] * len(MO_means)
    x_vals_evoked = [2] * len(evoked_means)
    ax3.set_title('Means', fontsize=10)

    ax3.plot(x_vals_mo, MO_means, 'o', color='blue')
    ax3.plot(x_vals_evoked, evoked_means, 'o', color='green')
    ax3.set_xticks([1, 2], labels=['MO Baseline', 'Evoked'], rotation=45, ha='right', )
    ax3.set_xlim([0.8, 2.2])

    baseline_mean = np.mean(MO_means)
    evoked_mean = np.mean(evoked_means)
    ax3.plot([[1], [2]], (baseline_mean, evoked_mean), '--ok', linewidth=3)


def _plot_evoked_baseline_means(ax2: plt.Axes, baseline_means, evoked_means):
    x_val = [[1], [2]]
    x_vals = np.tile(x_val, (1, len(baseline_means)))
    ax2.plot(x_vals, (baseline_means, evoked_means), '-o', linewidth=2)

    ax2.set_title('Means', fontsize=10)
    ax2.set_xticks([1, 2], labels=['Baseline', 'Evoked'], rotation=45, ha='right', )
    ax2.set_xlim([0.8, 2.2])

    baseline_mean = np.mean(baseline_means)
    evoked_mean = np.mean(evoked_means)
    ax2.plot(x_val, (baseline_mean, evoked_mean), '--ok', linewidth=3)


def _plot_odor_traces(FV_data: pd.DataFrame, response_duration: int, project_folder: ProjectFolder,
                      all_cells: bool, cell_data: tuple):
    cell_name, cell_df, combo_auroc_data, wilcoxon_totals = cell_data
    cell_df = cell_df.T[cell_name]
    odor_list = cell_df.columns.get_level_values(0).unique()
    odor_list = [odor for odor in odor_list if odor != 'MO']
    _, MO_baseline = trace_tools.get_evoked_baseline_means(cell_df['MO'], FV_data['MO'],
                                                           response_duration, None)

    for odor in odor_list:
        baseline_start = 0
        baseline_end = -response_duration
        evoked_start = 0
        evoked_end = response_duration

        odor_data = cell_df[odor]
        odor_times = FV_data[odor]
        wilcoxon_p_val = str(round(wilcoxon_totals[odor], 3))
        auroc_stats = combo_auroc_data.loc[odor]
        significance_val = auroc_stats['significance_chart'].astype(int)
        percentile = str(round(auroc_stats['percentiles'], 4) * 100)

        if all_cells is False and significance_val == 0:
            # If were not plotting everything and the cell is not significant; skip
            continue

        save_path = project_folder.analysis_dir.figures_dir.traces_dir.subdir(cell_name)

        if significance_val != 0:  # Tag the significant graphs
            significant = True
        else:
            significant = False
            save_path = save_path.subdir('insignificant')

        timestamps = [_item[1].values for _item in odor_times.items()]
        trial_data = [_item[1].values for _item in odor_data.items()]
        x_min, x_max = genminmax(np.array(timestamps), 0.05)
        y_min, y_max = genminmax(np.array(trial_data), 0.05)

        baseline_means, evoked_means = trace_tools.get_evoked_baseline_means(odor_data, odor_times,
                                                                             response_duration, -2)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, width_ratios=[5, 1, 1], sharey=True)

        plt.suptitle(f'Cell: {cell_name} Odor: {odor}', fontsize=14)
        ax1.set_title('Cell Traces', fontsize=10)
        ax1.set(xlabel="Time ((s) since FV On)", ylabel="dF/F")

        plot_data = zip(timestamps, trial_data)

        x_vals = []  # We want to expose the last set of x_vals for the average to use
        for x_vals, y_vals in plot_data:
            ax1.plot(x_vals, y_vals, linewidth=0.5, color='k', alpha=0.5)

        avg_data = odor_data.mean(axis=1)
        ax1.plot(x_vals, avg_data, "m", linewidth=1.5)

        ax1.set_xticks(np.arange(-3, 6), labels=np.arange(-3, 6))

        if significant:
            ax1.text(0.015, 0.965, 'Significant!', transform=ax1.transAxes,
                     fontsize='x-small', style='italic',
                     bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})

            ax1.text(0.015, 0.02, f'AUROC Value Percentile: {percentile}th | Wilcoxon p-value: {wilcoxon_p_val}',
                     transform=ax1.transAxes, fontsize='x-small', style='italic',
                     bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})

        rect_height = y_max + np.abs(y_min)
        baseline_rectangle = mpatches.Rectangle((baseline_start, y_min),
                                                baseline_end - baseline_start,
                                                rect_height, alpha=0.3, facecolor='blue')

        evoked_rectangle = mpatches.Rectangle((evoked_start, y_min), (evoked_end - evoked_start),
                                              rect_height,
                                              alpha=0.3, facecolor='green')

        ax1.add_patch(baseline_rectangle)
        ax1.add_patch(evoked_rectangle)

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        _plot_evoked_baseline_means(ax2, baseline_means, evoked_means)
        _plot_evoked_MO_means(ax3, MO_baseline, evoked_means)

        plt.subplots_adjust(wspace=0.1, bottom=0.18)

        fig_name = f'{cell_name}-{odor}.pdf'
        fig_save_path = save_path.joinpath(fig_name)
        fig.savefig(fig_save_path, dpi=300)
        plt.close(fig)


def pooled_cell_plotting(combined_data_shift, AUROC_data: pd.DataFrame, wilcoxon_total, FV_data: pd.DataFrame, response_duration: int,
                         project_folder: ProjectFolder, all_cells: bool = False, num_workers: int = None):
    num_cells = len(combined_data_shift.columns.get_level_values(0).unique())
    data_iterator = plotting_data_generator(combined_data_shift, AUROC_data, wilcoxon_total)
    plot_function = partial(_plot_odor_traces, FV_data, response_duration, project_folder, all_cells)

    process_map(plot_function, data_iterator, max_workers=num_workers, desc="Plotting Cell Traces: ", total=num_cells)


def _plot_auroc_distribution(auroc_dir, plot_all, cell_data):
    cell_name, cell_df = cell_data

    output_dir = auroc_dir.subdir(cell_name)
    for odor_name, data in cell_df.T.iterrows():
        data = data[cell_name]
        significance_value = data['significance_chart']
        auroc_val = data['auroc_values']
        bounds = data['bounds']
        shuffle = data['shuffles']

        if significance_value != 0 | plot_all:
            upper_bound, lower_bound = bounds

            fig, ax = plt.subplots()
            ax.hist(shuffle, bins=10)
            ax.axvline(x=upper_bound, color='b'), ax.axvline(x=lower_bound, color='b')
            ax.axvline(x=auroc_val, color='r')
            fig.suptitle(f'{cell_name} x {odor_name}')
            plt.close(fig)

            filename = f'{cell_name}-{odor_name}.pdf'
            filepath = output_dir.joinpath(filename)
            fig.savefig(filepath, dpi=300)


def pooled_auroc_distributions(AUROC_data, project_folder: ProjectFolder, all_cells: bool = False, num_workers: int = None):

    output_dir = project_folder.analysis_dir.figures_dir.auroc_dir

    iterator = AUROC_data.T.groupby('Cell')

    plot_function = partial(_plot_auroc_distribution, output_dir, all_cells)

    _ = process_map(plot_function, iterator, desc="Plotting AUROC Distributions: ", max_workers=num_workers)

    # with tqdm(desc=f"Plotting AUROC Distributions: ", total=len(cell_names)) as pbar:
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executer:
    #         for _ in executer.map(plot_function, cell_names):
    #             pbar.update()


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


def plot_trial_variances(input_data, significance_table: np.array, latentCells: bool = False) -> None:
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


def plot_distance_v_correlation(distances, correlations, project_folder):

    # Quick linear regression
    m, b = np.polyfit(distances, correlations, deg=1)
    reg_x = np.arange(np.max(distances))
    reg_y = np.add(np.multiply(m, reg_x), b)

    # Plot regression line, formula, and data
    fig, ax = plt.subplots()
    ax.plot(reg_x, reg_y, color='magenta')
    ax.scatter(distances, correlations)
    plt.xlabel("Euclidian Distance (au)")
    plt.ylabel("Neural Activity Correlation")
    plt.title("Activity vs. Spatial Distance")
    _ = ax.text(0, 1.05, f'y={np.round(m, 4)}x + {np.round(b, 4)}', transform=ax.transAxes)

    save_dir = project_folder.analysis_dir.figures_dir.path
    save_path = save_dir.joinpath('distance_v_correlation.pdf')
    fig.savefig(save_path, dpi=800)


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


def plot_svm_performance(mean_performance, shuffle_mean_performance, CI, shuffle_CI, descriptors, svm_fig_dir):
    CI_min, CI_max = CI
    shuffle_CI_min, shuffle_CI_max = shuffle_CI

    _exp_type, CELL_CLASS, ANALYSIS_VARIABLE, num_cells = descriptors

    # Plot SVM performance vs. Shuffled Data
    fig, ax = plt.subplots()
    x_vals = np.linspace(-2, 3.5, len(mean_performance), endpoint=True) + 0.25
    ax.plot(x_vals, mean_performance, color='#04BBC9', linewidth=3)
    ax.plot(x_vals, shuffle_mean_performance, color='#C500FF', linewidth=1)
    ax.fill_between(x_vals, CI_min, CI_max, alpha=0.5, color='red')
    ax.fill_between(x_vals, shuffle_CI_min, shuffle_CI_max, alpha=0.5, color='green')

    x_vals = np.linspace(-2, 4, 13)
    plt.xticks(x_vals)
    ax.vlines(x=0, ymin=-1, ymax=1, color='r')
    ax.set_ylim([-0.05, 1])
    ax.set_xlim([-2.1, 4.1])

    plt.suptitle(f'{CELL_CLASS} {_exp_type} {ANALYSIS_VARIABLE} SVM Classifier n={num_cells}', fontsize=18,
                 fontweight='bold')
    ax.set_ylabel('Classifier Performance', fontsize=12)
    ax.set_xlabel('Time Relative to Odor Onset (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(svm_fig_dir.joinpath(f'{CELL_CLASS}_{_exp_type}_{ANALYSIS_VARIABLE}_Classifier_Odor.pdf'), dpi=600)

    return fig


def plot_avg_cm(_labels, average_odor_cm, fig_save_path, title_with_index):
    cm = LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:093391-28.4:019C5C-50:66B946-85:D5FF00
        (0.000, (0.035, 0.200, 0.569)),
        (0.284, (0.004, 0.612, 0.361)),
        (0.500, (0.400, 0.725, 0.275)),
        (0.850, (0.835, 1.000, 0.000)),
        (1.000, (0.835, 1.000, 0.000))))

    avg_cm = metrics.ConfusionMatrixDisplay(average_odor_cm, display_labels=_labels, )
    avg_cm.plot(include_values=False, im_kw={'vmin':0, 'vmax':1, 'cmap': cm})


    avg_cm.ax_.set_title(title_with_index)
    avg_cm.ax_.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    avg_cm.figure_.savefig(fig_save_path, dpi=900)

    return avg_cm.figure_, avg_cm.ax_


def plot_all_avg_dff(combined_data_dff, fig_save_path):
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(12,30))

    # cm = LinearSegmentedColormap.from_list('my_gradient', (
    # # Edit this gradient at https://eltos.github.io/gradient/#0:FF0008-7.5:830307-15:000000-25:000000-32.5:0A7000-100:18FF00
    # (0.000, (1.000, 0.000, 0.031)),
    # (0.1, (0.514, 0.012, 0.027)),
    # (0.150, (0.000, 0.000, 0.000)),
    # (0.2250, (0.000, 0.000, 0.000)),
    # (0.325, (0.039, 0.439, 0.000)),
    # (1.000, (0.094, 1.000, 0.000))))

    # New Orleans (VGLUT ID)
    cm = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:5D2A9B-19.3:C18FFD-25:FFFFFF-39:FFFFFF-44.5:58C153-100:008C00
    (0.000, (0.365, 0.165, 0.608)),
    (0.193, (0.757, 0.561, 0.992)),
    (0.250, (1.000, 1.000, 1.000)),
    (0.390, (1.000, 1.000, 1.000)),
    (0.445, (0.345, 0.757, 0.325)),
    (1.000, (0.000, 0.549, 0.000))))

    im = ax.imshow(combined_data_dff, aspect=.25, vmax=0.1, vmin=-0.05, cmap=cm)
    ax.set_xticks(np.arange(combined_data_dff.shape[1]), labels=[])
    ax.set_yticks(np.arange(combined_data_dff.shape[0]), labels=[])
    _ = ax.vlines(x=[3.5, 7.5, 11.5, 15.5, 19.5], ymin=0, ymax=combined_data_dff.shape[0]-1, linestyles='--', color='k')
    _ = fig.colorbar(im, ticks=np.linspace(-0.05, 0.1, 7, endpoint=True), shrink=0.5, aspect=40)
    plt.tight_layout()

    output_path = fig_save_path.joinpath('combined_DFF.pdf')
    fig.savefig(output_path, dpi=600, transparent=True)


def plot_avg_dff(dff_for_bin, cell_names, odor_names, _bin, fig_save_path):
    fig, ax = plt.subplots(figsize=(30, 30))
    cm = LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FF0008-8.8:000000-16:000000-39.4:0E9B00-100:18FF00
    (0.000, (1.000, 0.000, 0.031)),
    (0.088, (0.000, 0.000, 0.000)),
    (0.160, (0.000, 0.000, 0.000)),
    (0.394, (0.055, 0.608, 0.000)),
    (1.000, (0.094, 1.000, 0.000))))


    im = ax.imshow(dff_for_bin, cmap=cm, vmin=-0.57, vmax=3.46)
    ax.set_title(f'{_bin[0] * 100} - {_bin[1] * 100} ms')
    fig.colorbar(im, ax=ax)
    plt.show()
