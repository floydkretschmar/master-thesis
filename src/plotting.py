import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

import tikzplotlib as tpl
from src.training_evaluation import MEAN, MEDIAN

COLORS = [
    {"edgecolor": '#060080', "facecolor": '#060080'},
    {"edgecolor": '#558f47', "facecolor": '#558f47'}
]


class Plot():
    def __init__(self, x_axis, x_label, x_scale, y_label, *statistics):
        self._statistics = statistics
        self._x_axis = x_axis
        self._x_label = x_label
        self._x_scale = x_scale
        self._y_label = y_label

    @property
    def statistics(self):
        return self._statistics

    @property
    def x_axis(self):
        return self._x_axis

    @property
    def x_label(self):
        return self._x_label

    @property
    def x_scale(self):
        return self._x_scale

    @property
    def y_label(self):
        return self._y_label


def _plot_results(plotting_dictionary, file_path, figsize, plots_per_row, y_lim):
    # x = plotting_dictionary["plot_info"]["x_axis"]
    # x_scale = plotting_dictionary["plot_info"]["x_scale"]
    # x_label = plotting_dictionary["plot_info"]["x_label"]

    performance_measures = plotting_dictionary["performance"]
    fairness_measures = plotting_dictionary["fairness"]

    num_columns = min(len(performance_measures.items()), plots_per_row)
    if num_columns < plots_per_row:
        num_columns = min(max(len(fairness_measures.items()), num_columns), plots_per_row)

    # get num_rows for maximum of plots_per_row graphs per row
    num_rows = (len(performance_measures.items()) // num_columns) + (
        1 if len(performance_measures.items()) % num_columns > 0 else 0)
    num_rows += (len(fairness_measures.items()) // num_columns) + (
        1 if len(fairness_measures.items()) % num_columns > 0 else 0)

    if figsize is None:
        figure = plt.figure(constrained_layout=True)
    else:
        figure = plt.figure(constrained_layout=True, figsize=figsize, dpi=80)

    grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)

    current_row = 0
    current_column = 0

    for measure_dict in [performance_measures, fairness_measures]:
        for y_label, y_dict in measure_dict.items():
            x = y_dict["x_axis"]
            x_label = y_dict["x_label"]
            x_scale = y_dict["x_scale"]
            axis = figure.add_subplot(grid[current_row, current_column])
            axis.title.set_text(y_label)
            axis.set_xlabel(x_label)
            axis.set_xscale(x_scale)

            for i, y_value in enumerate(y_dict["y_values"]):
                y = y_value["value"]
                y_uncertainty_lower = y_value["uncertainty_lower_bound"]
                y_uncertainty_upper = y_value["uncertainty_upper_bound"]

                axis.plot(x, y, COLORS[i]["edgecolor"])
                axis.fill_between(x,
                                  y_uncertainty_lower,
                                  y_uncertainty_upper,
                                  alpha=0.3,
                                  edgecolor=COLORS[i]["edgecolor"],
                                  color=COLORS[i]["facecolor"])
                if y_lim is not None:
                    axis.set_ylim(y_lim[0], y_lim[1])

            if current_column < num_columns - 1:
                current_column += 1
            else:
                current_column = 0
                current_row += 1

        if current_column > 0:
            current_row += 1
            current_column = 0

    plt.savefig(file_path)
    tpl.save(file_path.replace(".png", ".tex"),
             figure=figure,
             axis_width='\\figwidth',
             axis_height='\\figheight',
             tex_relative_path_to_data='.',
             extra_groupstyle_parameters={"horizontal sep=1.2cm"},
             extra_axis_parameters={
                 "scaled y ticks = false, \n yticklabel style = {/pgf/number format/fixed, /pgf/number format/precision=3}"})
    plt.close('all')


def plot_median(performance_plots,
                fairness_plots,
                file_path,
                figsize=None,
                plots_per_row=4,
                y_lim=None):
    plotting_dict = _build_plot_dict(performance_plots, fairness_plots, MEDIAN)
    _plot_results(plotting_dict, file_path, figsize, plots_per_row, y_lim)


def plot_mean(performance_plots,
              fairness_plots,
              file_path,
              figsize=None,
              plots_per_row=4,
              y_lim=None):
    plotting_dict = _build_plot_dict(performance_plots, fairness_plots, MEAN)
    _plot_results(plotting_dict, file_path, figsize, plots_per_row, y_lim)


def _build_plot_dict(performance_plots, fairness_plots, result_format):
    plotting_dict = {
        "performance": {},
        "fairness": {}
    }

    for plots, plot_type in [(performance_plots, "performance"), (fairness_plots, "fairness")]:
        for plot in plots:
            plotting_dict[plot_type][plot.y_label] = {
                "x_axis": plot.x_axis,
                "x_label": plot.x_label,
                "x_scale": plot.x_scale,
                "y_values": []
            }
            for num_statistic, statistic in enumerate(plot.statistics):
                if result_format == MEAN:
                    value = statistic.mean()
                    measure_stddev = statistic.standard_deviation()
                    lower_bound = value - measure_stddev
                    upper_bound = value + measure_stddev
                elif result_format == MEDIAN:
                    value = statistic.median()
                    lower_bound = statistic.first_quartile()
                    upper_bound = statistic.third_quartile()
                plotting_dict[plot_type][plot.y_label]["y_values"].append({
                    "value": value,
                    "uncertainty_lower_bound": lower_bound,
                    "uncertainty_upper_bound": upper_bound
                })

    return plotting_dict


def plot_epoch_statistics(path, fairness, lambdas, gradients, utils):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.plot(range(0, len(fairness)), fairness, 'g')
    green_patch = mpatches.Patch(color='green', label='Fairness Function (Test Set)')
    ax1.legend(handles=[green_patch])

    ax2.plot(range(0, len(lambdas)), lambdas, 'b')
    black_patch = mpatches.Patch(color='blue', label='Lambda')
    ax2.legend(handles=[black_patch])

    ax3.plot(range(0, len(gradients)), gradients, 'r')
    red_patch = mpatches.Patch(color='red', label='Gradient of lambda')
    ax3.legend(handles=[red_patch])

    ax4.plot(range(0, len(utils)), utils, 'm')
    orange_patch = mpatches.Patch(color='magenta', label='Utility')
    ax4.legend(handles=[orange_patch])

    plt.savefig(path)
    plt.close('all')
