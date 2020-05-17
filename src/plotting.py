import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tikzplotlib as tpl
from src.training_evaluation import Statistics, ModelParameters


def _plot_results(plotting_dictionary, file_path):
    x = plotting_dictionary["plot_info"]["x_axis"]
    x_scale = plotting_dictionary["plot_info"]["x_scale"]
    x_label = plotting_dictionary["plot_info"]["x_label"]

    performance_measures = plotting_dictionary["performance_measures"]
    fairness_measures = plotting_dictionary["fairness_measures"]

    num_columns = min(len(performance_measures.items()), 3)
    if num_columns < 3:
        num_columns = min(max(len(fairness_measures.items()), num_columns), 3)

    # get num_rows for maximum of 3 graphs per row
    num_rows = (len(performance_measures.items()) // num_columns) + (
        1 if len(performance_measures.items()) % num_columns > 0 else 0)
    num_rows += (len(fairness_measures.items()) // num_columns) + (
        1 if len(fairness_measures.items()) % num_columns > 0 else 0)

    figure = plt.figure(constrained_layout=True, figsize=(24, 15), dpi=80)
    grid = GridSpec(nrows=num_rows, ncols=num_columns, figure=figure)

    current_row = 0
    current_column = 0

    for measure_dict in [performance_measures, fairness_measures]:
        for y_label, y_dict in measure_dict.items():
            y = y_dict["value"]
            y_uncertainty_lower = y_dict["uncertainty_lower_bound"]
            y_uncertainty_upper = y_dict["uncertainty_upper_bound"]

            axis = figure.add_subplot(grid[current_row, current_column])
            axis.plot(x, y)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_xscale(x_scale)
            axis.fill_between(x,
                              y_uncertainty_lower,
                              y_uncertainty_upper,
                              alpha=0.3,
                              edgecolor='#060080',
                              facecolor='#928CFF')

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
             tex_relative_path_to_data='.')
    plt.close('all')


def plot_median(statistics, performance_measures, fairness_measures, file_path, model_parameters=None):
    plotting_dict = _build_plot_dict(statistics, performance_measures, fairness_measures, Statistics.MEDIAN,
                                     model_parameters)
    _plot_results(plotting_dict, file_path)


def plot_mean(statistics, performance_measures, fairness_measures, file_path, model_parameters=None):
    plotting_dict = _build_plot_dict(statistics, performance_measures, fairness_measures, Statistics.MEAN,
                                     model_parameters)
    _plot_results(plotting_dict, file_path)


def _build_plot_dict(statistics, performance_measures, fairness_measures, result_format, model_parameters=None):
    p_measures = performance_measures if isinstance(performance_measures, list) else [performance_measures]
    f_measures = fairness_measures if isinstance(fairness_measures, list) else [fairness_measures]

    plotting_dict = {
        "plot_info": {
            "x_axis": statistics.results[Statistics.X_VALUES],
            "x_label": statistics.results[Statistics.X_NAME],
            "x_scale": statistics.results[Statistics.X_SCALE]
        },
        "performance_measures": {},
        "fairness_measures": {}
    }

    for p_measure in p_measures:
        measure_label = Statistics.measure_as_human_readable_string(p_measure)

        if result_format == Statistics.MEAN:
            value = statistics.performance(measure_key=p_measure, result_format=Statistics.MEAN)
            measure_stddev = statistics.performance(measure_key=p_measure, result_format=Statistics.STANDARD_DEVIATION)
            lower_bound = value - measure_stddev
            upper_bound = value + measure_stddev
        elif result_format == Statistics.MEDIAN:
            value = statistics.performance(measure_key=p_measure, result_format=Statistics.MEDIAN)
            lower_bound = statistics.performance(measure_key=p_measure, result_format=Statistics.FIRST_QUARTILE)
            upper_bound = statistics.performance(measure_key=p_measure, result_format=Statistics.THIRD_QUARTILE)

        plotting_dict["performance_measures"][measure_label] = {
            "value": value,
            "uncertainty_lower_bound": lower_bound,
            "uncertainty_upper_bound": upper_bound,
        }

    if model_parameters is not None:
        if result_format == Statistics.MEAN:
            plotting_dict["performance_measures"]["Lagrangian Multiplier"] = {
                "value": model_parameters.get_lagrangians(result_format=ModelParameters.MEAN),
                "uncertainty_lower_bound": model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) -
                                           model_parameters.get_lagrangians(
                                               result_format=ModelParameters.STANDARD_DEVIATION),
                "uncertainty_upper_bound": model_parameters.get_lagrangians(result_format=ModelParameters.MEAN) +
                                           model_parameters.get_lagrangians(
                                               result_format=ModelParameters.STANDARD_DEVIATION),
            }
        elif result_format == Statistics.MEDIAN:
            plotting_dict["performance_measures"]["Lagrangian Multiplier"] = {
                "value": model_parameters.get_lagrangians(result_format=ModelParameters.MEDIAN),
                "uncertainty_lower_bound": model_parameters.get_lagrangians(
                    result_format=ModelParameters.FIRST_QUARTILE),
                "uncertainty_upper_bound": model_parameters.get_lagrangians(
                    result_format=ModelParameters.THIRD_QUARTILE),
            }

    for f_measure in f_measures:
        measure_label = Statistics.measure_as_human_readable_string(f_measure)

        if result_format == Statistics.MEAN:
            value = statistics.fairness(measure_key=f_measure, result_format=Statistics.MEAN)
            measure_stddev = statistics.fairness(measure_key=f_measure, result_format=Statistics.STANDARD_DEVIATION)
            lower_bound = value - measure_stddev
            upper_bound = value + measure_stddev
        elif result_format == Statistics.MEDIAN:
            value = statistics.fairness(measure_key=f_measure, result_format=Statistics.MEDIAN)
            lower_bound = statistics.fairness(measure_key=f_measure, result_format=Statistics.FIRST_QUARTILE)
            upper_bound = statistics.fairness(measure_key=f_measure, result_format=Statistics.THIRD_QUARTILE)

        plotting_dict["fairness_measures"][measure_label] = {
            "value": value,
            "uncertainty_lower_bound": lower_bound,
            "uncertainty_upper_bound": upper_bound,
        }

    return plotting_dict
