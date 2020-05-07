import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.training_evaluation import Statistics, _unserialize_dictionary, MultiStatistics
from src.util import load_dictionary
from src.training import _save_results, _process_results
from src.plotting import plot_mean, plot_median

# get imput and output path from args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, required=True, help="the path containing the raw data")
parser.add_argument('-o', '--output_path', type=str, required=True,
                    help="the path into which the processed data will be saved")
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-f', '--fairness', action='store_true')
parser.add_argument('-a', '--analyze', action='store_true')
args = parser.parse_args()


def fairness(runs):
    fairness_rates = np.array([float(run) for run_path, run in runs], dtype=float)
    run_paths = [run_path for run_path, run in runs]
    sorted_fairness_idx = np.argsort(fairness_rates)
    overall_statistics = MultiStatistics.build("log", fairness_rates[sorted_fairness_idx], "Lambda")

    # load statistics and model parameters for all runs of a parameter setting
    for lambda_idx in sorted_fairness_idx:
        statistics_path = os.path.join(run_paths[lambda_idx], "statistics.json")
        serialized_statistics = load_dictionary(statistics_path)
        statistics = Statistics.build_from_serialized_dictionary(serialized_statistics)
        overall_statistics.log_run(statistics)

    return overall_statistics


def no_fairness(runs):
    run_results = []

    # load statistics and model parameters for all runs of a parameter setting
    for run_path, run in runs:
        statistics_path = os.path.join(run_path, "statistics.json")
        serialized_statistics = load_dictionary(statistics_path)
        statistics = Statistics.build_from_serialized_dictionary(serialized_statistics)

        model_parameter_path = os.path.join(run_path, "models.json")
        serialized_model_parameters = load_dictionary(model_parameter_path)
        model_parameters = _unserialize_dictionary(serialized_model_parameters)
        model_parameters = model_parameters['0']['0']

        run_results.append([(statistics, model_parameters)])

    # merge statistics and model parameters
    statistcs_over_runs, model_parameters = _process_results(run_results)

    return statistcs_over_runs[0], model_parameters


cost_directories = [os.path.join(args.input_path, cost) for cost in os.listdir(args.input_path)
                    if os.path.isdir(os.path.join(args.input_path, cost))]

results = []

for cost in cost_directories:
    learning_rates = [(os.path.join(cost, lr), lr) for lr in os.listdir(cost)
                      if os.path.isdir(os.path.join(cost, lr))]

    for learning_rate_path, learning_rate in learning_rates:
        parameter_settings = [(os.path.join(learning_rate_path, parameter_setting), parameter_setting) for
                              parameter_setting in os.listdir(learning_rate_path) if
                              os.path.isdir(os.path.join(learning_rate_path, parameter_setting))]

        for parameter_setting_path, parameter_setting in parameter_settings:
            runs = [(os.path.join(parameter_setting_path, run), run) for run in os.listdir(parameter_setting_path)
                    if os.path.isdir(os.path.join(parameter_setting_path, run))]

            if args.analyze:
                result_row = [learning_rate]
                result_row.append(parameter_setting)
            else:
                result_row = None

            if args.fairness:
                statistics = fairness(runs)
                model_parameters = None

                if result_row is not None:
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.MEDIAN).max())
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.MEDIAN).min())
                    result_row.append((statistics.performance(Statistics.UTILITY, Statistics.THIRD_QUARTILE) -
                                       statistics.performance(Statistics.UTILITY, Statistics.FIRST_QUARTILE)).mean(
                        axis=0))
                    result_columns = ['lr', 'parameters', 'max_median_util', 'min_median_util', 'avg_iqr_util']
                    results.append(result_row)
            else:
                statistics, model_parameters = no_fairness(runs)

                if result_row is not None:
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.MEDIAN)[-1])
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.THIRD_QUARTILE)[-1] -
                                      statistics.performance(Statistics.UTILITY, Statistics.FIRST_QUARTILE)[-1])
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.MEAN)[-1])
                    result_row.append(statistics.performance(Statistics.UTILITY, Statistics.STANDARD_DEVIATION)[-1])
                    result_columns = ['lr', 'parameters', 'final_median', 'final_iqr', 'final_mean', 'final_std_dev']
                    results.append(result_row)

            if args.save:
                output_path = parameter_setting_path.replace(args.input_path, args.output_path)
                # mkdir output path if necessary
                Path(output_path).mkdir(parents=True, exist_ok=True)

                # save the merged statistics, parameters and plot them
                _save_results(output_path, statistics, model_parameters)
                plot_mean(statistics, file_path=os.path.join(output_path, "results_mean.png"))
                plot_median(statistics, file_path=os.path.join(output_path, "results_median.png"))

            print("finished processing {}".format(parameter_setting_path))

if args.analyze:
    results_df = pd.DataFrame(results)
    results_df.columns = result_columns
    results_df.to_csv(os.path.join(args.output_path, 'results.csv'))
    results_df.to_excel(os.path.join(args.output_path, 'results.xls'))
