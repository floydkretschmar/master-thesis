import argparse
import json
import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.training_evaluation import Statistics, _unserialize_dictionary
from src.util import load_dictionary
from src.training import _save_results, _process_results
from src.plotting import plot_mean, plot_median

# get imput and output path from args
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', type=str, required=True, help="the path containing the raw data")
parser.add_argument('-o', '--output_path', type=str, required=True,
                    help="the path into which the processed data will be saved")
args = parser.parse_args()

cost_directories = [os.path.join(args.input_path, cost) for cost in os.listdir(args.input_path)
                    if os.path.isdir(os.path.join(args.input_path, cost))]

best_parameter_settings = []

for cost in cost_directories:
    learning_rates = [os.path.join(cost, lr) for lr in os.listdir(cost)
                      if os.path.isdir(os.path.join(cost, lr))]

    for learning_rate in learning_rates:
        parameter_settings = [os.path.join(learning_rate, parameter_setting) for parameter_setting in
                              os.listdir(learning_rate)
                              if os.path.isdir(os.path.join(learning_rate, parameter_setting))]

        best_utility = float('-inf')
        best_utility_path = ""

        for parameter_setting in parameter_settings:
            runs = [os.path.join(parameter_setting, run) for run in os.listdir(parameter_setting)
                    if os.path.isdir(os.path.join(parameter_setting, run))]

            run_results = []

            # load statistics and model parameters for all runs of a parameter setting
            for run in runs:
                statistics_path = os.path.join(run, "statistics.json")
                serialized_statistics = load_dictionary(statistics_path)
                statistics = Statistics.build_from_serialized_dictionary(serialized_statistics)

                # if the x values = range then we are merging old buggy results > hardcode x range
                if statistics.results[Statistics.X_VALUES] == "range":
                    statistics.results[Statistics.X_VALUES] = list(
                        range(0, statistics.results["all"][Statistics.UTILITY].shape[0]))

                model_parameter_path = os.path.join(run, "models.json")
                serialized_model_parameters = load_dictionary(model_parameter_path)
                model_parameters = _unserialize_dictionary(serialized_model_parameters)
                model_parameters = model_parameters['0']['0']

                run_results.append([(statistics, model_parameters)])

            # merge statistics and model parameters
            statistcs_over_runs, model_parameters = _process_results(run_results)
            output_path = parameter_setting.replace(args.input_path, args.output_path)

            # mkdir output path if necessary
            Path(output_path).mkdir(parents=True, exist_ok=True)

            statistics = statistcs_over_runs[0]

            # save the merged statistics, parameters and plot them
            _save_results(output_path, statistics, model_parameters)
            plot_mean(statistics, file_path=os.path.join(output_path, "results_mean.png"))
            plot_median(statistics, file_path=os.path.join(output_path, "results_median.png"))

            # update the best median utility parameter setting for the current learning rate
            current_utility = statistics.performance(Statistics.UTILITY, Statistics.MEDIAN)[-1]
            if current_utility > best_utility:
                best_utility = current_utility
                best_utility_path = output_path

        best_parameter_settings.append((best_utility_path, best_utility))

res = os.path.join(args.output_path, "best_results.txt")
with open(res, 'w') as file:
    json.dump(best_parameter_settings, file)
