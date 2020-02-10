import os
import sys
root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import multiprocessing as mp
import numbers
from copy import deepcopy

from src.util import stack

# Result Format
MEAN = "MEAN"
STANDARD_DEVIATION = "STDDEV"
MEDIAN = "MEDIAN"
FIRST_QUARTILE = "FIRST_QUARTILE"
THIRD_QUARTILE = "THIRD_QUARTILE"

def build_result_dictionary(measure):
    return {
        MEAN: np.mean(measure, axis=1),
        MEDIAN: np.median(measure, axis=1),
        STANDARD_DEVIATION: np.std(measure, axis=1),
        FIRST_QUARTILE: np.percentile(measure, q=25, axis=1),
        THIRD_QUARTILE: np.percentile(measure, q=75, axis=1)
    }

class ModelParameters():
    # Result Format
    MEAN = MEAN
    STANDARD_DEVIATION = STANDARD_DEVIATION
    MEDIAN = MEDIAN
    FIRST_QUARTILE = FIRST_QUARTILE
    THIRD_QUARTILE = THIRD_QUARTILE

    def __init__(self, model_parameter_dict):
        self.dict = model_parameter_dict
        self.lagrangians = None

        def get_lambda_over_iterations(iteration_dict):
            lambdas_over_iterations = []
            for _, parameters in iteration_dict.items():
                lambdas_over_iterations.append(parameters["lambda"])
            return np.array(lambdas_over_iterations, dtype=float)
        
        # multiple lambdas
        if len(self.dict) > 1:
            for _, iteration_dict in self.dict.items():
                self.lagrangians = stack(self.lagrangians, get_lambda_over_iterations(iteration_dict).reshape(1, -1), axis=0)
        # single lambda
        else:
            self.lagrangians = get_lambda_over_iterations(self.dict[0]).reshape(1, -1)

    def get_lagrangians(self, result_format):
        return build_result_dictionary(self.lagrangians)[result_format]

    def to_dict(self):
        return deepcopy(self.dict)

class BaseStatistics():
    # Scale Measures:
    X_VALUES = "X_VALUES"
    X_SCALE = "X_SCALE"
    X_NAME = "X_NAME"

    # Performance Measures
    UTILITY = "U"
    NUM_INDIVIDUALS = "A"
    NUM_NEGATIVES = "N"
    NUM_POSITIVES = "P"
    NUM_PRED_NEGATIVES = "NPRED"
    NUM_PRED_POSITIVES = "NPRED"
    TRUE_POSITIVES = "TP"
    TRUE_NEGATIVES = "TN"
    FALSE_POSITIVES = "FP"
    FALSE_NEGATIVES = "FN"
    TRUE_POSITIVE_RATE = "TPR"
    FALSE_POSITIVE_RATE = "FPR"
    TRUE_NEGATIVE_RATE = "TNR"
    FALSE_NEGATIVE_RATE = "FNR"
    POSITIVE_PREDICTIVE_VALUE = "PPV"
    NEGATIVE_PREDICTIVE_VALUE = "NPV"
    FALSE_DISCOVERY_RATE = "FDR" 
    FALSE_OMISSION_RATE = "FOR"
    ACCURACY = "ACC"
    ERROR_RATE = "ERR"
    SELECTION_RATE = "SEL"
    F1 = "F1"

    # Fairness Measures
    DISPARATE_IMPACT = "DI"
    DEMOGRAPHIC_PARITY = "DP"
    EQUALITY_OF_OPPORTUNITY = "EOP"
    
    # Result Format
    MEAN = MEAN
    STANDARD_DEVIATION = STANDARD_DEVIATION
    MEDIAN = MEDIAN
    FIRST_QUARTILE = FIRST_QUARTILE
    THIRD_QUARTILE = THIRD_QUARTILE

    def __init__(self):
        self.results = {}
        self.functions = {
            BaseStatistics.TRUE_POSITIVE_RATE: lambda statistics, prot: statistics.results[prot][BaseStatistics.TRUE_POSITIVES] / statistics.results[prot][BaseStatistics.NUM_POSITIVES],
            BaseStatistics.FALSE_POSITIVE_RATE: lambda statistics, prot: statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][Statistics.NUM_POSITIVES],
            BaseStatistics.TRUE_NEGATIVE_RATE: lambda statistics, prot: statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES],
            BaseStatistics.FALSE_NEGATIVE_RATE: lambda statistics, prot:statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES],
            BaseStatistics.POSITIVE_PREDICTIVE_VALUE: lambda statistics, prot: statistics.results[prot][Statistics.TRUE_POSITIVES] / statistics.results[prot][Statistics.NUM_PRED_POSITIVES],
            BaseStatistics.NEGATIVE_PREDICTIVE_VALUE: lambda statistics, prot: statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_PRED_NEGATIVES],
            BaseStatistics.FALSE_DISCOVERY_RATE: lambda statistics, prot: statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][Statistics.NUM_PRED_POSITIVES],
            BaseStatistics.FALSE_OMISSION_RATE: lambda statistics, prot: statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][Statistics.NUM_PRED_NEGATIVES],
            BaseStatistics.ACCURACY: lambda statistics, prot: (statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][Statistics.TRUE_NEGATIVES]) / statistics.results[prot][Statistics.NUM_INDIVIDUALS],
            BaseStatistics.ERROR_RATE: lambda statistics, prot: 1 - statistics._get_measure(prot, Statistics.ACCURACY),
            BaseStatistics.SELECTION_RATE: lambda statistics, prot: statistics.results[prot][Statistics.NUM_PRED_POSITIVES] / statistics.results[prot][Statistics.NUM_INDIVIDUALS],
            BaseStatistics.F1: lambda statistics, prot: (2 * statistics.results[prot][Statistics.TRUE_POSITIVES]) / (2 * statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][Statistics.FALSE_POSITIVES] + statistics.results[prot][Statistics.FALSE_POSITIVES]),
            BaseStatistics.DISPARATE_IMPACT: lambda statistics, prot: statistics._get_measure("protected", Statistics.SELECTION_RATE) / statistics._get_measure("unprotected", Statistics.SELECTION_RATE),
            BaseStatistics.DEMOGRAPHIC_PARITY: lambda statistics, prot: statistics._get_measure("protected", Statistics.SELECTION_RATE) - statistics._get_measure("unprotected", Statistics.SELECTION_RATE),
            BaseStatistics.EQUALITY_OF_OPPORTUNITY: lambda statistics, prot: statistics._get_measure("protected", Statistics.TRUE_POSITIVE_RATE) - statistics._get_measure("unprotected", Statistics.TRUE_POSITIVE_RATE)
        }

    def _get_measure(self, prot, measure_key):
        if self.results[prot][measure_key] is None:
            measure = self.functions[measure_key](self, prot)
            self.results[prot][measure_key] = measure
        else:
            measure = self.results[prot][measure_key]
        
        return measure


    def performance(self, measure, result_format, protected=None):
        if protected:
            prot = "protected"
        elif protected is None:
            prot = "all"
        else:
            prot = "unprotected"
        
        measure = self._get_measure(prot, measure)
        return build_result_dictionary(measure)[result_format]

    def fairness(self, measure, result_format):
        measure = self._get_measure("all", measure)
        return build_result_dictionary(measure)[result_format]

    def to_dict(self):
        return deepcopy(self.results)

class Statistics(BaseStatistics):
    def __init__(self):
        super(Statistics, self).__init__()

    def merge(self, statistics):
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key != Statistics.X_VALUES and protected_key != Statistics.X_SCALE and protected_key != Statistics.X_NAME:
                for (measure_key, measure_value) in protected_value.items():
                    if measure_key != Statistics.NUM_INDIVIDUALS and measure_key != Statistics.NUM_NEGATIVES and measure_key != Statistics.NUM_POSITIVES: 
                        # if the measure value has not been calculated for either of the merging statistics: reset and let it be recalculated during next call
                        if measure_value is None or self.results[protected_key][measure_key] is None:
                            self.results[protected_key][measure_key] = None 
                        else:
                            self.results[protected_key][measure_key] = stack(self.results[protected_key][measure_key], measure_value, axis=1)
        return self

    @staticmethod
    def calculate_statistics(predictions, observations, protected_attributes, ground_truths, utility_function):
        statistics = Statistics()
        statistics.results[Statistics.X_VALUES] = range(0, predictions.shape[1])
        statistics.results[Statistics.X_SCALE] = "linear"
        statistics.results[Statistics.X_NAME] = "Timestep"

        for prot in ["all", "unprotected", "protected"]:
            if prot == "unprotected":
                filtered_predictions = predictions[(protected_attributes == 0).squeeze(), :, :]
                filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 0], axis=1)
            elif prot == "protected":
                filtered_predictions = predictions[(protected_attributes == 1).squeeze(), :, :]
                filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 1], axis=1)
            else:
                filtered_predictions = predictions
                filtered_ground_truths = ground_truths

            utility_matching_gt = np.repeat(filtered_ground_truths, filtered_predictions.shape[1], axis=1)
            utility_matching_gt = np.expand_dims(utility_matching_gt, axis=2)
            utility_matching_gt = np.repeat(utility_matching_gt, filtered_predictions.shape[2], axis=2)

            # calculate base statistics during creation of statistics object
            statistics.results[prot] = {
                Statistics.UTILITY: np.mean(utility_function(x=observations, decisions=filtered_predictions, y=utility_matching_gt, s=protected_attributes), axis=0),
                Statistics.NUM_INDIVIDUALS: len(filtered_ground_truths),
                Statistics.NUM_NEGATIVES: len(filtered_ground_truths[filtered_ground_truths==0]),
                Statistics.NUM_POSITIVES: len(filtered_ground_truths[filtered_ground_truths==1]),
                Statistics.NUM_PRED_NEGATIVES: np.sum((1 - filtered_predictions), axis=0),
                Statistics.NUM_PRED_POSITIVES: np.sum(filtered_predictions, axis=0),
                Statistics.TRUE_POSITIVES: np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 1), axis=0),
                Statistics.TRUE_NEGATIVES: np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 0), axis=0),
                Statistics.FALSE_POSITIVES: np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 0), axis=0),
                Statistics.FALSE_NEGATIVES: np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 1), axis=0)
            }

            # calculate futher statistics based on the base statistics only on demand to save memory
            statistics.results[prot][Statistics.TRUE_POSITIVE_RATE] = None
            statistics.results[prot][Statistics.FALSE_POSITIVE_RATE] = None
            statistics.results[prot][Statistics.TRUE_NEGATIVE_RATE] = None
            statistics.results[prot][Statistics.FALSE_NEGATIVE_RATE] = None
            statistics.results[prot][Statistics.POSITIVE_PREDICTIVE_VALUE] = None
            statistics.results[prot][Statistics.NEGATIVE_PREDICTIVE_VALUE] = None
            statistics.results[prot][Statistics.FALSE_DISCOVERY_RATE] = None
            statistics.results[prot][Statistics.FALSE_OMISSION_RATE] = None
            statistics.results[prot][Statistics.ACCURACY] = None
            statistics.results[prot][Statistics.ERROR_RATE] = None
            statistics.results[prot][Statistics.SELECTION_RATE] = None
            statistics.results[prot][Statistics.F1] = None
        
        statistics.results["all"][Statistics.DISPARATE_IMPACT] = None
        statistics.results["all"][Statistics.DEMOGRAPHIC_PARITY] = None
        statistics.results["all"][Statistics.EQUALITY_OF_OPPORTUNITY] = None
        
        return statistics

class MultiStatistics(BaseStatistics):
    def __init__(self, x_scale, x_values, x_name):
        super(MultiStatistics, self).__init__()
        self.results[MultiStatistics.X_VALUES] = x_values
        self.results[Statistics.X_SCALE] = x_scale
        self.results[Statistics.X_NAME] = x_name

        for prot in ["all", "unprotected", "protected"]:
            self.results[prot] = {
                MultiStatistics.UTILITY: None,
                MultiStatistics.NUM_INDIVIDUALS: None,
                MultiStatistics.NUM_NEGATIVES: None,
                MultiStatistics.NUM_POSITIVES: None,
                MultiStatistics.NUM_PRED_NEGATIVES: None,
                MultiStatistics.NUM_PRED_POSITIVES: None,
                MultiStatistics.TRUE_POSITIVES: None,
                MultiStatistics.TRUE_NEGATIVES: None,
                MultiStatistics.FALSE_POSITIVES: None,
                MultiStatistics.FALSE_NEGATIVES: None,
                MultiStatistics.TRUE_POSITIVE_RATE: None,
                MultiStatistics.FALSE_POSITIVE_RATE: None,
                MultiStatistics.TRUE_NEGATIVE_RATE: None,
                MultiStatistics.FALSE_NEGATIVE_RATE: None,
                MultiStatistics.POSITIVE_PREDICTIVE_VALUE: None,
                MultiStatistics.NEGATIVE_PREDICTIVE_VALUE: None,
                MultiStatistics.FALSE_DISCOVERY_RATE: None,
                MultiStatistics.FALSE_OMISSION_RATE: None,
                MultiStatistics.ACCURACY: None,
                MultiStatistics.ERROR_RATE: None,
                MultiStatistics.SELECTION_RATE: None,
                MultiStatistics.F1: None
            }
        
        self.results["all"][MultiStatistics.DISPARATE_IMPACT] = None
        self.results["all"][MultiStatistics.DEMOGRAPHIC_PARITY] = None
        self.results["all"][MultiStatistics.EQUALITY_OF_OPPORTUNITY] = None

    def log_run(self, statistics):            
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key != MultiStatistics.X_VALUES and protected_key != MultiStatistics.X_SCALE and protected_key != MultiStatistics.X_NAME:
                for measure_key in protected_value:
                    measure_value = statistics._get_measure(protected_key, measure_key)
                    
                    if isinstance(measure_value, numbers.Number):
                        value = measure_value
                    else:
                        value = measure_value[-1,:].reshape(1, -1)

                    if self.results[protected_key][measure_key] is None:
                        self.results[protected_key][measure_key] = value
                    else:
                        self.results[protected_key][measure_key] = np.vstack((self.results[protected_key][measure_key], value))

                    