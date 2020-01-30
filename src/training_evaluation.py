import numpy as np
from src.consequential_learning import consequential_learning
from src.util import stack

import multiprocessing as mp
import numbers
from copy import deepcopy

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

    def __init__(self):
        self.model_parameters = {}
        self.lagrangians_over_iterations = None

    def add(self, iteration, parameters, lagrangian):
        self.model_parameters[iteration] = {
            "model_parameters": parameters,
            "lagrangian_multiplier": lagrangian
        }
        self.lagrangians_over_iterations = stack(self.lagrangians_over_iterations, lagrangian.reshape(-1, 1), axis=1)

    def get_lagrangians(self, result_format):
        return build_result_dictionary(self.lagrangians_over_iterations)[result_format]

    def to_dict(self):
        return deepcopy(self.model_parameters)

class BaseStatistics():
    # Performance Measures
    TIMESTEPS = "T"
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

    def performance(self, measure, result_format, protected=None):
        if protected:
            prot = "protected"
        elif protected is None:
            prot = "all"
        else:
            prot = "unprotected"

        measure = self.results[prot][measure]
        return build_result_dictionary(measure)[result_format]

    def fairness(self, measure, result_format):
        measure = self.results["all"][measure]
        return build_result_dictionary(measure)[result_format]

class Statistics(BaseStatistics):
    def __init__(self):
        super(Statistics, self).__init__()
        self.results = {}

    @staticmethod
    def calculate_statistics(predictions, observations, protected_attributes, ground_truths, utility_function):
        statistics = Statistics()
        statistics.results[Statistics.TIMESTEPS] = predictions.shape[1]
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

            statistics.results[prot][Statistics.TRUE_POSITIVE_RATE] = statistics.results[prot][Statistics.TRUE_POSITIVES] / statistics.results[prot][Statistics.NUM_POSITIVES]
            statistics.results[prot][Statistics.FALSE_POSITIVE_RATE] = statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][Statistics.NUM_POSITIVES]
            statistics.results[prot][Statistics.TRUE_NEGATIVE_RATE] = statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES]
            statistics.results[prot][Statistics.FALSE_NEGATIVE_RATE] = statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][Statistics.NUM_NEGATIVES]
            statistics.results[prot][Statistics.POSITIVE_PREDICTIVE_VALUE] = statistics.results[prot][Statistics.TRUE_POSITIVES] / statistics.results[prot][Statistics.NUM_PRED_POSITIVES]
            statistics.results[prot][Statistics.NEGATIVE_PREDICTIVE_VALUE] = statistics.results[prot][Statistics.TRUE_NEGATIVES] / statistics.results[prot][Statistics.NUM_PRED_NEGATIVES]
            statistics.results[prot][Statistics.FALSE_DISCOVERY_RATE] = statistics.results[prot][Statistics.FALSE_POSITIVES] / statistics.results[prot][Statistics.NUM_PRED_POSITIVES]
            statistics.results[prot][Statistics.FALSE_OMISSION_RATE] = statistics.results[prot][Statistics.FALSE_NEGATIVES] / statistics.results[prot][Statistics.NUM_PRED_NEGATIVES]
            statistics.results[prot][Statistics.ACCURACY] = (statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][Statistics.TRUE_NEGATIVES]) / statistics.results[prot][Statistics.NUM_INDIVIDUALS]
            statistics.results[prot][Statistics.ERROR_RATE] = 1 - statistics.results[prot][Statistics.ACCURACY]
            statistics.results[prot][Statistics.SELECTION_RATE] = statistics.results[prot][Statistics.NUM_PRED_POSITIVES] / statistics.results[prot][Statistics.NUM_INDIVIDUALS]
            statistics.results[prot][Statistics.F1] = (2 * statistics.results[prot][Statistics.TRUE_POSITIVES]) / (2 * statistics.results[prot][Statistics.TRUE_POSITIVES] + statistics.results[prot][Statistics.FALSE_POSITIVES] + statistics.results[prot][Statistics.FALSE_POSITIVES])
        
        statistics.results["all"][Statistics.DISPARATE_IMPACT] = statistics.results["protected"][Statistics.SELECTION_RATE] / statistics.results["unprotected"][Statistics.SELECTION_RATE]
        statistics.results["all"][Statistics.DEMOGRAPHIC_PARITY] = statistics.results["protected"][Statistics.SELECTION_RATE] - statistics.results["unprotected"][Statistics.SELECTION_RATE]
        statistics.results["all"][Statistics.EQUALITY_OF_OPPORTUNITY] = statistics.results["protected"][Statistics.TRUE_POSITIVE_RATE] - statistics.results["unprotected"][Statistics.TRUE_POSITIVE_RATE]
        
        return statistics

class MultipleRunStatistics(BaseStatistics):
    LAMBDAS = "LAMBDAS"

    def __init__(self):
        super(MultipleRunStatistics, self).__init__()
        self.results[MultipleRunStatistics.LAMBDAS] = []
        self.results[MultipleRunStatistics.TIMESTEPS] = None

        for prot in ["all", "unprotected", "protected"]:
            self.results[prot] = {
                MultipleRunStatistics.UTILITY: None,
                MultipleRunStatistics.NUM_INDIVIDUALS: None,
                MultipleRunStatistics.NUM_NEGATIVES: None,
                MultipleRunStatistics.NUM_POSITIVES: None,
                MultipleRunStatistics.NUM_PRED_NEGATIVES: None,
                MultipleRunStatistics.NUM_PRED_POSITIVES: None,
                MultipleRunStatistics.TRUE_POSITIVES: None,
                MultipleRunStatistics.TRUE_NEGATIVES: None,
                MultipleRunStatistics.FALSE_POSITIVES: None,
                MultipleRunStatistics.FALSE_NEGATIVES: None,
                MultipleRunStatistics.TRUE_POSITIVE_RATE: None,
                MultipleRunStatistics.FALSE_POSITIVE_RATE: None,
                MultipleRunStatistics.TRUE_NEGATIVE_RATE: None,
                MultipleRunStatistics.FALSE_NEGATIVE_RATE: None,
                MultipleRunStatistics.POSITIVE_PREDICTIVE_VALUE: None,
                MultipleRunStatistics.NEGATIVE_PREDICTIVE_VALUE: None,
                MultipleRunStatistics.FALSE_DISCOVERY_RATE: None,
                MultipleRunStatistics.FALSE_OMISSION_RATE: None,
                MultipleRunStatistics.ACCURACY: None,
                MultipleRunStatistics.ERROR_RATE: None,
                MultipleRunStatistics.SELECTION_RATE: None,
                MultipleRunStatistics.F1: None
            }
        
        self.results["all"][MultipleRunStatistics.DISPARATE_IMPACT] = None
        self.results["all"][MultipleRunStatistics.DEMOGRAPHIC_PARITY] = None
        self.results["all"][MultipleRunStatistics.EQUALITY_OF_OPPORTUNITY] = None

    def log_run(self, statistics, fairness_rate):
        self.results[MultipleRunStatistics.LAMBDAS].append(fairness_rate)
            
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key == MultipleRunStatistics.TIMESTEPS:
                if self.results[Statistics.TIMESTEPS] is None:
                    self.results[MultipleRunStatistics.TIMESTEPS] = protected_value
            else:
                for (measure_key, measure_value) in protected_value.items():
                    if isinstance(measure_value, numbers.Number):
                        value = measure_value
                    else:
                        value = measure_value[-1,:].reshape(1, -1)

                    if self.results[protected_key][measure_key] is None:
                        self.results[protected_key][measure_key] = value
                    else:
                        self.results[protected_key][measure_key] = np.vstack((self.results[protected_key][measure_key], value))

                    