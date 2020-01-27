import numpy as np
from src.consequential_learning import consequential_learning
import multiprocessing as mp
import numbers


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
    EXACT_VALUE = "EXACT"
    MEAN = "MEAN"
    STANDARD_DEVIATION = "STDDEV"
    MEDIAN = "MEDIAN"
    FIRST_QUARTILE = "FIRST_QUARTILE"
    THIRD_QUARTILE = "THIRD_QUARTILE"

    @staticmethod
    def _build_return_dictionary(measure):
        return {
            BaseStatistics.MEAN: np.mean(measure, axis=1),
            BaseStatistics.MEDIAN: np.median(measure, axis=1),
            BaseStatistics.STANDARD_DEVIATION: np.std(measure, axis=1),
            BaseStatistics.FIRST_QUARTILE: np.percentile(measure, q=25, axis=1),
            BaseStatistics.THIRD_QUARTILE: np.percentile(measure, q=75, axis=1)
        }

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
        return BaseStatistics._build_return_dictionary(measure)[result_format]

    def fairness(self, measure, result_format):
        measure = self.results["all"][measure]
        return BaseStatistics._build_return_dictionary(measure)[result_format]

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

class LambdaStatistics(BaseStatistics):
    LAMBDAS = "LAMBDAS"

    def __init__(self):
        super(LambdaStatistics, self).__init__()
        self.results[LambdaStatistics.LAMBDAS] = []
        self.results[LambdaStatistics.TIMESTEPS] = None

        for prot in ["all", "unprotected", "protected"]:
            self.results[prot] = {
                LambdaStatistics.UTILITY: None,
                LambdaStatistics.NUM_INDIVIDUALS: None,
                LambdaStatistics.NUM_NEGATIVES: None,
                LambdaStatistics.NUM_POSITIVES: None,
                LambdaStatistics.NUM_PRED_NEGATIVES: None,
                LambdaStatistics.NUM_PRED_POSITIVES: None,
                LambdaStatistics.TRUE_POSITIVES: None,
                LambdaStatistics.TRUE_NEGATIVES: None,
                LambdaStatistics.FALSE_POSITIVES: None,
                LambdaStatistics.FALSE_NEGATIVES: None,
                LambdaStatistics.TRUE_POSITIVE_RATE: None,
                LambdaStatistics.FALSE_POSITIVE_RATE: None,
                LambdaStatistics.TRUE_NEGATIVE_RATE: None,
                LambdaStatistics.FALSE_NEGATIVE_RATE: None,
                LambdaStatistics.POSITIVE_PREDICTIVE_VALUE: None,
                LambdaStatistics.NEGATIVE_PREDICTIVE_VALUE: None,
                LambdaStatistics.FALSE_DISCOVERY_RATE: None,
                LambdaStatistics.FALSE_OMISSION_RATE: None,
                LambdaStatistics.ACCURACY: None,
                LambdaStatistics.ERROR_RATE: None,
                LambdaStatistics.SELECTION_RATE: None,
                LambdaStatistics.F1: None
            }
        
        self.results["all"][LambdaStatistics.DISPARATE_IMPACT] = None
        self.results["all"][LambdaStatistics.DEMOGRAPHIC_PARITY] = None
        self.results["all"][LambdaStatistics.EQUALITY_OF_OPPORTUNITY] = None

    def log_run(self, statistics, fairness_rate):
        self.results[LambdaStatistics.LAMBDAS].append(fairness_rate)
            
        for (protected_key, protected_value) in statistics.results.items():
            if protected_key == LambdaStatistics.TIMESTEPS:
                if self.results[Statistics.TIMESTEPS] is None:
                    self.results[LambdaStatistics.TIMESTEPS] = protected_value
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

                    