import os
import sys

root_path = os.path.abspath(os.path.join('.'))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import numbers
from copy import deepcopy

from src.util import stack, serialize_dictionary, mean

# Result Format
MEAN = "MEAN"
STANDARD_DEVIATION = "STDDEV"
MEDIAN = "MEDIAN"
FIRST_QUARTILE = "FIRST_QUARTILE"
THIRD_QUARTILE = "THIRD_QUARTILE"

# Performance Measures
NUM_INDIVIDUALS = "A"
NUM_NEGATIVES = "N"
NUM_POSITIVES = "P"
NUM_PRED_NEGATIVES = "NPRED"
NUM_PRED_POSITIVES = "PPRED"
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

# Special Performance Measures:
UTILITY = "U"
COVARIANCE_OF_DECISION_DP = "COV_DP"


def true_positive_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][TRUE_POSITIVES] / statistics.results[protected_string][NUM_POSITIVES]


def false_positive_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][FALSE_POSITIVES] / statistics.results[protected_string][
        NUM_POSITIVES]


def true_negative_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][TRUE_NEGATIVES] / statistics.results[protected_string][
        NUM_NEGATIVES]


def false_negative_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][FALSE_NEGATIVES] / statistics.results[protected_string][
        NUM_NEGATIVES]


def positive_predictive_value(statistics, protected_string, protected):
    return statistics.results[protected_string][TRUE_POSITIVES] / statistics.results[protected_string][
        NUM_PRED_POSITIVES]


def negative_predictive_value(statistics, protected_string, protected):
    return statistics.results[protected_string][TRUE_NEGATIVES] / statistics.results[protected_string][
        NUM_PRED_NEGATIVES]


def false_discovery_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][FALSE_POSITIVES] / statistics.results[protected_string][
        NUM_PRED_POSITIVES]


def false_omission_rate(statistics, protected_string, protected):
    return statistics.results[protected_string][FALSE_NEGATIVES] / statistics.results[protected_string][
        NUM_PRED_NEGATIVES]


def accuracy(statistics, protected_string, protected):
    return (statistics.results[protected_string][TRUE_POSITIVES] + statistics.results[protected_string][
        TRUE_NEGATIVES]) / \
           statistics.results[protected_string][NUM_INDIVIDUALS]


def error_rate(statistics, protected_string, protected):
    return 1 - statistics.accuracy(protected)._measure


def selection_rate(statistics, protected_string, protected):
    pred_pos = statistics.results[protected_string][NUM_PRED_POSITIVES]
    num_ind = statistics.results[protected_string][NUM_INDIVIDUALS]

    return pred_pos / num_ind


def f1(statistics, protected_string, protected):
    return 2 * (statistics.true_positive_rate(protected)._measure * statistics.positive_predictive_value(
        protected)._measure) / (statistics.true_positive_rate(protected)._measure + statistics.positive_predictive_value(
        protected)._measure)


def disparate_impact(statistics, protected_string, protected):
    return statistics.selection_rate(protected=True)._measure / statistics.selection_rate(protected=False)._measure


def demographic_parity(statistics, protected_string, protected):
    sel_prot = statistics.selection_rate(protected=True)._measure
    sel_unprot = statistics.selection_rate(protected=False)._measure
    dp = sel_prot - sel_unprot
    return dp


def equality_of_opportunity(statistics, protected_string, protected):
    return statistics.true_positive_rate(protected=True)._measure - statistics.true_positive_rate(
        protected=False)._measure


def _unserialize_value(value, recursive_func=None):
    if isinstance(value, dict) and recursive_func:
        return recursive_func(value)
    elif isinstance(value, list):
        return np.array(value)
    elif value == "NoneType":
        return None
    elif isinstance(value, numbers.Number):
        return np.array(value).reshape(1, 1)
    else:
        return value


class ModelParameters():
    # Result Format
    MEAN = MEAN
    STANDARD_DEVIATION = STANDARD_DEVIATION
    MEDIAN = MEDIAN
    FIRST_QUARTILE = FIRST_QUARTILE
    THIRD_QUARTILE = THIRD_QUARTILE

    def __init__(self, model_parameter_dict):
        self.dict = deepcopy(model_parameter_dict)

        lambdas = np.array(self.dict["lambdas"], dtype=float)
        self.dict["lambdas"] = {}
        if len(lambdas.shape) > 1:
            for l in range(lambdas.shape[1]):
                self.dict["lambdas"][l] = lambdas[:, l].reshape(-1, 1)
        else:
            self.dict["lambdas"][0] = lambdas.reshape(-1, 1)

    @staticmethod
    def _unserialize_dictionary(dictionary):
        unserialized_dict = deepcopy(dictionary)
        unserialized_dict["lambdas"] = _unserialize_value(unserialized_dict["lambdas"])
        unserialized_dict["model_parameters"] = unserialized_dict["model_parameters"]

        return unserialized_dict

    @staticmethod
    def build_from_serialized_dictionary(serialized_dict):
        results = ModelParameters._unserialize_dictionary(serialized_dict)
        return ModelParameters(results)

    def merge(self, model_parameters):
        self.dict["model_parameters"].append(model_parameters.dict["model_parameters"])

        for l_idx in self.dict["lambdas"].keys():
            self.dict["lambdas"][l_idx] = stack(self.dict["lambdas"][l_idx],
                                         model_parameters.dict["lambdas"][l_idx].reshape(-1, 1),
                                         axis=1)

    def get_lagrangians(self):
        lagrangians = []
        for l_idx in self.dict["lambdas"].keys():
            lagrangians.append(Statistic(self.dict["lambdas"][l_idx], "Lagrangian Multiplier"))
        return lagrangians

    def to_dict(self):
        return deepcopy(self.dict)


class Statistic:
    def __init__(self, measure, name):
        self._measure = deepcopy(measure)
        self._name = name

    @property
    def length(self):
        return self._measure.shape[0]

    @property
    def name(self):
        return self._name

    def mean(self):
        return mean(self._measure, axis=1)

    def median(self):
        return np.median(self._measure, axis=1)

    def standard_deviation(self):
        return np.std(self._measure, axis=1)

    def first_quartile(self):
        return np.percentile(self._measure, q=25, axis=1)

    def third_quartile(self):
        return np.percentile(self._measure, q=75, axis=1)


class Statistics:
    def __init__(self, predictions, protected_attributes, ground_truths, additonal_measures=None):
        self._functions = {
            TRUE_POSITIVE_RATE: true_positive_rate,
            FALSE_POSITIVE_RATE: false_positive_rate,
            TRUE_NEGATIVE_RATE: true_negative_rate,
            FALSE_NEGATIVE_RATE: false_negative_rate,
            POSITIVE_PREDICTIVE_VALUE: positive_predictive_value,
            NEGATIVE_PREDICTIVE_VALUE: negative_predictive_value,
            FALSE_DISCOVERY_RATE: false_discovery_rate,
            FALSE_OMISSION_RATE: false_omission_rate,
            ACCURACY: accuracy,
            ERROR_RATE: error_rate,
            SELECTION_RATE: selection_rate,
            F1: f1,
            DISPARATE_IMPACT: disparate_impact,
            DEMOGRAPHIC_PARITY: demographic_parity,
            EQUALITY_OF_OPPORTUNITY: equality_of_opportunity
        }
        self.results = {}

        if predictions is not None and protected_attributes is not None and ground_truths is not None:
            for prot in ["all", "unprotected", "protected"]:
                if prot == "unprotected":
                    filtered_protected_attributes = protected_attributes[protected_attributes == 0]
                    filtered_predictions = predictions[(protected_attributes == 0).squeeze(), :]
                    filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 0], axis=1)
                elif prot == "protected":
                    filtered_protected_attributes = protected_attributes[protected_attributes == 1]
                    filtered_predictions = predictions[(protected_attributes == 1).squeeze(), :]
                    filtered_ground_truths = np.expand_dims(ground_truths[protected_attributes == 1], axis=1)
                else:
                    filtered_protected_attributes = protected_attributes
                    filtered_predictions = predictions
                    filtered_ground_truths = ground_truths

                utility_matching_gt = np.repeat(filtered_ground_truths, filtered_predictions.shape[1], axis=1)

                # calculate base statistics during creation of statistics object
                self.results[prot] = {
                    NUM_INDIVIDUALS: np.array(len(filtered_ground_truths)),
                    NUM_NEGATIVES: np.array(len(filtered_ground_truths[filtered_ground_truths == 0])),
                    NUM_POSITIVES: np.array(len(filtered_ground_truths[filtered_ground_truths == 1])),
                    NUM_PRED_NEGATIVES: np.expand_dims(np.sum((1 - filtered_predictions), axis=0), axis=1),
                    NUM_PRED_POSITIVES: np.expand_dims(np.sum(filtered_predictions, axis=0), axis=1),
                    TRUE_POSITIVES: np.expand_dims(
                        np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 1), axis=0), axis=1),
                    TRUE_NEGATIVES: np.expand_dims(
                        np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 0), axis=0), axis=1),
                    FALSE_POSITIVES: np.expand_dims(
                        np.sum(np.logical_and(filtered_predictions == 1, utility_matching_gt == 0), axis=0), axis=1),
                    FALSE_NEGATIVES: np.expand_dims(
                        np.sum(np.logical_and(filtered_predictions == 0, utility_matching_gt == 1), axis=0), axis=1)
                }

                # calculate futher statistics based on the base statistics only on demand to save memory
                self.results[prot][TRUE_POSITIVE_RATE] = None
                self.results[prot][FALSE_POSITIVE_RATE] = None
                self.results[prot][TRUE_NEGATIVE_RATE] = None
                self.results[prot][FALSE_NEGATIVE_RATE] = None
                self.results[prot][POSITIVE_PREDICTIVE_VALUE] = None
                self.results[prot][NEGATIVE_PREDICTIVE_VALUE] = None
                self.results[prot][FALSE_DISCOVERY_RATE] = None
                self.results[prot][FALSE_OMISSION_RATE] = None
                self.results[prot][ACCURACY] = None
                self.results[prot][ERROR_RATE] = None
                self.results[prot][SELECTION_RATE] = None
                self.results[prot][F1] = None

                if additonal_measures is not None:
                    for measure_key, measure_item in additonal_measures.items():
                        if measure_item["detailed"] or not measure_item["detailed"] and prot == "all":
                            measure_function = measure_item["measure_function"]

                            def _func(column):
                                measure = measure_function(s=filtered_protected_attributes.squeeze(),
                                                           y=filtered_ground_truths.squeeze(),
                                                           decisions=column)
                                return measure

                            self.results[prot][measure_key] = np.apply_along_axis(_func,
                                                                                  0,
                                                                                  filtered_predictions).reshape(-1, 1)

            self.results["all"][DISPARATE_IMPACT] = None
            self.results["all"][DEMOGRAPHIC_PARITY] = None
            self.results["all"][EQUALITY_OF_OPPORTUNITY] = None

    @staticmethod
    def _unserialize_dictionary(dictionary):
        unserialized_dict = deepcopy(dictionary)
        for key, value in unserialized_dict.items():
            unserialized_dict[key] = _unserialize_value(value, Statistics._unserialize_dictionary)

        return unserialized_dict

    @staticmethod
    def build_from_serialized_dictionary(serialized_dict):
        results = Statistics._unserialize_dictionary(serialized_dict)
        stats = Statistics(None, None, None, None)
        stats.results = results
        return stats

    def get_additonal_measure(self, measure_key, measure_name, protected=None):
        measure = self._get_measure(protected, measure_key, lambda: None)

        if measure is None:
            if protected is not None:
                if protected:
                    group = "the protected individuals in the dataset"
                else:
                    group = "the unprotected individuals in the dataset"
            else:
                group = "the entire dataset"

            raise TypeError(
                "The measure {} has not been calculated for {} on this particular instance of the Statistics class. "
                "Please make sure you have specified the additional measure in your training parameters.".format(
                    measure_key,
                    group
                ))
        else:
            return Statistic(measure, measure_name)

    def _get_measure(self, protected, measure_key, function=None):
        if protected:
            prot = "protected"
        elif protected is None:
            prot = "all"
        else:
            prot = "unprotected"

        if self.results[prot][measure_key] is None:
            measure = function(self, prot, protected)
            self.results[prot][measure_key] = measure
        else:
            measure = self.results[prot][measure_key]

        return measure

    def to_dict(self):
        return serialize_dictionary(self.results)

    def merge(self, statistics):
        for protected in ["all", "unprotected", "protected"]:
            for measure in statistics.results[protected].keys():
                self_measure = deepcopy(self.results[protected][measure])
                other_measure = deepcopy(statistics.results[protected][measure])
                if self_measure is None or other_measure is None:
                    self.results[protected][measure] = None
                else:
                    self.results[protected][measure] = stack(self_measure, other_measure, axis=1)

    def number_of_samples(self, protected=None):
        return self._get_measure(protected, NUM_INDIVIDUALS)

    def number_of_negative_samples(self, protected=None):
        return self._get_measure(protected, NUM_NEGATIVES)

    def number_of_positive_samples(self, protected=None):
        return self._get_measure(protected, NUM_POSITIVES)

    def number_of_negative_predictions(self, protected=None):
        return self._get_measure(protected, NUM_PRED_NEGATIVES)

    def number_of_positive_predictions(self, protected=None):
        return self._get_measure(protected, NUM_PRED_POSITIVES)

    def number_of_true_positives(self, protected=None):
        return self._get_measure(protected, TRUE_POSITIVES)

    def number_of_true_negatives(self, protected=None):
        return self._get_measure(protected, TRUE_NEGATIVES)

    def number_of_false_positives(self, protected=None):
        return self._get_measure(protected, FALSE_POSITIVES)

    def number_of_false_negatives(self, protected=None):
        return self._get_measure(protected, FALSE_NEGATIVES)

    def true_positive_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    TRUE_POSITIVE_RATE,
                                    self._functions[TRUE_POSITIVE_RATE])
        return Statistic(measure, "True Positive Rate")

    def false_positive_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    FALSE_POSITIVE_RATE,
                                    self._functions[FALSE_POSITIVE_RATE])
        return Statistic(measure, "False Positive Rate")

    def true_negative_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    TRUE_NEGATIVE_RATE,
                                    self._functions[TRUE_NEGATIVE_RATE])
        return Statistic(measure, "True Negative Rate")

    def false_negative_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    FALSE_NEGATIVE_RATE,
                                    self._functions[FALSE_NEGATIVE_RATE])
        return Statistic(measure, "False Negative Rate")

    def positive_predictive_value(self, protected=None):
        measure = self._get_measure(protected,
                                    POSITIVE_PREDICTIVE_VALUE,
                                    self._functions[POSITIVE_PREDICTIVE_VALUE])
        return Statistic(measure, "Positive Predictive Value")

    def negative_predictive_value(self, protected=None):
        measure = self._get_measure(protected,
                                    NEGATIVE_PREDICTIVE_VALUE,
                                    self._functions[NEGATIVE_PREDICTIVE_VALUE])
        return Statistic(measure, "Negative Predictive Value")

    def false_discovery_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    FALSE_DISCOVERY_RATE,
                                    self._functions[FALSE_DISCOVERY_RATE])
        return Statistic(measure, "False Discovery Rate")

    def false_omission_rate(self, protected=None):
        measure = self._get_measure(protected,
                                    FALSE_OMISSION_RATE,
                                    self._functions[FALSE_OMISSION_RATE])
        return Statistic(measure, "False Omission Rate")

    def accuracy(self, protected=None):
        measure = self._get_measure(protected,
                                    ACCURACY,
                                    self._functions[ACCURACY])
        return Statistic(measure, "Accuracy")

    def error_rate(self, protected):
        measure = self._get_measure(protected,
                                    ERROR_RATE,
                                    self._functions[ERROR_RATE])
        return Statistic(measure, "Error Rate")

    def selection_rate(self, protected):
        measure = self._get_measure(protected,
                                    SELECTION_RATE,
                                    self._functions[SELECTION_RATE])
        return Statistic(measure, "Selection Rate")

    def f1_score(self, protected):
        measure = self._get_measure(protected, F1, self._functions[F1])
        return Statistic(measure, "F1 Score")

    def disparate_impact(self):
        measure = self._get_measure(None, DISPARATE_IMPACT, self._functions[DISPARATE_IMPACT])
        return Statistic(measure, "Disparate Impact")

    def demographic_parity(self):
        measure = self._get_measure(None, DEMOGRAPHIC_PARITY, self._functions[DEMOGRAPHIC_PARITY])
        return Statistic(measure, "Demographic Parity")

    def equality_of_opportunity(self):
        measure = self._get_measure(None, EQUALITY_OF_OPPORTUNITY, self._functions[EQUALITY_OF_OPPORTUNITY])
        return Statistic(measure, "Equality of Opportunity")


class MultiStatistics(Statistics):
    def __init__(self):
        super(MultiStatistics, self).__init__(None, None, None, None)
        for prot in ["all", "unprotected", "protected"]:
            self.results[prot] = {
                UTILITY: None,
                NUM_INDIVIDUALS: None,
                NUM_NEGATIVES: None,
                NUM_POSITIVES: None,
                NUM_PRED_NEGATIVES: None,
                NUM_PRED_POSITIVES: None,
                TRUE_POSITIVES: None,
                TRUE_NEGATIVES: None,
                FALSE_POSITIVES: None,
                FALSE_NEGATIVES: None,
                TRUE_POSITIVE_RATE: None,
                FALSE_POSITIVE_RATE: None,
                TRUE_NEGATIVE_RATE: None,
                FALSE_NEGATIVE_RATE: None,
                POSITIVE_PREDICTIVE_VALUE: None,
                NEGATIVE_PREDICTIVE_VALUE: None,
                FALSE_DISCOVERY_RATE: None,
                FALSE_OMISSION_RATE: None,
                ACCURACY: None,
                ERROR_RATE: None,
                SELECTION_RATE: None,
                F1: None
            }

        self.results["all"][DISPARATE_IMPACT] = None
        self.results["all"][DEMOGRAPHIC_PARITY] = None
        self.results["all"][EQUALITY_OF_OPPORTUNITY] = None

    def log_run(self, statistics):
        for (protected_key, protected_value) in statistics.results.items():
            for measure_key in protected_value:
                if protected_key == "all":
                    protected = None
                elif protected_key == "protected":
                    protected = True
                else:
                    protected = False

                measure_value = statistics._get_measure(protected,
                                                        measure_key,
                                                        self._functions[measure_key] if measure_key in
                                                                                        self._functions else None)

                if isinstance(measure_value, numbers.Number):
                    value = measure_value
                else:
                    value = measure_value[-1, :].reshape(1, -1)

                if measure_key not in self.results[protected_key] or self.results[protected_key][
                    measure_key] is None:
                    self.results[protected_key][measure_key] = value
                else:
                    self.results[protected_key][measure_key] = np.vstack(
                        (self.results[protected_key][measure_key], value))
