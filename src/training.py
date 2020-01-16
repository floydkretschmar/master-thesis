import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import multiprocessing as mp

from src.consequential_learning import consequential_learning

result_list = []
def result_worker(result):
    global result_list
    result_list.append(result)

def train_single(training_parameters):
    np.random.seed()
    for u, bd in consequential_learning(**training_parameters):
        utility, benefit_delta = u, bd

    return utility, benefit_delta

def train_multiple(training_parameters, iterations, lambdas=[0.0], verbose=False):
    mean_utilities = []
    max_utilities = []
    min_utilities = []
    mean_benefit_delta = []
    max_benefit_delta = []
    min_benefit_delta = []

    for fairness_rate in lambdas:
        global result_list

        print("Processing Lambda: {}".format(fairness_rate))
        benefit_deltas = []
        utilities = []
        training_parameters["fairness_rate"] = fairness_rate
        
        # multithreaded runs of training
        pool = mp.Pool(mp.cpu_count())
        for _ in range(0, iterations):
            pool.apply_async(train_single, args=(training_parameters,), callback=result_worker) 
        pool.close()
        pool.join()

        results = np.array(result_list).squeeze()
        result_list = []
        utilities = results[:,0]
        benefit_deltas = results[:,1]

        mean_utilities.append(utilities.mean())
        max_utilities.append(utilities.max())
        min_utilities.append(utilities.min())
        mean_benefit_delta.append(benefit_deltas.mean())
        max_benefit_delta.append(benefit_deltas.max())
        min_benefit_delta.append(benefit_deltas.min())

        if verbose:
            print("Mean utility: {}".format(utilities.mean()))
            print("Mean benefit delta: {}".format(benefit_deltas.mean()))

    return {
        "utility_stats": {
            "mean": mean_utilities,
            "min": min_utilities,
            "max": max_utilities
        },
        "benefit_delta_stats": {
            "mean": mean_benefit_delta,
            "min": max_benefit_delta,
            "max": min_benefit_delta
        }
    }