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

def train_multiple(training_parameters, iterations, lambdas=[0.0], verbose=False, asynchronous=True):
    mean_utilities = []
    stddev_utilites = []
    mean_benefit_delta = []
    stddev_benefit_delta = []

    for fairness_rate in lambdas:
        global result_list

        print("Processing Lambda: {}".format(fairness_rate))
        benefit_deltas = []
        utilities = []
        training_parameters["fairness_rate"] = fairness_rate
        
        # multithreaded runs of training
        if asynchronous:
            pool = mp.Pool(mp.cpu_count())
            for _ in range(0, iterations):
                pool.apply_async(train_single, args=(training_parameters,), callback=result_worker) 
            pool.close()
            pool.join()
        else:
            for _ in range(0, iterations):
                result = train_single(training_parameters)
                result_worker(result)

        results = np.array(result_list).squeeze()
        result_list = []

        utilities = results[:,0]
        benefit_deltas = results[:,1]

        mean_utilities.append(utilities.mean())
        stddev_utilites.append(utilities.std())
        mean_benefit_delta.append(benefit_deltas.mean())
        stddev_benefit_delta.append(benefit_deltas.std())

        if verbose:
            print("Mean utility: {}".format(utilities.mean()))
            print("Stddev utility: {}".format(utilities.std()))
            print("Mean benefit delta: {}".format(benefit_deltas.mean()))
            print("Stddev benefit delta: {}".format(benefit_deltas.std()))

    return {
        "utility_stats": {
            "mean": np.array(mean_utilities),
            "stddev": np.array(stddev_utilites)
        },
        "benefit_delta_stats": {
            "mean": np.array(mean_benefit_delta),
            "stddev": np.array(stddev_benefit_delta)
        }
    }