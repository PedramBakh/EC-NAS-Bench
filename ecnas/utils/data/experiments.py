import sys
import os
from pathlib import Path
import pickle
import numpy as np
from math import inf

baselines_core = os.path.join(Path(__file__).parent.parent.parent.parent, "baselines")
sys.path.append(baselines_core)

from core import pareto


def load_pickle_data(fn):
    return pickle.load(open(fn, "rb"))


def filter_data(data, dvs):
    return data[data["mean"].isin(dvs) == False]


def convert_data(data):
    accs = data[data["metric_name"] == "val_acc"]["mean"].values
    energies = data[data["metric_name"] == "energy"]["mean"].values
    accs *= 0.01  # Convert to percentage
    energies *= 1000  # Convert to mJ
    return np.array([accs, energies]).T


def create_front(front, max_acc):
    if max_acc:
        front = front[front[:, 0].argsort()[::-1]]
        front = front[-1:, :]
    else:
        is_front = pareto.pareto(front)
        front = front[is_front == 1]
    return front


def load_experiments(fn, n_trials, dummy_values=None, max_acc=False):
    base_fn = fn + "_"
    max_samples = 0
    dvs = [inf, -inf, 0.0, 1.0] if dummy_values is not None else dummy_values + [inf, -inf, 0.0, 1.0]
    all_experiments = []
    for i in range(n_trials):
        data = load_pickle_data(base_fn + str(i) + ".pickle")
        if dummy_values is not None:
            data = filter_data(data, dvs)
        front = convert_data(data)
        front = create_front(front, max_acc)
        sols = list(zip(front[:, 0], front[:, 1]))
        all_experiments.append(sols)
        if len(sols) > max_samples:
            max_samples = len(sols)
    for i in range(n_trials):
        diff = max_samples - len(all_experiments[i])
        all_experiments[i].extend([dummy_values] * diff)
    X = np.ndarray((n_trials, max_samples, 2))
    for i in range(n_trials):
        X[i] = np.array(all_experiments[i])
    return X


def load_experiments_full(fn, n_trials, dummy_values=None):
    return load_experiments(fn, n_trials, dummy_values, max_acc=False)


def load_not_front(fn, n_trials, dummy_values=None):
    return load_experiments(fn, n_trials, dummy_values, max_acc=True)
