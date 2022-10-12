import pandas as pd
import numpy as np


def get_ranges(benchmark, budget):
    """Get the minimum and maximum values for the benchmark and a budget."""
    all_lookup_keys = list(benchmark.hash_iterator())

    min_acc, max_acc = 1, 0
    min_time, max_time = 1e16, 0
    min_energy, max_energy = 1e16, 0
    min_params, max_params = 1e16, 0

    for key in all_lookup_keys:
        fixed_stat, computed_stat = benchmark.get_metrics_from_hash(key)
        computed_stat = computed_stat[budget][0]

        if computed_stat["final_validation_accuracy"] < min_acc:
            min_acc = computed_stat["final_validation_accuracy"]
        if computed_stat["final_validation_accuracy"] > max_acc:
            max_acc = computed_stat["final_validation_accuracy"]

        if computed_stat["final_training_time"] < min_time:
            min_time = computed_stat["final_training_time"]
        if computed_stat["final_training_time"] > max_time:
            max_time = computed_stat["final_training_time"]

        if computed_stat["energy (kWh)"] < min_energy:
            min_energy = computed_stat["energy (kWh)"]
        if computed_stat["energy (kWh)"] > max_energy:
            max_energy = computed_stat["energy (kWh)"]

        if fixed_stat["trainable_parameters"] < min_params:
            min_params = fixed_stat["trainable_parameters"]
        if fixed_stat["trainable_parameters"] > max_params:
            max_params = fixed_stat["trainable_parameters"]
    return [(min_time, max_time), (min_acc, max_acc), (min_energy, max_energy), (min_params, max_params)]


def get_dataframe(archs, benchmark, budget=108):
    df = pd.DataFrame()
    for i in range(len(archs)):
        matrix, labels = archs[i][0]
        spec = benchmark.get_model_spec(matrix, labels)
        fixed_stats, computed_stats = benchmark.get_metrics_from_spec(spec)
        metrics = {**fixed_stats, **computed_stats[budget][0]}
        df = df.append(metrics, ignore_index=True)

    return df[["final_training_time", "final_validation_accuracy", "energy (kWh)", "trainable_parameters"]]


def get_bend_angles(x, y):
    x = np.array(x)
    y = np.array(y)

    # diff of x to points right
    x_diff = np.diff(x)
    # diff of y to points right
    y_diff = np.diff(y)

    # diff of x to points left
    x_diff_left = np.diff(x[::-1])[::-1]
    # diff of y to points left
    y_diff_left = np.diff(y[::-1])[::-1]

    # angle of line to points right
    angles_right = np.arctan(y_diff, x_diff)
    # angle of line to points left
    angles_left = np.arctan(y_diff_left, x_diff_left)

    # difference of angles
    angles_diff = np.abs(angles_right - angles_left)

    return angles_diff
