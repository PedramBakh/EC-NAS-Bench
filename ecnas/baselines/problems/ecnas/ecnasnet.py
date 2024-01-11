import numpy as np
import sys

sys.path.append("../../../")
from ecnas.api import nasbench101


vertex = 7
fn = f"path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_7V9E_surrogate"
benchmark = nasbench101.ECNASBench(fn)
benchmark.info()


def evaluate_network(config, budget=None):
    """
    Evaluate the performance of the model on the given architecture.
    """
    budget = budget if budget else config["budget"]
    vertices = config["vertices"]
    edges = config["edges"]
    max_edges = 9

    # Initialize the adjacency matrix
    matrix = np.zeros([vertices, vertices], dtype=np.int8)

    # Fill the adjacency matrix based on the configuration
    num_edges = 0
    for i in range(vertices):
        for j in range(i + 1, vertices):
            if num_edges == edges:
                break
            if config["edge_{}_{}".format(i, j)] > 0:
                matrix[i, j] = 1
                num_edges += 1
            if num_edges == edges:
                break

    # Define a default return value for unsuccessful evaluations
    bad_return = {
        "val_acc": (0.0, 0.0),
        "tst_acc": (0.0, 0.0),
        "train_time": (0.0, 0.0),
        "energy": (0.0, 0.0),
        "co2eq": (0.0, 0.0),
        "params": (0.0, 0.0),
    }

    # Check if the number of edges is greater than the maximum allowed
    if np.sum(matrix) > max_edges:
        return bad_return

    # Get the operation labeling for each node
    labeling = [config["op_node_%d" % i] for i in range(vertices - 2)]
    labeling = ["input"] + list(labeling) + ["output"]

    # Get the model specification from the adjacency matrix and labeling
    model_spec = benchmark.get_model_spec(matrix, labeling)

    try:
        metrics = benchmark.query(model_spec, epochs=budget)
    except KeyError:
        return bad_return
    except nasbench101.OutOfDomainError:
        return bad_return

    # Return the evaluation results
    res = {
        "val_acc": (-100.0 * metrics["validation_accuracy"], 0.0),
        "tst_acc": (-100.0 * metrics["test_accuracy"], 0.0),
        "train_time": (metrics["training_time"], 0.0),
        "energy": (metrics["energy (kWh)"], 0.0),
        "co2eq": (metrics["co2eq (g)"], 0.0),
        "params": (np.log10(metrics["trainable_parameters"]), 0.0),
    }

    return res


def extract_energy(config):
    """
    Extract the energy of the model on the given architecture.
    """
    res = evaluate_network(config, budget=config["budget"])
    if res["energy"][0] == 0.0:
        return 1e6
    else:
        return res["energy"][0]
