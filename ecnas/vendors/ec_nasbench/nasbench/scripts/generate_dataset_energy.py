import argparse

import tensorflow as tf
import json
import base64
import numpy as np
import os
from nasbench.lib import model_metrics_pb2
from nasbench.lib import model_metrics_energy_pb2
from nasbench.lib import config as _config
from absl import app
from absl import flags
from nasbench import api
import re

# Define data paths
CHECKPOINTS_PER_MODEL = 3

# Build config to access flags
config = _config.build_config()

models_file = "/path/to/ecnas/vendors/ec_nasbench/nasbench/data/graphs/generated_graphs_5V9E.json"
eval_dir = "/path/to/ecnas/vendors/ec_nasbench/nasbench/data/train_model_results/energy/5V9E/4_epochs"
base_tfrecord = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/nasbench_full.tfrecord"
name = "5V9E_estimate"

flags.DEFINE_string("base_tfrecord", base_tfrecord, "Original tfrecord for performance statistics (e.g. accuracy).")

flags.DEFINE_string("models_file", models_file, "JSON file containing models.")

flags.DEFINE_string(name="eval_dir", default=eval_dir, help="Directory of trained model results.")

flags.DEFINE_integer(name="examples", default=1, help="Number of example datapoints to print. Default is 1.")

FLAGS = flags.FLAGS


def get_model_hash_and_ops(graphs_file):
    """
    Loads model file as JSON to obtain model hash and operations for generated specs.
    """
    f = open(graphs_file)
    graphs = json.loads(f.read())
    f.close()

    return graphs


def get_epochs(evaluation_msg):
    """Get the max epoch budget for current evaluation."""
    return evaluation_msg.current_epoch


def get_raw_operations_and_matrix(matrix, labels):
    """
    Get raw operation and matrix encoding from generated model spec (graph).
    First and last label is input and output, respectively, with no operations.
    """
    matrix_str = ""
    for lst in matrix:
        matrix_str = matrix_str + "".join(map(str, lst))

    # Add input and output to avail_ops corresponding to indices -1 and -2, respectively.
    ops = config["available_ops"] + ["output", "input"]
    ops_translated = np.take(ops, labels)
    label_str = ",".join(map(str, ops_translated))

    return matrix_str, label_str


def get_sub_folders(dir):
    """
    Get all sub-folders in dir if sub-folder contains folders.
    """
    sub_folders = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    return sub_folders


def get_metrics(eval_dir, graph_info):
    """
    Creates metrics for each model repeat. Each data point is the metrics obtain for a single repeat of a model.
    The same graph hash is therefore present the number of repeats for a model.
    """
    # Open original benchmark
    nasbench = api.NASBench(FLAGS.base_tfrecord)

    # Get all sub-folders
    model_folders = []
    for subfolder in get_sub_folders(eval_dir):
        model_folders += get_sub_folders(subfolder)
    metrics = []
    for hash_path in model_folders:
        model_hash = os.path.basename(os.path.normpath(hash_path))
        # Store model hash and metrics for later pairing
        model_info = graph_info[model_hash]
        raw_adjacency, raw_ops = get_raw_operations_and_matrix(model_info[0], model_info[1])

        # Get original model metrics
        _, nb_metrics = nasbench.get_metrics_from_hash(model_hash)

        for budget in [4, 12, 36, 108]:
            temp_path = os.path.join(
                hash_path, "repeat_1"
            )  # There is only one repeat for each model for the estimates.
            # Open repeat folder and locate the results.json file
            with open(os.path.join(temp_path, "results.json"), "r") as results:
                data = json.loads(results.read())
                eval_msgs = []
                # Create evaluation messages for a single repeat
                for i in range(CHECKPOINTS_PER_MODEL):
                    eval_dict = data["evaluation_results"][i]
                    # Get average accuracies over all repeats from nasbench
                    nb_train_acc = np.mean(
                        [nb_metrics[budget][i]["final_train_accuracy"] for i in range(len(nb_metrics[budget]))]
                    )
                    nb_test_acc = np.mean(
                        [nb_metrics[budget][i]["final_test_accuracy"] for i in range(len(nb_metrics[budget]))]
                    )
                    nb_validation_acc = np.mean(
                        [nb_metrics[budget][i]["final_validation_accuracy"] for i in range(len(nb_metrics[budget]))]
                    )
                    msg = model_metrics_energy_pb2.EvaluationDataEnergy(
                        current_epoch=eval_dict["epochs"],
                        training_time=eval_dict["training_time"],
                        train_accuracy=nb_train_acc,
                        validation_accuracy=nb_validation_acc,
                        test_accuracy=nb_test_acc,
                        energy=eval_dict["energy (kWh)"],
                        co2eq=eval_dict["co2eq (g)"],
                        avg_intensity=float(eval_dict["avg_intensity (gCO2/kWh)"]),
                        start_emission=eval_dict["start_emission"],
                        stop_emission=eval_dict["stop_emission"],
                    )
                    if i + 1 == CHECKPOINTS_PER_MODEL:
                        # Linear scaling of energy, co2eq and training time (GPU time)
                        scale = budget / msg.current_epoch
                        msg.current_epoch = budget
                        msg.energy *= scale
                        msg.co2eq *= scale
                        msg.training_time *= scale

                        # Carbontracker predictions (wall-clock)
                        msg.pred_energy = eval_dict["predicted_energy (kWh)"]
                        msg.pred_co2eq = eval_dict["predicted_co2eq (g)"]
                        msg.pred_training_time = eval_dict["predicted_training_time"]
                        msg.pred_avg_intensity = float(eval_dict["predicted_avg_intensity (gCO2/kWh)"])
                    eval_msgs.append(msg)

                gpu_names = []
                gpu_usages_w = []
                gpu_usages_j = []
                for (device_name, w, j) in zip(
                    data["avg_power_usages:"]["gpu"]["devices"],
                    data["avg_power_usages:"]["gpu"]["avg_power_usages (W)"],
                    data["avg_power_usages:"]["gpu"]["avg_energy_usages (J)"],
                ):
                    gpu_names.append(model_metrics_energy_pb2.Device(name=device_name))
                    gpu_usages_w.append(model_metrics_energy_pb2.UsageW(avg_power_usage=w[0]))
                    gpu_usages_j.append(model_metrics_energy_pb2.UsageJ(avg_power_usage=j[0]))
                gpu_data = model_metrics_energy_pb2.GPU(
                    avg_power_usages_w=gpu_usages_w, avg_power_usages_j=gpu_usages_j, devices=gpu_names
                )
                cpu_names = []
                cpu_usages_w = []
                cpu_usages_j = []
                for (device_name, w, j) in zip(
                    data["avg_power_usages:"]["cpu"]["devices"],
                    data["avg_power_usages:"]["cpu"]["avg_power_usages (W)"],
                    data["avg_power_usages:"]["cpu"]["avg_energy_usages (J)"],
                ):
                    cpu_names.append(model_metrics_energy_pb2.Device(name=device_name))
                    cpu_usages_w.append(model_metrics_energy_pb2.UsageW(avg_power_usage=w[0]))
                    cpu_usages_j.append(model_metrics_energy_pb2.UsageJ(avg_power_usage=j[0]))
                cpu_data = model_metrics_energy_pb2.CPU(
                    avg_power_usages_w=cpu_usages_w, avg_power_usages_j=cpu_usages_j, devices=cpu_names
                )

                power_usage_data = model_metrics_energy_pb2.PowerUsageData(gpu=gpu_data, cpu=cpu_data)

                metric_msg = model_metrics_energy_pb2.ModelMetricsEnergy(
                    evaluation_data_energy=eval_msgs,
                    trainable_parameters=data["trainable_params"],
                    total_time=data["total_time"],
                    total_energy=data["total_energy (kWh)"],
                    total_co2eq=data["total_co2eq (g)"],
                    overall_avg_intensity=float(data["avg_intensity (gCO2/kWh)"]),
                    avg_power_usages=power_usage_data,
                )

                ser_metric = metric_msg.SerializeToString()
                utf8_metric = base64.encodebytes(ser_metric)

            # Get max epochs
            max_epochs = max(eval_msgs, key=get_epochs)
            max_epochs = int(max_epochs.current_epoch)

            # Append and encode
            row = [model_hash, int(max_epochs), raw_adjacency, raw_ops, utf8_metric]
            metrics.append(np.array(row))
    return metrics


def main(args):
    metrics = get_metrics(FLAGS.eval_dir, get_model_hash_and_ops(FLAGS.models_file))
    print("Generating records..")
    # The script generates the dataset from all results available for a specific graph_spec (i.e. all budgets).

    root_path = os.path.dirname(os.path.dirname(os.getcwd()))
    tfrecord_name = (
        os.path.join(root_path, "ecnas", "utils", "data", "tabular_benchmarks", "energy_" + f"{name}") + ".tfrecord"
    )

    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for metric in metrics:
            row = metric.tolist()
            row[1] = int(row[1])  # Convert epoch from float to int
            row = json.dumps(row)
            row = row.encode("utf-8")
            writer.write(row)
    print("Done!")

    i = 0
    for serialized_row in tf.compat.v1.python_io.tf_record_iterator(tfrecord_name):
        if i < FLAGS.examples:
            print("-" * os.get_terminal_size()[0])
            print(f"EXAMPLE {i + 1}:")
            print("-" * os.get_terminal_size()[0])
            # Parse the data from the data file.
            module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = json.loads(serialized_row.decode("utf-8"))

            print(f"module hash: {module_hash}")
            print(f"Raw Adjacency: {raw_adjacency}")
            print(f"Raw operations: {raw_operations}")
            print(f"Raw metrics: {raw_metrics}")

            metrics = model_metrics_energy_pb2.ModelMetricsEnergy.FromString(base64.b64decode(raw_metrics))
            operations = raw_operations.split(",")
            dim = int(np.sqrt(len(raw_adjacency)))
            adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
            adjacency = np.reshape(adjacency, (dim, dim))

            print(f"Adjacency:\n {adjacency}")
            print(f"Metrics: {metrics}")
            print(f"Operations: {operations}")
            i = i + 1
        else:
            break


if __name__ == "__main__":
    app.run(main)
