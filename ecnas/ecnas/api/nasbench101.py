import math
import os
import random
import sys
from collections import Counter
from google.protobuf import json_format
import base64
import copy
import json

from ecnas.api.benchmark import Benchmark
from ecnas.utils.data.get_benchmarks import (
    get_nasbench101_full,
    get_nasbench101_only108,
)
from nasbench.lib import config, model_metrics_pb2, model_metrics_energy_pb2
from nasbench.lib import model_spec as _model_spec

import numpy as np
import tensorflow as tf

# Add submodule to path
path_nasbench = os.path.join(os.path.dirname(os.getcwd()), "ecnas", "vendors", "ec_nasbench")
path_carbontracker = os.path.join(os.path.dirname(os.getcwd()), "ec_carbontracker")
sys.path.append(path_nasbench)
sys.path.append(path_carbontracker)
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())),
        "ecnas",
        "vendors",
        "ec_nasbench",
    )
)

TABULAR_BENCHMARKS = os.path.join(os.getcwd(), "ecnas", "utils", "data", "tabular_benchmarks")
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""


class InconsistentModelRepeatError(Exception):
    """Indicates that the dataset file has inconsistent number of model repeats or less than 3 repeats per model."""


class NASBench101(Benchmark):
    def __init__(self, variant="full", dataset_file=None, seed=None):
        if variant == "full":
            get_nasbench101_full()
            self.dataset_file = os.path.join(TABULAR_BENCHMARKS, "nasbench_full.tfrecord")
            self._name = "NASBench101 (nasbench_full.tfrecord)"
            super().__init__(dataset_file=self.dataset_file, seed=seed)
        elif variant == "only108":
            get_nasbench101_only108()
            self.dataset_file = os.path.join(TABULAR_BENCHMARKS, "nasbench_only108.tfrecord")
            self._name = "NASBench101 (nasbench_only108.tfrecord)"
            super().__init__(dataset_file=self.dataset_file, seed=seed)

    def _setup(self):
        self.config = config.build_config()

        # Store statistic of dataset
        self._num_datapoints = 0
        self._unique_models = 0
        self._aggregate_training_time_days = 0
        self._all_ops = set()

        # Stores the fixed statistics that are independent of evaluation (i.e.,
        # adjacency matrix, operations, and number of parameters).
        # hash --> metric name --> scalar
        self.fixed_statistics = {}

        # Stores the statistics that are computed via training and evaluating the
        # model on CIFAR-10. Statistics are computed for multiple repeats of each
        # model at each max epoch length.
        # hash --> epochs --> repeat index --> metric name --> scalar
        self.computed_statistics = {}

        # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
        # {108} for the smaller dataset with only the 108 epochs.
        self.valid_epochs = set()

        self._module_vertices = []

        self._max_edges = set()

        for serialized_row in tf.compat.v1.python_io.tf_record_iterator(self.dataset_file):
            # Parse the data from the data file.
            (
                module_hash,
                epochs,
                raw_adjacency,
                raw_operations,
                raw_metrics,
            ) = json.loads(serialized_row.decode("utf-8"))

            dim = int(np.sqrt(len(raw_adjacency)))
            adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
            adjacency = np.reshape(adjacency, (dim, dim))
            operations = raw_operations.split(",")
            metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))

            self._all_ops.update(operations)
            self._max_edges.add(adjacency.sum())

            if module_hash not in self.fixed_statistics:
                # First time seeing this module, initialize fixed statistics.
                new_entry = {}
                new_entry["module_adjacency"] = adjacency
                new_entry["module_operations"] = operations
                new_entry["trainable_parameters"] = metrics.trainable_parameters
                self.fixed_statistics[module_hash] = new_entry
                self.computed_statistics[module_hash] = {}
                self._unique_models += 1

            self.valid_epochs.add(epochs)

            if epochs not in self.computed_statistics[module_hash]:
                self.computed_statistics[module_hash][epochs] = []

            # Each data_point consists of the metrics recorded from a single
            # train-and-evaluation of a model at a specific epoch length.
            data_point = {}

            # Note: metrics.evaluation_data[0] contains the computed metrics at the
            # start of training (step 0) but this is unused by this API.

            # Evaluation statistics at the half-way point of training
            half_evaluation = metrics.evaluation_data[1]
            data_point["halfway_training_time"] = half_evaluation.training_time
            data_point["halfway_train_accuracy"] = half_evaluation.train_accuracy
            data_point["halfway_validation_accuracy"] = half_evaluation.validation_accuracy
            data_point["halfway_test_accuracy"] = half_evaluation.test_accuracy

            # Evaluation statistics at the end of training
            final_evaluation = metrics.evaluation_data[2]
            data_point["final_training_time"] = final_evaluation.training_time
            data_point["final_train_accuracy"] = final_evaluation.train_accuracy
            data_point["final_validation_accuracy"] = final_evaluation.validation_accuracy
            data_point["final_test_accuracy"] = final_evaluation.test_accuracy

            self.computed_statistics[module_hash][epochs].append(data_point)

            seconds_in_day = 60 * 60 * 24
            self._num_datapoints += 1
            self._aggregate_training_time_days += data_point["final_training_time"] / seconds_in_day
            self._module_vertices.append(math.sqrt(len(list(raw_adjacency))))

        self._module_vertices = dict(Counter(self._module_vertices))
        self._max_edges = max(self._max_edges)

        self.history = {}
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def query(self, model_spec, epochs=108, stop_halfway=False):
        if epochs not in self.valid_epochs:
            raise OutOfDomainError("invalid number of epochs, must be one of %s" % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        sampled_index = random.randint(0, self.config["num_repeats"] - 1)
        computed_stat = computed_stat[epochs][sampled_index]

        data = {}
        data["module_adjacency"] = fixed_stat["module_adjacency"]
        data["module_operations"] = fixed_stat["module_operations"]
        data["trainable_parameters"] = fixed_stat["trainable_parameters"]

        if stop_halfway:
            data["training_time"] = computed_stat["halfway_training_time"]
            data["train_accuracy"] = computed_stat["halfway_train_accuracy"]
            data["validation_accuracy"] = computed_stat["halfway_validation_accuracy"]
            data["test_accuracy"] = computed_stat["halfway_test_accuracy"]
        else:
            data["training_time"] = computed_stat["final_training_time"]
            data["train_accuracy"] = computed_stat["final_train_accuracy"]
            data["validation_accuracy"] = computed_stat["final_validation_accuracy"]
            data["test_accuracy"] = computed_stat["final_test_accuracy"]

        self.training_time_spent += data["training_time"]
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data

    def get_metrics_from_hash(self, module_hash):
        fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
        return fixed_stat, computed_stat

    def get_metrics_from_spec(self, model_spec):
        self._check_spec(model_spec)
        module_hash = self._hash_spec(model_spec)
        return self.get_metrics_from_hash(module_hash)

    def get_budget_counters(self):
        """Returns the time and budget counters."""
        return self.training_time_spent, self.total_epochs_spent

    def reset_budget_counters(self):
        """Reset the time and epoch budget counters."""
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError("invalid spec, provided graph is disconnected.")

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.config["module_vertices"]:
            raise OutOfDomainError(
                "too many vertices, got %d (max vertices = %d)" % (num_vertices, config["module_vertices"])
            )

        if num_edges > self.config["max_edges"]:
            raise OutOfDomainError("too many edges, got %d (max edges = %d)" % (num_edges, self.config["max_edges"]))

        if model_spec.ops[0] != "input":
            raise OutOfDomainError("first operation should be 'input'")
        if model_spec.ops[-1] != "output":
            raise OutOfDomainError("last operation should be 'output'")
        for op in model_spec.ops[1:-1]:
            if op not in self.config["available_ops"]:
                raise OutOfDomainError("unsupported op %s (available ops = %s)" % (op, self.config["available_ops"]))

    def _hash_spec(self, model_spec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec(self.config["available_ops"])

    def has_iterator(self):
        return self.fixed_statistics.keys()

    def info(self):
        # valid epochs, number of models, vertices/edges of dataset
        print(self._name + ":")
        print(f"# of datapoints: {self._num_datapoints}")
        print(f"# of unique models: {self._unique_models}")

        print(f"Module vertices: {json.dumps(self._module_vertices, indent=2)}")
        print(f"Max edges: {self._max_edges}")
        print(f"Operations: {self._all_ops}")

        print(f"Epoch budgets: {sorted(self.valid_epochs)}")
        print(f"Aggregate training time: {self._aggregate_training_time_days:.3f} day(s)")


class ECNASBench(Benchmark):
    def __init__(self, dataset_file=None, seed=None):
        self.dataset_file = os.path.join(TABULAR_BENCHMARKS, dataset_file)
        self._name = f"EC-NAS-Bench ({dataset_file})"
        super().__init__(dataset_file=self.dataset_file, seed=seed)

    def _setup(self):

        # Store statistic of dataset
        self._num_datapoints = 0
        self._unique_models = 0
        self._aggregate_training_time_days = 0
        self._all_ops = set()
        self._aggregate_energy = 0
        self._aggregate_co2 = 0

        # Stores the fixed statistics that are independent of evaluation (i.e.,
        # adjacency matrix, operations, and number of parameters).
        # hash --> metric name --> scalar
        self.fixed_statistics = {}

        # Stores the statistics that are computed via training and evaluating the
        # model on CIFAR-10. Statistics are computed for multiple repeats of each
        # model at each max epoch length.
        # hash --> epochs --> repeat index --> metric name --> scalar
        self.computed_statistics = {}

        # Valid queriable epoch lengths. {4, 12, 36, 108} for the full dataset or
        # {108} for the smaller dataset with only the 108 epochs.
        self.valid_epochs = set()

        self._module_vertices = []

        self._max_edges = set()

        for serialized_row in tf.compat.v1.python_io.tf_record_iterator(self.dataset_file):
            # Parse the data from the data file.
            (
                module_hash,
                epochs,
                raw_adjacency,
                raw_operations,
                raw_metrics,
            ) = json.loads(serialized_row.decode("utf-8"))

            dim = int(np.sqrt(len(raw_adjacency)))
            adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
            adjacency = np.reshape(adjacency, (dim, dim))
            operations = raw_operations.split(",")
            metrics = model_metrics_energy_pb2.ModelMetricsEnergy.FromString(base64.b64decode(raw_metrics))

            self._all_ops.update(operations)
            self._max_edges.add(adjacency.sum())

            if module_hash not in self.fixed_statistics:
                # First time seeing this module, initialize fixed statistics.
                new_entry = {}
                new_entry["module_adjacency"] = adjacency
                new_entry["module_operations"] = operations
                new_entry["trainable_parameters"] = metrics.trainable_parameters
                self.fixed_statistics[module_hash] = new_entry
                self.computed_statistics[module_hash] = {}
                self._unique_models += 1
                self._module_vertices.append(dim)

            self.valid_epochs.add(epochs)

            if epochs not in self.computed_statistics[module_hash]:
                self.computed_statistics[module_hash][epochs] = []

            # Each data_point consists of the metrics recorded from a single
            # train-and-evaluation of a model at a specific epoch length.
            data_point = {}

            # Note: metrics.evaluation_data[0] contains the computed metrics at the
            # start of training (step 0) but this is unused by this API.

            # Evaluation statistics at the half-way point of training
            half_evaluation = metrics.evaluation_data_energy[1]
            data_point["halfway_training_time"] = half_evaluation.training_time
            data_point["halfway_train_accuracy"] = half_evaluation.train_accuracy
            data_point["halfway_validation_accuracy"] = half_evaluation.validation_accuracy
            data_point["halfway_test_accuracy"] = half_evaluation.test_accuracy

            # Evaluation statistics at the end of training
            final_evaluation = metrics.evaluation_data_energy[2]
            data_point["final_training_time"] = final_evaluation.training_time
            data_point["final_train_accuracy"] = final_evaluation.train_accuracy
            data_point["final_validation_accuracy"] = final_evaluation.validation_accuracy
            data_point["final_test_accuracy"] = final_evaluation.test_accuracy

            # Total Metrics
            data_point["total_time"] = metrics.total_time
            data_point["total_energy (kWh)"] = metrics.total_energy
            data_point["total_co2eq (g)"] = metrics.total_co2eq
            data_point["total_start_emission"] = metrics.start_emission
            data_point["total_stop_emission"] = metrics.stop_emission
            data_point["overall_avg_intensity (gCO2/kWh)"] = metrics.overall_avg_intensity
            data_point["avg_power_usages"] = json_format.MessageToDict(metrics.avg_power_usages)

            # For max epoch budget
            data_point["energy (kWh)"] = final_evaluation.energy
            data_point["co2eq (g)"] = final_evaluation.co2eq
            data_point["avg_intensity (gCO2/kWh)"] = final_evaluation.avg_intensity
            data_point["start_emission"] = final_evaluation.start_emission
            data_point["stop_emission"] = final_evaluation.stop_emission
            data_point["predicted_energy (kWh)"] = final_evaluation.pred_energy
            data_point["predicted_co2eq (g)"] = final_evaluation.pred_co2eq
            data_point["predicted_training_time"] = final_evaluation.pred_training_time
            data_point["predicted_avg_intensity (gCO2/kWh)"] = final_evaluation.pred_avg_intensity

            self.computed_statistics[module_hash][epochs].append(data_point)

            seconds_in_day = 60 * 60 * 24
            self._num_datapoints += 1

            # Uncomment for actual training costs
            if epochs == 4:
                self._aggregate_training_time_days += final_evaluation.training_time / seconds_in_day
                self._aggregate_energy += final_evaluation.energy
                self._aggregate_co2 += final_evaluation.co2eq

            # self._aggregate_training_time_days += data_point["final_training_time"] / seconds_in_day
            # self._aggregate_energy += data_point["energy (kWh)"]
            # self._aggregate_co2 += data_point["co2eq (g)"]

        self._module_vertices = dict(Counter(self._module_vertices))
        self._max_edges = max(self._max_edges)

        self.history = {}
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

        self.config = {
            "module_vertices": int(max(self._module_vertices)),
            "max_edges": self._max_edges,
            "available_ops": ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"],
            "num_repeats": 3,
        }

    def query(self, model_spec, epochs=108, stop_halfway=False):
        if epochs not in self.valid_epochs:
            raise OutOfDomainError("invalid number of epochs, must be one of %s" % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        computed_stat = computed_stat[epochs][0]

        data = {}
        data["total_start_emission"] = computed_stat["start_emission"]
        data["total_stop_emission"] = computed_stat["stop_emission"]

        data["total_time"] = computed_stat["total_time"]
        data["trainable_parameters"] = fixed_stat["trainable_parameters"]
        data["overall_avg_intensity (gCO2/kWh)"] = computed_stat["overall_avg_intensity (gCO2/kWh)"]

        data["total_energy (kWh)"] = computed_stat["total_energy (kWh)"]
        data["total_co2eq (g)"] = computed_stat["total_co2eq (g)"]
        data["avg_power_usages"] = computed_stat["avg_power_usages"]

        data["module_adjacency"] = fixed_stat["module_adjacency"]
        data["module_operations"] = fixed_stat["module_operations"]

        if stop_halfway:
            data["training_time"] = computed_stat["halfway_training_time"]
            data["train_accuracy"] = computed_stat["halfway_train_accuracy"]
            data["validation_accuracy"] = computed_stat["halfway_validation_accuracy"]
            data["test_accuracy"] = computed_stat["halfway_test_accuracy"]
        else:
            data["training_time"] = computed_stat["final_training_time"]
            data["train_accuracy"] = computed_stat["final_train_accuracy"]
            data["validation_accuracy"] = computed_stat["final_validation_accuracy"]
            data["test_accuracy"] = computed_stat["final_test_accuracy"]

        data["start_emission"] = computed_stat["start_emission"]
        data["stop_emission"] = computed_stat["stop_emission"]
        data["avg_intensity (gCO2/kWh)"] = computed_stat["avg_intensity (gCO2/kWh)"]

        data["predicted_training_time"] = computed_stat["predicted_training_time"]
        data["predicted_avg_intensity (gCO2/kWh)"] = computed_stat["predicted_avg_intensity (gCO2/kWh)"]

        data["energy (kWh)"] = computed_stat["energy (kWh)"]
        data["co2eq (g)"] = computed_stat["co2eq (g)"]
        data["predicted_energy (kWh)"] = computed_stat["predicted_energy (kWh)"]
        data["predicted_co2eq (g)"] = computed_stat["predicted_co2eq (g)"]

        self.training_time_spent += data["training_time"]
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data

    def get_metrics_from_hash(self, module_hash):
        fixed_stat = copy.deepcopy(self.fixed_statistics[module_hash])
        computed_stat = copy.deepcopy(self.computed_statistics[module_hash])
        return fixed_stat, computed_stat

    def get_metrics_from_spec(self, model_spec):
        self._check_spec(model_spec)
        module_hash = self._hash_spec(model_spec)
        return self.get_metrics_from_hash(module_hash)

    def get_budget_counters(self):
        """Returns the time and budget counters."""
        return self.training_time_spent, self.total_epochs_spent

    def reset_budget_counters(self):
        """Reset the time and epoch budget counters."""
        self.training_time_spent = 0.0
        self.total_epochs_spent = 0

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError("invalid spec, provided graph is disconnected.")

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.config["module_vertices"]:
            raise OutOfDomainError(
                "too many vertices, got %d (max vertices = %d)" % (num_vertices, config["module_vertices"])
            )

        if num_edges > self.config["max_edges"]:
            raise OutOfDomainError("too many edges, got %d (max edges = %d)" % (num_edges, self.config["max_edges"]))

        if model_spec.ops[0] != "input":
            raise OutOfDomainError("first operation should be 'input'")
        if model_spec.ops[-1] != "output":
            raise OutOfDomainError("last operation should be 'output'")
        for op in model_spec.ops[1:-1]:
            if op not in self.config["available_ops"]:
                raise OutOfDomainError("unsupported op %s (available ops = %s)" % (op, self.config["available_ops"]))

    def _hash_spec(self, model_spec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec(self.config["available_ops"])

    def hash_iterator(self):
        return self.fixed_statistics.keys()

    def info(self):
        print(f"{self._name}:")
        print(f"Datapoints: {self._num_datapoints}")
        print(f"Unique models: {self._unique_models}")

        module_vertices = dict((str(int(k)) + "V", str(v) + " models") for k, v in self._module_vertices.items())
        print(f"Module vertices: {json.dumps(module_vertices, indent=2)}")
        print(f"Max edges: {self._max_edges}")
        print(f"Operations: {self._all_ops}")

        print(f"Epoch budgets: {sorted(self.valid_epochs)}")
        print(f"Aggregate training time: {self._aggregate_training_time_days:.3f} day(s)")
        print(f"Aggregate energy consumption (kWh): {self._aggregate_energy}")
        print(f"Aggregate co2eq emission (g): {self._aggregate_co2}")

    def get_model_spec(self, matrix, labels):
        return ModelSpec(matrix, labels)

    def get_intermediate_ops(self):
        return [op for op in self._all_ops if op not in ["input", "output"]]
