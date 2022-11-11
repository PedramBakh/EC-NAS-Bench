# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Performs training and evaluation of the proposed model spec on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

# Add submodule to path
path = os.path.join(os.path.dirname(os.getcwd()), "ec_carbontracker")
path_extended = os.path.join(os.path.dirname(os.getcwd()), "ecnas", "vendors", "ec_carbontracker")

sys.path.append(path)
sys.path.append(path_extended)

from carbontracker.tracker import CarbonTracker
from carbontracker import parser
from absl import flags

from nasbench.lib import cifar
from nasbench.lib import model_builder
from nasbench.lib import training_time
import numpy as np
import tensorflow as tf

VALID_EXCEPTIONS = (
    tf.estimator.NanLossDuringTrainingError,  # NaN loss
    tf.errors.ResourceExhaustedError,  # OOM
    tf.errors.InvalidArgumentError,  # NaN gradient
    tf.errors.DeadlineExceededError,  # Timed out
)

LOGGER_COUNTER = 0


def _logger_counter():
    global LOGGER_COUNTER
    LOGGER_COUNTER += 1
    return LOGGER_COUNTER


class AbortError(Exception):
    """Signals that evaluation failed for a valid reason."""

    pass


def train_and_evaluate(spec, config, model_dir):
    """Train and evaluate the proposed model.

    This method trains and evaluates the model for the creation of the benchmark
    dataset. The default values from the config.py are exactly the values used.

    Args:
      spec: ModelSpec object.
      config: config dict generated from config.py.
      model_dir: directory to store the checkpoint files.

    Returns:
      dict containing the evaluation metadata.
    """
    return _train_and_evaluate_impl(spec, config, model_dir)


def augment_and_evaluate(spec, config, model_dir, epochs_per_eval=5):
    """Trains the model on the full training set and evaluates on test set.

    "Augment" specifically refers to training the same spec in a larger network on
    the full training set.  Typically this involves increasing the epoch count,
    number of modules/stacks, and changing the LR schedule. These changes should
    be made to the config dict before calling this method.

    Note: this method was not used for generating the NAS Benchmark dataset. See
    train_and_evaluate instead.

    Args:
      spec: ModelSpec object.
      config: config dict generated from config.py.
      model_dir: directory to store the checkpoint files.
      epochs_per_eval: number of epochs per evaluation run. Evaluation is always
        run at the very start and end.

    Returns:
      dict containing the evaluation metadata.
    """
    return _augment_and_evaluate_impl(spec, config, model_dir, epochs_per_eval)


def _train_and_evaluate_impl(spec, config, model_dir):
    """Train and evaluate implementation, see train_and_evaluate docstring."""
    evaluator = _TrainAndEvaluator(spec, config, model_dir)
    return evaluator.run()


class _TrainAndEvaluator(object):
    """Runs the training and evaluation."""

    def __init__(self, spec, config, model_dir):
        """Initialize evaluator. See train_and_evaluate docstring."""
        self.input_train = cifar.CIFARInput("train", config)
        self.input_train_eval = cifar.CIFARInput("train_eval", config)
        self.input_valid = cifar.CIFARInput("valid", config)
        self.input_test = cifar.CIFARInput("test", config)
        self.input_sample = cifar.CIFARInput("sample", config)
        self.estimator = _create_estimator(
            spec, config, model_dir, self.input_train.num_images, self.input_sample.num_images
        )

        self.spec = spec
        self.config = config
        self.model_dir = model_dir

    def run(self):
        """Runs training and evaluation."""
        attempts = 0
        while True:
            # Delete everything in the model dir at the start of each attempt
            try:
                tf.io.gfile.rmtree(self.model_dir)
            except tf.errors.NotFoundError:
                pass
            tf.io.gfile.makedirs(self.model_dir)

            try:
                # Train
                if self.config["train_seconds"] > 0.0:
                    timing = training_time.limit(self.config["train_seconds"])
                else:
                    timing = training_time.limit(None)

                evaluations = list(map(float, self.config["intermediate_evaluations"]))
                if not evaluations or evaluations[-1] != 1.0:
                    evaluations.append(1.0)
                assert evaluations == sorted(evaluations)

                metadata = {}
                evaluation_results = []
                start_time = time.time()

                total_carbontracker = CarbonTracker(
                    epochs=1,
                    log_dir=os.path.join(self.config["train_model_dir"], "emissions_log_total"),
                    logging_mode=1,
                    logger_name=f"carbontracker_total_{_logger_counter()}",
                    epochs_before_pred=0,
                )
                total_carbontracker.epoch_start()

                # Train for 1 step with 0 LR to initialize the weights, then evaluate
                # once at the start for completeness, accuracies expected to be around
                # random selection. Note that batch norm moving averages change during
                # the step but the trainable weights do not.
                init_carbontracker = CarbonTracker(
                    epochs=1,
                    log_dir=os.path.join(self.config["train_model_dir"], "emissions_log_init"),
                    logging_mode=1,
                    logger_name=f"carbontracker_init_{_logger_counter()}",
                    epochs_before_pred=0,
                )
                init_carbontracker.epoch_start()
                self.estimator.train(
                    input_fn=self.input_train.input_fn,
                    max_steps=1,
                    hooks=[timing.train_hook],
                    saving_listeners=[timing.saving_listener],
                )
                init_carbontracker.epoch_end()
                init_carbontracker_std_stream, init_carbontracker_out_stream = init_carbontracker.get_logger_stream()
                init_log = parser.parse_streams(
                    std=init_carbontracker_std_stream,
                    out=init_carbontracker_out_stream,
                    timestamp=True,
                    avg_carbon_intensity=True,
                )
                init_log_filtered = {
                    "energy (kWh)": init_log["actual"]["energy (kWh)"],
                    "co2eq (g)": init_log["actual"]["co2eq (g)"],
                    "avg_intensity (gCO2/kWh)": init_log["actual"]["avg_intensity (gCO2/kWh)"],
                    "start_emission": init_log["actual"]["start_emission"],
                    "stop_emission": init_log["actual"]["stop_emission"],
                }
                evaluation_results.append(self._evaluate_all(0.0, 0, add_metrics=init_log_filtered))

                for next_evaluation in evaluations:
                    epoch = next_evaluation * self.config["train_epochs"]
                    train_steps = int(epoch * self.input_train.num_images / self.config["batch_size"])

                    # If max epoch budget, we do not want a prediction (i.e.epochs_before_pred = 0).
                    if epoch == self.config["train_epochs"]:
                        train_carbontracker = CarbonTracker(
                            epochs=self.config["train_epochs"],
                            log_dir=os.path.join(self.config["train_model_dir"], "emissions_log_train"),
                            logging_mode=1,
                            logger_name=f"carbontracker_init_{_logger_counter()}",
                            epochs_before_pred=0,
                        )
                    else:
                        train_carbontracker = CarbonTracker(
                            epochs=self.config["train_epochs"],
                            log_dir=os.path.join(self.config["train_model_dir"], "emissions_log_train"),
                            logging_mode=1,
                            logger_name=f"carbontracker_init_{_logger_counter()}",
                        )
                    train_carbontracker.epoch_start()
                    self.estimator.train(
                        input_fn=self.input_train.input_fn,
                        max_steps=train_steps,
                        hooks=[timing.train_hook],
                        saving_listeners=[timing.saving_listener],
                    )
                    train_carbontracker.epoch_end()
                    (
                        train_carbontracker_std_stream,
                        train_carbontracker_out_stream,
                    ) = train_carbontracker.get_logger_stream()
                    train_log = parser.parse_streams(
                        std=train_carbontracker_std_stream,
                        out=train_carbontracker_out_stream,
                        avg_carbon_intensity=True,
                        timestamp=True,
                    )

                    # If intermediate evaluation
                    epoch = round(epoch, 1)
                    # This currently only supports previous evaluation as weight initialization
                    if epoch != self.config["train_epochs"]:
                        train_log_filtered = {
                            "energy (kWh)": train_log["actual"]["energy (kWh)"]
                            + evaluation_results[-1]["energy (kWh)"],
                            "co2eq (g)": train_log["actual"]["co2eq (g)"] + evaluation_results[-1]["co2eq (g)"],
                            "avg_intensity (gCO2/kWh)": train_log["actual"]["avg_intensity (gCO2/kWh)"],
                            "start_emission": train_log["actual"]["start_emission"],
                            "stop_emission": train_log["actual"]["stop_emission"],
                        }

                        evaluation_results.append(
                            self._evaluate_all(epoch, train_steps, add_metrics=train_log_filtered)
                        )

                        train_log_prediction = {
                            "predicted_energy (kWh)": train_log["pred"]["energy (kWh)"],
                            "predicted_co2eq (g)": train_log["pred"]["co2eq (g)"],
                            "predicted_training_time": train_log["pred"]["duration (s)"],
                            "predicted_avg_intensity (gCO2/kWh)": train_log["pred"]["avg_intensity (gCO2/kWh)"],
                        }
                        metadata["tmp"] = train_log_prediction

                    # If last epoch: Note that we have to add any previous evaluations done to the emission metrics,
                    # as training continues after intermediate evaluations rather than restart (e.g. if there is a
                    # halfway evaluation, emission data for last epoch will only correspond to half the epoch budget
                    # trained).
                    if epoch == self.config["train_epochs"]:
                        train_log_filtered = {
                            "energy (kWh)": train_log["actual"]["energy (kWh)"]
                            + evaluation_results[-1]["energy (kWh)"],
                            "co2eq (g)": train_log["actual"]["co2eq (g)"] + evaluation_results[-1]["co2eq (g)"],
                            "avg_intensity (gCO2/kWh)": train_log["actual"]["avg_intensity (gCO2/kWh)"],
                            "start_emission": train_log["actual"]["start_emission"],
                            "stop_emission": train_log["actual"]["stop_emission"],
                        }
                        # If any intermediate evaluations
                        if len(evaluations) > 1:
                            train_log_filtered["predicted_energy (kWh)"] = metadata["tmp"]["predicted_energy (kWh)"]
                            train_log_filtered["predicted_co2eq (g)"] = metadata["tmp"]["predicted_co2eq (g)"]
                            train_log_filtered["predicted_training_time"] = metadata["tmp"]["predicted_training_time"]
                            train_log_filtered["predicted_avg_intensity (gCO2/kWh)"] = metadata["tmp"][
                                "predicted_avg_intensity (gCO2/kWh)"
                            ]
                            del metadata["tmp"]

                        evaluation_results.append(
                            self._evaluate_all(epoch, train_steps, add_metrics=train_log_filtered)
                        )

                all_time = time.time() - start_time
                total_carbontracker.epoch_end()
                total_std_stream, total_out_stream = total_carbontracker.get_logger_stream()
                total_log = parser.parse_streams(
                    std=total_std_stream, out=total_out_stream, timestamp=True, avg_carbon_intensity=True
                )

                break  # Break from retry loop on success
            except VALID_EXCEPTIONS as e:  # pylint: disable=catching-non-exception
                attempts += 1
                tf.compat.v1.logging.warning(str(e))
                if attempts >= self.config["max_attempts"]:
                    raise AbortError(str(e))

        metadata = {
            "model_hash": self.spec.hash_spec(self.config["available_ops"]),
            "trainable_params": _get_param_count(self.model_dir),
            "total_time": all_time,  # includes eval and other metric time
            "total_energy (kWh)": total_log["actual"]["energy (kWh)"],
            "total_co2eq (g)": total_log["actual"]["co2eq (g)"],
            "start_emission": total_log["actual"]["start_emission"],
            "stop_emission": total_log["actual"]["stop_emission"],
            "avg_intensity (gCO2/kWh)": total_log["actual"]["avg_intensity (gCO2/kWh)"],
            "avg_power_usages:": parser.filter_logs(total_log["components"], ["epoch_durations (s)"], nested=True),
            "evaluation_results": evaluation_results,
        }

        total_carbontracker.stop()
        init_carbontracker.stop()
        train_carbontracker.stop()

        del total_std_stream
        del total_out_stream
        del init_carbontracker_std_stream
        del init_carbontracker_out_stream
        del train_carbontracker_std_stream
        del train_carbontracker_out_stream

        return metadata

    def _evaluate_all(self, epochs, steps, add_metrics=None):
        """Runs all the evaluations."""
        train_accuracy = _evaluate(self.estimator, self.input_train_eval, self.config, name="train")
        valid_accuracy = _evaluate(self.estimator, self.input_valid, self.config, name="valid")
        test_accuracy = _evaluate(self.estimator, self.input_test, self.config, name="test")
        train_time = self.estimator.get_variable_value(training_time.TOTAL_TIME_NAME)

        now = time.time()
        sample_metrics = self._compute_sample_metrics()
        predict_time = time.time() - now

        metrics = {
            "epochs": epochs,
            "training_time": train_time,
            "training_steps": steps,
            "train_accuracy": train_accuracy,
            "validation_accuracy": valid_accuracy,
            "test_accuracy": test_accuracy,
            # 'sample_metrics': sample_metrics,
            "predict_time": predict_time,
        }
        if add_metrics is not None:
            for metric, value in add_metrics.items():
                metrics[metric] = value
            return metrics
        else:
            return metrics

    def _compute_sample_metrics(self):
        """Computes the metrics on a fixed batch."""
        sample_metrics = next(self.estimator.predict(input_fn=self.input_sample.input_fn, yield_single_examples=False))

        # Fix the extra batch dimension added by PREDICT
        for metric in sample_metrics:
            if metric in ["logits", "input_grad_norm"]:
                # Batch-shaped tensors take first batch
                sample_metrics[metric] = sample_metrics[metric][: self.input_sample.num_images, Ellipsis]
            else:
                # Other tensors remove batch dimension
                sample_metrics[metric] = sample_metrics[metric][0, Ellipsis]

        return sample_metrics


def _augment_and_evaluate_impl(spec, config, model_dir, epochs_per_eval=5):
    """Augment and evaluate implementation, see augment_and_evaluate docstring."""
    input_augment, input_test = [cifar.CIFARInput(m, config) for m in ["augment", "test"]]
    estimator = _create_estimator(spec, config, model_dir, input_augment.num_images)

    if config["train_seconds"] > 0.0:
        timing = training_time.limit(config["train_seconds"])
    else:
        timing = training_time.limit(None)

    steps_per_epoch = input_augment.num_images / config["batch_size"]  # float
    ckpt = tf.train.latest_checkpoint(model_dir)
    if not ckpt:
        current_step = 0
    else:
        current_step = int(ckpt.split("-")[-1])
    max_steps = int(config["train_epochs"] * steps_per_epoch)

    while current_step < max_steps:
        next_step = current_step + int(epochs_per_eval * steps_per_epoch)
        next_step = min(next_step, max_steps)
        estimator.train(
            input_fn=input_augment.input_fn,
            max_steps=next_step,
            hooks=[timing.train_hook],
            saving_listeners=[timing.saving_listener],
        )
        current_step = next_step

        test_accuracy = _evaluate(estimator, input_test, config)

    metadata = {
        "trainable_params": _get_param_count(model_dir),
        "test_accuracy": test_accuracy,
    }

    return metadata


def _create_estimator(spec, config, model_dir, num_train_images, num_sample_images=None):
    """Creates the TPUEstimator object."""
    # Estimator will save a checkpoint at the end of every train() call. Disable
    # automatic checkpoints by setting the time interval between checkpoints to
    # a very large value.
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        model_dir=model_dir,
        keep_checkpoint_max=3,  # Keeps ckpt at start, halfway, and end
        save_checkpoints_secs=2**30,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=config["tpu_iterations_per_loop"], num_shards=config["tpu_num_shards"]
        ),
    )

    # This is a hack to allow PREDICT on a fixed batch on TPU. By replicating the
    # batch by the number of shards, this ensures each TPU core operates on the
    # entire fixed batch.
    if num_sample_images and config["use_tpu"]:
        num_sample_images *= config["tpu_num_shards"]

    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=config["use_tpu"],
        model_fn=model_builder.build_model_fn(spec, config, num_train_images),
        config=run_config,
        train_batch_size=config["batch_size"],
        eval_batch_size=config["batch_size"],
        predict_batch_size=num_sample_images,
    )

    return estimator


def _evaluate(estimator, input_data, config, name=None):
    """Evaluate the estimator on the input data."""
    steps = input_data.num_images // config["batch_size"]
    results = estimator.evaluate(input_fn=input_data.input_fn, steps=steps, name=name)
    return results["accuracy"]


def _get_param_count(model_dir):
    """Get trainable param count from the model directory."""
    tf.compat.v1.reset_default_graph()
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(checkpoint.model_checkpoint_path + ".meta")
        saver.restore(sess, checkpoint.model_checkpoint_path)
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

    return params
