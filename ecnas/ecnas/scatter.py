import itertools
from ecnas.api import nasbench101
from ecnas.nas.algorithms import MOO
import matplotlib.pyplot as plt
import numpy as np
from ecnas.utils.scale import metrics
from ecnas.utils.plot.radar import ComplexRadar
from ecnas.utils.plot import latex

plt.rcParams["text.usetex"] = True

seed = 123
np.random.seed(seed)

fn_5v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_5V9E_estimate.tfrecord"
fn_4v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_4V9E_estimate.tfrecord"

nb5v = nasbench101.ECNASBench(fn_4v, seed)
nb4v = nasbench101.ECNASBench(fn_4v, seed)
nb4v.info()
nb5v.info()


def get_data(benchmark):
    model_test = []
    model_val = []
    model_energy = []
    model_co2 = []
    model_time = []
    model_param = []
    model_epochs = []
    for model in benchmark.computed_statistics:
        computed_stat, metrics = nb5v.get_metrics_from_hash(model)
        for epoch in metrics:
            test_acc = metrics[epoch][0]["final_test_accuracy"]
            val_acc = metrics[epoch][0]["final_validation_accuracy"]
            energy = metrics[epoch][0]["energy (kWh)"]
            co2 = metrics[epoch][0]["co2eq (g)"]
            time = metrics[epoch][0]["final_training_time"]
            param = computed_stat["trainable_parameters"]

            model_epochs.append(epoch)
            model_test.append(test_acc)
            model_val.append(val_acc)
            model_energy.append(energy)
            model_co2.append(co2)
            model_time.append(time)
            model_param.append(param)

    return model_epochs, model_test, model_val, model_energy, model_co2, model_time, model_param


model_epochs, model_test, model_val, model_energy, model_co2, model_time, model_param = get_data(nb5v)

order = [1, 2, 0, 3]
colors = itertools.cycle(["r", "g", "b", "k"])

fig = plt.figure(figsize=latex.get_size_iclr2023())
for i in range(len(model_val)):
    plt.scatter(
        model_time[i], model_val[i], color=next(colors), marker="^", alpha=0.5, label=str(model_epochs[i]) + " epochs"
    )
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", scatterpoints=1)
plt.ylabel("$P_{v}$")
plt.xlabel("$E$")
plt.ylim(0.0, 1.0)
plt.savefig("scatter_5v.pdf", dpi=300, bbox_inches="tight")

plt.clf()
model_epochs, model_test, model_val, model_energy, model_co2, model_time, model_param = get_data(nb4v)
for i in range(len(model_val)):
    plt.scatter(
        model_time[i], model_val[i], color=next(colors), marker="^", alpha=0.5, label=str(model_epochs[i]) + " epochs"
    )
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", scatterpoints=1)
plt.ylabel("$P_{v}$")
plt.xlabel("$E$")
plt.ylim(0.0, 1.0)
plt.savefig("scatter_4v.pdf", dpi=300, bbox_inches="tight")
