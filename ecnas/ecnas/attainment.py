import pandas
from ecnas.api import nasbench101
from ecnas.nas.algorithms import MOO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ecnas.utils.scale import metrics
from ecnas.utils.plot.radar import ComplexRadar
from ecnas.utils.plot import latex

plt.rcParams["text.usetex"] = True
from eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot

fn_5v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_5V9E_estimate.tfrecord"
fn_4v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_4V9E_estimate.tfrecord"

nb5v = nasbench101.ECNASBench(fn_5v)
nb4v = nasbench101.ECNASBench(fn_4v)

# objectives
acc = lambda x: (-x["validation_accuracy"])
ene = lambda x: (x["energy (kWh)"])
time = lambda x: (x["training_time"])
co2 = lambda x: (x["co2eq (g)"])

# search parameters
budget = 108
pop_size = 20
iter = int(1e2)
ref_point = [1, 1e8]

n_seeds = 100  # number of seeds to run
n_trials = 1  # number of trials per seed
n_obj = 2  # number of objectives
runs = n_seeds * n_trials  # total number of runs
n_samples = 5

levels = [runs // 4, runs // 2, 3 * runs // 4]

X = np.ndarray((runs, n_samples, n_obj), dtype=object)
k = 0
for i in tqdm(range(n_seeds), desc="seeds", position=True, leave=True):
    moo = MOO(nb5v, seed=i)
    for j in range(n_trials):
        ndom, ndom_archs = moo.optimize(budget, acc, ene, pop_size, int(iter), ref_point)
        ndom = np.array(ndom[:n_samples])
        X[k] = ndom
        k += 1

# plot attainment surface
surfs = get_empirical_attainment_surface(costs=X, levels=levels)

_, ax = plt.subplots()
eaf_plot = EmpiricalAttainmentFuncPlot()
eaf_plot.plot_surface_with_band(ax, color="red", label="median", surfs=surfs)
plt.xlabel("$r_{0},-P_v$")
plt.ylabel("$r_{1},E$(kWh)")
ax.legend()
ax.grid()
plt.savefig("attainment.pdf", dpi=300, bbox_inches="tight")
