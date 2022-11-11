from ecnas.api import nasbench101
from ecnas.nas.algorithms import MOO
from ecnas.utils.scale import metrics
from ecnas.utils.plot import latex
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

# Path to tfrecord
# Initialize benchmark
fn_5v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_5V9E_estimate.tfrecord"
nb5v = nasbench101.ECNASBench(fn_5v)

# Define some objectives functions (accessed pre-saved training statistics of trained architectures)
acc = lambda x: (-x["validation_accuracy"])
ene = lambda x: (x["energy (kWh)"])
time = lambda x: (x["training_time"])
co2 = lambda x: (x["co2eq (g)"])

# Search parameters for the MOO-algorithm
budget = 108
pop_size = 20
iter = int(1e2)
ref_point = [1, 1e8]
seed = 42

# Instantiate MOO algorithm
moo = MOO(benchmark=nb5v,seed=seed)

# Run MOO and return the pareto-efficient solutions (pareto_sol), 
# and the corresponding architectures with additional information (pareto_archs)
pareto_sol, pareto_archs = moo.optimize(budget, acc, ene, pop_size, int(iter), ref_point)
