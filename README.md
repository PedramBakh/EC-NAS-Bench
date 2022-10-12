# EC-NAS-Bench
![PythonVersion](https://img.shields.io/badge/Made%20with-Python%203.8-1f425f.svg?logo=python)
![Activity](https://img.shields.io/github/last-commit/PedramBakh/ec-nas-bench)
![License](https://img.shields.io/github/license/PedramBakh/ec-nas-bench)



## Abstract
The demand for large-scale computational resources for Neural Architecture Search (NAS) has been lessened by tabular benchmarks for NAS. Evaluating NAS strategies is now possible on extensive search spaces and at a moderate computational cost. But so far, NAS has mainly focused on maximising performance on some hold-out validation/test set. However, energy consumption is a partially conflicting objective that should not be neglected. We hypothesise that constraining NAS to include the energy consumption of training the models could reveal a subspace of undiscovered architectures that are more computationally efficient with a smaller carbon footprint. To support the hypothesis, an existing tabular benchmark for NAS is augmented with the energy consumption of each architecture. We then perform multi-objective optimisation that includes energy consumption as an additional objective. We demonstrate the usefulness of multi-objective NAS for uncovering the trade-off between performance and energy consumption as well as for finding more energy-efficient architectures. The updated tabular benchmark is open-sourced to encourage the further exploration of energy consumption-aware NAS.

**This repository contains code for the paper [Energy Consumption-Aware Tabular Benchmarks For NAS](google.com).**

## Getting started
To install the requirements, using Conda, run the following command:
```sh 
$ conda env create --name envname --file=environment.yml
```
Due to possible dependency issues for newer hardware, pip requirements are also included separately.
To install the requirements, using pip, un the following command:
```sh 
$ pip install -r requirements.txt
```

## Benchmarks
| Benchmark | Description | Dataset | Space | No. Architectures | Datapoints | Surrogate | Included
| --- | --- | --- | --- | --- | --- | --- | --- |
| EC-NAS-Bench | Image Classification | CIFAR-10 | 5V | 2632 | 10528 | Yes | Yes  
| EC-NAS-Bench | Image Classification | CIFAR-10 | 4V | 91 | 364 | Yes | Yes

It is also possible to use the MOO-algorithm with the original NAS-Bench-101 dataset, which can be downloaded using `ecnas/utils/get_bencmarks.py`.

## Example usage
The following code initializes the EC-Nas-Bench with the 5V space, runs the MOO algorithm for validation accuracy and energy consumption, and returns the set of Pareto-efficient solutions.
```python
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

# Run MOO and return the pareto-efficient solutions (pareto_sol), 
# and the corresponding architectures with additional information (pareto_archs)
pareto_sol, pareto_archs = moo.optimize(budget, acc, ene, pop_size, int(iter), ref_point)
```
Using the results from the optimization above, the Pareto-front, including the extreme points and the knee point, can be plotted with the following code. Additional examples can be found in `attainment.py`, `scatter.py` and `complex_radar.py`.
```python
# Define colors
yellow = "#FFD700"
dark_green = "#006400"
dark_blue = "#00008B"
grey = "#808080"

# Get x and y data points
x, y = [i[0] for i in ndom], [i[1] for i in ndom]

# Determine knee point
angles = metrics.get_bend_angles(x, y)
idx = np.argmin(angles)
knee_x, knee_y = x[idx], y[idx]

# Get metrics for conveniance
df = metrics.get_dataframe(ndom_archs, nb5v, budget)

# Plot pareto front
fig = plt.figure()
plt.scatter(x, y, color=grey)
plt.scatter(-df.iloc[0].values.tolist()[1], df.iloc[0].values.tolist()[2], color=dark_blue, marker="^")
plt.scatter(-df.iloc[1].values.tolist()[1], df.iloc[1].values.tolist()[2], color=dark_green, marker="^")
plt.scatter(knee_x, knee_y, color=yellow, marker="^")

plt.xlabel("$r_{0},-P_v$")
plt.ylabel("$r_{1},E$(kWh)")
plt.ylim(0.01, 1.4)
plt.xlim(-0.95, -0.83)
plt.savefig(f"front_{budget}.png", dpi=300, bbox_inches="tight")
```

## Citation
Kindly use the following BibTeX entry if you use the code in your work.
```
TBA
```
