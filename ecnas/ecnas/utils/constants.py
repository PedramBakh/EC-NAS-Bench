import os
from pathlib import Path
import numpy as np

RC_PARAMS = {
    "font.size": 16,
    #'font.sans-serif': 'Arial',
    "font.weight": "bold",
    "legend.frameon": True,
    "axes.labelsize": 20,
    "axes.titlesize": 16,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.fontsize": 18,
    "text.usetex": True,
    "text.latex.preamble": r"\boldmath",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}

dirname = os.path.dirname(__file__)
tab_benchmark = os.path.join(dirname, "data/tabular_benchmarks")
path_to_experiment_data = os.path.join(Path(dirname).parent, "experiments", "runs")

TABULAR_COLLECTION = {
    # Full benchmark surrogate
    "7V_surrogate": os.path.join(tab_benchmark, "energy_7V9E_surrogate.tfrecord"),
    # Hardware specific subspaces
    "4V_quadrortx6000": os.path.join(tab_benchmark, "hardware", "energy_4V9E_quadrortx6000.tfrecord"),
    "4V_3060": os.path.join(tab_benchmark, "hardware", "energy_4V9E_rtx3060.tfrecord"),
    "4V_3090": os.path.join(tab_benchmark, "hardware", "energy_4V9E_rtx3090.tfrecord"),
    "4V_titanxp": os.path.join(tab_benchmark, "hardware", "energy_4V9E_titanxp.tfrecord"),
    # Original NASBench-101
    "NB101": os.path.join(tab_benchmark, "nasbench_full.tfrecord"),
}

EXPERIMENT_COLLECTION = {
    "RS": os.path.join(path_to_experiment_data, "rs"),
    "SEMOA": os.path.join(path_to_experiment_data, "semoa"),
    "SHEMOA": os.path.join(path_to_experiment_data, "shemoa"),
    "MSEHVI": os.path.join(path_to_experiment_data, "msehvi"),
}

COLORS = {
    "dark_yellow": "#FF8C00",
    "dark_green": "#006400",
    "dark_blue": "#00008B",
    "red": "#FF0000",
    "black": "#000000",
    "grey": "#808080",
}
eps = 1e-10
REF_SOLUTIONS = {"ACC_ENERGY": np.array([0.5, 1e6])}
