from ecnas.api import nasbench101
from ecnas.nas.algorithms import MOO
import matplotlib.pyplot as plt
import numpy as np
from ecnas.utils.scale import metrics
from ecnas.utils.plot.radar import ComplexRadar
from ecnas.utils.plot import latex

plt.rcParams["text.usetex"] = True
from kneed import DataGenerator, KneeLocator

seed = 123
np.random.seed(seed)

fn_5v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_5V9E_estimate.tfrecord"
fn_4v = "/path/to/ecnas/ecnas/utils/data/tabular_benchmarks/energy_4V9E_estimate.tfrecord"

nb5v = nasbench101.ECNASBench(fn_5v, seed)
nb4v = nasbench101.ECNASBench(fn_4v, seed)

# objectives
acc = lambda x: (-x["validation_accuracy"])
ene = lambda x: (x["energy (kWh)"])
time = lambda x: (x["training_time"])
co2 = lambda x: (x["co2eq (g)"])

# search parameters
budget = 108
pop_size = 20
iter = 1e1
ref_point = [1, 1e8]

# run multi-objective optimization
moo = MOO(nb4v, seed=seed)
ndom, ndom_archs = moo.optimize(budget, acc, ene, pop_size, int(iter), ref_point)
so, so_arch = moo.optimize(budget, acc, acc, pop_size, int(iter), ref_point)

yellow = "#FFD700"
dark_yellow = "#FF8C00"
dark_green = "#006400"
dark_blue = "#00008B"
red = "#FF0000"
black = "#000000"
grey = "#808080"

# plot pareto front
fig = plt.figure(figsize=latex.get_size_iclr2023())
x = [i[0] for i in ndom]
y = [i[1] for i in ndom]
angles = metrics.get_bend_angles(x, y)
idx = np.argmin(angles)
knee_x, knee_y = x[idx], y[idx]

print(knee_x, knee_y)
df = metrics.get_dataframe(ndom_archs, nb5v, budget)
plt.scatter(x, y, color=grey)
plt.scatter(-df.iloc[0].values.tolist()[1], df.iloc[0].values.tolist()[2], color=dark_blue, marker="^")
plt.scatter(-df.iloc[1].values.tolist()[1], df.iloc[1].values.tolist()[2], color=dark_green, marker="^")
plt.scatter(knee_x, knee_y, color=yellow, marker="^")

plt.xlabel("$r_{0},-P_v$")
plt.ylabel("$r_{1},E$(kWh)")
plt.ylim(0.01, 1.4)
plt.xlim(-0.95, -0.83)
plt.savefig(f"front_{budget}.pdf", dpi=300, bbox_inches="tight")

# complex radar plot
format_cfg = {
    "rad_ln_args": {"visible": True},
    "outer_ring": {"visible": True},
    "angle_ln_args": {"visible": True},
    "rgrid_tick_lbls_args": {"fontsize": 6},
    "theta_tick_lbls": {"fontsize": 9},
    "theta_tick_lbls_pad": 3,
}


ranges = metrics.get_ranges(nb5v, budget)

fig = plt.figure(figsize=latex.get_size_iclr2023())
radar = ComplexRadar(fig, df.columns, ranges, n_ring_levels=5, show_scales=True, format_cfg=format_cfg)
radar.plot(df.iloc[0].values.tolist(), color=dark_blue, linewidth=0.8)
radar.fill(df.iloc[0].values.tolist(), alpha=0.4, color=dark_blue)

radar.plot(df.iloc[1].values.tolist(), color=dark_green, linewidth=0.8)
radar.fill(df.iloc[1].values.tolist(), alpha=0.4, color=dark_green)

radar.plot(df.iloc[idx].values.tolist(), color=yellow, linewidth=0.8)
radar.fill(df.iloc[idx].values.tolist(), alpha=0.4, color=yellow)

plt.savefig("radar.pdf", dpi=300, bbox_inches="tight")

plt.clf()
df_so = metrics.get_dataframe(so_arch, nb5v, budget)
fig = plt.figure(figsize=latex.get_size_iclr2023())
radar = ComplexRadar(fig, df.columns, ranges, n_ring_levels=5, show_scales=True, format_cfg=format_cfg)

radar.plot(df_so.iloc[0].values.tolist(), color=red, linewidth=0.8)
radar.fill(df_so.iloc[0].values.tolist(), alpha=0.4, color=red)
plt.savefig(f"so_radar_{budget}.pdf", dpi=300, bbox_inches="tight")
