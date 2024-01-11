from ax import Metric
from ax import MultiObjective
from ax import ObjectiveThreshold
from ax import MultiObjectiveOptimizationConfig
from baselines import MultiObjectiveSimpleExperiment
from .ecnasnet import evaluate_network
from .search_space import CustomSearchSpace


def get_ecnas(num_nodes, num_choices, name=None):
    val_acc = Metric("val_acc", True)
    tst_acc = Metric("tst_acc", True)
    train_time = Metric("train_time", True)
    energy = Metric("energy", True)
    co2eq = Metric("co2eq", True)
    params = Metric("params", True)

    objective = MultiObjective([energy, val_acc])
    thresholds = [ObjectiveThreshold(energy, 0.0), ObjectiveThreshold(val_acc, 0.0)]
    optimization_config = MultiObjectiveOptimizationConfig(objective=objective, objective_thresholds=thresholds)

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=CustomSearchSpace(num_nodes, num_choices).as_ax_space(),
        eval_function=evaluate_network,
        optimization_config=optimization_config,
        extra_metrics=[tst_acc, co2eq, params, train_time],
    )
