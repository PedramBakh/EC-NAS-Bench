from baselines import save_experiment
from baselines.methods.shemoa import SHEMOA
from baselines.methods.shemoa import Mutation, Recombination, ParentSelection

from baselines.problems.ecnas import ecnasSearchSpace
from baselines.problems import get_ecnas
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    num_nodes = 4
    ops_choices = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]

    # Parameters ecnas
    N_init = 10
    min_budget = 4
    max_budget = 108
    max_function_evals = 100
    mutation_type = Mutation.UNIFORM
    recombination_type = Recombination.UNIFORM
    selection_type = ParentSelection.TOURNAMENT

    #################
    #### SH-EMOA ####
    #################
    trials = 10

    for run in tqdm(range(trials)):
        np.random.seed(run)
        search_space = ecnasSearchSpace(num_nodes, ops_choices)
        experiment = get_ecnas(num_nodes, ops_choices, "SHEMOA")
        ea = SHEMOA(
            search_space,
            experiment,
            N_init,
            min_budget,
            max_budget,
            mutation_type=mutation_type,
            recombination_type=recombination_type,
            selection_type=selection_type,
            total_number_of_function_evaluations=max_function_evals,
        )
        ea.optimize()

        res = experiment.fetch_data().df
        save_experiment(res, f"experiments/shemoa/{num_nodes}v_{experiment.name}_{run}.pickle")
