from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.mobananas import get_MOSHBANANAS
from tqdm import tqdm
from baselines.problems import get_ecnas
from baselines.problems.ecnas import ecnasSearchSpace

if __name__ == "__main__":
    # Parameters Flowers
    # N_init = 10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 10000
    # num_arch=20
    # select_models=10
    # eta=3
    # search_space = FlowersSearchSpace()
    # experiment = get_flowers('MOSHBANANAS')

    # Parameters Fashion
    # N_init = 10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 400
    # num_arch=20
    # select_models=10
    # eta=3
    # search_space = FashionSearchSpace()
    # experiment = get_fashion('MOSHBANANAS')

    # Parameters ECNAS
    num_nodes = 4
    ops_choices = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]

    N_init = 10
    min_budget = 4
    max_budget = 108
    max_function_evals = 10000
    num_arch = 20
    select_models = 10
    eta = 3
    search_space = ecnasSearchSpace(num_nodes, ops_choices)
    experiment = get_ecnas(num_nodes, ops_choices, "MOSHBANANAS")

    #####################
    #### MOSHBANANAS ####
    #####################
    get_MOSHBANANAS(
        experiment,
        search_space,
        initial_samples=N_init,
        select_models=select_models,
        num_arch=num_arch,
        min_budget=min_budget,
        max_budget=max_budget,
        function_evaluations=max_function_evals,
        eta=eta,
    )
    save_experiment(experiment, f"{experiment.name}.pickle")
