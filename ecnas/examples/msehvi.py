import sys

sys.path.append("../")
from tqdm import tqdm
from baselines.problems import get_ecnas
from baselines.problems.ecnas import ecnasSearchSpace
from baselines.problems.ecnas import discrete_ecnas
from baselines.problems.flowers import discrete_flowers
from baselines.problems import get_flowers
from baselines.problems.fashion import discrete_fashion
from baselines.problems import get_fashion
from baselines.problems import get_branin_currin, BraninCurrinEvalFunction
from baselines import save_experiment
from baselines.methods.msehvi.msehvi import MSEHVI
from ax import Models
import numpy as np

if __name__ == "__main__":
    # Parameters Flowers
    # N_init = 50 # Number of initial random samples
    # N = 20000   # Number of MS-EHVI samples (it is not important)
    # discrete_f = discrete_flowers       # Discrete function
    # discrete_m = 'num_params'           # Name of the discrete metric
    # experiment = get_flowers('MSEHVI')  # Function to get the problem

    # Parameters Fashion
    # N_init = 10 # Number of initial random samples
    # N = 20000   # Number of MS-EHVI samples (it is not important)
    # discrete_f = discrete_fashion       # Discrete function
    # discrete_m = 'num_params'           # Name of the discrete metric
    # experiment = get_fashion('MSEHVI')  # Function to get the problem

    # Parameters Branin Crunin
    # N_init = 10
    # N = 100
    # discrete_f = BraninCurrinEvalFunction().discrete_call
    # discrete_m = 'a'
    # experiment = get_branin_currin('MSEHVI')

    # Parameters ECNAS
    N_init = 10
    N = 100
    num_nodes = 4
    ops_choices = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]
    discrete_f = discrete_ecnas
    discrete_m = "energy"

    #################
    #### MS-EHVI ####
    #################

    trials = 10

    for run in tqdm(range(0, trials)):
        # set random seed
        np.random.seed(run)
        experiment = get_ecnas(name="MSEHVI", num_nodes=num_nodes, num_choices=ops_choices)

        # Random search initialization
        for _ in range(N_init):
            experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
            experiment.fetch_data()

        # Proper guided search
        msehvi = MSEHVI(experiment, discrete_m, discrete_f)
        for _ in tqdm(range(N)):
            msehvi.step()

        res = experiment.fetch_data().df

        save_experiment(res, f"experiments/msehvi/{num_nodes}v_{experiment.name}_{run}.pickle")
