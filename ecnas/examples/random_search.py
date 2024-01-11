import sys
sys.path.append("../")

from ax import Models
from tqdm import tqdm
from baselines import save_experiment
from baselines.problems import get_ecnas
import numpy as np

if __name__ == "__main__":
    # Parameters Flowers
    N = 1000  # Number of samples (it is not important)

    num_nodes = 4
    ops_choices = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]

    trials = 10

    for run in tqdm(range(trials)):
        np.random.seed(run)
        experiment = get_ecnas(name="rs", num_nodes=num_nodes, num_choices=ops_choices)

        for _ in tqdm(range(N)):
            experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
            experiment.fetch_data()

        res = experiment.fetch_data().df
        save_experiment(res, f"experiments/rs/{num_nodes}v_{experiment.name}_{run}.pickle")