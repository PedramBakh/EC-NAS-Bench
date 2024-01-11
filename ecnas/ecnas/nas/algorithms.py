# Load libraries
import numpy as np
import random as rnd
from comocma import NonDominatedList
from tqdm import tqdm

class MOO:
    """
    This class implements the MOO algorithm.
    Note: This implementation is outdated and was not utilized in the paper.
    It is preserved here solely for reference purposes.
    """
    def __init__(self, benchmark, seed, single_objective=False):
        self.Benchmark = benchmark
        self.SingleObjective = single_objective
        # Set random seed
        rnd.seed(seed)
        np.random.seed(seed)

        # Store all results as a data frame
        self.Results = ([], [])

    def get_random_archs(self, n):
        archs = []
        all_lookup_keys = list(self.Benchmark.hash_iterator())
        for i in range(n + 1):
            rnd_key = rnd.choice(all_lookup_keys)
            model_spec, _ = self.Benchmark.get_metrics_from_hash(rnd_key)
            matrix, labels = model_spec["module_adjacency"], model_spec["module_operations"]
            archs.append((matrix, labels))
        return archs

    def evaluate_archs(self, archs, budgets, f1, f2):
        scores = []
        epochs = []
        for budget in budgets:
            for arch in archs:
                matrix, labels = arch
                try:
                    metrics = self.Benchmark.query(self.Benchmark.get_model_spec(matrix, labels), budget)
                    f_vals = f1(metrics), f2(metrics)
                    scores.append(f_vals)
                    epochs.append(budget)
                    self.Results[0].append((f_vals))
                    self.Results[1].append(())
                except:
                    scores.append((1, 1e6))
                    epochs.append(budget)
                    self.Results[0].append((1, 1e6))
                    self.Results[1].append(())

        return scores, epochs

    def linear_rank_sample(self, ndom, n):
        if len(ndom) == 1:
            return np.zeros(n, dtype=int)

        m = len(ndom)  # no. parents
        parent_indices = np.arange(m)

        eta_plus = 1.9  # parent selection pressure
        eta_minus = 2.0 - eta_plus  # offspring selection pressure
        p = np.array(
            [1.0 / m * (eta_plus - (eta_plus - eta_minus) * i / (m - 1)) for i in parent_indices]
        )  # parent probabilities
        eps = 1e-3
        assert p.sum() < 1.0 + eps
        # weighted probability for all parents
        offspring_indices = np.random.choice(parent_indices, size=n, p=p)

        return offspring_indices

    def perturb(self, matrix, labels, available_ops):
        v = len(matrix)
        # make matrix an np.int8 array
        matrix = np.array(matrix, dtype=np.int8)
        # remove first and last element of labels
        labels = labels[1 : v - 1]
        # matrix = np.zeros([v, v], dtype=np.int8)

        ops = np.zeros([len(labels) - 2], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        unchanged = True
        while unchanged:  # ensure that the point is indeed changed
            for i in range((v * (v - 1)) // 2):
                row = idx[0][i]
                col = idx[1][i]
                em = 2.0  # expected number of edge modifications
                p = em / (v * (v - 1) // 2)  # mutation probability
                modify = np.random.choice([0, 1], p=np.array([1 - p, p]))
                if modify:
                    matrix[row, col] = (matrix[row, col] + 1) % 2
                    unchanged = False

            # randomly select a number of nodes to mutate
            for j in range(v - 2):
                em = 0.5  # expected number of node changes
                p = em / (v - 2)  # mutation probability
                modify = np.random.choice([0, 1], p=np.array([1 - p, p]))
                if modify:
                    choice = rnd.choice(available_ops)
                    while choice != labels[j]:
                        choice = rnd.choice(available_ops)
                    labels[j] = choice
                    # add input and output nodes
                    unchanged = False

        labels = ["input"] + labels + ["output"]
        return matrix, labels

    def optimize(self, budgets, f1, f2, population_size, max_iter, ref_point):
        # get random architectures
        archs = self.get_random_archs(population_size)

        # init non-dominated set
        ndom = NonDominatedList(reference_point=ref_point)

        # list for storing archs, objectives and hypervolume
        ndom_archs = []
        for i in tqdm(
            range(int(max_iter / 4))
        ):  # We consider all 4 budgets per iteration so max_iter/4 is func. evals.
            # evaluate architectures
            f_vals, epochs = self.evaluate_archs(archs, budgets, f1, f2)
            print(f_vals)
            # store matrix and labels of non-dominated solutions if not dominated
            for arch, f_val, epoch in zip(archs, f_vals, epochs):
                if not ndom.dominates(f_val):
                    ndom_archs.append((arch, f_val, float(ndom.contributing_hypervolume(f_val)), epoch))
            # add evaluations to ndom
            ndom.add_list(f_vals)

            # remove dominated solutions from non-dominate set
            del_indices = []
            for j, dp in enumerate(ndom_archs):
                sols = [list(sol) for sol in ndom]
                if list(dp[1]) not in sols:
                    del_indices.append(j)
            del_indices = sorted(del_indices, reverse=True)
            for j in del_indices:
                del ndom_archs[j]

            # sort ndom by contributing hypervolume of ndom_archs
            ndom_temp, ndom_archs_temp = zip(*sorted(zip(ndom, ndom_archs), key=lambda x: x[1][2], reverse=True))
            ndom = NonDominatedList(list(ndom_temp), reference_point=ref_point)
            ndom_archs = list(ndom_archs_temp)

            # get offspring indices
            offspring_indices = self.linear_rank_sample(ndom, population_size)

            # perturb architectures until population size is reached
            offspring_archs = []
            for j in range(population_size):
                # get parent matrix and labels
                parent_matrix, parent_labels = ndom_archs[offspring_indices[j]][0]
                # get available operations
                available_ops = self.Benchmark.get_intermediate_ops()

                matrix, labels = None, None
                while True:
                    try:
                        # perturb parent
                        matrix, labels = self.perturb(parent_matrix, parent_labels, available_ops)
                        # check if architecture is valid
                        self.Benchmark._check_spec(self.Benchmark.get_model_spec(matrix, labels))
                    except Exception:
                        break
                    break

                # add to new_archs
                offspring_archs.append((matrix, labels))

            # update archs
            archs = offspring_archs.copy()

        return ndom, ndom_archs
