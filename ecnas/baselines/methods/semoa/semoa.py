from baselines.core.multiobjective_experiment import MultiObjectiveSimpleExperiment
import numpy as np
from .individual import Individual
from ax import Models, Experiment, Data, MultiObjective
from comocma import NonDominatedList

class SEMOA:
    def __init__(
        self,
        search_space,
        experiment: Experiment,
        population_size: int = 10,
        num_generations: int = 100,
        min_budget: int = 4,
        max_budget: int = 108,
        expected_node_changes: float = 0.5,  # expected number of mutations per individual
        expected_edge_changes: float = 2.0,  # expected number of edge changes per individual
        eta_p: float = 1.9,  # parent selection pressure
        eta_o: float = 2.0 - 1.9,  # offspring selection pressure
    ):
        assert min_budget <= max_budget
        assert min_budget in [4, 12, 36, 108] and max_budget in [4, 12, 36, 108]
        assert 0 < population_size
        assert 0 < num_generations
        assert 0 < expected_node_changes
        assert 0 < expected_edge_changes
        assert 0 < eta_p
        assert 0 < eta_o

        self.current_budget = min_budget
        self.experiment = experiment
        self.budgets = [min_budget * 3**i for i in range(int(np.log(max_budget / min_budget) / np.log(3)) + 1)]
        self.population_size = population_size
        self.num_generations = num_generations
        self.expected_node_changes = expected_node_changes
        self.expected_edge_changes = expected_edge_changes
        self.eta_p = eta_p
        self.eta_o = eta_o
        self.search_space = search_space
        self.best = NonDominatedList(reference_point=[1e6, 100])

        self.experiment_simple = MultiObjectiveSimpleExperiment(
            name=self.experiment.name,
            search_space=self.experiment.search_space,
            eval_function=self.experiment.evaluation_function,
            optimization_config=self.experiment.optimization_config,
        )

        self.metrics = [m for m in self.experiment.optimization_config.objective.metrics]

        self.population = [
            Individual(self.budgets[0], self.search_space, name_file="dummy.txt", experiment=self.experiment)
            for _ in range(self.population_size)
        ]

    def select_parents(self):
        """
        (Linear rank sample)
        Method for selecting offspring from the population
        """
        if len(self.population) == 1:
            return np.zeros(self.population_size, dtype=int)

        m = len(self.population)
        parent_indices = np.arange(m)
        p = np.array([1.0 / m * (self.eta_p - (self.eta_p - self.eta_o) * i / (m - 1)) for i in parent_indices])
        eps = 1e-3
        assert p.sum() < 1.0 + eps
        selection = np.random.choice(parent_indices, size=self.population_size, p=p)

        return selection

    def step(self):
        """
        Method for evolving the population
        """
        # Sort population by contributing hypervolume
        self.population.sort(
            key=lambda x: self.best.contributing_hypervolume((x.fitness[0], x.fitness[1])), reverse=True
        )

        # Select parents
        parent_indices = self.select_parents()
        parents = [self.population[idx] for idx in parent_indices]

        # Generate offspring
        offspring = self.generate_offspring(parents)

        # Extend offspring to population
        self.population = self.population + offspring

        # Evaluate population
        evals = []
        for individual in self.population:
            _fit = individual.fitness
            if not _fit == [0, 0]:
                evals.append((_fit[0], _fit[1]))

        # Update population based on survivors
        survivors = []
        survivors_fit = []
        for ind in self.population:
            ind_fit = (ind.fitness[0], ind.fitness[1])
            if ind_fit in evals and ind_fit not in survivors_fit:
                survivors.append(ind)
                survivors_fit.append(ind_fit)

        # Update non-dominated list
        self.best.add_list(evals)
        self.population = survivors.copy()

    def generate_offspring(self, parents):
        """
        Method for generating offspring from parents
        """
        offspring = []
        for i in range(len(parents)):
            offspring.append(parents[i].perturb(self.expected_node_changes, self.expected_edge_changes))

        return offspring

    def optimize(self):
        from tqdm import tqdm

        for _ in tqdm(range(self.num_generations // len(self.budgets))):
            for b in self.budgets:
                self.current_budget = b
                for individual in self.population:
                    individual.budget = b

                self.step()

        return self.best
