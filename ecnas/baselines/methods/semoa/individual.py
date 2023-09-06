import enum
import uuid
from copy import deepcopy

import numpy as np
from typing import Optional, Dict, List

from ax import Experiment, GeneratorRun, Arm, SearchSpace


class Individual:

    def __init__(
            self,
            budget: int,
            search_space,
            name_file: Optional[str] = 'ea',
            x_coordinate: Optional[Dict] = None,
            experiment: Experiment = None
    ) -> None:

        self._space = search_space
        self._id = uuid.uuid4()
        self._name_file = name_file
        self._x = search_space.sample_configuration().get_dictionary() if not x_coordinate else x_coordinate
        self._age = 0
        self._x_changed = True
        self._fit = None
        self._budget = budget
        self._experiment = experiment
        self._num_evals = 0

    @property
    def fitness(self):
        if self._x_changed:
            self._x_changed = False

            params = deepcopy(self._x)
            params['budget'] = int(self._budget)
            trial_name = '{}-{}'.format(self._id, self._num_evals)
            params['id'] = trial_name

            trial = self._experiment.new_trial(GeneratorRun([Arm(params, name=trial_name)]))
            data = self._experiment.eval_trial(trial)
            self._num_evals += 1

            acc = float(data.df[data.df['metric_name'] == 'val_acc']['mean'])
            ene = float(data.df[data.df['metric_name'] == 'energy']['mean'])

            self._fit = [ene, acc]

        return self._fit

    @property
    def x_coordinate(self) -> Dict:
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        self._x_changed = True
        self._x = value

    @property
    def budget(self):
        return self._budget

    @budget.setter
    def budget(self, value):
        self._x_changed = True
        self._budget = value

    @property
    def id(self):
        return self._id

    def perturb(self, expected_node_changes, expected_edge_changes):
        """
        Method for perturbing an individual to create offspring
        """

        new_x = self.x_coordinate.copy()
        edge_keys = [f"edge_{i}_{j}" for i in range(new_x['vertices']) for j in range(i + 1, new_x['vertices'])]

        # Modify edges
        num_edges = (new_x['vertices'] * (new_x['vertices'] - 1) // 2)
        p_edge_change = expected_edge_changes / num_edges
        for edge_key in edge_keys:
            if np.random.rand() < p_edge_change:
                new_x[edge_key] = 1 - new_x[edge_key]

        # Modify nodes
        if new_x['vertices'] > 2:
            p_node_change = expected_node_changes / (new_x['vertices'] - 2)
            op_keys = [f"op_node_{i}" for i in range(1, new_x['vertices'] - 2)]
            for op_key in op_keys:
                if np.random.rand() < p_node_change:
                    new_op = self._space.sample_configuration()[op_key]
                    while new_op == new_x[op_key]:
                        new_op = self._space.sample_configuration()[op_key]
                    new_x[op_key] = new_op

        child = Individual(self._budget, self._space, self._name_file, new_x, self._experiment)
        child._age = self._age + 1
        return child




