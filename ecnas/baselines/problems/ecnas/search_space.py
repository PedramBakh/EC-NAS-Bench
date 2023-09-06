import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class CustomSearchSpace(CS.ConfigurationSpace):

    def __init__(self, num_nodes, ops_choices):
        super(CustomSearchSpace, self).__init__()

        self.num_nodes = num_nodes
        self.ops_choices = ops_choices
        self.budget_choices = [4, 12, 36, 108]

        # Add number of edges hyperparameter
        self.edges = UniformIntegerHyperparameter("edges", lower=1, upper=num_nodes * (num_nodes - 1) / 2)
        self.add_hyperparameter(self.edges)

        # Add number of vertices hyperparameter
        self.vertices = UniformIntegerHyperparameter("vertices", lower=2, upper=num_nodes)
        self.add_hyperparameter(self.vertices)

        # Add budget hyperparameter
        budget = CategoricalHyperparameter("budget", self.budget_choices)
        self.add_hyperparameter(budget)

        # Add operation hyperparameters
        self.op_names = []
        for i in range(num_nodes - 2):
            op_node = CategoricalHyperparameter(f"op_node_{i}", ops_choices)
            self.add_hyperparameter(op_node)
            self.op_names.append(f"op_node_{i}")

        # Add edge hyperparameters
        self.edge_names = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = CategoricalHyperparameter(f"edge_{i}_{j}", [0, 1])
                self.add_hyperparameter(edge)
                self.edge_names.append(f"edge_{i}_{j}")

    def as_uniform_space(self):
        cs = CS.ConfigurationSpace()

        # Add operation hyperparameters
        for i in range(self.num_nodes - 2):
            op_node = self.get_hyperparameter(f"op_node_{i}")
            cs.add_hyperparameter(op_node)

        # Add edge hyperparameters
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                edge = UniformIntegerHyperparameter(f"edge_{i}_{j}", 0, 1)
                cs.add_hyperparameter(edge)

        # Add budget hyperparameter
        budget = self.get_hyperparameter("budget")
        cs.add_hyperparameter(budget)

        # Add number of vertices hyperparameter
        vertices = self.get_hyperparameter("vertices")
        cs.add_hyperparameter(vertices)

        # Add number of edges hyperparameter
        edges = self.get_hyperparameter("edges")
        cs.add_hyperparameter(edges)

        return cs

    def as_ax_space(self):
        from ax import ParameterType, ChoiceParameter, SearchSpace, FixedParameter

        parameters = []

        i = FixedParameter('id', ParameterType.STRING, 'dummy')
        parameters.append(i)

        # Add operation hyperparameters
        for i in range(self.num_nodes - 2):
            op_node = ChoiceParameter(f"op_node_{i}", ParameterType.STRING, values=self.ops_choices)
            parameters.append(op_node)

        # Add edge hyperparameters
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                edge = ChoiceParameter(f"edge_{i}_{j}", ParameterType.INT, values=[0, 1])
                parameters.append(edge)

        # Add budget hyperparameter
        budget = ChoiceParameter("budget", ParameterType.INT, values=self.budget_choices)
        parameters.append(budget)

        # Add number of vertices hyperparameter
        vertices = self.get_hyperparameter("vertices")
        vertices_param = ChoiceParameter("vertices", ParameterType.INT, values=[vertices.lower, vertices.upper])
        parameters.append(vertices_param)

        # Add number of edges hyperparameter
        edges = self.get_hyperparameter("edges")
        edges_param = ChoiceParameter("edges", ParameterType.INT, values=[edges.lower, edges.upper])
        parameters.append(edges_param)

        return SearchSpace(parameters=parameters)

    def sample_hyperparameter(self, hp_name):
        import numpy as np
        if hp_name in self.op_names:
            return np.random.choice(self.ops_choices)
        elif hp_name in self.edge_names:
            return np.random.randint(0, 2)
        else:
            raise ValueError(f"Hyperparameter {hp_name} not found in search space")

    def get_choices(self, hp_name):
        if hp_name in self.op_names:
            return self.ops_choices

        return [0, 1]

    def is_mutable_hyperparameter(self, hp_name):
        return hp_name in self.ops_choices or hp_name in self.edge_names
