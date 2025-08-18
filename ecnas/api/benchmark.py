import os
import sys
from abc import ABC, abstractmethod
import time

# Add submodule to path
path_nasbench = os.path.join(os.path.dirname(os.getcwd()), "ecnas", "vendors", "ec_nasbench")
path_carbontracker = os.path.join(os.path.dirname(os.getcwd()), "ec_carbontracker")
sys.path.append(path_nasbench)
sys.path.append(path_carbontracker)


class Benchmark(ABC):
    def __init__(self, dataset_file=None, seed=None):
        print("Setting up tabular benchmark from file... This may take a few minutes...")
        print(dataset_file)
        start = time.time()

        self._setup()

        elapsed = time.time() - start
        print("Loaded dataset in %d seconds" % elapsed)

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def query(self):
        pass

    @abstractmethod
    def get_metrics_from_hash(self):
        pass

    @abstractmethod
    def get_metrics_from_spec(self):
        pass

    @abstractmethod
    def _check_spec(self):
        pass

    @abstractmethod
    def info(self):
        pass
