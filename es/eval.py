from typing import Optional, Sequence

import numpy as np
import multiprocessing as mp

from npne import random

__all__ = ["SimpleEvaluator"]


class SimpleEvaluator:
    """
    Simple on-premise multiprocessing evaluator. Implement `evaluate_solution` to
    implement a new evaluator that inherits this class.

    Parameters:
        num_workers (Optional[int]): Number of parallel workers; if not specified, use
            the max number of available CPU cores (default: None).
    """

    def __init__(self, num_workers: Optional[int] = None) -> None:
        if num_workers is None:
            num_workers = mp.cpu_count()
        self.num_workers = num_workers

    def evaluate_solution(self, seed: int, solution: np.ndarray) -> None:
        """
        Evaluate a single solution candidate and return its fitness score. Implement
        a new evaluator class by implementing this method.

        Parameters:
            seed (int): A random seed for ensuring reproducibility of evaluation.
            solution (np.ndarray): A solution candidate represented with a NumPy array.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def evaluate(self, solutions: Sequence[np.ndarray]) -> np.ndarray:
        """
        Evaluate the argument set of solutions in subprocesses and return their fitness
        scores in a vector. Note that this method will fail if `evaluate_solution()` is
        not implemented.

        Parameters:
            solutions (Sequence[np.ndarray]): A sequence of solution candidates.

        Returns:
            A NumPy array of fitness scores.
        """
        results = []
        seeds = random.randint(2 ** 31 - 1, size=len(solutions)).tolist()
        with mp.Pool(self.num_workers) as pool:
            for seed, solution in zip(seeds, solutions):
                result = pool.apply_async(self.evaluate_solution, args=(seed, solution))
                results.append(result)
            fitness = [r.get() for r in results]
        return np.array(fitness)
