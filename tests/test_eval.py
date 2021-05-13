from unittest import TestCase

import numpy as np

from es import SimpleEvaluator


class _ExampleSimpleEvaluator(SimpleEvaluator):
    def evaluate_solution(self, seed: int, solution: np.ndarray) -> float:
        return -np.sum(solution ** 2)


class TestSimpleEvaluator(TestCase):
    def test_simple_evaluator__evaluate(self) -> None:
        evaluator = _ExampleSimpleEvaluator(1)
        solutions = [np.random.randn(10)]
        fitness = evaluator.evaluate(solutions)
        expected = np.array([-np.sum(solutions[0] ** 2)])

        self.assertTrue(np.array_equal(fitness, expected))

    def test_simple_evaluator__evaluate_multiple(self) -> None:
        evaluator = _ExampleSimpleEvaluator(1)
        solutions = np.random.randn(64, 10)
        fitness = evaluator.evaluate(solutions)
        expected = -np.sum(solutions ** 2, axis=1)

        self.assertTrue(np.array_equal(fitness, expected))

    def test_simple_evaluator__evaluate_multiprocessing(self) -> None:
        evaluator = _ExampleSimpleEvaluator(8)
        solutions = np.random.randn(64, 10)
        fitness = evaluator.evaluate(solutions)
        expected = -np.sum(solutions ** 2, axis=1)

        self.assertTrue(np.array_equal(fitness, expected))
