from typing import Literal
import pandas
import numpy as np

from src.simple_perceptron import SimplePerceptron

class BinaryPerceptron(SimplePerceptron):
    def __init__(
        self,
        dataset: pandas.DataFrame,
        learn_rate: float = 0.1,
        max_epochs=1000,
        random_weight_initialize: bool = True,
        copy_dataset = False,
    ) -> None:
        super().__init__(dataset, learn_rate, max_epochs, random_weight_initialize, copy_dataset)

    def has_next(self):
        return self.error != 0 and self.current_epoch < self.max_epochs

    def _calc_error_per_data(self, delta: float) -> float:
        return abs(delta)

    def _activation_func(self, weighted_sum: np.float64) -> Literal[-1, 1]:
        return 1 if weighted_sum >= 0 else -1

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.weights += self.learn_rate * delta * inputs
