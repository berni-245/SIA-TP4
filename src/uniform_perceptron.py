from typing import List
import pandas as pd
import numpy as np

from src.perceptron_function import PerceptronFunction
from src.simple_perceptron import SimplePerceptron

class UniformPerceptron(SimplePerceptron):
    def __init__(
            self,
            dataset: pd.DataFrame,
            learn_rate: float = 0.1,
            max_epochs=1000,
            random_weight_initialize: bool = True,
            activation_func: PerceptronFunction = PerceptronFunction.HYPERBOLIC,
            beta: float = 1,
            min_error: float = 1,
            batch_update: int = 1,
            copy_dataset = False,
        ) -> None: 
        super().__init__(dataset, learn_rate, max_epochs, random_weight_initialize, copy_dataset, min_error)
        self.activation_func = activation_func
        self.beta = beta
        self.accumulated_weights = np.zeros(len(self.col_labels))
        self.min_data_batch_for_update = max(1, min(batch_update, len(self.dataset.index))) 
        self.data_update_count = 0
        self.normalization_constant = 1
        if activation_func.image != None:
            Y = self.dataset['ev']
            self.min_ev = np.min(Y)
            self.max_ev = np.max(Y)
            (min_image, max_image) = activation_func.image
            self.normalization_constant = (max_image - min_image)/(self.max_ev - self.min_ev)
            self.dataset['ev'] = min_image + (Y - self.min_ev)*self.normalization_constant

    def has_next(self):
        return np.abs(self.error) > self.min_error and self.current_epoch < self.max_epochs
    
    def _calc_error_per_data(self, delta: float):
        return ((delta/self.normalization_constant)**2) / 2

    def _activation_func(self, weighted_sum: np.float64) -> float:
        return self.activation_func.func(weighted_sum, self.beta)

    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        self.accumulated_weights += self.learn_rate * delta * inputs * self.activation_func.deriv(np.dot(self.weights, inputs), self.beta)
        self.data_update_count += 1
        if self.data_update_count % self.min_data_batch_for_update == 0:
            self.weights += self.accumulated_weights
            self.accumulated_weights = np.zeros(len(self.col_labels))
            self.data_update_count = 0

    def try_current_epoch(self, inputs: List[float]):
        output = super().try_current_epoch(inputs)
        if self.activation_func.image != None:
            (min, max) = self.activation_func.image
            output = (output - min)/self.normalization_constant + self.min_ev
        return output
