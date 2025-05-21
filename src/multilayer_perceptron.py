from ctypes import ArgumentError
from enum import Enum
from re import S
from typing import List
import numpy as np
import pandas
from numpy.typing import NDArray

from src.perceptron_function import PerceptronFunction

class PerceptronOptimizer(Enum):
    GRADIENT_DESCENT = 0
    MOMENTUM = 1
    ADAM = 2

    @classmethod
    def from_string(cls, name: str):
        return cls[name.upper()]

class NeuralNet:
    def __init__(
    self,
    input_count: int,
    hidden_layers: List[int], 
    activation_func: PerceptronFunction, 
    optimizer: PerceptronOptimizer,
    beta_func: float = 1,             
    random_weight_initialize: bool = True
):
        """
        input_count: the amount of input arguments
        hidden_layers: list of the amount of neurons per layer (including output layer)
        activation_func: activation function used in all layers
        """

        if (len(hidden_layers) == 0):
            raise ArgumentError("One layer must be specified")

        self.activation_func = activation_func
        self.weights: List[NDArray[np.float64]] = [] # NDArray can be vector or matrix, in this case matrix
        self.beta_func = beta_func
        self.data_error = 1

        prev_neuron_count = input_count
        for neurons in hidden_layers:
            if random_weight_initialize:
                weight_matrix = np.random.uniform(-1, 1, (neurons, prev_neuron_count + 1)) # +1 for bias
            else:
                weight_matrix = np.zeros((neurons, prev_neuron_count + 1))
            self.weights.append(weight_matrix)

            prev_neuron_count = neurons

        if optimizer == PerceptronOptimizer.GRADIENT_DESCENT:
            self.update_weights_func = self._gradient_descent
        elif optimizer == PerceptronOptimizer.MOMENTUM:
            self.update_weights_func = self._momentum
            self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]
            self.alpha: float = 0.9  
        else: # it will default to adam
            self.update_weights_func = self._adam_step
            self.t = 0  
            self.m = [np.zeros_like(w) for w in self.weights] 
            self.v = [np.zeros_like(w) for w in self.weights] 
            self.beta1: float = 0.9
            self.beta2: float = 0.999
            self.epsilon: float = 1e-8

    def forward_pass(self, input_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        input_values: vector of input data (without bias)
        """
        self.values: List[NDArray[np.float64]] = [input_values] # the input values and neuron values of each layer size equals to amount of layers + 1
        self.sums: List[NDArray[np.float64]] = [] # the weighted sums without the activation, size equals to amount of layers
        current_values = input_values
        for weight_matrix in self.weights:
            current_values = np.insert(current_values, 0, 1.0)  # add bias
            z = np.dot(weight_matrix, current_values)
            self.sums.append(z)
            current_values: NDArray[np.float64] = self.activation_func.func(z, self.beta_func)
            self.values.append(current_values)
        return current_values  
    
    def update_weights(
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
        self.update_weights_func(input_values, expected_output, learning_rate)
        


    def _gradient_descent (  
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.1
    ):
        final_output = self.forward_pass(input_values)

        # Deltas for output layer
        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error**2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        # Backpropagate deltas and update weights
        for i in reversed(range(len(self.weights))):
            # Prepare values with bias
            values = np.insert(self.values[i], 0, 1.0)
            # If deltas is 3x1 and values 4x1, np.outer will transverse values to 1x4, to get the resulting matrix of size 3x4
            self.weights[i] += learning_rate * np.outer(deltas, values) 

            if i > 0: # if we are not on the last layer, we should calculate delta

                # Remove bias weights from current layer, they don't have associated delta
                weights_wo_bias = self.weights[i][:, 1:]

                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

    def _momentum(
    self,
    input_values: NDArray[np.float64],
    expected_output: NDArray[np.float64],
    learning_rate: float = 0.1,
):
        final_output = self.forward_pass(input_values)

        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error**2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)  # agregar bias
            gradient = np.outer(deltas, values)

            # Momentum update: Δw(t+1) = -η * grad + α * Δw(t)
            delta_w = learning_rate * gradient + self.alpha * self.prev_weight_updates[i]
            self.weights[i] += delta_w
            self.prev_weight_updates[i] = delta_w  

            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

    def _adam_step(
        self,
        input_values: NDArray[np.float64],
        expected_output: NDArray[np.float64],
        learning_rate: float = 0.001
    ):
        self.t += 1
        final_output = self.forward_pass(input_values)

        # Output layer delta
        self.data_error = 0
        error = expected_output - final_output
        self.data_error = np.sum(error ** 2)
        deltas = error * self.activation_func.deriv(self.sums[-1], self.beta_func)

        grads = []

        for i in reversed(range(len(self.weights))):
            values = np.insert(self.values[i], 0, 1.0)
            grad = - np.outer(deltas, values)
            grads.insert(0, grad)

            if i > 0:
                weights_wo_bias = self.weights[i][:, 1:]
                deltas = np.dot(weights_wo_bias.T, deltas) * self.activation_func.deriv(self.sums[i - 1], self.beta_func)

        # Update weights using Adam
        for i in range(len(self.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class MultiLayerPerceptron():
    def __init__(self, neural_net: NeuralNet, dataset: pandas.DataFrame, learn_rate: float = 0.1, min_error: float = 0.1, max_epochs = 10000):
        """
        dataset: DataFrame with cols 'x1', 'x2', ..., 'xn' and 'ev' (expected value) as final col

        learn_rate: value between 0 and 1, higher value = bigger steps, usually between 0 and 0.1
        """
 
        if 'ev' not in dataset.columns:
            raise ValueError("Missing 'ev' column for expected output.")
        
        try:
            list(dataset['ev'])
        except Exception:
            raise TypeError("Dataframe column 'ev' must be an array")

        input_cols = sorted(
            [col for col in dataset.columns if col.startswith('x')]
        )

        if len(input_cols) <= 0:
            raise ValueError("At least one input column 'x1', ..., 'xn' is required.")


        expected_cols = input_cols + ['ev']
        dataset = dataset[expected_cols] # sorts the dataset cols in the given order

        self.neural_net = neural_net
        self.col_labels = input_cols

        self.dataset = dataset
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

        self.min_error = min_error
        self.error = 100
        self.current_epoch = 0

    def has_next(self) -> bool:
        return self.min_error < self.error and self.current_epoch < self.max_epochs

    def next_epoch(self):
        if not self.has_next():
            raise Exception("Solution was already found or max epochs were reached")
        self.error = 0
        self.current_epoch += 1
        for _, row in self.dataset.iterrows():
            inputs = row[self.col_labels].values.astype(float)
            self.neural_net.update_weights(inputs, row['ev'], self.learn_rate)
            self.error += self.neural_net.data_error
        self.error /= 2
                        
    def try_current_epoch(self, inputs: List[float]):
        """
        inputs: an array containing the numeric parameters x1, ..., xn you want to test with this epoch
        """

        return self.neural_net.forward_pass(np.array(inputs))
    
    def try_testing_set(self, testing_set: pandas.DataFrame):
        """
        testing_set: DataFrame with columns 'x1', ..., 'xn'.
        Returns an array of the prediction for each row
        """
        predictions = []

        for _, row in testing_set.iterrows():
            inputs = row[self.col_labels].tolist()
            prediction = self.try_current_epoch(inputs)
            predictions.append(prediction)

        return predictions