import numpy as np
from numpy.typing import NDArray

class KohonenNet():
    def __init__(
        self,
        inputs: NDArray[np.float64],
        k_grid_dimension: int,
        use_weights_from_inputs: bool,
        max_epochs=1000,
        initial_learn_rate: float = 0.5,
    ) -> None:
        self.var_count = inputs.shape[1]
        self.means = np.mean(inputs, axis=0)
        self.stds = np.std(inputs, axis=0)
        self.inputs = (inputs - self.means) / self.stds

        self.k_grid_dimension = k_grid_dimension
        self.max_epochs = max_epochs
        self.learn_rate = initial_learn_rate
        self.neuron_count = k_grid_dimension * k_grid_dimension

        if use_weights_from_inputs:
            input_count = self.inputs.shape[0]
            
            if input_count >= self.neuron_count:
                indices = np.random.choice(input_count, self.neuron_count, replace=False)
            else:
                full_indices = np.arange(input_count)
                extra_indices = np.random.choice(input_count, self.neuron_count - input_count, replace=True)
                indices = np.concatenate([full_indices, extra_indices])
                np.random.shuffle(indices)
            
            self.weights = self.inputs[indices].copy()
        else:
            self.weights = np.random.uniform(0, 1, size=(self.neuron_count, self.var_count))

