import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

class KohonenNet():
    def __init__(
        self,
        inputs: NDArray[np.float64],
        k_grid_dimension: int,
        use_weights_from_inputs: bool,
        max_epochs=1000,
        initial_radius: int | None = None, # TODO: add possibility to start with fixed learn rate and fixed radius
        initial_learn_rate: float = 0.5,
    ) -> None:
        self.var_count = inputs.shape[1]
        self.k = k_grid_dimension
        self.max_epochs = max_epochs
        self.learn_rate = initial_learn_rate
        self.neuron_count = self.k * self.k
        if initial_radius is None:
            self.radius = self.neuron_count
        else:
            self.radius = initial_radius

        # Standardize inputs
        self.means = np.mean(inputs, axis=0)
        self.stds = np.std(inputs, axis=0)
        self.stds[self.stds == 0] = 1  
        self.inputs = (inputs - self.means) / self.stds

        if use_weights_from_inputs:
            input_count = self.inputs.shape[0]

            if input_count >= self.neuron_count:
                indices = np.random.choice(input_count, self.neuron_count, replace=False)
            else:
                full_indices = np.arange(input_count)
                extra_indices = np.random.choice(input_count, self.neuron_count - input_count, replace=True)
                indices = np.concatenate([full_indices, extra_indices])
                np.random.shuffle(indices)

            flat_weights = self.inputs[indices].copy()
        else:
            flat_weights = np.random.uniform(0, 1, size=(self.neuron_count, self.var_count))

        self.weights = flat_weights.reshape((self.k, self.k, self.var_count)) 
        self.current_epoch = 0

    def has_next(self) -> bool:
        return self.current_epoch < self.max_epochs
    
    def next_epoch(self):
        if not self.has_next():
            raise Exception("Max epochs reached")

        self.current_epoch += 1
        epoch_lr = self.learn_rate / self.current_epoch
        epoch_radius = self.radius / self.current_epoch + 1 # TODO: look for better radio update function, must converge to 1

        for input_vector in self.inputs:
            best_neuron_i, best_neuron_j = self.find_best_neuron(input_vector)

            self.radius = epoch_radius  
            neighbors = self.get_neighbors(best_neuron_i, best_neuron_j)

            for i, j in neighbors:
                self.weights[i, j] += epoch_lr * (input_vector - self.weights[i, j])
    
    def find_best_neuron(self, input_vector: NDArray[np.float64]) -> Tuple[int, int]:
        diffs = self.weights - input_vector  
        dists = np.linalg.norm(diffs, axis=2)

        best_neuron_indexes = np.unravel_index(np.argmin(dists), dists.shape)
        return int(best_neuron_indexes[0]), int(best_neuron_indexes[1])

    def get_neighbors(self, center_row: int, center_col: int) -> List[Tuple[int, int]]:
        neighbors = []
        radius_int = int(np.ceil(self.radius))

        for row_offset in range(-radius_int, radius_int + 1):
            for col_offset in range(-radius_int, radius_int + 1):
                neighbor_row = center_row + row_offset
                neighbor_col = center_col + col_offset

                if 0 <= neighbor_row < self.k and 0 <= neighbor_col < self.k:
                    distance = np.sqrt(row_offset**2 + col_offset**2)
                    if distance <= self.radius:
                        neighbors.append((neighbor_row, neighbor_col))

        return neighbors
