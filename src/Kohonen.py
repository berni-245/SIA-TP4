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
        initial_radius: int | None = None,
        decrease_radius: bool = True,
        ini_learn_rate: float = 0.5,
        decrease_learn_rate: bool = True,
        labels: List[str] | None = None
    ) -> None:
        self.var_count = inputs.shape[1]
        self.k = k_grid_dimension
        self.max_epochs = max_epochs

        self.ini_learn_rate = ini_learn_rate
        self.decrease_learn_rate = decrease_learn_rate

        self.neuron_count = self.k * self.k
        self.initial_radius = initial_radius if initial_radius is not None else (self.k / 2)
        self.radius = self.initial_radius
        self.decrease_radius = decrease_radius

        # Standardize inputs
        self.means = np.mean(inputs, axis=0)
        self.stds = np.std(inputs, axis=0)
        self.stds[self.stds == 0] = 1
        self.inputs = (inputs - self.means) / self.stds

        self.labels = labels

        # Use inputs as weights or initialize them random
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

        activations = []
        for i in range(len(flat_weights)):
            info = {}
            info["total_activations"] = 0
            if labels is not None: 
                for label in labels:
                    info[label] = 0
            activations.append(info)
        self.activations = np.array(activations).reshape((self.k, self.k))
        self.weights = flat_weights.reshape((self.k, self.k, self.var_count))
        self.current_epoch = 0

    def has_next(self) -> bool:
        return self.current_epoch < self.max_epochs

    def next_epoch(self):
        if not self.has_next():
            raise Exception("Max epochs reached")

        self.current_epoch += 1
        if self.decrease_learn_rate:
            epoch_lr = self.ini_learn_rate / self.current_epoch
        else:
            epoch_lr = self.ini_learn_rate
        if self.decrease_radius:
            tau = self.max_epochs / np.log(self.initial_radius)
            epoch_radius = 1 + (self.initial_radius - 1) * np.exp(-self.current_epoch / tau)
        else:
            epoch_radius = self.initial_radius

        for data_idx, input_vector in enumerate(self.inputs):
            best_neuron_i, best_neuron_j = self._find_best_neuron(input_vector)
            self.activations[best_neuron_i][best_neuron_j]["total_activations"] += 1
            if self.labels is not None:
                self.activations[best_neuron_i][best_neuron_j][self.labels[data_idx]] += 1

            self.radius = epoch_radius
            neighbors = self._get_neighbors(best_neuron_i, best_neuron_j)

            for i, j in neighbors:
                self.weights[i, j] += epoch_lr * (input_vector - self.weights[i, j])

    def _find_best_neuron(self, input_vector: NDArray[np.float64]) -> Tuple[int, int]:
        diffs = self.weights - input_vector
        dists = np.linalg.norm(diffs, axis=2)

        best_neuron_indexes = np.unravel_index(np.argmin(dists), dists.shape)
        return int(best_neuron_indexes[0]), int(best_neuron_indexes[1])

    def _get_neighbors(self, center_row: int, center_col: int) -> List[Tuple[int, int]]:
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
