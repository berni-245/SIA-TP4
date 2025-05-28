import numpy as np
from numpy.typing import NDArray

class OjaNet():
    def __init__(
            self,
            inputs: NDArray[np.float64],
            max_epochs=1000,
            ini_learn_rate: float = 0.5,
            decrease_learn_rate: bool = True,            
        ) -> None: 
        self.ini_learn_rate = ini_learn_rate
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.decrease_learn_rate = decrease_learn_rate

        self.weights = np.random.uniform(0, 1, inputs.shape[1])

        self.means = np.mean(inputs, axis=0)
        self.stds = np.std(inputs, axis=0)
        self.stds[self.stds == 0] = 1  

        # Standardize
        self.inputs = (inputs - self.means) / self.stds
    
    def has_next(self):
        return self.current_epoch < self.max_epochs

    def next_epoch(self) -> NDArray[np.float64]:
        if not self.has_next():
            raise Exception("Max epochs were reached")
        self.current_epoch += 1
        if self.decrease_learn_rate:
            epoch_learn_rate = self.ini_learn_rate / self.current_epoch
        else:
            epoch_learn_rate = self.ini_learn_rate
        for row in self.inputs:
            output = np.dot(row, self.weights)
            self.weights += epoch_learn_rate * (output*row - output**2 * self.weights)
        return self.weights