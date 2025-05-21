from abc import ABC, abstractmethod
from typing import Any, List
import pandas
import numpy as np
from numpy.typing import NDArray


class SimplePerceptron(ABC):
    def __init__(
        self,
        dataset: pandas.DataFrame,
        learn_rate: float = 0.1,
        max_epochs = 1000,
        random_weight_initialize: bool = True,
        copy_dataset = False,
        min_error: float = 0,
    ) -> None:
        """
        dataset: DataFrame with cols 'x1', 'x2', ..., 'xn' and 'ev' (expected value) as final col

        learn_rate: value between 0 and 1, higher value = bigger steps, usually between 0 and 0.1
        """

        if copy_dataset:
            self.dataset = dataset.copy(True)
        else:
            self.dataset = dataset

        if 'ev' not in self.dataset.columns:
            raise ValueError("Missing 'ev' column for expected output.")

        input_cols = sorted(
            [col for col in self.dataset.columns if col.startswith('x')]
        )

        if len(input_cols) <= 0:
            raise ValueError("At least one input column 'x1', ..., 'xn' is required.")

        self.dataset.insert(0, 'x0', 1) # for the bias
        input_cols = ['x0'] + input_cols

        self.col_labels = input_cols
        if random_weight_initialize:
            self.weights = np.random.uniform(-1, 1, len(input_cols))
        else:
            self.weights = np.zeros(len(input_cols))

        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

        self.min_error = min_error
        self.error = min_error + 1
        self.current_epoch = 0

    @abstractmethod
    def has_next(self) -> bool:
        pass
        
    def next_epoch(self) -> NDArray[np.float64]:
        if not self.has_next():
            raise Exception("Solution was already found or max epochs were reached")
        self.error = 0
        self.current_epoch += 1
        for _, row in self.dataset.iterrows():
            inputs = row[self.col_labels].values.astype(float) # type: ignore[assignment]
            output = self._activation_func(np.dot(self.weights, inputs))
            delta = row['ev'] - output
            self._calc_weight_adjustment(inputs, delta)
            self.error += self._calc_error_per_data(delta)        
        return self.weights
    
    @abstractmethod
    def _calc_error_per_data(self, delta: float) -> float: # AKA, error for each row of the dataset
        pass

    @abstractmethod
    def _activation_func(self, weighted_sum: np.float64) -> Any:
        pass

    @abstractmethod
    def _calc_weight_adjustment(self, inputs: np.ndarray, delta: float) -> None:
        pass

    def try_current_epoch(self, inputs: List[float]):
        """
        inputs: an array containing the numeric parameters x1, ..., xn you want to test with this epoch
        """
        inputs.insert(0, 1) # adds the bias parameter x0

        return self._activation_func(np.dot(self.weights, np.array(inputs)))
    
    def try_testing_set(self, testing_set: pandas.DataFrame):
        """
        testing_set: DataFrame with columns 'x1', ..., 'xn'.
        Returns an array of the prediction for each row
        """
        predictions: List[float] = []

        for _, row in testing_set.iterrows():
            inputs = row[self.col_labels[1:]].tolist()  # skip x0
            prediction = self.try_current_epoch(inputs)
            predictions.append(prediction)

        return np.array(predictions)
