from numpy.typing import NDArray
import numpy as np

class HopfieldNN():
    def __init__(self, patterns: NDArray[np.bool_], max_iters: int) -> None:
        """
        Initialize the Hopfield Neural Network with a set of binary patterns.

        Parameters:
            patterns (NDArray[np.bool_]): A 2D NumPy array of shape (N, M),
                where each column represents a binary pattern of length N.
                The array must have boolean dtype.

        Raises:
            AssertionError: If `patterns` is not a 2D boolean NumPy array.
        """
        assert patterns.dtype == bool, f"`patterns` should be a boolean 2D matrix. Found type: {patterns.dtype}"
        assert patterns.ndim == 2, f"`patterns` should be a boolean 2D matrix. Found dims: {patterns.ndim}D"

        self.patterns = np.where(patterns, 1, -1).astype(np.int8)
        self.pattern_length = patterns.shape[0]
        self.patterns_count = patterns.shape[1]

        for i in range(self.patterns_count):
            for j in range(i + 1, self.patterns_count):
                dot = np.dot(self.patterns[:, i], self.patterns[:, j])
                if dot != 0:
                    print(f"Columns {i} and {j} are not orthogonal. Dot product = {dot}")

        self.weights = (1/self.pattern_length) * (self.patterns @ self.patterns.T)
        # self.query_pattern: Union[NDArray[np.int8], None] = None
        self.max_iters = max_iters

    def find_pattern(self, pattern: NDArray[np.bool_]) -> None:
        """
        Accepts a boolean pattern, ensures it's a 1D column vector.
        
        Parameters:
            pattern (NDArray[np.bool_]): A boolean pattern of shape (N,) or (1, N).
            
        Raises:
            AssertionError: If the pattern is not a 1D vector of expected length.
        """
        assert pattern.ndim in (1, 2), f"`pattern` must be 1D or 2D. Got {pattern.ndim}D"

        if pattern.ndim == 2:
            if pattern.shape[0] == 1 and pattern.shape[1] == self.pattern_length:
                pattern = pattern.T
            elif pattern.shape[1] != 1 or pattern.shape[0] != self.pattern_length:
                raise ValueError(f"`pattern` must have shape ({self.pattern_length}, 1) or (1, {self.pattern_length})")
        else:
            assert pattern.shape[0] == self.pattern_length, \
                f"Pattern length mismatch: expected {self.pattern_length}, got {pattern.shape[0]}"
            pattern = pattern.reshape(-1, 1)

        query_pattern = np.where(pattern, 1, -1).astype(np.int8)
        print(query_pattern)
        query_pattern_prev = np.empty_like(query_pattern)
        for i in range(self.max_iters):
            query_pattern_prev = query_pattern.copy()
            query_pattern = self._update_pattern(query_pattern)
            print(i)
            print(query_pattern)
            print()
            if np.array_equal(query_pattern, query_pattern_prev):
                break
        else:
            print("Max iterations reached without finding a pattern match")
            return

        for col_index in range(self.patterns.shape[1]):
            if np.array_equal(query_pattern.flatten(), self.patterns[:, col_index]):
                print(f"Match found in column {col_index}")
                return
        print("No matching column found.")

    def _update_pattern(self, pattern: NDArray[np.int8]):
        return np.sign(self.weights @ pattern)
