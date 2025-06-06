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

        # for i in range(self.patterns_count):
        #     for j in range(i + 1, self.patterns_count):
        #         dot = np.dot(self.patterns[:, i], self.patterns[:, j])
        #         if dot != 0:
        #             print(f"Columns {i} and {j} are not orthogonal. Dot product = {dot}")

        self.weights = (1/self.pattern_length) * (self.patterns @ self.patterns.T)
        self.max_iters = max_iters

    def set_query_pattern(self, pattern: NDArray[np.bool_]):
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

        self.query_pattern = np.where(pattern, 1, -1).astype(np.int8)
        self.query_pattern_prev = np.empty_like(self.query_pattern)

    def pattern_next(self) -> None:
        self.query_pattern_prev = self.query_pattern
        updated = np.sign(self.weights @ self.query_pattern)
        self.query_pattern = np.where(updated == 0, self.query_pattern_prev, updated)

    # def _pattern_next_inefficient(self, n_convergence: None | int = None) -> np.ndarray:
    #     self.query_pattern_prev = self.query_pattern.copy()
    #     updates = []
    #
    #     for i in range(self.pattern_length):
    #         self.query_pattern_prev[i] = self.query_pattern[i]
    #         h = 0
    #         for j in range(self.pattern_length):
    #             h += self.weights[i, j] * self.query_pattern[j]
    #         if h != 0:
    #             self.query_pattern[i] = np.sign(h)
    #
    #         updates.append(self.query_pattern.copy())
    #         if n_convergence and h != 0 and self.pattern_converged():
    #             n_convergence -= 1
    #             if n_convergence == 0:
    #                 break
    #
    #     return np.column_stack(updates) # shape: (pattern_length, pattern_length)

    def _pattern_next_inefficient(self, i: int):
        self.query_pattern_prev = self.query_pattern.copy()
        h = 0
        for j in range(self.pattern_length):
            h += self.weights[i, j] * self.query_pattern[j]
        if h != 0:
            self.query_pattern[i] = np.sign(h)

    def run_until_converged(self, n_convergence: None | int = None) -> int:
        step_count = 0
        converged = False
        n_convergence = n_convergence if n_convergence else self.pattern_length
        n_conv = n_convergence
        while step_count < self.max_iters:
            for i in range(self.pattern_length):
                step_count += 1
                self._pattern_next_inefficient(i)
                if n_conv and self.pattern_converged():
                    n_conv -= 1
                    if n_conv == 0:
                        converged = True
                        break
                else:
                    n_conv = n_convergence
            if converged:
                break
        else:
            # Only executes if while loop didn't break (i.e., max_iters reached)
            return -1

        match_idx = self.pattern_match()
        if match_idx < 0:
            return -2
        else:
            return step_count

    def run_until_converged_with_history(self, n_convergence: None | int = None) -> np.ndarray:
        step_count = 0
        converged = False
        n_convergence = n_convergence if n_convergence else self.pattern_length
        n_conv = n_convergence
        full_history = []
        while step_count < self.max_iters:
            for i in range(self.pattern_length):
                step_count += 1
                self._pattern_next_inefficient(i)
                full_history.append(self.query_pattern.copy())
                if n_conv and self.pattern_converged():
                    n_conv -= 1
                    if n_conv == 0:
                        converged = True
                        break
                else:
                    n_conv = n_convergence
            if converged:
                break

        # Concatenate horizontally: final shape (pattern_length, total_updates)
        history_matrix = np.hstack(full_history)
        history_matrix = np.where(history_matrix == 1, True, False)

        if step_count >= self.max_iters:
            print("Max iterations reached without finding a pattern match")
            return history_matrix

        match_idx = self.pattern_match()
        if match_idx < 0:
            print("No matching column found.")
        else:
            print(f"Match found in column {match_idx}")

        return history_matrix

    def find_pattern(self) -> int:
        for _ in range(self.max_iters):
            self.pattern_next()
            if self.pattern_converged():
                break
        else:
            return -1 # Max iterations reached without finding a pattern match
        
        match_idx = self.pattern_match()
        if match_idx < 0:
            return -2 # Found an spurious pattern
        else:
            return match_idx
        
    def pattern_converged(self) -> bool:
        return np.array_equal(self.query_pattern, self.query_pattern_prev)

    def pattern_match(self) -> int:
        for col_index in range(self.patterns.shape[1]):
            if np.array_equal(self.query_pattern.flatten(), self.patterns[:, col_index]):
                return col_index
        return -1

    def energy(self):
        S = self.query_pattern
        return float(-0.5 * S.T @ self.weights @ S)
