import numpy as np
from numpy.typing import NDArray

class PCA():
    def __init__(self, inputs: NDArray[np.float64]) -> None:
        self.inputs = inputs
        x_dim, y_dim = inputs.shape  

        self.means = np.mean(inputs, axis=0)
        self.stds = np.std(inputs, axis=0, ddof=0)

        # Standardize
        self.standardized = (inputs - self.means) / self.stds

        self.covariance = (self.standardized.T @ self.standardized) / x_dim

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.covariance)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

    def transform_fixed_components(self, components: int) -> NDArray[np.float64]:
        """Transform the data using a fixed number of principal components."""
        selected_vectors = self.eigenvectors[:, :components]
        return self.standardized @ selected_vectors

    def transform_min_percentage(self, percentage: float) -> NDArray[np.float64]:
        """Transform the data using the minimum number of components needed to reach the target percentage of explained variance."""
        total_variance = np.sum(self.eigenvalues)
        explained_variance = np.cumsum(self.eigenvalues) / total_variance

        # Find how many components are needed to reach the desired percentage
        num_components = np.searchsorted(explained_variance, percentage) + 1

        selected_vectors = self.eigenvectors[:, :num_components]
        return self.standardized @ selected_vectors