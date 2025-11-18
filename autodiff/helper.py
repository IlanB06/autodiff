import numpy as np

class Standardizer:
    def standardize_data(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)

        self.sigma = sigma
        self.mean = mean

        standardized_data = (data - mean) / sigma
        return standardized_data

    def destandardize_data(self, data: np.ndarray) -> np.ndarray:
        standardized_weights = data[:-1]
        standardized_bias = data[-1]

        original_data = standardized_weights / self.sigma.reshape(-1, 1)
        original_bias = standardized_bias - np.sum(
                (self.mean / self.sigma).reshape(-1, 1) * standardized_weights
            )
        return np.vstack([original_data, original_bias])
    