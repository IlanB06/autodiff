import numpy as np

class Standardizer:
    """
    This is used to standardize and destandatize data,
    Its important to use the same instance of Standardizer
    to standardize and destandardize the data, since the
    standardizer stores the mean and std needed for 
    destandardization. 
    """    
    def standardize_data(self, data: np.ndarray) -> np.ndarray:
        """Standardize the given data

        Args:
            data (np.ndarray): Data to standardize

        Returns:
            np.ndarray: Standardized data
        """        
        mean = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)

        self._sigma = sigma
        self._mean = mean

        standardized_data = (data - mean) / sigma
        return standardized_data

    def destandardize_data(self, data: np.ndarray) -> np.ndarray:
        """Destandardize the given data

        Args:
            data (np.ndarray): Data to destandardize

        Returns:
            np.ndarray: Destandardized data
        """        
        standardized_weights = data[:-1]
        standardized_bias = data[-1]

        original_data = standardized_weights / self._sigma.reshape(-1, 1)
        original_bias = standardized_bias - np.sum(
                (self._mean / self._sigma).reshape(-1, 1) * standardized_weights
            )
        return np.vstack([original_data, original_bias])
    
    @property
    def mean(self) -> np.ndarray:
        """Getter for the means

        Returns:
            np.ndarray: Means
        """        
        safe_view = self._mean.view()
        safe_view.flags.writeable = False
        return safe_view
    
    @property
    def std(self) -> np.ndarray:
        """Getter for the standard deviations

        Returns:
            np.ndarray: Standard deviations
        """        
        safe_view = self._sigma.view()
        safe_view.flags.writeable = False
        return safe_view