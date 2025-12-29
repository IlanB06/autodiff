from collections.abc import Callable
from copy import deepcopy

import numpy as np

from autodiff.variable import Variable
from autodiff.helper import Standardizer


class MultipleLinearRegression:
    """A multiple linear regression model using the normal equation

    Attributes:
        _parameters (dict): Model parameters
        _hyperparameters (dict): Model hyperparameters
    """

    def __init__(self) -> None:
        """Initializes a multiple linear regression model.

        Attributes:
            _parameters (dict): Dict to store parameters
        """
        self._parameters: dict = {}
        self._hyperparameters: dict = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """Calculates the coefficients to best fit the observations

        Args:
            observations (np.ndarray): Input observations
            ground_truth (np.ndarray): Corresponding ground truth
        """
        rows = observations.shape[0]
        y = ground_truth
        augmented_obs = np.append(observations, np.ones((rows, 1)), axis=1)
        optimal_params = np.dot(
            (np.linalg.inv(np.dot(augmented_obs.T, augmented_obs))),
            np.dot(augmented_obs.T, y),
        )

        self._parameters["params"] = optimal_params

    def fit_gradient_descent(
        self,
        obs: np.ndarray,
        ground_truth: Variable,
        *,
        lr: float,
        loss_function: Callable,
        parameter_initialization: Callable,
        max_iter: int,
        standardize_data: bool = False,
    ) -> None:
        """Function to fit Multiple linear regression using
        gradient descent, you can specify your loss function and your
        parameter initialization.

        Args:
            observations (np.ndarray): The observations
            ground_truth (Variable): The ground truth
            lr (float): Learning rate
            loss_function (Callable): Loss function, should only take
            predictions and ground truth as arguments
            parameter_initialization (Callable): Function to initialize parameters
            max_iter (int): Maximum number of iterations
            standardize_data (bool, optional): Indicate if you want to standardize
            your data, this is mainly to show the difference between standardization
            and no standardization. Defaults to False.
        """
        self._hyperparameters["learning_rate"] = lr
        std = Standardizer()
        if standardize_data:
            obs = std.standardize_data(obs)

        n = obs.shape[0]
        aug_obs_np = np.append(obs, np.ones((n, 1)), axis=1)
        aug_obs = Variable(aug_obs_np)

        n_features = aug_obs.data.shape[1]
        initial_params: Variable = parameter_initialization((n_features, 1))
        self._parameters["var_params"] = initial_params

        for _ in range(max_iter):
            pred: Variable = aug_obs.matmul(self._parameters["var_params"])
            loss: Variable = loss_function(pred, ground_truth)
            loss.backward()

            new_params = (
                self._parameters["var_params"].data
                - lr * self._parameters["var_params"].gradient
            )
            self._parameters["var_params"] = Variable(new_params)
            # Set the np.ndarray as real params
            self._parameters["params"] = self._parameters["var_params"].data

            # Clean-up
            loss.delete_gradient()
            pred.delete_gradient()
            self._parameters["var_params"].delete_gradient()

        # Destandardize params, since they were trained on standardized data
        if standardize_data:
            standardized_params = self._parameters["params"]
            self._parameters["params"] = std.destandardize_data(standardized_params)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predicts output values of given input data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Predicted output values
        """
        params = self._parameters["params"]
        rows, _ = data.shape
        tilde_x = np.append(data, np.ones((rows, 1)), axis=1)
        return np.dot(tilde_x, params)

    @property
    def parameters(self) -> dict:
        """Get the parameters of the model

        Returns:
            dict: parameters of the model
        """
        return deepcopy(self._parameters)

    @property
    def hyperparameters(self) -> dict:
        """Get the models hyperparameters

        Returns:
            dict: Hyperparameters of the model
        """
        return deepcopy(self._hyperparameters)
