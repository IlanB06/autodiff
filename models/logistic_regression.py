from collections.abc import Callable
from copy import deepcopy

import numpy as np

from autodiff.variable import Variable
from autodiff.helper import Standardizer


class LogisticRegression:
    """A logistic regression model trained using gradient descent

    Attributes:
        _parameters (dict): Parameters of the model
        _hyperparameters (dict): Hyperparameters of the model
        _mean (float): Mean used in standardizing the data during fitting,
        such that predictions can also be standardized.
        _sigma (float): Std used in standardizng the data during fitting,
        such that predictions can also be standardized.
    """

    def __init__(self) -> None:
        """Initialize logistic regression"""
        self._parameters: dict = {}
        self._hyperparameters: dict = {}
        self._mean: float = None
        self._sigma: float = None

    def fit(
        self,
        obs: np.ndarray,
        gt: Variable,
        lr: float,
        *,
        loss_function: Callable,
        parameter_initialization: Callable,
        max_iter: int,
    ) -> None:
        """Fit the logistic regression model using gradient descent

        ruff PLR0913 does not agree with the amount of arguments passed,
        however every argument is neccesary.

        Args:
            observations (Variable): The observations
            gt (Variable): The ground truth
            lr (float): The learning rate
            loss_function (Callable): The loss function you want to use,
            should only have predictions and ground truth as argument
            parameter_initialization (Callable): Function to initialize parameters
            max_iter (int): Maximum number of iterations.
        """
        self._hyperparameters["learning_rate"] = lr
        self.std = Standardizer()
        obs = self.std.standardize_data(obs)

        n = obs.shape[0]
        aug_obs_np = np.append(obs, np.ones((n, 1)), axis=1)
        aug_obs = Variable(aug_obs_np)

        n_features = aug_obs.data.shape[1]
        initial_params: Variable = parameter_initialization((n_features, 1))
        self._parameters["var_params"] = initial_params

        for _ in range(max_iter):
            pred: Variable = aug_obs.matmul(self._parameters["var_params"])
            sigmoid_pred = pred.sigmoid()
            loss = loss_function(sigmoid_pred, gt)
            loss.backward()

            new_params = (
                self._parameters["var_params"].data
                - lr * self._parameters["var_params"].gradient
            )
            self._parameters["var_params"] = Variable(new_params)
            # Set the np.ndarray as real params
            self._parameters["params"] = self._parameters["var_params"].data

            loss.delete_gradient()
            pred.delete_gradient()
            sigmoid_pred.delete_gradient()
            self._parameters["var_params"].delete_gradient()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict data based on fitted parameters

        Args:
            data (np.ndarray): Data you want to use to predict y_hat

        Returns:
            np.ndarray: y_hat, as probalities
        """
        standardize_data = (data - self.std.mean) / self.std.std
        n = standardize_data.shape[0]
        aug_data = np.append(standardize_data, np.ones((n, 1)), axis=1)
        return 1.0 / (1.0 + np.exp(np.matmul(-aug_data, self._parameters["params"])))

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
