from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from autodiff.variable import Variable
from models.logistic_regression import LogisticRegression
from models.mlr import MultipleLinearRegression


def main() -> None:
    """Main function to showcase all parts of the assignment"""
    # Showcase part 1 through 4
    part_1_4()

    # Showcase part 5
    part_5()

    # Showcase part 6
    part_6()


def part_1_4() -> None:
    """
    This function shows the functionality required by
    part 1 through 4. We go over basic variable creation,
    gradient calculated, gradient deleting and multiplying by
    scalars, both using __mul__ and __rmul__.
    """
    # Create 5 different vectors
    a = np.array([4, 4, 5], dtype=float)
    b = np.array([3.14, 2, 5], dtype=float)
    c = np.array([2, 6, 7], dtype=float)
    d = np.array([1, 4, 2], dtype=float)
    e = np.array([5, 9, 6], dtype=float)

    # Make every vector a variable
    v1 = Variable(a)
    v2 = Variable(b)
    v3 = Variable(c)
    v4 = Variable(d)
    v5 = Variable(e)
    v6 = Variable(a)

    # Multiple operations
    out = (v1 * v2 + v3.sin() * v4 + v2.exp() * v5.log() + v3 / v1).sum()

    # Print out the intitial variables and their
    print("\n===Initial Variables: Data and Gradients===")
    out.backward()
    variables = [out, v1, v2, v3, v4, v5]
    [print(var) for var in variables]

    # Part 4
    # Delete gradients
    print("\n===Delete Gradients===")
    out.delete_gradient()
    [print(var) for var in variables]

    print("\n===Test mult for scalars===")
    t1 = v6 * 3
    t2 = 3 * v6

    # Should both be [12, 12, 15]
    print(t1)
    print(t2)


def part_5() -> None:
    """
    Demonstrate the different MLR fitting methods.
    We compare both GD with MSE and L1 to the Analitic solution.
    We see that both GD fitting methods do not perform well, and also
    perform about the same, however, one of the issues is data standardization.
    We will see, among other things, the impact of this in Part 6.

    Though not originally part of part 5, we will show how L2 performs
    without standardized data.
    """
    showcase_mlr()

    # L2 loss
    showcase_l2_loss()


def part_6() -> None:
    """
    We observe that GD performs significantly better with standardization of
    the data.

    We will also see logistic regression with gradient descent. It will try
    to predict breast cancer. We observe that only
    """
    showcase_mlr(standardize=True)

    # === L2 Loss ===
    showcase_l2_loss(standardize=True)

    # === LOGISTIC REGRESSION ===
    data = load_breast_cancer()
    obs = pd.DataFrame(data.data, columns=data.feature_names)
    obs = obs.to_numpy()

    ground_truth = data.target
    ground_truth = ground_truth.astype("float64")
    ground_truth = ground_truth.reshape(-1, 1)

    gaussian_initialization = make_gaussian(seed=42)

    log_res = LogisticRegression()
    log_res.fit(
        obs,
        Variable(ground_truth),
        lr=0.1,
        loss_function=log_loss,
        parameter_initialization=gaussian_initialization,
        max_iter=100,
    )

    print("\nPARAMETERS")
    print("===LOGISTIC REGRESSION===")
    print(log_res.parameters["params"])

    preds = log_res.predict(obs)
    threshold = 0.5
    pred_bin = (preds >= threshold).astype("float64")
    count = 0
    for i in range(len(pred_bin)):
        if pred_bin[i] == ground_truth[i]:
            count += 1
    print(f"Accuracy: {count / len(pred_bin):.4f}")


def showcase_l2_loss(*, standardize: bool = False) -> None:
    """
    This function test L2 loss

    Args:
        standardize (bool, optional): This is to show
        the difference standardized data makes when it comes
        to convergence using Gradient Descent. Defaults to False.
    """
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    observations = diabetes_df[["age", "bmi", "bp"]]
    obs = observations.to_numpy()
    ground_truth = diabetes.target
    ground_truth = ground_truth.reshape(-1, 1)
    mlr_gd_l2 = MultipleLinearRegression()

    ridge_loss = make_ridge_loss(
        lam=0.1, get_params=lambda: mlr_gd_l2.parameters["var_params"]
    )

    gaussian_initialization_69 = make_gaussian(seed=69)
    mlr_gd_l2.fit_gradient_descent(
        obs,
        Variable(ground_truth),
        lr=0.1,
        loss_function=ridge_loss,
        parameter_initialization=gaussian_initialization_69,
        max_iter=100,
        standardize_data=standardize,
    )
    print("\nPARAMETERS")
    print("===GRADIENT DESCENT WITH L2===")
    print(mlr_gd_l2.parameters["params"])

    print("\nMEAN ABSOLUTE ERROR")
    print("===GD With L2===")
    l2_pred = mlr_gd_l2.predict(obs)
    print(abs(l2_pred - ground_truth).mean())


def showcase_mlr(*, standardize: bool = False) -> None:
    """
    This is a function to test out the Multiple Linear Regression models. Using
    different fit and loss functions.

    Args:
        standardize (bool, optional): This is to show
        the difference standardized data makes when it comes
        to convergence using Gradient Descent. Defaults to False.
    """
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    observations = diabetes_df[["age", "bmi", "bp"]]
    obs = observations.to_numpy()
    ground_truth = diabetes.target
    ground_truth = ground_truth.reshape(-1, 1)
    mlr_gd = MultipleLinearRegression()
    mlr_analitic = MultipleLinearRegression()
    mlr_gd_l1 = MultipleLinearRegression()

    gaussian_initialization = make_gaussian(seed=42)
    mlr_gd.fit_gradient_descent(
        obs,
        Variable(ground_truth),
        lr=0.1,
        loss_function=mean_squared_error,
        parameter_initialization=gaussian_initialization,
        max_iter=100,
        standardize_data=standardize,
    )

    mlr_analitic.fit(obs, ground_truth)

    lasso_loss = make_lasso_loss(
        lam=0.1, get_params=lambda: mlr_gd_l1.parameters["var_params"]
    )

    gaussian_initialization_69 = make_gaussian(seed=69)
    mlr_gd_l1.fit_gradient_descent(
        obs,
        Variable(ground_truth),
        lr=0.1,
        loss_function=lasso_loss,
        parameter_initialization=gaussian_initialization_69,
        max_iter=100,
        standardize_data=standardize,
    )

    print("\nPARAMETERS")
    print("===ANALITIC SOLUTION ===")
    print(mlr_analitic.parameters["params"])

    print("\n==GRADIENT DESCENT WITH MSE===")
    print(mlr_gd.parameters["params"])

    print("\n===GRADIENT DESCENT WITH L1===")
    print(mlr_gd_l1.parameters["params"])

    # Calculate absolute error
    print("\nMEAN ABSOLUTE ERROR")

    print("===Analitic===")
    analitic_pred = mlr_analitic.predict(obs)
    print(abs(analitic_pred - ground_truth).mean())

    print("\n===GD With MSE===")
    gd_pred = mlr_gd.predict(obs)
    print(abs(gd_pred - ground_truth).mean())

    print("\n===GD With L1===")
    l1_pred = mlr_gd_l1.predict(obs)
    print(abs(l1_pred - ground_truth).mean())


def mean_squared_error(predictions: Variable, ground_truth: Variable) -> Variable:
    """
    Calculate the mean squared error between predictions and the ground truth.

    Args:
        predictions (Variable): The predictions
        ground_truth (Variable): The ground truth

    Returns:
        Variable: A Variable containing the loss and, by virtue of being of
        type Variable, a way to backward pass gradients.
    """
    return ((predictions - ground_truth) ** 2).sum() * (1 / ground_truth.data.shape[0])


def make_gaussian(*, seed: int = 42) -> Callable:
    """To conform to ruff NPY002, we need to use np.random.Generator, this however
    requires a seed to be explitily used, since the generator does not
    relay on global state RNG, i.e. it does not work with np.random.seed(42).
    Using a make function to set this seed makes the gaussian initialization
    plug and play, when it comes to passing it as an argument to a fitting function.

    Args:
        seed (int, optional): The random seed you want to use. Defaults to 42.

    Returns:
        Callable: A function to initialize the parameters with a guassian distrubtion,
        with its random seed set as the seed passed into the make_gaussian function.
    """
    rng = np.random.default_rng(seed)

    def gaussian_initialization(shape: tuple[int, ...]) -> Variable:
        """Initialize parameters randomly by drawing from a standard gaussian

        Args:
            shape (tuple[int,...]): The shape the initial parameters should be

        Returns:
            Variable: Returns a Variable containing the initial random parameters.
        """
        return Variable(rng.standard_normal(shape))

    return gaussian_initialization


def make_lasso_loss(lam: float, get_params: Callable) -> Callable:
    """
    We want our fit_gradient_descent function to work regardless
    of which loss function we choose, however, LASSO loss requires
    more arguments than MSE, since we need the parameters to take the one
    norm (||w||_1), we will achieve this using an anonousmous function.
    We need a lambda also, since MSE also does not take a lambda.
    So we need a work around.

    I am not that familiar with L1 loss, so I somewhat freely interpreted
    the wiki page: https://en.wikipedia.org/wiki/Regularization_(mathematics)

    One such interpretation, L1 loss is only for features, so we so disregard
    the intercept. I did this using a mask (I cast it as a Variable, and
    flagged it is a scalar to prevent a backward pass on the mask)

    Args:
        lam (float): Lambda value
        get_params (Callable): A function to get the parameters, for our current
        implementation you it should call self._parameters["var_params"], since
        this returns the variable of the parameters, this makes us able to
        calculate the gradients

    Returns:
        Callable: A loss function that can correctly be called using only
        predictions and ground_truth, just like MSE, making it plug and play
        for the fit_gradient_descent function
    """

    def lasso_loss(predictions: Variable, ground_truth: Variable) -> Variable:
        """The function to perform the LASSO loss with. This has been
        constructed using make_lasso_loss, see its documentation for further
        details.

        Args:
            predictions (Variable): The predictions: f(X)
            ground_truth (Variable): The ground truth: y

        Returns:
            Variable: A variable containing the loss. By virtue of being
            of type Variable, we can perform a backward pass on it to
            get the gradients.
        """
        mse = mean_squared_error(predictions, ground_truth)
        params: Variable = get_params()

        mask = np.ones_like(params.data)
        mask[-1, 0] = 0.0
        var_mask = Variable(mask, is_scalar=True)

        l1 = (abs(params) * var_mask).sum()
        return mse + (lam * l1)

    return lasso_loss


def make_ridge_loss(lam: float, get_params: Callable) -> Callable:
    """
    This is the same as lasso loss, but we square the params, instead of
    taking the absolute.

    Args:
        lam (float): Lambda value
        get_params (Callable): A function to get the parameters, for our current
        implementation you it should call self._parameters["var_params"], since
        this returns the variable of the parameters, this makes us able to
        calculate the gradients

    Returns:
        Callable: A loss function that can correctly be called using only
        predictions and ground_truth, just like MSE, making it plug and play
        for the fit_gradient_descent function
    """

    def ridge_loss(predictions: Variable, ground_truth: Variable) -> Variable:
        """Perform L2/Ridge loss calculation.

        Args:
            predictions (Variable): The predictions: f(X)
            ground_truth (Variable): The ground truth: y

        Returns:
            Variable: A variable containing the loss. By virtue of being
            of type Variable, we can perform a backward pass on it to
            get the gradients.
        """
        mse = mean_squared_error(predictions, ground_truth)
        params: Variable = get_params()

        mask = np.ones_like(params.data)
        mask[-1, 0] = 0.0
        var_mask = Variable(mask, is_scalar=True)

        l1 = ((params**2) * var_mask).sum()
        return mse + (lam * l1)

    return ridge_loss


def log_loss(pred: Variable, gt: Variable) -> Variable:
    """Log loss function for logistic regression

    Args:
        pred (Variable): predicted data
        gt (Variable): ground truth

    Returns:
        Variable: Variable containing the log loss,
        and a way to backward pass the loss
    """
    log_ls = -(gt * pred.log() + (1.0 - gt) * (1.0 - pred).log()).sum()
    return log_ls * (1 / gt.data.shape[0])


if __name__ == "__main__":
    main()
