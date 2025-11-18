from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from . import variable


class Operation(ABC):
    """An operation can be called on one or two Variables
    Every operator has a way to perform a backward pass to get
    gradients for gradient descent.
    """

    def __init__(self) -> None:
        """Initialize the operation, with parents, can be None if initial node"""
        self._parents = None

    @abstractmethod
    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Every operation needs a backward pass,
        implentation depends on what operation

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """

    def __str__(self) -> str:
        """String representation of the class

        Returns:
            str: Class name
        """
        return self.__class__.__name__

    def _check_instances(self, args: tuple[variable.Variable, ...]) -> None:
        """Check if the arguments given are of instance Variable, because
        operations only work on variables

        Args:
            args (tuple[variable.Variable,...]): The arguments, can be only one
            or two, since these mathematical operations only work on
            one or two Variables at a time

        Raises:
            TypeError: All args should be of type Variable
        """
        unary_operator_length = 1
        binary_operator_length = 2
        if len(args) == unary_operator_length:
            a = args[0]
            if not isinstance(a, variable.Variable):
                raise TypeError(
                    f"The argument should be of type Variable\
                                not {type(a).__name__}"
                )

        if len(args) == binary_operator_length:
            a, b = args
            if not isinstance(a, variable.Variable) or not isinstance(
                b, variable.Variable
            ):
                raise TypeError(
                    f"Both arguments should be of type Variable \
                                not {type(a).__name__} and {type(b).__name__}"
                )

    def _check_args_len(
        self, args: tuple[variable.Variable, ...], max_args_length: int
    ) -> None:
        """Check the length of the amount of arguments provides. In theory this
        should not be necessary, but just in case...

        Args:
            args (tuple[variable.Variable,...]): The arguments
            max_args_length (int): Amount of arguments there should be

        Raises:
            ValueError: _description_
        """
        if len(args) != max_args_length:
            raise ValueError(
                f"{self.__class__.__name__} expected\
                             {max_args_length} argument(s), not {len(args)}"
            )

    @abstractmethod
    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Every operation should be callable

        Returns:
            variable.Variable: A new variable with an operation applied
        """

    @property
    def parents(self) -> tuple[variable.Variable, variable.Variable] | None:
        """Getter for parents

        Returns:
            tuple[variable.Variable, variable.Variable] | None: When defined return
        """
        return self._parents


class Add(Operation):
    """Add two arrays together, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and perform addiction on the two arrays

        Returns:
            variable.Variable: Variable with the added array
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)
        a, b = self._parents = args
        out_data = a.data + b.data
        return variable.Variable(np.array(out_data), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of addition
        dOut/da = (dOut/d(a + b) * (d(a + b)/da) = incoming_gradient * 1
        dOut/da = (dOut/d(a + b) * (d(a + b)/db) = incoming_gradient * 1

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_add, b_add = self._parents

        gradient_a = incoming_gradient
        gradient_b = incoming_gradient

        a_add.backward(incoming_gradient=gradient_a)
        b_add.backward(incoming_gradient=gradient_b)


class Multiply(Operation):
    """Multiply two arrays, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and perform multiplication

        Returns:
            variable.Variable: Variable with the multiplied array
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)
        a, b = self._parents = args

        product = np.array(a.data * b.data)
        return variable.Variable(np.array(product), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of multiplication
        dOut/da = (dOut/d(a * b)) * (d(a * b)/da) = incoming_gradient * b
        dOut/db = (dOut/d(a * b)) * (d(a * b)/db) = incoming_gradient * a
        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_mult, b_mult = self._parents

        gradient_a = incoming_gradient * b_mult.data
        gradient_b = incoming_gradient * a_mult.data

        a_mult.backward(incoming_gradient=gradient_a)
        b_mult.backward(incoming_gradient=gradient_b)


class Sine(Operation):
    """Apply sin(x) to every element in the array, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and apply sin to every element in Variable data

        Returns:
            variable.Variable: Variable with sin applied to every element
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)
        a = args[0]
        self._parents = args

        sine = np.sin(a.data)
        return variable.Variable(sine, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of sine
        dOut/da = (dOut/dsin(a)) * (dsin(a)/dx) = incoming_gradient * cos(a)

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_sin = self._parents[0]
        gradient_a = incoming_gradient * np.cos(a_sin.data)
        a_sin.backward(incoming_gradient=gradient_a)


class Sum(Operation):
    """Sum all elements in an array to get a scalar, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and sum over all elements in data

        Returns:
            variable.Variable: Variable with scalar value
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)
        a = args[0]
        self._parents = args

        scalar_a = np.array(a.data.sum())
        return variable.Variable(np.array(scalar_a), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of sum
        This works the same as addition, the derivative is 1,
        we did not use broadcast because it was not handling dtypes the way
        we wanted it. Since we check the shape of all data passed in our
        method should work fine.

        Args:
            incoming_gradient (np.ndarray): The incoming data
        """
        a_sum = self._parents[0]
        grad = np.array(np.squeeze(incoming_gradient))
        if grad.ndim != 0:
            raise ValueError(
                f"Sum.backward() expects scalar, got shape {incoming_gradient.shape}"
            )
        gradient_a = np.ones_like(a_sum.data) * grad

        a_sum.backward(incoming_gradient=gradient_a)


class Log(Operation):
    """Apply log(x) (ln) to every element in the array, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and apply log to every element of the data of the Variable

        Returns:
            variable.Variable: Variable with ln(x) applied to every element
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)
        a = args[0]
        self._parents = args

        pos_data = np.maximum(a.data, 1e-8)
        natural_log = np.log(pos_data)
        return variable.Variable(natural_log, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of ln
        dOut/da = (dOut/dlog(a)) * (dlog(a)/da) = incoming_gradient * (1/a)

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_log = self._parents[0]
        # ln(x) is only defined for x>0
        safe_data = np.maximum(a_log.data, 1e-8)
        gradient_a = incoming_gradient * (1 / safe_data)
        a_log.backward(incoming_gradient=gradient_a)


class Exp(Operation):
    """Apply exp(x) to every element in the array, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and apply exponentation to every element in the data of Variable

        Returns:
            variable.Variable: Variable with exp(x) apply to every element
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)
        a = args[0]
        self._parents = args

        exponential_data = np.exp(a.data)
        return variable.Variable(exponential_data, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of exp
        dOut/da = (dOut/dexp(a)) * (dexp(a)/da) = incoming_gradient * exp(a)

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_exp = self._parents[0]
        gradient_a = incoming_gradient * np.exp(a_exp.data)
        a_exp.backward(incoming_gradient=gradient_a)


class Divide(Operation):
    """Divide two arrays, also perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """When class operation is called accept arguments of type Variable
        and divide the two Variables

        Returns:
            variable.Variable: Variable with the divided array
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)
        a, b = self._parents = args

        non_zero_b = b.data.copy()
        non_zero_b[non_zero_b == 0] = 1e-8
        divided_data = a.data / non_zero_b
        return variable.Variable(divided_data, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of divide
        dOut/da = (dOut/d(a/b)) * (d(a/b)/da) = incoming_gradient * 1/b
        dOut/db = (dOut/d(a/b)) * (d(a/b)/db) = -incoming_gradient * (a/b^2)

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_div, b_div = self._parents
        non_zero_b = b_div.data.copy()
        non_zero_b[non_zero_b == 0] = 1e-8
        gradient_a = incoming_gradient * (1 / non_zero_b)
        gradient_b = -incoming_gradient * (a_div.data / (non_zero_b**2))
        a_div.backward(incoming_gradient=gradient_a)
        b_div.backward(incoming_gradient=gradient_b)


class MatrixMult(Operation):
    """Perform matrix multiplication on two Variables"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Perform matrix multiplication of two variables, since
        matrix multiplication is somewhat involved when it comes to
        dimensionality, we have extra checks to make sure matrix
        multiplication is possible

        Raises:
            ValueError: Both arrays should be matrices
            ValueError: The dimensions should allow for multiplication

        Returns:
            variable.Variable: A variable with the multiplied array
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)

        a, b = self._parents = args
        matrix_dim = 2

        if a.data.ndim != matrix_dim or b.data.ndim != matrix_dim:
            err_msg = "Matrix multiplication requires 2D arrays"
            raise ValueError(err_msg)
        if a.data.shape[1] != b.data.shape[0]:
            raise ValueError(
                f"Matrix shape mismatch: \
                              {a.data.shape} @ {b.data.shape} "
            )

        matrix_mult = np.matmul(a.data, b.data)
        return variable.Variable(matrix_mult, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of matrix multiplication
        dOut/da = (dOut/d(a@b)) * b.T = incoming_gradient @ b.T
        dOut/db = a.T * (dOut/d(a@b)) = a.T @ incoming_gradient

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_matmul, b_matmul = self._parents
        if incoming_gradient.shape != (a_matmul.data.shape[0], b_matmul.data.shape[1]):
            err_msg = "Incoming gradient has wrong shape"
            raise ValueError(err_msg)

        gradient_a = np.matmul(incoming_gradient, b_matmul.data.T)
        gradient_b = np.matmul(a_matmul.data.T, incoming_gradient)
        a_matmul.backward(incoming_gradient=gradient_a)
        b_matmul.backward(incoming_gradient=gradient_b)


class Absolute(Operation):
    """Take the absolute values of every element, and perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Take the absolute values of every element in the data of the Variable

        Returns:
            variable.Variable: A Variable with an array of absolute values
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)

        a = args[0]
        self._parents = args

        absolute_values = np.abs(a.data)
        return variable.Variable(absolute_values, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of the absolute value
        dOut/da = dOut/d|a| * d|a|/da = incoming_gradient * sign(b)
        Technically the derivative for x = 0 is not defined, but we just put 0
        there.
        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_abs = self._parents[0]

        gradient_a = incoming_gradient * np.sign(a_abs.data)
        a_abs.backward(incoming_gradient=gradient_a)


class Transpose(Operation):
    """ "Transpose the data in a Variable, and perform backward pass"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Take the transpose of the data in a Variable

        Returns:
            variable.Variable: A transposes Variable
        """
        self._check_args_len(args, max_args_length=1)
        self._check_instances(args)

        a = args[0]
        self._parents = args

        transpose_a = a.data.T
        return variable.Variable(transpose_a, operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """The backward pass of transpose
        We just take the transpose of the incoming gradient as backward pass

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_transpose = self._parents[0]

        gradient_a = incoming_gradient.T
        a_transpose.backward(incoming_gradient=gradient_a)


class Pow(Operation):
    """Raise every element in the array to some integer n"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Raise every element in the Variable's data to some
        integer n.

        Raises:
            ValueError: The integer n should b a scalar, i.e. we can't raise
            an array to more than one value

        Returns:
            variable.Variable: Variable raised to the n-th power
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)
        a, b = self._parents = args
        if b.is_scalar:
            n = b.data
        else:
            err_msg = "Second argument to Pow should be a scalar"
            raise ValueError(err_msg)

        a_power_n = a.data**n
        return variable.Variable(np.array(a_power_n), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of power function

        dOut/da = dOut/d(a^n) * (d(a^n)/a) = incoming_gradient * n *(a^(n-1))

        Args:
            incoming_gradient (np.ndarray): _description_
        """
        a_pow, n = self._parents

        gradient_a = incoming_gradient * n.data * (a_pow.data ** (n.data - 1))
        a_pow.backward(incoming_gradient=gradient_a)


class Minus(Operation):
    """Subtract two Variables from eachother"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Subtract two Variables form eachother

        Returns:
            variable.Variable: A Variable containing the subtracted array
        """
        self._check_args_len(args, max_args_length=2)
        self._check_instances(args)
        a, b = self._parents = args

        a_minus_b = a.data - b.data
        return variable.Variable(np.array(a_minus_b), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Backward pass of Minus operator
        dOut/da = dOut/d(a-b) / d(a-b)/da = 1
        dOut/db = dOut/d(a-b) / d(a-b)/db = -1

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_minus, b_minus = self._parents
        gradient_a = incoming_gradient
        gradient_b = -1 * incoming_gradient
        a_minus.backward(incoming_gradient=gradient_a)
        b_minus.backward(incoming_gradient=gradient_b)


class Sigmoid(Operation):
    """Perform a sigmoid on every element in the array"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Perform a sigmoid on every element in a Variables data

        Returns:
            variable.Variable: A Variable with Sigmoid applies to every element
        """
        a = args[0]
        self._parents = args

        sigmoid_a = 1 / (1 + np.exp(-a.data))
        return variable.Variable(np.array(sigmoid_a), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Perform a backward pass of a sigmoid
        dOut/da = dOut/dSigmoid(a) * dSigmoid(a)/da =
                    incoming_gradient * (sigmoid(a) * (1 - sigmoid(a)))

        Args:
            incoming_gradient (np.ndarray): The incoming gradient
        """
        a_sigmoid = self._parents[0]

        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        sigm = sigmoid(a_sigmoid.data)

        gradient_a = incoming_gradient * (sigm * (1 - sigm))
        a_sigmoid.backward(incoming_gradient=gradient_a)


class UnaryMinus(Operation):
    """ "Perform Unary minus on a Variables"""

    def __call__(self, *args: variable.Variable) -> variable.Variable:
        """Perfrom a unary minus an the data of a Variable, so just
        times the array by -1

        Returns:
            variable.Variable: Variable with its signs flipped
        """
        a = args[0]
        self._parents = args

        umin_a = -a.data
        return variable.Variable(np.array(umin_a), operation=self)

    def backward(self, incoming_gradient: np.ndarray) -> None:
        """Perform backward pass of unary minus
        dOut/da = dOut/d(-a) * d(-a)/da = incoming_gradient * -1

        Args:
            incoming_gradient (np.ndarray): _description_
        """
        a_umin = self._parents[0]
        gradient_a = -incoming_gradient
        a_umin.backward(gradient_a)
