from __future__ import annotations

from typing import Union

import numpy as np

from . import operation as operations

NumOrVariable = Union[int, float, "Variable"]


class Variable:
    """
    Variable class:
    This class contains the data of the variable, as well was the gradient and the
    operations. Functionallity for a backward pass is provided as well as serveral
    operations.
    """

    def __init__(
        self,
        data: np.ndarray,
        operation: operations.Operation | None = None,
        *,
        is_scalar: bool = False,
    ) -> None:
        """Initialize variable

        Args:
            data (np.ndarray): The data of said variable
            operation (Optional[operations.Operation], optional): When we do
            an operation we want to set an operation, but not by default.
            Defaults to None.
            is_scalar (bool): True if the argument is a scalar, because we
            dont need to perform a backward pass on a scalar, but we need
            a scalar Variable to perform a forward pass.
        """
        self._data: np.ndarray = data
        self._operation: operations.Operation | None = operation
        self._gradient: np.ndarray = np.zeros_like(self._data)
        self._is_scalar = is_scalar

    def __add__(self, other: Variable) -> Variable:
        """Add two arrays of the same shape

        Args:
            other (Variable): Array to be added to self.data

        Returns:
            Variable: Added array, with Add as operator
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)
        
        return operations.Add()(self, checked_other)
    
    def __radd__(self, other: Variable) -> Variable:
        """Add two arrays of the same shape

        Args:
            other (Variable): Array to be added to self.data

        Returns:
            Variable: Added array, with Add as operator
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)
        
        return operations.Add()(self, checked_other)

    def __mul__(self, other: NumOrVariable) -> Variable:
        """Multiply two arrays of the same shape, or an array and scalar

        Args:
            other (Variable): The array to be multiplied with self.data

        Returns:
            Variable: Multiplied array, with Multiply as operator
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)

        return operations.Multiply()(self, checked_other)

    def __rmul__(self, other: NumOrVariable) -> Variable:
        """Multiply an array and a scalar from right to left

        Args:
            other (Variable): The array to be multiplied with self.data

        Returns:
            Variable: Array multiplied with scalar
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)

        return operations.Multiply()(checked_other, self)

    def __truediv__(self, other: Variable) -> Variable:
        """Divide two arrays by eachother

        Args:
            other (Variable): Array to divide by

        Returns:
            Variable: Variable with divided array and Divide as Operator
        """
        self._check_shape(self, other, "divide")
        return operations.Divide()(self, other)

    def __pow__(self, other: int) -> Variable:
        """Raise array to the power other

        Args:
            other (int): The power to raise the matrix to

        Raises:
            TypeError: Only raising to the power of an integer is implemented

        Returns:
            Variable: Variable with every element raised to other, and Pow as Operator
        """
        if not isinstance(other, int):
            err_msg = "You can only raise to the power of an int"
            raise TypeError(err_msg)
        var_other = Variable(np.array(other), is_scalar=True)
        return operations.Pow()(self, var_other)

    def __sub__(self, other: NumOrVariable) -> Variable:
        """Subtract two arrays

        Args:
            other (Variable): Array being substracted

        Returns:
            Variable: Subtracted array with operator Minus
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)
        return operations.Minus()(self, checked_other)

    def __rsub__(self, other: NumOrVariable) -> Variable:
        """Subtract two arrays from right to left

        Args:
            other (Variable): Array being substracted

        Returns:
            Variable: Subtracted array with operator Minus
        """
        checked_other: Variable = self._check_if_variable(other)
        if checked_other is NotImplemented:
            return NotImplemented

        self._check_if_scalar(checked_other)
        return operations.Minus()(checked_other, self)

    def __abs__(self) -> Variable:
        """Take absolute value of every element in the array

        Returns:
            Variable: Array with only absolute values with Absolute as operator
        """
        return operations.Absolute()(self)

    def __neg__(self) -> Variable:
        """Take the unary minus of array

        Returns:
            Variable: Array with all values having their sign flipped
            and UnaryMinus as operator
        """
        return operations.UnaryMinus()(self)

    def sigmoid(self) -> Variable:
        """Get the sigmoid of every element in the array

        Returns:
            Variable: Array with sigmoid applied to every element
            with Sigmoid as operator
        """
        return operations.Sigmoid()(self)

    def matmul(self, other: Variable) -> Variable:
        """Multiply two matrices (if possible)

        Args:
            other (Variable): Matrix to multiply with

        Returns:
            Variable: Multiplied matrix with MatrixMult as Operator
        """
        return operations.MatrixMult()(self, other)

    def transpose(self) -> Variable:
        """Transpose an array

        Returns:
            Variable: Transposed array with Transpose as Operator
        """
        return operations.Transpose()(self)

    def exp(self) -> Variable:
        """Apply e^x for every x in in the array

        Returns:
            Variable: Variable with exp taken of every element and Exp as operator
        """
        return operations.Exp()(self)

    def log(self) -> Variable:
        """Take the natural logarithm of every element in the array

        Returns:
            Variable: Variable with log taken of every element and Log as operator
        """
        return operations.Log()(self)

    def sin(self) -> Variable:
        """Take the sine of every element in the array

        Returns:
            Variable: Variable with sine taken of every element and Sine as opperator
        """
        return operations.Sine()(self)

    def sum(self) -> Variable:
        """Reduce array to scalar, by summing over every element in the array

        Returns:
            Variable: Variable with scalar value and Sum as operator
        """
        return operations.Sum()(self)

    def delete_gradient(self) -> None:
        """Recursive function for deleting all gradients, from terminal to root"""
        self.gradient = np.zeros_like(self.data)
        if self.operation is None:
            return

        parents = self.operation.parents
        if parents is None:
            return

        unary_operation_length = 1
        binary_operation_length = 2

        if len(parents) == unary_operation_length:
            a = parents[0]
            a.delete_gradient()

        if len(parents) == binary_operation_length:
            a, b = parents
            a.delete_gradient()
            b.delete_gradient()

    def backward(self, incoming_gradient: np.ndarray | None = None) -> None:
        """Perform a backward pass

        Args:
            incoming_gradient (np.ndarray | None, optional): The incoming gradient
            of another variable to perform the backward pass with. Defaults to None,
            so you can just call backward(), without specifying a gradient.

        Returns:
            None: If we hit an initial node we return nothing, just stop the
            backward pass.
        """
        if self._is_scalar:
            return

        if incoming_gradient is None:
            incoming = np.ones_like(self._data)
        else:
            incoming = incoming_gradient

        # Accumalate gradients
        self._gradient += incoming

        # Hit initial node
        if self.operation is None:
            return

        self.operation.backward(incoming)

    def __str__(self) -> str:
        """String representation of Variable when called by print()

        Returns:
            str: Useful data, like Data, Operation, Gradient
        """
        return f"Data: {self.data}\nOperation: {self.operation}\
            \nGradient: {self.gradient}\n"

    def _read_only_array(self, array: np.ndarray) -> np.ndarray:
        """Return read-only version of array, does not affect the array passed
        in as argument.

        Args:
            array (np.ndarray): array to convert

        Returns:
            np.ndarray: Read-only version of array (does not affect argument array)
        """
        safe_view = array.view()
        safe_view.flags.writeable = False
        return safe_view

    def _check_if_scalar(self, checked_other: Variable) -> None:
        """Check if an array is a scalar, because than a shape mismatch should
        be allowed, otherwise it is not

        Args:
            checked_other (Variable): The other array or scalar to compare

        Raises:
            ValueError: If the shapes mismatch raise a ValueError
        """
        try:
            self._check_shape(self, checked_other, "multiply")
        except ValueError as e:
            if self.is_scalar or checked_other.is_scalar:
                pass
            else:
                raise ValueError(
                    f"Cannot {operations} variables of shapes: \
                        {self.data.shape} and {checked_other.data.shape}"
                ) from e

    def _check_shape(self, a: Variable, b: Variable, operation: str) -> None:
        """Check if the shape of the two arrays are compatible

        Args:
            a (Variable): First array
            b (Variable): Second array
            operation (str): Operation you are carrying out

        Raises:
            ValueError: If the two arrays do not have the same shape
        """
        if a.data.shape != b.data.shape:
            raise ValueError(
                f"Cannot {operation} variables of shapes:\
                    {a.data.shape} and {b.data.shape}"
            )

    def _check_if_variable(self, obj: object) -> Variable:
        """Check if on object is of type Variable

        Args:
            obj (object): The object we want to check

        Returns:
            Variable: Should return variable if scalar, or NotImplemented
        """
        if isinstance(obj, Variable):
            return obj
        if isinstance(obj, (float, int, np.ndarray)):
            return Variable(np.array(obj, dtype=float), is_scalar=True)
        return NotImplemented

    @property
    def data(self) -> np.ndarray:
        """Getter for _data

        Returns:
            np.ndarray: returns read-only version of data
        """
        return self._read_only_array(self._data)

    @property
    def operation(self) -> operations.Operation | None:
        """Getter for _operation

        Returns:
            operations.Operation | None: If an operation is defined return it
        """
        return self._operation

    @operation.setter
    def operation(self, new_operation: operations.Operation) -> None:
        """Setter for _operation

        Args:
            new_operation (operations.Operation): New operation

        Raises:
            TypeError: If the new operation is not of type Operation, we raise an error
        """
        if not isinstance(new_operation, operations.Operation):
            raise TypeError(
                f"Expected type: Operation, got: {type(new_operation).__name__}"
            )
        self._operation = new_operation

    @property
    def gradient(self) -> np.ndarray:
        """Getter for _gradient

        Returns:
            np.ndarray: return a read-only array of the gradient
        """
        return self._read_only_array(self._gradient)

    @gradient.setter
    def gradient(self, gradient: np.ndarray) -> None:
        """Setter for _gradient

        Args:
            gradient (np.ndarray): The gradient we want to set

        Raises:
            TypeError: If the gradient is not of type np.ndarray, we raise an error
        """
        if not isinstance(gradient, np.ndarray):
            raise TypeError(
                f"Gradient should be a NumPy array, not {type(gradient).__name__}"
            )
        self._gradient = gradient

    @property
    def is_scalar(self) -> bool:
        """Getter for _is_scalar

        Returns:
            bool: True if Variabel is a scalar
        """
        return self._is_scalar
