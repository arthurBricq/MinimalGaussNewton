import numpy as np
from numpy._typing import NDArray

"""
Represents the model to fit using a gauss newton method.

To fit a model, three functions must be overriden
- `jacobian`: computes the jacobian of the residuals
    at the current evaluation point.
- `residuals`: computes the vector of residuals
    at the current evaluation point.
- `initial_condition`: returns the initial condition where to
    start the iterative method.
"""
class GenericFunctor:
    """
    Constructor will be inherited by subclasses

    INPUTS
    - n: number of dimensions of the model to fit
    - m: number of datapoints
    """
    def __init__(self, n: int, m: int):
        self._n = n;
        self._m = m;

    def residual(self, x) -> np.ndarray:
        return np.array([x]);

    def jacobian(self, x) -> np.ndarray:
        return np.array([x]);

    def initial_condition(self) -> np.ndarray:
        return np.zeros(self._n);

    def set_data(self, data: np.ndarray):
        self._data = data;

    def n(self) -> int:
        return self._n;

    def m(self) -> int:
        return self._m;

"""
Guass Newton Solver

Provides the solution x so that the following cost is minimized

    cost(x) = sum ( r_i(x) ^ 2 ) over datapoints
    with r_i the i-th residuals

Must be provided with a functor to evaluate:
- r_i the residual
- J_i, the jacobian of the residual
- x_0, the initial condition
"""
class GaussNewtonSolver:
    MAX_ITER = 1000;
    TOL = 0.000001;

    def __init__(self, functor: GenericFunctor):
        self._functor = functor;

    """
    Solve the gauss-newton problem with the provided data.

    INPUTS
    - data: input data of 2D points. Must have the shape (M, 2).
    """
    def solve(self, data: np.ndarray):
        self._functor.set_data(data)

        x = self._functor.initial_condition();
        n_iter = 0;

        should_continue = True;
        while should_continue:
            if n_iter > self.MAX_ITER:
                should_continue = False;

            # Compute jacobian and residuals
            J = self._functor.jacobian(x);
            res = self._functor.residual(x);

            # Compute A = J^T @ J
            A = J.transpose() @ J;

            # Compute b = J^T @ res;
            b = J.transpose() @ res;

            # Invert A
            A_inv= np.linalg.inv(A)

            # Compute increment
            delta = A_inv @ b;

            # Convergence criteria
            for i in range(self._functor.n()):
                if abs(delta[i]) < self.TOL:
                    should_continue = False;

            # Apply the increment
            print("----------------------")
            print(f"iter = {n_iter}, x = {x}, delta = {delta}")
            x -= delta;
            n_iter += 1;

        # Algorithm is finished
        print("Convergence finished");
        print("iter = ", n_iter);
        print("Solution: ", x);
        return x;

