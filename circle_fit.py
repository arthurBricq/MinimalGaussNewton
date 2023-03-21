import numpy as np
import matplotlib.pyplot as plt;

from gauss_netwon import GenericFunctor, GaussNewtonSolver

"""
A circle model to be fitted.
"""
class CircleFunctor(GenericFunctor):

    def residual(self, x) -> np.ndarray:
        residuals = np.zeros(self._m);
        for i in range(self._m):
            residuals[i] = np.sqrt((self._data[i, 0] - x[0]) ** 2 + (self._data[i, 1] - x[1]) ** 2) - x[2];
        return residuals;

    def jacobian(self, x) -> np.ndarray:
        jac = np.zeros([self._m, self._n]);
        for i in range(self._m):
            denominator = np.sqrt((self._data[i, 0] - x[0]) ** 2 + (self._data[i, 1] - x[1]) ** 2);
            jac[i,0] = (x[0] - self._data[i, 0]) / denominator;
            jac[i,1] = (x[1] - self._data[i, 1]) / denominator;
            jac[i,2] = -1;
        return jac;

    def initial_condition(self) -> np.ndarray:
        return np.array([0, 0, 1.0]);

"""
Function to generate points from a circle with noise
"""
def generate_circle_data(cx: float, cy: float, r: float, n: int, noise: float) -> np.ndarray:
    # Generate angles
    angles = np.linspace(0, 2*np.pi, n);

    # For each angle, generate a point
    x = r * np.cos(angles) + cx + np.random.randn(n) * noise
    y = r * np.sin(angles) + cy + np.random.randn(n) * noise
    return np.array([x,y]).transpose()


"""
Function to draw the results of the circle fit.
"""
def draw(points, c: np.ndarray, gt: np.ndarray):
    fig, axs = plt.subplots(1,1)
    axs.plot(points[:,0], points[:, 1], 'x')
    axs.axis('equal')
    axs.grid(True)
    circle1 = plt.Circle((c[0], c[1]), c[2], color='k', fill=False)
    circle2 = plt.Circle((gt[0], gt[1]), gt[2], color='r', fill=False)
    axs.add_patch(circle1)
    axs.add_patch(circle2)
    plt.savefig("/mnt/c/Users/abricq/fig.png")
    plt.show()

if __name__ == "__main__":
    circle = np.array([0.1, 0.1, 0.8])
    data = generate_circle_data(
            circle[0],
            circle[1],
            circle[2],
            50,
            0.05
            );
    circle_model = CircleFunctor(3, data.shape[0])
    solver = GaussNewtonSolver(circle_model)
    result = solver.solve(data)
    draw(data, result, circle);

