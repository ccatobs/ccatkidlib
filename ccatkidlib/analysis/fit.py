import numpy as np
from numba import njit

@njit(parallel=True, cache=True)
def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    '''Calculate 2D linear regression for given x and y np.arrays by solving matrix equation

    Args:
        x (np.ndarray): Array of independent variable data
        y (np.ndarray): Array of dependent variable data
    Returns:
        tuple[int, int]: Slope, intercept
    '''
    A = np.ones((2, x.size))
    A[0] = x

    m, b = np.linalg.solve(A @ A.T, A @ y.T)
    return m, b