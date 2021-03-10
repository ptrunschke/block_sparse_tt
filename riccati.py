import numpy as np
from scipy import linalg as la


def riccati_matrices(_n, _nu=1.0, _lambda=0.1, _boundary='Neumann'):
    """
    Builds the Riccati matrices for the optimization problem
        minimize integral(yQy + uRu, dt)
        subject to y' = Ay + Bu
    Here A is the 1-dimensional diffusion operator and Bu is a uniform forcing of size u on the interval [-0.4, 0.4].
    The solution Pi of the Riccati equation represents the value function as v(x) = x Pi x.

    Parameters
    ----------
    _n : int
        spatial discretization points that are considered
    _nu : float
        diffusion constant (default: 1.0)
    _lambda : float
        cost parameter (default: 0.1)
    _boundary : 'Dirichlet' or 'Neumann'
        the boundary condition to use (default: 'Neumann')

    Author: Leon Sallandt
    """
    assert _boundary in ['Dirichlet', 'Neumann']
    domain = (-1, 1)
    s = np.linspace(*domain, num=_n)    # gridpoints
    A = -2*np.eye(_n) + np.eye(_n, k=1) + np.eye(_n, k=-1)
    Q = np.eye(_n)
    if _boundary == 'Dirichlet':
        h = (domain[1]-domain[0]) / (_n+1)
    elif _boundary == 'Neumann':
        h = (domain[1]-domain[0]) / (_n-1)  # step size in space
        A[[0,-1], [1,-2]] *= 2
        Q[[0,-1], [0,-1]] /= 2
    A *= _nu / h**2
    Q *= h
    B = ((-0.4 < s) & (s < 0.4)).astype(float).reshape(-1,1)
    R = _lambda * np.eye(1)
    Pi = la.solve_continuous_are(A, B, Q, R)
    return A, B, Q, R, Pi
